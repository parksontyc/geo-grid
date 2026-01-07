from __future__ import annotations

import math
import gc
from pathlib import Path
from typing import Union, Tuple, Optional, Literal, Generator, List

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt

import warnings


# ================= 基礎工具 (保持不變) =================
def read_vector(path): return gpd.read_file(path)

def ensure_crs(gdf, crs): return gdf.set_crs(crs) if gdf.crs is None else gdf

def project_to_epsg(gdf, epsg=3826): return gdf.to_crs(epsg=epsg)

def aligned_bounds(bounds, cell_size):
    minx, miny, maxx, maxy = bounds
    minx = math.floor(minx / cell_size) * cell_size
    miny = math.floor(miny / cell_size) * cell_size
    maxx = math.ceil(maxx / cell_size) * cell_size
    maxy = math.ceil(maxy / cell_size) * cell_size
    return float(minx), float(miny), float(maxx), float(maxy)

def make_grid(bounds, cell_size, crs, align=True):
    if align: bounds = aligned_bounds(bounds, cell_size)
    minx, miny, maxx, maxy = bounds
    xs = np.arange(minx, maxx, cell_size)
    ys = np.arange(miny, maxy, cell_size)
    cells = [box(x, y, x + cell_size, y + cell_size) for x in xs for y in ys]
    return gpd.GeoDataFrame({"geometry": cells}, crs=crs)

def apply_clip_mode(grid, mask_gdf, clip_mode="filter"):
    if clip_mode == "none": return grid
    # 優化：這裡不需要 dissolve，因為傳進來的已經是單一 Polygon (explode 後)
    # mask = mask_gdf.dissolve() 
    
    if clip_mode == "clip":
        return gpd.clip(grid, mask_gdf)
    elif clip_mode == "filter":
        joined = grid.sjoin(mask_gdf, how="inner", predicate="intersects")
        if "index_right" in joined.columns: joined = joined.drop(columns=["index_right"])
        return joined.reset_index(drop=True)
    return grid

def add_grid_id(grid, origin, cell_size, scheme="xy"):
    minx0, miny0 = origin
    b = grid.geometry.bounds
    grid_col = np.floor((b["minx"].to_numpy() - minx0) / cell_size).astype("int64")
    grid_row = np.floor((b["miny"].to_numpy() - miny0) / cell_size).astype("int64")
    out = grid.copy()
    out["grid_col"] = grid_col
    out["grid_row"] = grid_row
    
    if scheme == "xy":
        # 防止溢位，改用字串或更安全的 hash 方式，這裡維持簡單邏輯但轉 int64
        out["grid_id"] = (out["grid_col"] * 100000 + out["grid_row"]).astype("int64")
    else:
        out["grid_id"] = "C" + out["grid_col"].astype(str) + "_R" + out["grid_row"].astype(str)
    return out


# ================= 核心函式 (修復 MemoryError 版) =================
def build_projected_grid_generator(
    path: Union[str, Path],
    target_epsg: int = 3826,
    cell_size: float = 100.0,
    clip_mode: str = "filter",
    id_scheme: str = "xy",
) -> Tuple[gpd.GeoDataFrame, Generator[Tuple[str, gpd.GeoDataFrame], None, None]]:
    
    # 1. 前置處理
    print("正在讀取並處理基礎圖資...")
    gdf = read_vector(path)
    if gdf.crs is None: gdf.set_crs(epsg=3826, inplace=True)
    gdf_proj = project_to_epsg(gdf, epsg=target_epsg)

    # 2. 計算全域原點
    global_origin = aligned_bounds(tuple(map(float, gdf_proj.total_bounds)), cell_size)[:2]
    
    # 3. 定義內部的 Generator 函式
    def _grid_generator():
        county_col = next((col for col in ['COUNTYNAME', 'county', 'CNAME'] if col in gdf_proj.columns), None)
        if not county_col:
            print("未找到縣市欄位，無法分批處理。")
            return

        counties = gdf_proj[county_col].unique()
        print(f"啟動分批產生器，共 {len(counties)} 個縣市...")

        for county in counties:
            # 取出該縣市 (這是一個 MultiPolygon，可能包含本島+離島)
            county_gdf = gdf_proj[gdf_proj[county_col] == county]
            
            # 【關鍵修復 Step 1】將縣市炸開成獨立的 Polygon
            # 這樣高雄本島和東沙島會變成兩筆獨立資料，不會連在一起畫大框框
            exploded_parts = county_gdf.explode(index_parts=True)
            
            county_grids = []
            
            # 【關鍵修復 Step 2】針對每個「碎片 (島嶼/區塊)」單獨生成網格
            for _, part in exploded_parts.iterrows():
                # 轉回 GeoDataFrame 以利處理
                part_gdf = gpd.GeoDataFrame([part], crs=gdf_proj.crs)
                
                # 只針對這個小島的範圍畫網格 (範圍極小，不會 OOM)
                part_bounds = aligned_bounds(tuple(map(float, part_gdf.total_bounds)), cell_size)
                
                # 這裡的 make_grid 現在只會產生幾千/幾萬個格子，非常安全
                grid_part = make_grid(part_bounds, cell_size, gdf_proj.crs, align=False)
                grid_part = apply_clip_mode(grid_part, part_gdf, clip_mode=clip_mode)
                
                if not grid_part.empty:
                    county_grids.append(grid_part)
            
            # 【關鍵修復 Step 3】將所有碎片網格合併回「一個縣市網格」
            if county_grids:
                full_county_grid = pd.concat(county_grids, ignore_index=True)
                
                # 加上 ID (依然使用全域原點，確保 ID 連續性)
                full_county_grid = add_grid_id(full_county_grid, global_origin, cell_size, id_scheme)
                full_county_grid = full_county_grid[['grid_id', 'grid_col', 'grid_row', 'geometry']]
                
                # 交貨
                yield county, full_county_grid
            
            # 清理
            del county_grids, exploded_parts, county_gdf
            gc.collect()

    return gdf_proj, _grid_generator()


#============網格建置流程============#
def generate_joined_grids(
    shp_path: str,
    output_dir: str,
    cell_size: int = 300,
    county_col: str = 'COUNTYNAME'
):
    """
    生成網格並與行政區進行空間連結（使用中心點法避免邊界重複）。
    
    Args:
        shp_path (str): 縣市界線 SHP 檔案路徑。
        output_dir (str): 輸出 Parquet 檔案的資料夾路徑。
        cell_size (int): 網格大小 (公尺)，預設 300。
        county_col (str): SHP 檔中縣市名稱的欄位，預設 'COUNTYNAME'。
    """
    
    # 0. 環境設定
    warnings.filterwarnings('ignore')
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== 開始執行任務: 網格大小 {cell_size}m ===")
    print(f"來源: {shp_path}")
    print(f"輸出: {out_dir}")

    # 1. 呼叫產生器
    # (假設 build_projected_grid_generator 已定義在您的環境中)
    print("正在初始化網格產生器...")
    gdf_tm2, grid_gen = build_projected_grid_generator(
        shp_path,
        cell_size=cell_size,
        clip_mode="filter",
        id_scheme="xy"
    )

    # 儲存基礎行政區資料 (作為對照用)
    gdf_out_path = out_dir / "gdf_tm2_base.parquet"
    gdf_tm2.to_parquet(gdf_out_path)
    print(f"行政區底圖準備完成，共 {len(gdf_tm2)} 筆。")

    # 2. 執行迴圈與空間連結
    print("開始執行空間連結 (Spatial Join - Centroid Method)...")
    
    count = 0
    for county_name, local_grid in grid_gen:
        count += 1
        print(f"-> [{count}] 處理中: {county_name} | 原始網格數: {len(local_grid):,}")

        # --- A. 準備行政區資料 ---
        # 篩選出該縣市的行政區
        if county_col not in gdf_tm2.columns:
            raise ValueError(f"欄位錯誤: SHP 中找不到 '{county_col}' 欄位，請檢查來源檔案。")
            
        local_admin = gdf_tm2[gdf_tm2[county_col] == county_name].copy()
        
        # --- B. 中心點法空間連結 ---
        # 1. 備份原始方格幾何 (Polygon)
        local_grid['poly_geom'] = local_grid.geometry
        
        # 2. 切換為中心點 (Point)
        local_grid = local_grid.set_geometry(local_grid.centroid)
        
        # 3. 執行 sjoin (Point within Polygon)
        joined_grid = gpd.sjoin(
            local_grid, 
            local_admin, 
            how="inner", 
            predicate="within"
        )
        
        # --- C. 清理與復原 ---
        # 1. 移除 sjoin 產生的索引
        if 'index_right' in joined_grid.columns:
            joined_grid = joined_grid.drop(columns=['index_right'])

        # 2. 移除目前的幾何欄位 (Point) 以避免改名衝突
        current_geo_col = joined_grid.geometry.name
        joined_grid = joined_grid.drop(columns=[current_geo_col])
        
        # 3. 將備份的方格 (poly_geom) 改名回 'geometry' 並重新指定
        joined_grid = joined_grid.rename(columns={'poly_geom': 'geometry'}).set_geometry('geometry')

        print(f"   合併後: {len(joined_grid):,} (已歸屬行政區)")

        # --- D. 存檔 ---
        # 檔名包含網格大小，方便識別
        out_filename = f"Grid_{county_name}_{cell_size}m_joined.parquet"
        joined_grid.to_parquet(out_dir / out_filename)
        
        # 釋放記憶體
        del local_grid, local_admin, joined_grid

    print(f"=== 全部作業完成！共處理 {count} 個縣市 ===")


#=====[[繪圖檢視]]=====#
def plot_local_grid_by_targets(
    town_gdf: gpd.GeoDataFrame,
    grid_gdf: gpd.GeoDataFrame,
    targets: List[Tuple[str, Optional[str]]], # 允許第二個元素為 None
    county_col: str = "COUNTYNAME",
    town_col: str = "TOWNNAME",
    figsize: Tuple[int, int] = (10, 12),
    town_edgecolor: str = "black",
    town_linewidth: float = 2.0,
    grid_edgecolor: str = "red",
    grid_linewidth: float = 0.5,
    show_labels: bool = True,
    label_fontsize: int = 12,
    label_color: str = "black",
    label_bbox: bool = True,
    title_prefix: str = "Zoomed View",
    return_grid: bool = True,
    rasterized: bool = False,
):
    """
    依指定的「縣市 × 行政區」配對清單，裁切網格並繪製地圖。
    
    支援整縣市選取：
    若 targets 中的行政區為 None, "*", 或 "全區"，則選取該縣市所有行政區。
    例如: [("臺北市", None), ("新北市", "板橋區")]
    """

    if not targets:
        raise ValueError("targets 不可為空清單")

    # --- 修改點 1: 解析 targets，區分「整縣市」與「特定行政區」 ---
    whole_county_targets = set()
    specific_town_targets = set()

    for county, town in targets:
        # 定義萬用字元，遇到這些代表選取整個縣市
        if town is None or town in ["*", "全區", "ALL"]:
            whole_county_targets.add(county)
        else:
            specific_town_targets.add((county, town))

    # --- 修改點 2: 更新篩選邏輯 (Mask) ---
    def filter_condition(row):
        c_name = row[county_col]
        t_name = row[town_col]
        
        # 條件 A: 該列屬於「整縣市清單」
        if c_name in whole_county_targets:
            return True
        # 條件 B: 該列屬於「特定行政區清單」
        if (c_name, t_name) in specific_town_targets:
            return True
        return False

    mask = town_gdf.apply(filter_condition, axis=1)
    town_geom = town_gdf.loc[mask].copy() # copy 以避免警告

    if town_geom.empty:
        raise ValueError(f"找不到任何符合 targets 的區域：{targets}")

    # 2. 裁切網格 (邏輯不變)
    local_grid = gpd.clip(grid_gdf, town_geom)

    # 3. 繪圖 (邏輯不變)
    fig, ax = plt.subplots(figsize=figsize)

    town_geom.plot(
        ax=ax,
        facecolor="none",
        edgecolor=town_edgecolor,
        linewidth=town_linewidth,
        zorder=2,
    )

    if not local_grid.empty:
        local_grid.plot(
            ax=ax,
            facecolor="none",
            edgecolor=grid_edgecolor,
            linewidth=grid_linewidth,
            zorder=1,
            rasterized=rasterized
        )

    # 4. 標籤處理 (邏輯微調：避免整縣市時標籤過於擁擠，可考慮只標特定或維持原樣)
    # 這裡維持原樣，顯示所有被選中的行政區名稱
    if show_labels:
        label_points = town_geom.representative_point()

        for (_, row), point in zip(town_geom.iterrows(), label_points):
            # 標籤顯示格式
            label = f"{row[town_col]}" # 如果看整縣市，通常只顯示區名比較乾淨，可依需求改回 f"{row[county_col]} {row[town_col]}"
            
            ax.text(
                point.x,
                point.y,
                label,
                fontsize=label_fontsize,
                color=label_color,
                ha="center",
                va="center",
                zorder=3,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    fc="white",
                    ec="none",
                    alpha=0.7
                ) if label_bbox else None,
            )

    # --- 修改點 3: 標題顯示優化 ---
    title_parts = []
    # 先加整縣市
    for c in sorted(whole_county_targets):
        title_parts.append(f"{c}(全)")
    # 再加特定區
    for c, t in sorted(specific_town_targets):
        # 如果該縣市已經在整縣市清單中，就不重複顯示特定區
        if c not in whole_county_targets:
            title_parts.append(f"{c}-{t}")

    title_str = "、".join(title_parts)
    # 避免標題太長
    if len(title_str) > 50: 
        title_str = title_str[:50] + "..."

    ax.set_title(f"{title_prefix}: {title_str}")
    ax.set_axis_off()
    plt.show()

    return local_grid if return_grid else None



