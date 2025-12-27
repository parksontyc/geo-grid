from __future__ import annotations

from pathlib import Path
from typing import Union, Tuple, Optional, Literal, Tuple, List

import math
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt


ClipMode = Literal["clip", "filter", "none"]
IdScheme = Literal["xy", "string"]

#=====[[讀取shape檔]]=====#
def read_vector(path: Union[str, Path]) -> gpd.GeoDataFrame:
    """
    讀取向量資料（shp/geojson/gpkg...），由 geopandas 自動判斷格式
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"檔案不存在：{path}")
    return gpd.read_file(path)


#=====[[轉換座標系]]=====#
def ensure_crs(gdf: gpd.GeoDataFrame, crs: Union[str, int]) -> gpd.GeoDataFrame:
    """
    若 gdf 沒有 CRS，則設定 CRS（不做座標轉換，只是宣告原始 CRS）
    """
    if gdf.crs is None:
        return gdf.set_crs(crs)
    return gdf

def project_to_epsg(gdf: gpd.GeoDataFrame, epsg: int = 3826) -> gpd.GeoDataFrame:
    """
    轉換到指定 EPSG（例：3826 = TWD97 / TM2, 單位公尺）
    """
    if gdf.crs is None:
        raise ValueError("GeoDataFrame 沒有 CRS，請先用 ensure_crs() 指定原始 CRS")
    return gdf.to_crs(epsg=epsg)


#=====[[生成網格]]=====#
def aligned_bounds(
    bounds: Tuple[float, float, float, float],
    cell_size: float,
) -> Tuple[float, float, float, float]:
    """
    將 bounds 對齊到 cell_size 的整數倍，讓格網落點穩定（重跑一致）
    """
    if cell_size <= 0:
        raise ValueError("cell_size 必須 > 0")

    minx, miny, maxx, maxy = bounds
    minx = math.floor(minx / cell_size) * cell_size
    miny = math.floor(miny / cell_size) * cell_size
    maxx = math.ceil(maxx / cell_size) * cell_size
    maxy = math.ceil(maxy / cell_size) * cell_size
    return float(minx), float(miny), float(maxx), float(maxy)

def make_grid(
    bounds: Tuple[float, float, float, float],
    cell_size: float = 500.0,
    crs: Optional[Union[str, int]] = None,
    align: bool = True,
) -> gpd.GeoDataFrame:
    """
    依 bounds 產生規則格網（Polygon）
    - align=True：會先對齊 bounds（推薦）
    """
    if cell_size <= 0:
        raise ValueError("cell_size 必須 > 0")

    if align:
        bounds = aligned_bounds(bounds, cell_size)

    minx, miny, maxx, maxy = bounds
    xs = np.arange(minx, maxx, cell_size)
    ys = np.arange(miny, maxy, cell_size)

    cells = [box(x, y, x + cell_size, y + cell_size) for x in xs for y in ys]
    return gpd.GeoDataFrame({"geometry": cells}, crs=crs)

def apply_clip_mode(
    grid: gpd.GeoDataFrame,
    mask_gdf: gpd.GeoDataFrame,
    clip_mode: ClipMode = "filter",
    dissolve_mask: bool = True,
) -> gpd.GeoDataFrame:
    """
    依 clip_mode 處理格網：
    - "clip"   : 直接裁切，格子會被切碎（gpd.clip）
    - "filter" : 保留完整格子，只保留與 mask 相交的格子（sjoin intersects）
    - "none"   : 不處理，回傳原 grid
    """
    if clip_mode == "none":
        return grid

    mask = mask_gdf.dissolve() if dissolve_mask else mask_gdf

    if clip_mode == "clip":
        return gpd.clip(grid, mask)

    if clip_mode == "filter":
        # 保留完整格子：只做相交篩選，不裁切 geometry
        joined = grid.sjoin(mask, how="inner", predicate="intersects")
        if "index_right" in joined.columns:
            joined = joined.drop(columns=["index_right"])
        return joined.reset_index(drop=True)

    raise ValueError(f"clip_mode 不合法：{clip_mode}（請用 'clip'/'filter'/'none'）")

def add_grid_id(
    grid: gpd.GeoDataFrame,
    origin: Tuple[float, float],
    cell_size: float,
    id_col: str = "grid_id",
    scheme: IdScheme = "string",
) -> gpd.GeoDataFrame:
    """
    依格網左下角座標計算 (col,row) index，再產出 grid_id
    - origin: 使用「對齊後 bounds 的 (minx, miny)」當原點，ID 才會穩定
    - scheme:
        - "xy": 產出整數ID（row-major）+ grid_col/grid_row
        - "string": 產出 "C{col}_R{row}" 字串ID
    """
    if cell_size <= 0:
        raise ValueError("cell_size 必須 > 0")

    minx0, miny0 = origin
    b = grid.geometry.bounds

    grid_col = np.floor((b["minx"].to_numpy() - minx0) / cell_size).astype("int64")
    grid_row = np.floor((b["miny"].to_numpy() - miny0) / cell_size).astype("int64")

    out = grid.copy()
    out["grid_col"] = grid_col
    out["grid_row"] = grid_row

    if scheme == "xy":
        base = int(out["grid_col"].max()) + 1
        out[id_col] = (out["grid_row"] * base + out["grid_col"]).astype("int64")
    else:
        out[id_col] = ("C" + out["grid_col"].astype(str) + "_R" + out["grid_row"].astype(str))

    return out


#=====[[網格Pipeline]]=====#
def build_projected_grid(
    path: Union[str, Path],
    target_epsg: int = 3826,
    cell_size: float = 500.0,
    clip_mode: ClipMode = "filter",
    id_scheme: IdScheme = "string",
    assume_source_crs: Optional[Union[str, int]] = None,
    verbose: bool = True,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    完整流程：
    讀取 →（必要時補 CRS）→ 轉投影 → 對齊邊界 → 分格 → clip/filter/none → 網格ID
    回傳：(投影後行政區 gdf, 格網 grid)
    """
    # 1) 讀取
    gdf = read_vector(path)

    # 2) 補 CRS（若原始資料沒 CRS 才需要）
    if assume_source_crs is not None:
        gdf = ensure_crs(gdf, assume_source_crs)

    # 3) 投影（公尺座標）
    gdf_proj = project_to_epsg(gdf, epsg=target_epsg)

    # 4) bounds 對齊（穩定格網落點 & 穩定 ID）
    raw_bounds = tuple(map(float, gdf_proj.total_bounds))
    b_aligned = aligned_bounds(raw_bounds, cell_size)

    # 5) 生完整格網（先不裁切）
    grid = make_grid(b_aligned, cell_size=cell_size, crs=gdf_proj.crs, align=False)

    # 6) clip/filter/none
    grid = apply_clip_mode(grid, gdf_proj, clip_mode=clip_mode, dissolve_mask=True)

    # 7) 網格 ID（用對齊後的原點）
    origin = (b_aligned[0], b_aligned[1])
    grid = add_grid_id(grid, origin=origin, cell_size=cell_size, scheme=id_scheme)
    grid = grid[['grid_id', 'grid_col', 'grid_row', 'geometry']]

    if verbose:
        print(f"原始 CRS: {gdf.crs}")
        print(f"投影後 CRS: {gdf_proj.crs}")
        print(f"raw bounds: {raw_bounds}")
        print(f"aligned bounds: {b_aligned}")
        print(f"cell_size: {cell_size}m")
        print(f"clip_mode: {clip_mode}")
        print(f"grid cells: {len(grid):,}")
        print(f"grid columns: {list(grid.columns)}")

    return gdf_proj, grid


#=====[[繪圖檢視]]=====#
def plot_local_grid_by_targets(
    town_gdf: gpd.GeoDataFrame,
    grid_gdf: gpd.GeoDataFrame,
    targets: List[Tuple[str, str]],
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
):
    """
    依指定的「縣市 × 行政區」配對清單，裁切網格並繪製地圖，
    可選擇是否在圖上顯示縣市與行政區標籤。
    """

    if not targets:
        raise ValueError("targets 不可為空清單")

    target_set = set(targets)

    # 1. 精準篩選指定的「縣市 × 行政區」
    mask = town_gdf.apply(
        lambda r: (r[county_col], r[town_col]) in target_set,
        axis=1
    )
    town_geom = town_gdf.loc[mask]

    if town_geom.empty:
        raise ValueError(f"找不到任何符合 targets 的行政區：{targets}")

    # 2. 裁切網格
    local_grid = gpd.clip(grid_gdf, town_geom)

    # 3. 繪圖
    fig, ax = plt.subplots(figsize=figsize)

    town_geom.plot(
        ax=ax,
        facecolor="none",
        edgecolor=town_edgecolor,
        linewidth=town_linewidth,
        zorder=2,
    )

    local_grid.plot(
        ax=ax,
        facecolor="none",
        edgecolor=grid_edgecolor,
        linewidth=grid_linewidth,
        zorder=1,
    )

    # 4. 視需要加上縣市＋行政區標籤
    if show_labels:
        label_points = town_geom.representative_point()

        for (_, row), point in zip(town_geom.iterrows(), label_points):
            label = f"{row[county_col]} {row[town_col]}"
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

    title_targets = "、".join([f"{c}-{t}" for c, t in targets])
    ax.set_title(f"{title_prefix}: {title_targets}")
    ax.set_axis_off()
    plt.show()

    return local_grid if return_grid else None
