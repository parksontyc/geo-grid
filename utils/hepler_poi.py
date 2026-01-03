from __future__ import annotations
import re
import os
import time
from typing import Dict, Tuple, Union, List, Optional

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import requests
from urllib.parse import quote


from utils.id_category import MediumCategory, Subcategory, DetailedSubcategory

corrections = {
    '36857915': ('嘉義市西區培元里仁愛路316號1樓', '西區'),
    '93938137': ('嘉義市西區磚里中興路626號1樓', '西區'),
    '47911319': ('屏東縣東港鎮東隆街76號', '東港鎮'),
    '60054638': ('新北市中和區忠孝街124巷8號7樓', '中和區'),
    '60242576': ('新北市中和區捷運路133號1樓', '中和區'),
    '46582802': ('新竹市香山區元培街226號', '香山區'),
    '47319843': ('新竹市香山區埔前里牛埔南路127號', '香山區'),
    '02876359': ('新竹市香山區大庄里5鄰大庄路237巷22號1樓', '香山區'),
    '53279073': ('新竹市東區志平路111號4樓', '東區'),
    '53805842': ('新竹市北區成德路271號', '北區'),
    '23521565': ('新竹市香山區牛埔北路257巷20號', '香山區'),
    '24431057': ('新竹市香山區牛埔南路61號', '香山區'),
    '25025936': ('新竹市北區石坊街1號', '北區'),
    '22702672': ('新竹市香山區茄苳里7鄰茄苳北街835巷16號', '香山區'),
    '26157694': ('新竹市香山區莊敬街49巷8號1樓', '香山區'),
    '27809890': ('新竹市香山區香北路240號3樓', '香山區'),
    '10586261': ('新竹市香山區香山里2鄰瑞光街13號', '香山區'),
    '18210268': ('新竹市香山區2鄰牛埔北路37巷63號', '香山區'),
    '13878328': ('新竹市香山區9鄰元培街179號', '香山區'),
    '88316911': ('臺北市南港區向陽路246號', '南港區'),
    '24573007': ('臺北市南港區重陽路203巷29號二樓', '南港區'),
    '53750375': ('臺北市南港區重陽路203巷29號二樓', '南港區'),
    '95465339': ('臺北市南港區重陽里重陽路125巷16號6樓', '南港區'),
    '43909164': ('臺北市南港區重陽里重陽路203巷29號七樓', '南港區'),
    '44043132': ('臺北市南港區重陽里重陽路203巷29號九樓', '南港區'),
    '50761362': ('臺北市南港區重陽里重陽路203巷29號九樓', '南港區'),
    '24932691': ('臺北市南港區重陽里重陽路223號4樓', '南港區'),
    '94212127': ('臺北市南港區重陽里重陽路263巷3號3樓', '南港區'),
    '60515415': ('臺北市南港區重陽里重陽路269號六樓', '南港區'),
    '88466755': ('雲林縣斗六市文生路120巷10號', '斗六市'),

    # 原 correction_adds
    '20973179': ('臺北市信義區松光里松山路615巷1號', '信義區'),
    '20974178': ('臺北市信義區松光里松山路615巷1號', '信義區'),
}

company_name = {
    '83908991' : ['社團法人中華民國軍人之友社', '附設餐廳'],
    '00535958' : ['社團法人中華民國軍人之友社','崇仁店'],
    '26053939' : ['大力果汁', '鼎山分店'],
    '36967390' : ['大力果汁', '建興分店'],
    '81373578' : ['阿珍小吃店', '中正營業所'],
    '81373562' : ['阿珍小吃店', '復興營業所'],
    '91607940' : ['樸樹咖啡', '樹林店'],
    '93168019' : ['樸樹咖啡', '文山店'],
    '93160754' : ['財團法人中華基督教新生命小組教會', '台北東區社區關懷交誼中心'],
    '93269728' : ['財團法人中華基督教新生命小組教會', '高雄社區關懷交誼中心'],
    '92404850' : ['財團法人中華基督教新生命小組教會', '台南社區關懷交誼中心'],
    '40665761' : ['七分漢堡', '漢口店'],
    '26994186' : ['七分漢堡', '壹分店'],
    '39840376' : ['七分漢堡', '東海店'],
    '00324335' : ['禾豐商行', '新仁門市'],
    '00319810' : ['禾豐商行', '永康門市'],
    '92503492' : ['家悅美食小站', '營業所'],
    '92666973' : ['家悅美食小站', '中原分店'],
    '93220057' : ['家悅美食小站', '內湖分店'],
    '40990078' : ['廟口鴨香飯', '中正店'],
    '85195982' : ['廟口鴨香飯', '竹北店'],
    '93161310' : ['蘭丸美食館', '中山店'],
    '93160298' : ['蘭丸美食館', '信義店'],
    '47728827' : ['臺南市文化建設發展基金', '總爺藝文中心'],
    '89084241' : ['臺南市文化建設發展基金', '原臺南運河海關'],
    '69804025' : ['龍坤冷飲店', '內湖店'],
    '85531985' : ['龍坤冷飲店', '茄萣分店'],
    '92149486' : ['北平京廚北方麵食館', '華夏店'],
    '85474984' : ['北平京廚北方麵食館', '自強店'],
    '97836272' : ['華味香飲食冰菓室', '新進店'],
    '10462629' : ['華味香飲食冰菓室', '民治店'],
    '00583894' : ['盛食早午餐', '中山捷運門市'],
    '00583705' : ['盛食早午餐', '中正捷運門市'],
    '88396745' : ['金花商行', '南港三店'],
    '88172267' : ['金花商行', '內湖貳店'],
    '00507411' : ['好糧商號', '立功門市'],
    '88183530' : ['好糧商號', '華碩門市'],
    '93173815' : ['明明福門企業社', '中山營業所'],
    '00582184' : ['明明福門企業社', '北投營業所'],
    '85235083' : ['財團法人台灣敦睦聯誼會', '高雄圓山大飯店'],
    '11729908' : ['財團法人台灣敦睦聯誼會', '台北圓山聯誼會'],
    '11731806' : ['財團法人台灣敦睦聯誼會', '圓山大飯店'],
    '89042307' : ['茶厘家手調茶飲', '歸仁店'],
    '89042279' : ['茶厘家手調茶飲', '後壁厝店'],
    '89042290' : ['茶厘家手調茶飲', '仁德店'],
    '80006001' : ['懷念福隆古早味便當', '金城分店'],
    '87507976' : ['懷念福隆古早味便當', '學府分店'],
    '00517678' : ['好伙食小吃', '中北營業所'],
    '93173717' : ['好伙食小吃', '信義營業所'],
    '92101384' : ['鉅創企業社', '吉林店'],
    '00875203' : ['鉅創企業社', '左營店'],
    '00578344' : ['連城蔬食小吃店', '松山分店'],
    '00605026' : ['連城蔬食小吃店', '雙連分店'],
    '88271061' : ['采津餐飲', '華碩分店'],
    '00580355' : ['采津餐飲', '品珍分店'],
    '00843028' : ['豚十三福岡拉麵', '一心店'],
    '00845898' : ['豚十三福岡拉麵', '新富店'],
    '61102927' : ['飄鄉鍋物', '復興分店'],
    '94713163' : ['飄鄉鍋物', '華夏分店'],
    '94856582' : ['飄鄉鍋物', '博愛分店'],
    '94712707' : ['飄鄉鍋物', '楠梓分店'],
    '94712853' : ['飄鄉鍋物', '五甲分店'],
    '00909196' : ['飄鄉鍋物', '明誠分店	'],

}


#==== 讀取儲存的原始資料=================================
def csv_extractor(file_name: str) -> pd.DataFrame:
    try:
        total_rows = sum(1 for _ in open(file_name, 'r', encoding='utf8'))
        print(f'total_rows: {total_rows}')
        
        df_list = []
        with tqdm(total=total_rows, desc="Extracting rows") as pbar:
            for chunk in pd.read_csv(file_name, encoding='utf8', chunksize=10000):
                df_list.append(chunk)
                pbar.update(len(chunk))
        
        df = pd.concat(df_list, ignore_index=True)
        print('\nExtracting finished')
        return df
    except FileNotFoundError:
        print(f'Error: The file {file_name} was not found.')
    except Exception as e:
        print(f'An error occurred: {e}')
    return pd.DataFrame()


#=====新增及轉換欄位====================================   
def convert_columns( df: pd.DataFrame) -> pd.DataFrame:
    # 新增「縣市」欄位
    # 假設前3個字為縣市，只要其中含有「市」或「縣」
    df['縣市'] = df['營業地址'].apply(
        lambda x: x[:3] if ('市' in x[:3] or '縣' in x[:3]) else None
    )
    
    # 新增行政區欄位
    def extract_district(address):
        """
        從地址中提取行政區
        args:
            address(str): 完整地址

        returns:
            str: 行政區名稱，若無匹配則回傳none
        """
        if not address:
            return None
        
        # 縣市列表
        counties = [
        '台北市', '臺北市', '新北市', '桃園市', '台中市', '臺中市', 
        '台南市', '臺南市', '高雄市', '基隆市', '新竹市', '嘉義市',
        '新竹縣', '苗栗縣', '彰化縣', '南投縣', '雲林縣', '嘉義縣',
        '屏東縣', '宜蘭縣', '花蓮縣', '台東縣', '臺東縣', '澎湖縣',
        '金門縣', '連江縣'
        ]

        # 移除縣市名稱
        remaining = address
        for county in counties:
            if address.startswith(county):
                remaining = address[len(county): ]
                break

        # 使用正則表達式匹配行政區
        # 考慮多可能的模式
        patterns = [
            r'^(.{1,3}[鄉鎮市區])',
            r'^(.{1,2}[鄉鎮市區])'
        ]

        for pattern in patterns:
            match = re.match(pattern, remaining)
            if match:
                district = match.group(1)
                
                # 移除所有空格（全形和半形）和特殊字元
                district = district.replace(' ', '')  # 半形空格
                district = district.replace('　', '')  # 全形空格
                district = district.replace('\t', '')  # Tab
                district = district.replace('\n', '')  # 換行
                district = district.replace('\r', '')  # 回車

                return district

        return None

    # 轉換格式
    df['行政區'] = df['營業地址'].apply(extract_district)
    df['總機構統一編號'] = pd.to_numeric(df['總機構統一編號']).astype('Int64')  # 欄位中有空值
    df['資本額'] = pd.to_numeric(df['資本額']).astype('Int64')
    df['組織別名稱'] = pd.Categorical(df['組織別名稱'])
    df['使用統一發票'] = pd.Categorical(df['使用統一發票'])
    df['名稱'] = pd.Categorical(df['名稱'])

    # 先轉換為數值
    df['行業代號'] = pd.to_numeric(df['行業代號'], errors='coerce')
    df['統一編號'] = pd.to_numeric(df['統一編號'], errors='coerce')
    # 補0到6/8位數
    df['行業代號'] = df['行業代號'].apply(lambda x: str(int(x)).zfill(6) if pd.notna(x) else x)
    df['統一編號'] = df['統一編號'].apply(lambda x: str(int(x)).zfill(8) if pd.notna(x) else x)
    
    # 轉為 Categorical
    df['統一編號'] = pd.Categorical(df['統一編號'])

    # 「行業代號」拆分出中類、小類、細類
    # 定義函數來依據字典查找中類
    def medium_category_sector(code):
        return MediumCategory.get(str(code)[:2], '')
    
    # 定義函數來依據字典查找小類
    def subcategory_sector(code):
        return Subcategory.get(str(code)[:3], '')
    
    # 定義函數來依據字典查找細類
    def detail_subcategory_sector(code):
        return DetailedSubcategory.get(str(code)[:4], '')

    # 先將行業代號轉為字串型態
    df['行業代號'] = df['行業代號'].astype(str)
    # 新增「行業代號」拆分出中類、小類、細類
    df['中類'] = df['行業代號'].apply(medium_category_sector)
    df['小類'] = df['行業代號'].apply(subcategory_sector)
    df['細類'] = df['行業代號'].apply(detail_subcategory_sector)
    # 轉為 Categorical
    df['行業代號'] = pd.Categorical(df['行業代號'])


    # 「設立日期」欄位
    def parse_roc_date(val):
        """
        將民國年月日數值（如 1040413.0、400711.0、1130503.0）轉成西元日期。
        - 自動處理 float / int / str 類型
        - 不合法值轉為 NaT
        """
        if pd.isna(val):
            return pd.NaT
        try:
            # 先轉字串並補零到 7 碼（例如 400711 -> '0400711'）
            s = str(int(val)).zfill(7)
            y, m, d = int(s[:3]) + 1911, int(s[3:5]), int(s[5:7])
            return pd.Timestamp(f"{y}-{m:02d}-{d:02d}")
        except Exception:
            return pd.NaT
        
    df['設立日期'] = df['設立日期'].apply(parse_roc_date)
    df['設立季度'] = df['設立日期'].dt.to_period('Q')

    def fullwidth_to_halfwidth(text: str) -> str:
        """
        將文字中的全形字元轉為半形，並清理非法字元與空白。
        適用於欄位如「營業地址」「公司名稱」等。

        處理內容：
        1. 全形空白（12288）→ 半形空白（32）
        2. 一般全形字元（65281–65374）→ 對應半形
        3. 移除非法 Unicode 字元
        4. 將「―」替換為 "-"
        5. 移除多餘空白
        """
        if pd.isna(text):
            return text

        # 全形轉半形
        converted = ''.join(
            chr(32) if ord(uchar) == 12288 else
            chr(ord(uchar) - 65248) if 65281 <= ord(uchar) <= 65374 else
            uchar
            for uchar in text
        )

        # 處理特殊字元和格式
        converted = re.sub(r'[^\u0000-\uFFFF]', '', converted)  # 移除非法 Unicode
        converted = converted.replace("―", "-")  # 處理破折號
        converted = re.sub(r'\s+', '', converted)  # 移除所有空白（含全形空格）

        return converted

    df["營業地址"] = df["營業地址"].apply(fullwidth_to_halfwidth)

    return df


#=====資料修正==================================
def apply_manual_corrections(
    df: pd.DataFrame,
    corrections: Dict[str, Tuple[str, str]],
    uid_col: str = "統一編號",
    address_col: str = "正規化營業地址",
    district_col: str = "行政區",
) -> pd.DataFrame:
    """
    依據人工校正表，更新指定欄位內容（地址 / 行政區）。

    Parameters
    ----------
    df : pd.DataFrame
        欲修正的資料表
    corrections : dict[str, tuple[str, str]]
        {統一編號: (正規化營業地址, 行政區)}
    uid_col : str, default="統一編號"
        統一編號欄位名稱
    address_col : str, default="正規化營業地址"
        地址欄位名稱
    district_col : str, default="行政區"
        行政區欄位名稱

    Returns
    -------
    pd.DataFrame
        已套用修正的新 DataFrame（原 df 不會被改動）
    """
    df = df.copy()

    for uid, (address, district) in corrections.items():
        mask = df[uid_col] == uid
        if mask.any():
            df.loc[mask, [address_col, district_col]] = [address, district]

    return df


#======新增樓層資訊================================
def split_address_and_floor(address):
    """
    將地址拆分成正規化地址（不含樓層）和樓層兩個部分
    
    Parameters:
    address: 原始地址字串
    
    Returns:
    tuple: (正規化地址, 樓層)
    """
    if pd.isna(address) or not address:
        return (address, None)
    
    address = str(address).strip()
    
    # 轉換全形數字為半形
    address = address.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    
    # 先移除括號內容（如：前鎮第一臨時市場(D41號）
    address_clean = re.sub(r'[\(（〈\[《【][^)\]】》〉）]*[\)）〉\]》】]', '', address)

    # === 新增特殊處理 ===
    # 處理「號之X11樓」這種格式（X是個位數，11是樓層）
    special_pattern = re.search(r'(\d+號之\d)(\d{1,2}樓)', address_clean)
    if special_pattern:
        # 檢查是否為「號之5」+「11樓」的格式
        house_with_suffix = special_pattern.group(1)  # 例如：15號之5
        floor = special_pattern.group(2)               # 例如：11樓
        
        # 取得號之前的部分
        before_match = address_clean[:special_pattern.start()]
        
        # 組合正規化地址和樓層
        normalized_address = before_match + house_with_suffix
        return (normalized_address, floor)
    
    # 定義各種樓層模式（順序很重要，複雜的放前面）
    floor_patterns = [
        (r'(地下街[A-Z]\d+號?)', r'地下街[A-Z]\d+號?'),      # 地下街B70號
        (r'(地下街\d+號?)', r'地下街\d+號?'),                # 地下街70號


        (r'(\d+之\d+樓)', r'\d+之\d+樓'),                    # 4之3樓
        (r'(\d+樓之\d+)', r'\d+樓之\d+'),                    # 4樓之3
        (r'(\d+樓\-\d+)', r'\d+樓\-\d+'),                    # 4樓-3
        (r'([B地下]\d+樓)', r'[B地下]\d+樓'),                # B1樓
        (r'(\d+樓)', r'\d+樓'),                              # 4樓
        (r'([一二三四五六七八九十壹貳參肆伍陸柒捌玖拾百]+之\d+樓)', 
         r'[一二三四五六七八九十壹貳參肆伍陸柒捌玖拾百]+之\d+樓'),  # 四之3樓
        (r'([一二三四五六七八九十壹貳參肆伍陸柒捌玖拾百]+樓之\d+)', 
         r'[一二三四五六七八九十壹貳參肆伍陸柒捌玖拾百]+樓之\d+'),  # 四樓之3
        (r'([一二三四五六七八九十壹貳參肆伍陸柒捌玖拾百]+樓)', 
         r'[一二三四五六七八九十壹貳參肆伍陸柒捌玖拾百]+樓'),      # 三樓
        (r'(地下[一二三四五六七八九十]+樓)', r'地下[一二三四五六七八九十]+樓'),  # 地下一樓
        (r'(地下室)', r'地下室'),                             # 地下室
        (r'([B地下]\d+)', r'[B地下]\d+'),                     # B1（沒有樓字）
    ]
    
    # 處理地號格式（不會有樓層）
    if re.search(r'\d+地號', address_clean):
        # 地號地址，提取到地號為止
        match = re.match(r'^(.*?\d+地號)', address_clean)
        if match:
            return (match.group(1), None)
    
    # 處理一般地址格式
    # 先找到門牌號的位置
    house_no_match = re.search(r'(\d+之\d+|\d+\-\d+|\d+)號', address_clean)
    
    if house_no_match:
        base_address = address_clean[:house_no_match.end()]  # 包含號的基本地址
        after_house_no = address_clean[house_no_match.end():]  # 號之後的部分
        
        # 清理號後面可能的房間號（但不是樓層）
        after_house_no = re.sub(r'^[\s\-之]*(\d+[室房]|[A-Z]\d*室).*$', '', after_house_no)
        
        # 在號之後的部分尋找樓層
        floor_info = None
        clean_after = after_house_no
        
        for floor_capture, floor_pattern in floor_patterns:
            # 尋找樓層資訊
            floor_match = re.search(floor_capture, after_house_no)
            if floor_match:
                floor_info = floor_match.group(1)
                # 移除找到的樓層及其後的房間號
                clean_after = after_house_no[:floor_match.start()]
                # 只保留樓層資訊，移除樓層後的房間號
                remaining = after_house_no[floor_match.end():]
                if not re.match(r'^之\d+|^\-\d+', remaining):  # 不是樓之幾的格式
                    remaining = re.sub(r'^[\s\-之]*\d*[室房號A-Z].*$', '', remaining)
                break
        
        # 組合最終的正規化地址（不含樓層）
        normalized_address = base_address + clean_after.strip()
        normalized_address = normalized_address.strip().rstrip('、').rstrip('，')
        
        return (normalized_address, floor_info)
    
    # 如果沒有找到門牌號，返回原始地址
    return (address_clean, None)

def process_dataframe_addresses(df, address_column='地址', 
                               normalized_column='正規化地址', 
                               floor_column='樓層'):
    """
    處理DataFrame中的地址欄位，拆分成正規化地址和樓層
    
    Parameters:
    df: DataFrame
    address_column: 原始地址欄位名稱
    normalized_column: 正規化地址欄位名稱
    floor_column: 樓層欄位名稱
    
    Returns:
    DataFrame with new columns
    """
    # 應用拆分函數
    df[[normalized_column, floor_column]] = df[address_column].apply(
        lambda x: pd.Series(split_address_and_floor(x))
    )
    
    return df

#=====新增營業型態：連鎖/非連鎖===========================
def enrich_chain_status(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    處理資料集的連鎖狀態標記。

    Args:
        df_input (pd.DataFrame): 原始資料框，需包含「統一編號」與「總機構統一編號」欄位。

    Returns:
        pd.DataFrame: 處理後的新資料框，包含補全的統編與新增的「營業型態」欄位。
    """
    # 建立副本以免影響原始 DataFrame
    df: pd.DataFrame = df_input.copy()
    
    # ---- 1. 補全總機構統一編號 ----
    valid_org_ids: set = set(df['總機構統一編號'].dropna().unique())

    mask: pd.Series = (
        df['總機構統一編號'].isna()
        & df['統一編號'].notna()
        & df['統一編號'].isin(valid_org_ids)
    )

    df.loc[mask, '總機構統一編號'] = df.loc[mask, '統一編號']
    
    print(f"已補全 {mask.sum():,} 筆『總機構統一編號』。")

    # ---- 2. 新增營業型態欄位 ----
    counts: pd.Series = (
        df.groupby('總機構統一編號')['總機構統一編號']
          .transform('count')
    )

    df['營業型態'] = np.where(
        counts >= 2,
        '連鎖',
        '非連鎖'
    )

    # ---- 3. 連鎖家數（非連鎖為 NaN）----
    df['連鎖家數'] = np.where(
        counts >= 2,
        counts,
        np.nan
    )

    # ---- 4. 統計家數（以總機構為單位）----
    org_counts = (
        df.groupby('總機構統一編號')
          .size()
    )

    chain_orgs = (org_counts >= 2).sum()
    non_chain_orgs = (org_counts == 1).sum()
    total_orgs = org_counts.size

    print(
        f"連鎖家數：{chain_orgs:,}  "
        f"非連鎖家數：{non_chain_orgs:,}  "
        f"合計：{total_orgs:,} 家"
    )

    return df


#=====新增公司名稱/分公司名稱===========================
def split_chain_company_name(
    df: pd.DataFrame,
    name_col: str = "營業人名稱",
    type_col: str = "營業型態",
    company_col: str = "公司名稱",
    branch_col: str = "分公司名稱",
    keyword: Union[str, List[str]] = "股份有限公司",
    chain_value: str = "連鎖",
) -> pd.DataFrame:
    """
    依規則拆分連鎖型態之營業人名稱，產生公司名稱與分公司名稱欄位。

    規則：
    - 僅當「營業型態 == 連鎖」
    - 且「營業人名稱」包含 keyword（可為 list）
      → 使用第一個命中的 keyword 進行拆分
        - keyword 前 → 公司名稱
        - keyword 後 → 分公司名稱
    - 其餘情況填入 NaN
    """

    df = df.copy()

    df[company_col] = pd.Series(pd.NA, dtype="string")
    df[branch_col] = pd.Series(pd.NA, dtype="string")

    # keyword 正規化為 list
    keywords = [keyword] if isinstance(keyword, str) else list(keyword)


    base_mask = (
        (df[type_col] == chain_value)
        & df[name_col].notna()
    )

    # 逐一 keyword 嘗試拆分（只處理尚未被拆過的列）
    for kw in keywords:
        mask = (
            base_mask
            & df[name_col].str.contains(kw)
            & df[company_col].isna()
        )

        if not mask.any():
            continue

        split_result = (
            df.loc[mask, name_col]
            .str.split(kw, n=1, expand=True)
        )

        df.loc[mask, company_col] = split_result[0].str.strip()
        df.loc[mask, branch_col] = split_result[1].str.strip()

    return df


#=====補正公司名稱及分公司名稱===========================
def fill_company_and_branch_by_id(
    df: pd.DataFrame,
    mapping: Dict[str, List[str]],
    id_col: str = "統一編號",
    company_col: str = "公司名稱",
    branch_col: str = "分公司名稱",
) -> pd.DataFrame:
    """
    依據統一編號，補齊公司名稱與分公司名稱。

    Parameters
    ----------
    df : pd.DataFrame
        來源資料表
    mapping : dict
        key = 統一編號(str)
        value = [公司名稱, 分公司名稱]
    id_col : str
        統一編號欄位名稱
    company_col : str
        公司名稱欄位
    branch_col : str
        分公司名稱欄位

    Returns
    -------
    pd.DataFrame
        補值後的新 DataFrame（不修改原 df）
    """
    df = df.copy()

    # 統一轉成 string，避免 dtype 問題
    df[id_col] = df[id_col].astype("string")

    # 若欄位不存在，先建立，避免 SettingWithExpansion 問題
    for col in [company_col, branch_col]:
        if col not in df.columns:
            df[col] = pd.NA
        else:
            df[col] = df[col].astype("string")

    # 批次補值
    for uid, (company, branch) in mapping.items():
        mask = df[id_col] == uid
        df.loc[mask, company_col] = company
        df.loc[mask, branch_col] = branch

    return df

#=====tdx-token===================================
load_dotenv()

url = 'https://tdx.transportdata.tw/auth/realms/TDXConnect/protocol/openid-connect/token'
headers = {'content-type': 'application/x-www-form-urlencoded'}

data = {
    'grant_type': 'client_credentials',
    'client_id': f'{os.getenv("client_id")}',
    'client_secret': f'{os.getenv("client_secret")}'
}

response = requests.post(url, headers=headers, data=data)
print(response.json())
token = response.json()['access_token']


#=====取得座標===================================
POINT_RE = re.compile(r"POINT\s*\(\s*([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s*\)")

def _parse_point_wkt(geometry: str) -> Optional[tuple]:
    if not geometry:
        return None
    m = POINT_RE.search(geometry)
    if not m:
        return None
    lon = float(m.group(1))
    lat = float(m.group(2))
    return lon, lat

def geocode_tdx_df(
    df: pd.DataFrame,
    token: str,
    addr_col: str = "正規化營業地址",
    out_addr_col: str = "coords_address",
    out_lon_col: str = "longitude",
    out_lat_col: str = "latitude",
    requests_per_second: int = 10,
    pause_every: int = 45,
    pause_seconds: int = 60,
    timeout: int = 12,
    max_retries: int = 3,
    only_fill_na: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    修改說明：
    1. 新增 api_request_count 變數，統計實際發查次數。
    2. 在暫停與結束時輸出統計資訊。
    """
    # 欄位準備
    if out_addr_col not in df.columns:
        df.insert(df.columns.get_loc(addr_col) + 1, out_addr_col, pd.NA)
    if out_lon_col not in df.columns:
        df.insert(df.columns.get_loc(out_addr_col) + 1, out_lon_col, pd.NA)
    if out_lat_col not in df.columns:
        df.insert(df.columns.get_loc(out_lon_col) + 1, out_lat_col, pd.NA)

    # 速率限制參數
    min_interval = 1.0 / max(1, requests_per_second)
    last_ts = 0.0

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    total = len(df)
    processed = 0          # 總共遍歷了幾列 (包含跳過的)
    api_request_count = 0  # 實際發查了幾筆 (沒有跳過的)

    print(f"--- 開始處理，共 {total} 筆資料 ---")

    for idx, row in df.iterrows():
        # 1. 檢查輸入地址是否存在
        addr = row.get(addr_col, None)
        if not isinstance(addr, str) or not addr.strip():
            processed += 1
            continue

        # 2. 檢查是否需要發查 (only_fill_na 邏輯)
        if only_fill_na is True:
            existing_val = row.get(out_addr_col)
            if pd.notna(existing_val) and str(existing_val).strip() != "":
                processed += 1
                continue

        # --- 確定要發查了，計數器 +1 ---
        api_request_count += 1

        # 3. 批次暫停機制 (修改顯示資訊)
        # 注意：這裡用 api_request_count 來判斷暫停可能比較合理(避免跳過的也算次數)，
        # 但為了維持您原本的邏輯(每處理N行暫停)，這裡維持用 processed 或 api_request_count 皆可。
        # 這裡建議：如果擔心跳過太多導致連續發查過快，可以維持用 processed；
        # 但通常是為了省 API 額度或休息，這裡我加入 顯示實際發查數。
        if pause_every > 0 and processed > 0 and processed % pause_every == 0:
            if verbose:
                print(f"[Info] 進度 {processed}/{total} (實際發查: {api_request_count} 筆)，暫停 {pause_seconds} 秒...")
            time.sleep(pause_seconds)

        # 4. 速率限制
        now = time.time()
        elapsed = now - last_ts
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        # 5. 呼叫 API
        url = f"https://tdx.transportdata.tw/api/advanced/V3/Map/GeoCode/Coordinate/Address/{quote(addr)}?format=JSON"

        success = False
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(url, headers=headers, timeout=timeout)
                last_ts = time.time()

                if resp.status_code in (401, 403):
                    raise RuntimeError(f"TDX token 失效或權限不足（HTTP {resp.status_code}）。")

                if resp.status_code != 200:
                    if verbose:
                        print(f"[Warn] {addr} 回應 {resp.status_code}，第 {attempt}/{max_retries} 次重試。")
                    time.sleep(min(2 ** attempt, 5))
                    continue

                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    geometry = data[0].get("Geometry")
                    coords_addr = data[0].get("Address")
                    pt = _parse_point_wkt(geometry)
                    
                    if pt:
                        lon, lat = pt
                        df.at[idx, out_addr_col] = coords_addr
                        df.at[idx, out_lon_col] = lon
                        df.at[idx, out_lat_col] = lat
                        success = True
                        if verbose:
                            # 顯示目前是第幾筆實際發查
                            print(f"[OK] #{api_request_count} | {addr} → ({lon:.5f}, {lat:.5f})")
                        break
                    else:
                        if verbose:
                            print(f"[Warn] 幾何解析失敗，重試中...")
                        time.sleep(min(2 ** attempt, 5))
                        continue
                else:
                    if verbose:
                        print(f"[Info] #{api_request_count} | {addr} 查無資料。")
                    success = True 
                    break

            except Exception as e:
                if attempt >= max_retries:
                    if verbose:
                        print(f"[Error] {addr} 最終失敗：{e}")
                else:
                    if verbose:
                        print(f"[Warn] 發生錯誤：{e}，重試中...")
                    time.sleep(min(2 ** attempt, 5))

        processed += 1

    # --- 最終統計 ---
    if verbose:
        print("-" * 30)
        print(f"處理完成！")
        print(f"總筆數 (Total Rows): {total}")
        print(f"實際發查 (API Calls): {api_request_count}")
        print(f"跳過筆數 (Skipped): {total - api_request_count}")
        print("-" * 30)

    return df


#=====型態：目的/補給型===========================
def add_catering_type_column(df, source_col) -> pd.DataFrame:
    """
    根據指定的來源欄位，為 DataFrame 新增一個 '型態' 欄位，
    區分為 '目的型' 與 '補給型'。

    參數:
    df (pd.DataFrame): 您的 catering_services 資料框
    source_col (str): 用來判斷類別的欄位名稱 (例如 '細類' 或 '營業項目')

    回傳:
    pd.DataFrame: 處理完成的資料框
    """
    
    # 1. 定義分類清單
    # 目的型清單
    destination_type = [
        '餐廳', '吃到飽餐廳', '咖啡館', '茶館', '飲酒店', '有娛樂節目餐廳'
    ]

    # 補給型清單
    supply_type = [
        '便當、自助餐店', '早餐店', '麵店、小吃店', '連鎖速食店', '餐食攤販', 
        '調理飲料攤販', '手搖飲店', '其他飲料店']
    
    # 2. 建立副本以免影響原始傳入的變數 (良好的函式習慣)
    df_result = df.copy()
    
    # 3. 初始化 '型態' 欄位，預設為 '其他'
    df_result['型態'] = '其他'

    # 4. 執行分類 (使用向量化操作加速)
    # 判斷是否在目的型清單中
    mask_dest = df_result[source_col].isin(destination_type)
    df_result.loc[mask_dest, '型態'] = '目的型'
    
    # 判斷是否在補給型清單中
    mask_supply = df_result[source_col].isin(supply_type)
    df_result.loc[mask_supply, '型態'] = '補給型'
    
    return df_result


#=====回補座標===================================
def patch_coordinates(df: pd.DataFrame, 
                      coord_batch: pd.DataFrame, 
                      on_key: str = '統一編號') -> pd.DataFrame:
    """
    使用分批的座標資料來「更新」主資料表。
    避免重複 merge 產生多餘欄位。
    """
    # 1. 建立副本，避免修改原始變數
    df_result = df.copy()
    
    # 2. 定義我們要補的目標欄位
    target_cols = ['coords_address', 'longitude', 'latitude']
    
    # 3. [關鍵步驟] 檢查主表是否已有這些欄位，沒有的話先創立空欄位
    # 這是為了第一次執行時，確保欄位存在，後續才能 update
    for col in target_cols:
        if col not in df_result.columns:
            df_result[col] = pd.NA  # 或 np.nan

    # 4. 設定 Index 以便對齊
    # update 是根據 Index 進行對應的，所以必須暫時將 Key 設為 Index
    df_result = df_result.set_index(on_key)
    batch_data = coord_batch.set_index(on_key)
    
    # 5. 過濾 Batch 資料
    # 只取我們關心的欄位，避免 Batch 裡有其他雜訊欄位影響主表
    # intersect 確保只取 batch 裡實際存在的目標欄位
    cols_to_update = batch_data.columns.intersection(target_cols)
    batch_subset = batch_data[cols_to_update]

    # 6. 執行更新 (In-place update)
    # 這會將 batch_subset 裡有的值，填入 df_result 對應的位置
    df_result.update(batch_subset)
    
    # 7. 還原 Index 並回傳
    return df_result.reset_index()

