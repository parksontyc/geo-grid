import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import re
from typing import List, Dict, Union, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd

from utils.step import Step


# ----------------------------
# 1) 型別與常數
# ----------------------------
class SkipReason(Enum):
    """跳過原因的列舉"""
    INVALID_TYPE = '非字串格式'
    LAND_NUMBER = '包含地號'
    SECTION = '包含小段'
    INVALID_FORMAT = '非正規地址格式'

@dataclass
class AddressChange:
    """地址變更紀錄"""
    original: str
    modified: str
    changes_made: List[str] = field(default_factory=list)

@dataclass
class SkippedAddress:
    """跳過處理的地址資料"""
    address: str
    reason: SkipReason
    type: str

DEFAULT_DELIMITERS = [',', '、', '及', '.', '﹒', '~', '–', '，', '˙', '丶']
EXEMPT_ENDINGS = (
    '地下', '層', '樓', 'B1', ')', '室', '旁', '地', 
    '邊', '園', '公尺', '前', '面', '場', '路', '街', '巷', '段', 'F', '屋', '口',
    '側', '位', '攤', '棟', '區', '壁', '部', '棚', ']', '房', '櫃', '﹞', '1', '一', '內', '方', '鋪', '市', '庭','類', '分', '舖', '庫', '販', '右', '廊', '處', '外', '空', '慺','耬',
    '〕', '份', '半', '絡', '檈', '物', '間', '廳', '商', '〉', '後', '門', '樓`', '編', '院', '二', '摟', '橋', '等', '臨', '熡', '屢', '角', '下', '﹚', '牌', '近', '左', '>', '}', '厝', '址'
)

# ----------------------------
# 2) 類別實作
# ----------------------------
class AddressSplitter:
    """分割複數地址 + 規則補字（不主動清理；可選 preprocessor）"""

    def __init__(
        self,
        input_column: str = '正規化營業地址',
        output_column: str = '正規化營業地址',
        status_column: str = '地址狀態',
        delimiters: Optional[List[str]] = None,
        exempt_endings: Tuple[str, ...] = EXEMPT_ENDINGS,
        batch_size: int = 10000,
        print_report: bool = True,
        preprocessor: Optional[Callable[[str], str]] = None  # 預設不做任何前處理
    ):
        self.input_column = input_column
        self.output_column = output_column
        self.status_column = status_column
        self.batch_size = batch_size
        self.print_report = print_report
        self.preprocessor = preprocessor

        self.delimiters = delimiters or DEFAULT_DELIMITERS
        self.delimiters_pattern, self.exempt_endings_pattern = self._compile_patterns(
            self.delimiters, exempt_endings
        )
        self.reset_changes()

    # ----------------------------
    # 工具與狀態
    # ----------------------------
    def reset_changes(self):
        self.changes: Dict[str, List[Union[AddressChange, SkippedAddress]]] = {
            'modified': [],
            'skipped': []
        }
        self.total: int = 0

    @staticmethod
    def _compile_patterns(delimiters: List[str], exempt_endings_tuple: Tuple[str, ...]):
        delimiters_pattern = re.compile('|'.join(map(re.escape, delimiters)))
        exempt_endings = '|'.join(map(re.escape, exempt_endings_tuple))
        exempt_endings_pattern = re.compile(
            f"({exempt_endings}|之\\d+|號-\\d+|號\\d+|[A-Za-z]\\d+|樓-\\d+|樓\\d+|[A-Za-z]|附\\d+|位\\d+|層-\\d+|編號\\d+|層\\d+|編號\\d+)$"
        )
        return delimiters_pattern, exempt_endings_pattern

    @staticmethod
    def _validate_address(addr: str) -> Union[SkippedAddress, None]:
        """驗證地址的有效性（不自動清理）"""
        if not isinstance(addr, str):
            return SkippedAddress(addr, SkipReason.INVALID_TYPE, 'invalid')
        if '地號' in addr:
            return SkippedAddress(addr, SkipReason.LAND_NUMBER, 'land_number')
        if '小段' in addr:
            return SkippedAddress(addr, SkipReason.SECTION, 'section')
        # 基本格式（鬆綁版）：出現 市/區/鄉/鎮 且 之後某處有「號」
        if not re.search(r'[市區鄉鎮].*號', addr):
            return SkippedAddress(addr, SkipReason.INVALID_FORMAT, 'invalid')
        return None

    # ----------------------------
    # 單筆與批次
    # ----------------------------
    def _process_single_address(self, addr: str) -> Tuple[str, str, Union[AddressChange, SkippedAddress, None]]:
        """回傳 (處理後地址, 狀態, 變更/跳過紀錄)"""
        # 可選外部前處理（若有給）
        working = self.preprocessor(addr) if (self.preprocessor and isinstance(addr, str)) else addr

        # 驗證
        validation = self._validate_address(working)
        if validation:
            return working if isinstance(working, str) else addr, validation.reason.value, validation

        change_record = AddressChange(original=working, modified=working)

        # 2) 先「去尾括號內容」：若地址以「(...)」結尾，整段拿掉
        m = re.search(r'^(.*?號)(\([^)]*\))$', working)
        removed_paren = False
        paren_text = None
        if m:
            working = m.group(1)    # 僅保留「...號」
            paren_text = m.group(2) # 例如 "(26、27、28攤位)"、"(第三市場65、66)"、"(編號67、69)"
            removed_paren = True
            change_record.modified = working
            change_record.changes_made.append('去尾括號')

        # 3) 再做「複數地址分割」（只取第一段）
        is_multiple = bool(self.delimiters_pattern.search(working))
        split_addr = self.delimiters_pattern.split(working, maxsplit=1)[0]
        if split_addr != working:
            working = split_addr
            change_record.modified = working
            change_record.changes_made.append('分割')

        # ★ 新增：若以「路/街/巷/弄/段/大道 + 數字」結尾，先補「號」
        if re.search(r'(?:大道|路|街|段|巷|弄)\d+$', working) and not working.endswith('號'):
            working += '號'
            change_record.modified = working
            change_record.changes_made.append('補充號(數字結尾)')

        
        # 4) 規則補「號」
        if not working.endswith('號') and not self.exempt_endings_pattern.search(working):
            working += '號'
            change_record.modified = working
            change_record.changes_made.append('補充號')

        # 5) 規則補「樓」
        #   (a) 原規則：結尾為「號+數字」（排除「攤」）
        if re.search(r'(?<!攤)號\d+$', working) and not working.endswith('樓'):
            working += '樓'
            change_record.modified = working
            change_record.changes_made.append('補充樓')

        #   (b) 去掉的括號含「編號」→ 也補「樓」
        if removed_paren and paren_text and '編號' in paren_text and not working.endswith('樓'):
            working += '樓'
            change_record.modified = working
            change_record.changes_made.append('括號含編號→補樓')

        # 狀態
        if change_record.original != working:
            status = '複數地址' if is_multiple else '已修改'
            return working, status, change_record
        else:
            return working, '未變動', None

    def _process_batch(self, batch: pd.Series) -> Tuple[pd.Series, pd.Series, List[Union[AddressChange, SkippedAddress]]]:
        records: List[Union[AddressChange, SkippedAddress]] = []
        addr_out, status_out = [], []
        for v in batch:
            processed, status, rec = self._process_single_address(v)
            addr_out.append(processed)
            status_out.append(status)
            if rec is not None:
                records.append(rec)
        return pd.Series(addr_out, index=batch.index), pd.Series(status_out, index=batch.index), records

    # ----------------------------
    # 報告
    # ----------------------------
    def _print_report(self):
        print("\n地址處理報告")
        print("=" * 80)
        if self.changes['modified']:
            print("\n【已修改的地址（含去尾括號/分割/補號/補樓）】")
            for i, rec in enumerate(self.changes['modified'][:100], 1):
                print(f"{i:03d} 原：{rec.original}  ->  新：{rec.modified}  |  {', '.join(rec.changes_made)}")
            if len(self.changes['modified']) > 100:
                print(f"... 其餘 {len(self.changes['modified']) - 100} 筆略")

        if self.changes['skipped']:
            print("\n【跳過處理的地址】")
            for i, rec in enumerate(self.changes['skipped'][:100], 1):
                print(f"{i:03d} 原：{rec.address}  |  原因：{rec.reason.value}")
            if len(self.changes['skipped']) > 100:
                print(f"... 其餘 {len(self.changes['skipped']) - 100} 筆略")

        print("\n處理統計：")
        print(f"- 總地址數：{self.total}")
        print(f"- 已修改數：{len(self.changes['modified'])}")
        print(f"- 跳過數：{len(self.changes['skipped'])}")

    # ----------------------------
    # 對 DataFrame 執行
    # ----------------------------
    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        只做：
          - （可選）自訂 preprocessor(text) 前處理；預設 None -> 不處理
          - 尾端括號移除（若以 (...) 結尾）
          - 複數地址分割（只取第一段）
          - 規則補「號」/「樓」
          - 產出狀態欄位（未變動/已修改/複數地址/跳過原因）
        """
        if self.input_column not in df.columns:
            raise ValueError(f"找不到輸入欄位：{self.input_column}")

        self.reset_changes()
        self.total = len(df)

        result_df = df.copy()

        # 可選：對輸入欄位建立工作序列（若有 preprocessor）
        working_series = result_df[self.input_column].copy()
        if self.preprocessor is not None:
            working_series = working_series.apply(self.preprocessor)

        processed_addresses_all: List[pd.Series] = []
        processed_status_all: List[pd.Series] = []

        for start in range(0, len(result_df), self.batch_size):
            batch = working_series.iloc[start:start + self.batch_size]
            addrs, stats, recs = self._process_batch(batch)
            processed_addresses_all.append(addrs)
            processed_status_all.append(stats)
            # 累計紀錄
            for r in recs:
                if isinstance(r, AddressChange):
                    self.changes['modified'].append(r)
                elif isinstance(r, SkippedAddress):
                    self.changes['skipped'].append(r)

        processed_addresses = pd.concat(processed_addresses_all).reindex(result_df.index)
        processed_statuses = pd.concat(processed_status_all).reindex(result_df.index)

        # 先移除既有欄位，再按位置插入
        input_idx = result_df.columns.get_loc(self.input_column)
        for col in [self.output_column, self.status_column]:
            if col in result_df.columns:
                result_df = result_df.drop(columns=[col])

        result_df.insert(input_idx + 1, self.output_column, processed_addresses)
        result_df.insert(input_idx + 2, self.status_column, processed_statuses)

        if self.print_report:
            self._print_report()

        return result_df, {'modified': self.changes['modified'],
                           'skipped': self.changes['skipped'],
                           'total': self.total,
                           'counts': {'modified': len(self.changes['modified']),
                                      'skipped': len(self.changes['skipped'])}}