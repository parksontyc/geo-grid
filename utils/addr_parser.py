import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from typing import List, Dict, Union, Tuple
import re
from dataclasses import dataclass, field

import pandas as pd

from utils.step import Step


@dataclass
class AddressChange:
    """地址變更記錄"""
    original: str
    details: List[str] = field(default_factory=list)

@dataclass
class SkippedAddress:
    """跳過處理的地址"""
    address: str
    reason: str

class NumberConverter:
    """數字轉換工具類"""
    
    CH_TO_ARABIC = {
        '0': 0, '一': 1, '二': 2, '三': 3, '四': 4, 
        '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
        '零': 0, '○': 0
    }
    
    UNIT_MAPPING = {
        '十': 10, '百': 100, '千': 1000, '萬': 10000
    }
    
    ARABIC_TO_CH = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
    
    @classmethod
    def ch_to_num(cls, target_ch: str) -> str:
        """將中文數字轉換為阿拉伯數字"""
        result = 0
        temp_num = 0
        last_unit = 1
        
        for char in target_ch:
            if char in cls.CH_TO_ARABIC:
                temp_num = temp_num * 10 + cls.CH_TO_ARABIC[char]
            elif char in cls.UNIT_MAPPING:
                unit = cls.UNIT_MAPPING[char]
                if temp_num == 0:
                    temp_num = 1
                result += temp_num * unit
                temp_num = 0
                last_unit = unit
            else:
                continue
                
        if temp_num > 0:
            result += temp_num
        return str(result)
    
    @classmethod
    def num_to_ch(cls, target_num: str) -> str:
        """將阿拉伯數字轉換為中文數字"""
        if not target_num.isdigit():
            return target_num
            
        n = len(target_num)
        if n == 1:
            return cls.ARABIC_TO_CH[int(target_num)]
        elif n == 2:
            if target_num[0] == '1':
                return '十' + (cls.ARABIC_TO_CH[int(target_num[1])] if target_num[1] != '0' else '')
            else:
                return cls.ARABIC_TO_CH[int(target_num[0])] + '十' + (cls.ARABIC_TO_CH[int(target_num[1])] if target_num[1] != '0' else '')
        elif n >= 3:
            result = ''
            for i, digit in enumerate(target_num):
                if digit != '0':
                    result += cls.ARABIC_TO_CH[int(digit)] + ('十百千萬'[n - i - 2] if n - i - 1 > 0 else '')
                else:
                    if not result.endswith('零'):
                        result += '零'
            return result.rstrip('零')
        else:
            return target_num

class AddressParser(Step):
    """地址解析類"""
    
    KEYWORDS = [
        '縣', '市', '區', '鄉', '鎮', '村', '里', '鄰',
        '大道', '路', '街', '段', '巷', '弄', '衖', '號',
        '樓', '室'
    ]

    SKIP_KEYWORDS = ['地號', '小段']  # 新增需跳過處理的關鍵字列表
    
    def __init__(self):
        super().__init__()
        self.input_column = '正規化營業地址'
        self.output_column = '正規化營業地址'
        self.reset_changes()
        self.regex_arabic = re.compile(r'\d')
        self.regex_chinese = re.compile(r'[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e]')

    def _should_skip_address(self, address: str) -> Tuple[bool, str]:
        """檢查是否應該跳過處理此地址"""
        for keyword in self.SKIP_KEYWORDS:
            if keyword in address:
                return True, f"包含{keyword}"
        return False, ""

    def reset_changes(self):
        """重置變更追蹤狀態"""
        self.changes = {
            'modified': [],
            'skipped': []
        }
        self.total = 0

    def _reconstruct_address(self, parsed_dict: Dict[str, str]) -> str:
        """將解析後的地址字典重組為完整地址字串"""
        # 原有邏輯保持不變
        address_order = ['縣', '市', '區', '鄉', '鎮', '村', '里', '鄰','大道',
                        '路', '街',  '段', '巷', '弄', '衖', '號', '樓', '室']
        
        reconstructed = ''
        for key in address_order:
            if key in parsed_dict:
                reconstructed += parsed_dict[key] + key
        
        return reconstructed

    def _parse_single_address(self, row: pd.Series) -> pd.Series:  
        address = row[self.input_column]

        # 檢查是否需要跳過處理
        should_skip, reason = self._should_skip_address(address)
        if should_skip:
            # 將原始地址直接複製到輸出欄位，不做任何轉換
            row[self.output_column] = address
            # 記錄跳過的地址
            self.changes['skipped'].append(SkippedAddress(
                address=address,
                reason=reason
            ))
            return row
        
        """解析單一地址"""
        # 原有邏輯保持不變
        address = row[self.input_column]
        result = {}
        current_pos = 0
        original_address = address
        converted = False
        convert_details = []
        previous_key = None
        
        for key in self.KEYWORDS:
            if key in address[current_pos:]:
                idx = address.index(key, current_pos)
                value = address[current_pos:idx].strip()
                
                if key == '路' and self.regex_arabic.search(value):
                    if previous_key in ['縣', '市', '區', '鄉', '鎮', '村', '里', '鄰']:
                        match = re.search(r'(\d+)', value)
                        if match:
                            arabic_num = match.group(1)
                            chinese_num = NumberConverter.num_to_ch(arabic_num)
                            new_value = value.replace(arabic_num, chinese_num)
                            result[key] = new_value
                            convert_details.append(f'{value}{key}→{new_value}{key}')
                            converted = True
                        else:
                            result[key] = value
                    else:
                        result[key] = value

                elif key == '段' and self.regex_arabic.search(value):
                    if previous_key in ['大道', '路', '街']:
                        result[key] = NumberConverter.num_to_ch(value)
                        convert_details.append(f'{value}{key}→{result[key]}{key}')
                        converted = True
                    else:
                        result[key] = value

                elif key == '號' and self.regex_chinese.search(value):
                    if '之' not in value and previous_key in ['路', '街', '巷', '弄']:
                        result[key] = NumberConverter.ch_to_num(value)
                        convert_details.append(f'{value}{key}→{result[key]}{key}')
                        converted = True
                    elif '之' in value and previous_key in ['路', '街', '巷', '弄']:
                        parts = value.split('之') 
                        result[key] = f"{NumberConverter.ch_to_num(parts[0])}之{NumberConverter.ch_to_num(parts[1])}"
                        convert_details.append(f'{value}{key}→{result[key]}{key}')
                        converted = True
                    else:
                        result[key] = value

                elif key == '巷' and self.regex_chinese.search(value):
                    if previous_key in ['路', '街']:
                        result[key] = NumberConverter.ch_to_num(value)
                        convert_details.append(f'{value}{key}→{result[key]}{key}')
                        converted = True
                    else:
                        result[key] = value
                else:
                    result[key] = value
                    
                current_pos = idx + len(key)
                previous_key = key
        
        if converted:
            self.changes['modified'].append(AddressChange(
                original=original_address,
                details=convert_details
            ))
            
        row[self.output_column] = self._reconstruct_address(result)
        return row
    
    def _print_report(self):
        """印出處理報告"""
        print("\n地址解析報告:")
        print("=" * 80)
        
        if self.changes['modified']:
            print("\n【數字格式轉換的地址】")
            for idx, record in enumerate(self.changes['modified'], 1):
                print("-" * 80)
                print(f"地址 {idx}:")
                print(f"原始地址：{record.original}")
                print(f"轉換明細：{' | '.join(record.details)}")
        
        if self.changes['skipped']:
            print("\n【跳過處理的地址】")
            for idx, record in enumerate(self.changes['skipped'], 1):
                print("-" * 80)
                print(f"地址 {idx}:")
                print(f"原始地址：{record.address}")
                print(f"跳過原因：{record.reason}")
        
        print("\n處理統計：")
        print(f"轉換的地址數：{len(self.changes['modified'])}個")
        print(f"跳過的地址數：{len(self.changes['skipped'])}個")
        print(f"總地址數：{self.total}個")

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """處理 DataFrame 中的地址
        
        Args:
            df (pd.DataFrame): 輸入的DataFrame
            
        Returns:
            pd.DataFrame: 處理後的DataFrame
            
        Raises:
            ValueError: 當找不到輸入欄位或處理失敗時
        """
        try:
            # 檢查輸入欄位
            if self.input_column not in df.columns:
                raise ValueError(f"找不到輸入欄位：{self.input_column}")
            
            # 重置變更追蹤
            self.reset_changes()
            
            # 更新總數
            self.total = len(df)
            
            # 複製DataFrame以避免修改原始資料
            result_df = df.copy()
            
            # 處理每一行
            result_df = result_df.apply(self._parse_single_address, axis=1)
            
            # 顯示報告
            self._print_report()
            
            return result_df
            
        except Exception as e:
            raise ValueError(f"地址解析失敗：{str(e)}")

