import json
import pandas as pd
import numpy as np
import re
pd.set_option('future.no_silent_downcasting', True)
from typing import Dict, List, Any, Union, Tuple


class BBEHBuggyTablesSolver:
    def __init__(self):
        """初始化求解器"""
        pass

    def _fix_missing_null_values(self, table: List[Dict], bug_description: str) -> pd.DataFrame:
        """修复缺失的null值问题，使用统计方法推断和恢复缺失值"""
        df = pd.DataFrame(table)

        # 对每一列进行处理
        for col in df.columns:
            # 检查是否为数值列
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            if numeric_col.notna().sum() > 0:  # 如果列包含数值
                # 计算列的统计特征
                mean_val = numeric_col.mean()
                std_val = numeric_col.std()

                # 检测异常值的阈值
                lower_bound = mean_val - 2 * std_val
                upper_bound = mean_val + 2 * std_val

                # 找出可能缺失null值的位置（数据异常断点）
                values = numeric_col.dropna().values
                gaps = np.diff(sorted(values))
                median_gap = np.median(gaps)

                # 如果发现异常大的间隔，在这些位置插入null
                large_gaps = np.where(gaps > 3 * median_gap)[0]
                if len(large_gaps) > 0:
                    df[col] = df[col].astype(object)
                    for gap_idx in large_gaps:
                        # 在异常间隔处插入null值
                        df.loc[len(df)] = {col: np.nan for col in df.columns}

        return df.sort_index()

    def _fix_appended_random_values(self, table: List[Dict], bug_description: str) -> pd.DataFrame:
        """修复添加的随机值"""
        df = pd.DataFrame(table)

        if "每行末尾添加了一个随机值列" in bug_description:
            # 随机值被添加为新列，识别并删除它
            match = re.search(r"'(.+?)'", bug_description)
            if match:
                random_col = match.group(1)
                if random_col in df.columns:
                    df = df.drop(columns=[random_col])
            else:
                # 如果没有找到列名，尝试删除最后一列
                df = df.iloc[:, :-1]

        elif "表格末尾添加了一行随机值" in bug_description:
            # 随机值被添加为新行，删除最后一行
            df = df.iloc[:-1]

        return df

    def _fix_merged_rows(self, table: List[Dict], bug_description: str) -> pd.DataFrame:
        """修复合并的行"""
        df = pd.DataFrame(table)
        unmerged_data = []

        for _, row in df.iterrows():
            for col in df.columns:
                value = row[col]
                # 检查是否是字符串并包含逗号
                if isinstance(value, str) and "," in value:
                    # 这一行是合并的，需要拆分
                    split_values = {}
                    for c in df.columns:
                        vals = str(row[c]).split(",")
                        split_values[c] = [v if v != "null" else None for v in vals]

                    # 确定每行有多少个值
                    max_vals = max(len(split_values[c]) for c in df.columns)

                    # 创建拆分后的行
                    for i in range(max_vals):
                        new_row = {}
                        for c in df.columns:
                            vals = split_values[c]
                            if i < len(vals):
                                new_row[c] = vals[i]
                            else:
                                new_row[c] = None
                        unmerged_data.append(new_row)

                    break
            else:
                # 这行没有合并值，直接添加
                unmerged_data.append(row.to_dict())

        return pd.DataFrame(unmerged_data)

    def _fix_rotated_data(self, table: List[Dict], bug_description: str) -> pd.DataFrame:
        """修复旋转的数据"""
        df = pd.DataFrame(table)

        if "行被旋转" in bug_description and "最后一行被移到了第一行" in bug_description:
            # 将第一行移到最后一行
            rows = df.values.tolist()
            if len(rows) > 1:
                rows = rows[1:] + [rows[0]]
                df = pd.DataFrame(rows, columns=df.columns)

        elif "列被旋转" in bug_description and "最后一列被移到了第一列" in bug_description:
            # 将第一列移到最后一列
            cols = list(df.columns)
            if len(cols) > 1:
                new_order = cols[1:] + [cols[0]]
                df = df[new_order]

        return df

    def _fix_replaced_values(self, table: List[Dict], bug_description: str) -> pd.DataFrame:
        df = pd.DataFrame(table)

        for col in df.columns:
            df[col] = df[col].astype(object)

            error_mask = df[col] == "ERROR"
            if error_mask.any():
                numeric_values = pd.to_numeric(df[~error_mask][col], errors='coerce')

                if numeric_values.notna().sum() > 0:
                    mean_val = numeric_values.mean()
                    median_val = numeric_values.median()

                    for idx in df[error_mask].index:
                        prev_vals = df.loc[:idx - 1, col]
                        next_vals = df.loc[idx + 1:, col]

                        # 使用 concat 替代 append
                        nearby_values = pd.to_numeric(
                            pd.concat([prev_vals.tail(2), next_vals.head(2)]),
                            errors='coerce'
                        ).dropna()

                        if len(nearby_values) > 0:
                            df.loc[idx, col] = nearby_values.mean()
                        else:
                            df.loc[idx, col] = median_val
                else:
                    df[col] = df[col].replace("ERROR", np.nan)

        return df

    def fix_table(self, table: List[Dict], bug_description: str) -> pd.DataFrame:
        """根据bug描述修复表格"""
        if "空值(null)被错误地移除" in bug_description:
            return self._fix_missing_null_values(table, bug_description)
        elif "添加了一个随机值列" in bug_description or "添加了一行随机值" in bug_description:
            return self._fix_appended_random_values(table, bug_description)
        elif "每两行被错误地合并" in bug_description:
            return self._fix_merged_rows(table, bug_description)
        elif "行被旋转" in bug_description or "列被旋转" in bug_description:
            return self._fix_rotated_data(table, bug_description)
        elif "值被错误地替换为'ERROR'" in bug_description:
            return self._fix_replaced_values(table, bug_description)
        else:
            # 未识别的bug类型，返回原始表格
            return pd.DataFrame(table)

    def _convert_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """改进的值转换方法，更好地处理混合类型数据"""
        for col in df.columns:
            # 保存原始数据类型
            original_dtype = df[col].dtype

            # 尝试转换为数值类型
            try:
                numeric_values = pd.to_numeric(df[col], errors='coerce')
                non_numeric = df[col][numeric_values.isna() & df[col].notna()]

                # 检查是否主要是数值类型
                if numeric_values.notna().sum() / len(df) > 0.5:
                    df[col] = numeric_values

                    # 特殊处理非数值项
                    if len(non_numeric) > 0:
                        for idx in non_numeric.index:
                            if df.loc[idx, col] == 'null':
                                df.loc[idx, col] = np.nan
                            elif isinstance(df.loc[idx, col], str):
                                try:
                                    # 尝试清理并转换字符串
                                    cleaned_val = df.loc[idx, col].strip().replace(',', '')
                                    df.loc[idx, col] = float(cleaned_val)
                                except:
                                    df.loc[idx, col] = np.nan
            except:
                # 如果转换失败，保持原始类型
                df[col] = df[col].astype(original_dtype)

            # 统一处理null值
            df[col] = df[col].replace(['null', 'NULL', 'None', ''], np.nan)

        return df

    def apply_condition(self, df: pd.DataFrame, condition: Dict) -> pd.DataFrame:
        """应用查询条件"""
        if not condition:
            return df

        column = condition.get('column')
        operator = condition.get('operator')
        value = condition.get('value')

        if not column or not operator or column not in df.columns:
            return df

        if operator == ">":
            return df[df[column] > value]
        elif operator == "<":
            return df[df[column] < value]
        elif operator == "==":
            return df[df[column] == value]
        elif operator == "between":
            low = condition.get('low')
            high = condition.get('high')
            if low is not None and high is not None:
                return df[(df[column] >= low) & (df[column] <= high)]

        return df

    def execute_query(self, df: pd.DataFrame, query_info: Dict) -> float:
        query_type = query_info.get('type')
        column = query_info.get('column')
        condition = query_info.get('condition', {})

        if not query_type or not column or column not in df.columns:
            return np.nan

        try:
            # 创建副本避免 SettingWithCopyWarning
            filtered_df = self.apply_condition(df, condition).copy()
            filtered_df.loc[:, column] = pd.to_numeric(filtered_df[column], errors='coerce')

            values = filtered_df[column].dropna()

            if len(values) == 0:
                return np.nan

            # 执行查询
            if query_type == "count":
                return len(values)
            elif query_type == "sum":
                return float(values.sum())
            elif query_type == "mean":
                return float(values.mean())
            elif query_type == "stdev":
                return float(values.std()) if len(values) > 1 else np.nan
            elif query_type == "median":
                return float(values.median())
        except Exception as e:
            print(f"Query execution error: {str(e)}")
            return np.nan

        return np.nan

    def parse_query_string(self, query_string: str) -> Dict:
        """从查询字符串解析查询信息"""
        query_info = {}

        # 匹配查询类型和列名
        query_match = re.match(r"(\w+)\((.+?)\)(\s+where\s+(.+))?", query_string)
        if not query_match:
            return query_info

        query_type, column = query_match.group(1), query_match.group(2)
        query_info["type"] = query_type
        query_info["column"] = column

        # 提取条件（如果有）
        if query_match.group(4):
            condition_str = query_match.group(4)

            # 处理不同类型的条件
            between_match = re.match(r"(\d+)\s*<=\s*(.+?)\s*<=\s*(\d+)", condition_str)
            gt_match = re.match(r"(.+?)\s*>\s*(\d+)", condition_str)
            lt_match = re.match(r"(.+?)\s*<\s*(\d+)", condition_str)
            eq_match = re.match(r"(.+?)\s*==\s*(\d+)", condition_str)

            if between_match:
                low, col, high = between_match.groups()
                query_info["condition"] = {
                    "column": col,
                    "operator": "between",
                    "low": int(low),
                    "high": int(high)
                }
            elif gt_match:
                col, val = gt_match.groups()
                query_info["condition"] = {
                    "column": col,
                    "operator": ">",
                    "value": int(val)
                }
            elif lt_match:
                col, val = lt_match.groups()
                query_info["condition"] = {
                    "column": col,
                    "operator": "<",
                    "value": int(val)
                }
            elif eq_match:
                col, val = eq_match.groups()
                query_info["condition"] = {
                    "column": col,
                    "operator": "==",
                    "value": int(val)
                }

        return query_info

    def solve(self, example: Dict) -> float:
        """解决一个示例问题"""
        # 提取输入数据
        table = example['input']['table']
        bug_description = example['input']['bug_description']
        query = example['input']['query']

        # 修复表格
        fixed_df = self.fix_table(table, bug_description)

        # 转换数据类型
        fixed_df = self._convert_values(fixed_df)

        # 解析查询信息
        if 'query_info' in example:
            query_info = example['query_info']
        else:
            query_info = self.parse_query_string(query)

        # 执行查询
        result = self.execute_query(fixed_df, query_info)

        # 根据查询类型做适当的四舍五入
        if result is not None:
            if query_info.get('type') in ['mean', 'stdev']:
                result = round(result, 2)
            elif query_info.get('type') in ['sum', 'median']:
                result = round(result, 1)

        return result


# 使用示例
if __name__ == "__main__":
    # 从文件加载测试数据
    with open("generated_dataset.json", 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    solver = BBEHBuggyTablesSolver()

    # 解决每个示例
    for i, example in enumerate(test_data['examples']):
        result = solver.solve(example)
        print(f"示例 {i + 1} 的计算结果: {result}")

