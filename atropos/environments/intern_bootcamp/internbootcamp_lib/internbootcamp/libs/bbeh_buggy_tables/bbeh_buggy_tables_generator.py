import json
import random
import copy
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Union


class BBEHBuggyTablesGenerator:
    def __init__(self, task_file_path: str):
        """初始化生成器，加载task.json文件"""
        self.task_data = self._load_task_data(task_file_path)
        self.bug_types = [
            "missing_null_values",  # 空值被错误移除
            "appended_random_values",  # 添加随机值
            "merged_rows",  # 行合并错误
            "rotated_data",  # 数据旋转
            "replaced_values"  # 数据替换
        ]
        self.query_types = ["count", "sum", "mean", "stdev", "median"]

    def _load_task_data(self, file_path: str) -> Dict:
        """加载task.json文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载任务数据文件失败: {e}")
            return {}

    def _extract_examples(self) -> List[Dict]:
        """从task.json提取示例"""
        if 'examples' in self.task_data:
            return self.task_data['examples']
        return []

    def _generate_clean_table(self, n_rows: int, n_cols: int) -> pd.DataFrame:
        """生成一个干净的表格数据"""
        data = {}
        for col in range(n_cols):
            col_name = f"col_{col}"
            # 随机选择列的数据类型：数值或分类
            if random.random() > 0.3:  # 70%概率是数值列
                # 生成数值列，包含一些空值
                values = np.random.randint(1, 100, size=n_rows).astype(float)
                # 随机添加一些空值
                null_indices = random.sample(range(n_rows), k=random.randint(0, n_rows // 5))
                values[null_indices] = np.nan
            else:  # 30%概率是分类列
                # 生成分类列
                categories = [f"cat_{i}" for i in range(random.randint(3, 7))]
                values = [random.choice(categories) for _ in range(n_rows)]
                # 随机添加一些空值
                null_indices = random.sample(range(n_rows), k=random.randint(0, n_rows // 5))
                for idx in null_indices:
                    values[idx] = None
            data[col_name] = values

        return pd.DataFrame(data)

    def _apply_bug(self, df: pd.DataFrame, bug_type: str) -> Tuple[pd.DataFrame, str]:
        """对表格应用指定类型的bug"""
        buggy_df = copy.deepcopy(df)
        bug_description = ""

        if bug_type == "missing_null_values":
            # 删除空值（不保留空值标记）
            buggy_df = buggy_df.dropna()
            bug_description = "表格中的所有空值(null)被错误地移除了。"

        elif bug_type == "appended_random_values":
            # 在每行末尾添加随机值
            if random.choice([True, False]):  # 随机选择添加到行还是列
                # 添加到行
                n_rows, n_cols = buggy_df.shape
                new_col = f"random_col_{n_cols}"
                buggy_df[new_col] = [random.randint(1, 100) for _ in range(n_rows)]
                bug_description = f"表格中每行末尾添加了一个随机值列 '{new_col}'。"
            else:
                # 添加到列
                random_row = {}
                for col in buggy_df.columns:
                    if pd.api.types.is_numeric_dtype(buggy_df[col]):
                        random_row[col] = random.randint(1, 100)
                    else:
                        random_row[col] = f"random_value_{random.randint(1, 10)}"
                buggy_df = pd.concat([buggy_df, pd.DataFrame([random_row])], ignore_index=True)
                bug_description = "表格末尾添加了一行随机值。"

        elif bug_type == "merged_rows":
            # 每两行合并成一行
            merged_data = []
            n_rows = len(buggy_df)

            for i in range(0, n_rows, 2):
                if i + 1 < n_rows:
                    merged_row = {}
                    for col in buggy_df.columns:
                        # 合并两行的值，用逗号分隔
                        val1 = str(buggy_df.iloc[i][col]) if pd.notna(buggy_df.iloc[i][col]) else "null"
                        val2 = str(buggy_df.iloc[i + 1][col]) if pd.notna(buggy_df.iloc[i + 1][col]) else "null"
                        merged_row[col] = f"{val1},{val2}"
                    merged_data.append(merged_row)
                else:
                    # 如果是奇数行，最后一行单独保留
                    merged_row = {}
                    for col in buggy_df.columns:
                        val = str(buggy_df.iloc[i][col]) if pd.notna(buggy_df.iloc[i][col]) else "null"
                        merged_row[col] = val
                    merged_data.append(merged_row)

            buggy_df = pd.DataFrame(merged_data)
            bug_description = "表格中每两行被错误地合并成了一行，值之间用逗号分隔。"

        elif bug_type == "rotated_data":
            # 旋转数据
            if random.choice([True, False]):  # 随机选择旋转行还是列
                # 旋转行: 将最后一行移到第一行
                rows = buggy_df.values.tolist()
                if len(rows) > 1:
                    rows = [rows[-1]] + rows[:-1]
                    buggy_df = pd.DataFrame(rows, columns=buggy_df.columns)
                    bug_description = "表格的行被旋转了，最后一行被移到了第一行。"
            else:
                # 旋转列: 将最后一列移到第一列
                cols = list(buggy_df.columns)
                if len(cols) > 1:
                    new_order = [cols[-1]] + cols[:-1]
                    buggy_df = buggy_df[new_order]
                    bug_description = "表格的列被旋转了，最后一列被移到了第一列。"

        elif bug_type == "replaced_values":
            # 随机替换一些值为"ERROR"
            n_rows, n_cols = buggy_df.shape
            replacements = min(random.randint(1, n_rows), 5)  # 最多替换5个值

            for _ in range(replacements):
                row_idx = random.randint(0, n_rows - 1)
                col_idx = random.randint(0, n_cols - 1)
                col_name = buggy_df.columns[col_idx]
                buggy_df[col_name] = buggy_df[col_name].astype(object)
                buggy_df.loc[row_idx, col_name] = "ERROR"

            bug_description = f"表格中有{replacements}个值被错误地替换为'ERROR'。"

        return buggy_df, bug_description

    def _generate_condition(self, df: pd.DataFrame) -> Tuple[str, Dict]:
        """生成查询条件"""
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        if not numeric_cols:
            return "", {}

        selected_col = random.choice(numeric_cols)
        condition_type = random.choice(["gt", "lt", "eq", "between"])

        values = df[selected_col].dropna().values
        if len(values) == 0:
            return "", {}

        min_val, max_val = int(np.min(values)), int(np.max(values))

        condition_str = ""
        condition_dict = {}

        if condition_type == "gt":
            threshold = random.randint(min_val, max_val)
            condition_str = f"{selected_col} > {threshold}"
            condition_dict = {"column": selected_col, "operator": ">", "value": threshold}
        elif condition_type == "lt":
            threshold = random.randint(min_val, max_val)
            condition_str = f"{selected_col} < {threshold}"
            condition_dict = {"column": selected_col, "operator": "<", "value": threshold}
        elif condition_type == "eq":
            if len(values) > 0:
                value = int(random.choice(values))
                condition_str = f"{selected_col} == {value}"
                condition_dict = {"column": selected_col, "operator": "==", "value": value}
        elif condition_type == "between":
            val1 = random.randint(min_val, max_val)
            val2 = random.randint(min_val, max_val)
            low, high = min(val1, val2), max(val1, val2)
            condition_str = f"{low} <= {selected_col} <= {high}"
            condition_dict = {"column": selected_col, "operator": "between", "low": low, "high": high}

        return condition_str, condition_dict

    def _generate_query(self, df: pd.DataFrame) -> Tuple[str, Dict]:
        """生成查询信息"""
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        if not numeric_cols:
            return "", {}

        query_type = random.choice(self.query_types)
        selected_col = random.choice(numeric_cols)

        condition_str, condition_dict = self._generate_condition(df)

        query_str = f"{query_type}({selected_col})"
        if condition_str:
            query_str += f" where {condition_str}"

        query_dict = {
            "type": query_type,
            "column": selected_col
        }

        if condition_dict:
            query_dict["condition"] = condition_dict

        return query_str, query_dict

    def generate_example(self) -> Dict:
        """生成一个新的示例"""
        # 生成原始表格
        n_rows = random.randint(5, 15)
        n_cols = random.randint(3, 6)
        clean_table = self._generate_clean_table(n_rows, n_cols)

        # 选择一个bug类型
        bug_type = random.choice(self.bug_types)

        # 应用bug
        buggy_table, bug_description = self._apply_bug(clean_table, bug_type)

        # 生成查询
        query_str, query_dict = self._generate_query(clean_table)

        # 创建示例
        example = {
            "input": {
                "table": buggy_table.fillna("null").to_dict(orient="records"),
                "bug_description": bug_description,
                "query": query_str
            },
            "clean_table": clean_table.fillna("null").to_dict(orient="records"),
            "query_info": query_dict
        }

        return example

    def generate_dataset(self, n_examples: int) -> List[Dict]:
        """生成指定数量的示例"""
        dataset = []
        for _ in range(n_examples):
            example = self.generate_example()
            dataset.append(example)

        return dataset

    def save_dataset(self, dataset: List[Dict], file_path: str) -> None:
        """保存数据集到文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({"examples": dataset}, f, ensure_ascii=False, indent=2)
            print(f"数据集已保存至 {file_path}")
        except Exception as e:
            print(f"保存数据集失败: {e}")


# 使用示例
if __name__ == "__main__":
    generator = BBEHBuggyTablesGenerator("task.json")
    new_dataset = generator.generate_dataset(5)  # 生成5个新示例
    generator.save_dataset(new_dataset, "generated_dataset.json")
    print("生成了5个新的BBEH Buggy Tables示例")

