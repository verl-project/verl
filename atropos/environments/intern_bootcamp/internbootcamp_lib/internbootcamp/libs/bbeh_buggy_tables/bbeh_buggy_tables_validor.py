import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union
from internbootcamp.libs.bbeh_buggy_tables.bbeh_buggy_tables_solver import BBEHBuggyTablesSolver


class BBEHBuggyTablesValidator:
    def __init__(self):
        """初始化验证器"""
        self.solver = BBEHBuggyTablesSolver()

    def _get_expected_result(self, example: Dict) -> float:
        """计算期望的结果"""
        # 如果示例中包含干净的表格和查询信息，使用这些来计算参考答案
        if 'clean_table' in example and 'query_info' in example:
            clean_df = pd.DataFrame(example['clean_table'])

            # 转换数据类型
            for col in clean_df.columns:
                clean_df[col] = clean_df[col].replace('null', np.nan)
                try:
                    clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
                except:
                    pass

            # 使用求解器执行查询
            expected_result = self.solver.execute_query(clean_df, example['query_info'])

            # 根据查询类型做适当的四舍五入
            if expected_result is not None:
                if example['query_info'].get('type') in ['mean', 'stdev']:
                    expected_result = round(expected_result, 2)
                elif example['query_info'].get('type') in ['sum', 'median']:
                    expected_result = round(expected_result, 1)

            return expected_result

        return None

    def validate_example(self, example: Dict) -> Dict:
        """验证单个示例"""
        # 使用求解器计算结果
        actual_result = self.solver.solve(example)

        # 获取期望的结果
        expected_result = self._get_expected_result(example)

        # 检查结果是否匹配
        is_correct = abs(
            actual_result - expected_result) < 1e-6 if actual_result is not None and expected_result is not None else False

        # 创建验证报告
        validation_report = {
            "input": example['input'],
            "expected_result": expected_result,
            "actual_result": actual_result,
            "is_correct": is_correct,
            "error_details": None
        }

        # 如果结果不匹配，添加错误详情
        if not is_correct:
            validation_report["error_details"] = {
                "difference": abs(
                    actual_result - expected_result) if actual_result is not None and expected_result is not None else None,
                "error_type": self._get_error_type(actual_result, expected_result)
            }

        return validation_report

    def _get_error_type(self, actual: float, expected: float) -> str:
        """确定错误类型"""
        if actual is None:
            return "求解器未返回结果"
        if expected is None:
            return "无法确定期望结果"

        diff = actual - expected
        if abs(diff) < 1e-6:
            return "结果正确"
        elif diff > 0:
            return "计算结果过大"
        else:
            return "计算结果过小"

    def validate_dataset(self, dataset: Dict) -> Dict:
        """验证整个数据集"""
        examples = dataset.get('examples', [])
        validation_results = []
        error_statistics = {
            "求解器未返回结果": 0,
            "无法确定期望结果": 0,
            "计算结果过大": 0,
            "计算结果过小": 0
        }

        for example in examples:
            result = self.validate_example(example)
            validation_results.append(result)

            # 统计错误类型
            if not result['is_correct'] and result['error_details']:
                error_type = result['error_details']['error_type']
                error_statistics[error_type] = error_statistics.get(error_type, 0) + 1

        # 计算统计信息
        total = len(validation_results)
        correct = sum(1 for r in validation_results if r['is_correct'])
        accuracy = correct / total if total > 0 else 0

        # 创建验证摘要
        validation_summary = {
            "total_examples": total,
            "correct_examples": correct,
            "accuracy": accuracy,
            "error_statistics": error_statistics,
            "detailed_results": validation_results
        }

        return validation_summary

    def save_validation_report(self, validation_summary: Dict, file_path: str) -> None:
        """保存验证报告到文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(validation_summary, f, ensure_ascii=False, indent=2)
            print(f"验证报告已保存至 {file_path}")
        except Exception as e:
            print(f"保存验证报告失败: {e}")

    def print_validation_summary(self, validation_summary: Dict) -> None:
        """打印验证摘要"""
        print("\n=== 验证摘要 ===")
        print(f"总示例数: {validation_summary['total_examples']}")
        print(f"正确示例数: {validation_summary['correct_examples']}")
        print(f"准确率: {validation_summary['accuracy']:.2%}")

        print("\n错误统计:")
        for error_type, count in validation_summary['error_statistics'].items():
            if count > 0:
                print(f"- {error_type}: {count}次")

    def analyze_errors(self, validation_summary: Dict) -> Dict:
        """分析错误模式"""
        error_patterns = {}

        for result in validation_summary['detailed_results']:
            if not result['is_correct']:
                # 获取bug类型
                bug_description = result['input']['bug_description']
                bug_type = self._extract_bug_type(bug_description)

                # 统计每种bug类型的错误
                if bug_type not in error_patterns:
                    error_patterns[bug_type] = {
                        "count": 0,
                        "examples": []
                    }

                error_patterns[bug_type]["count"] += 1
                error_patterns[bug_type]["examples"].append({
                    "expected": result['expected_result'],
                    "actual": result['actual_result'],
                    "error_details": result['error_details']
                })

        return error_patterns

    def _extract_bug_type(self, bug_description: str) -> str:
        """从bug描述中提取bug类型"""
        if "空值(null)被错误地移除" in bug_description:
            return "missing_null_values"
        elif "添加了一个随机值列" in bug_description or "添加了一行随机值" in bug_description:
            return "appended_random_values"
        elif "每两行被错误地合并" in bug_description:
            return "merged_rows"
        elif "行被旋转" in bug_description or "列被旋转" in bug_description:
            return "rotated_data"
        elif "值被错误地替换为'ERROR'" in bug_description:
            return "replaced_values"
        else:
            return "unknown_bug_type"


# 使用示例
if __name__ == "__main__":
    # 从文件加载测试数据
    with open("generated_dataset.json", 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 创建验证器实例
    validator = BBEHBuggyTablesValidator()

    # 验证数据集
    validation_summary = validator.validate_dataset(test_data)

    # 打印验证摘要
    validator.print_validation_summary(validation_summary)

    # 分析错误模式
    error_patterns = validator.analyze_errors(validation_summary)

    # 保存验证报告
    validator.save_validation_report(validation_summary, "validation_report.json")

    # 打印错误模式分析
    print("\n=== 错误模式分析 ===")
    for bug_type, pattern in error_patterns.items():
        print(f"\n{bug_type}:")
        print(f"错误次数: {pattern['count']}")
        if pattern['examples']:
            print("示例错误:")
            for i, example in enumerate(pattern['examples'][:3], 1):  # 只显示前3个示例
                print(f"  {i}. 期望值: {example['expected']}, 实际值: {example['actual']}")
                print(f"     错误类型: {example['error_details']['error_type']}")
