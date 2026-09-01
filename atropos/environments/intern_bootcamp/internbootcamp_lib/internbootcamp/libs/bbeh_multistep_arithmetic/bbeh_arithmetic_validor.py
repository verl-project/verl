import math
import logging
from typing import Dict, Union


class BBEHArithmeticVerifier:
    def __init__(self):
        self.epsilon = 1e-10
        self.logger = logging.getLogger(__name__)
        self.stats = {
            "total": 0,
            "correct": 0,
            "by_difficulty": {
                "easy": {"total": 0, "correct": 0},
                "medium": {"total": 0, "correct": 0},
                "hard": {"total": 0, "correct": 0}
            },
            "by_operator": {},
            "by_expression_length": {
                "short": {"total": 0, "correct": 0},
                "medium": {"total": 0, "correct": 0},
                "long": {"total": 0, "correct": 0}
            }
        }

    def verify_answer(self, case: Dict, answer: float) -> bool:
        """验证答案是否正确"""
        try:
            expected = case["answer"]
            difficulty = case.get("difficulty", "medium")
            expression = case.get("expression", "")

            # 验证答案
            is_correct = self._validate_solution(expected, answer)

            # 更新统计信息
            self._update_statistics(is_correct, difficulty, expression)

            return is_correct

        except Exception as e:
            self.logger.error(f"Error in verification: {str(e)}")
            return False

    def _validate_solution(self, expected: float, calculated: float) -> bool:
        """验证解决方案"""
        try:
            # 处理无穷大的情况
            if math.isinf(expected) and math.isinf(calculated):
                return 1 if expected * calculated > 0 else 0 # 确保符号相同

            # 处理NaN的情况
            if math.isnan(expected) or math.isnan(calculated):
                return 0

            # 处理零附近的值
            if abs(expected) < self.epsilon and abs(calculated) < self.epsilon:
                return 1

            # 处理普通情况
            if abs(expected) > self.epsilon:
                error = abs(expected - calculated)
                relative_error = 1 - min(abs((expected - calculated) / abs(expected)), 1.0)
                return relative_error

            return abs(expected - calculated) < self.epsilon

        except Exception as e:
            self.logger.error(f"Error in solution validation: {str(e)}")
            return 0

    def _update_statistics(self, is_correct: bool, difficulty: str, expression: str) -> None:
        """更新统计信息"""
        try:
            # 更新总计数
            self.stats["total"] += 1
            if is_correct:
                self.stats["correct"] += 1

            # 更新难度统计
            if difficulty in self.stats["by_difficulty"]:
                self.stats["by_difficulty"][difficulty]["total"] += 1
                if is_correct:
                    self.stats["by_difficulty"][difficulty]["correct"] += 1

            # 更新表达式长度统计
            length_category = self._categorize_expression_length(expression)
            self.stats["by_expression_length"][length_category]["total"] += 1
            if is_correct:
                self.stats["by_expression_length"][length_category]["correct"] += 1

            # 更新运算符统计
            operators = self._extract_operators(expression)
            for op in operators:
                if op not in self.stats["by_operator"]:
                    self.stats["by_operator"][op] = {"total": 0, "correct": 0}
                self.stats["by_operator"][op]["total"] += 1
                if is_correct:
                    self.stats["by_operator"][op]["correct"] += 1

        except Exception as e:
            self.logger.error(f"Error updating statistics: {str(e)}")

    def _categorize_expression_length(self, expression: str) -> str:
        """根据表达式长度进行分类"""
        length = len(expression)
        if length < 30:
            return "short"
        elif length < 60:
            return "medium"
        else:
            return "long"

    def _extract_operators(self, expression: str) -> set:
        """提取表达式中的运算符"""
        operators = set()
        operator_chars = {'+', '-', '*', '/', '><', ';', '@', '<>', '[]', '#', '!', '~', '&', ':', ']['}

        i = 0
        while i < len(expression):
            # 检查两字符运算符
            if i + 1 < len(expression):
                two_char = expression[i:i + 2]
                if two_char in operator_chars:
                    operators.add(two_char)
                    i += 2
                    continue

            # 检查单字符运算符
            if expression[i] in operator_chars:
                operators.add(expression[i])

            i += 1

        return operators

    def get_statistics(self) -> Dict:
        """获取验证统计信息"""
        stats = {
            "total_cases": self.stats["total"],
            "correct_answers": self.stats["correct"],
            "success_rate": 0 if self.stats["total"] == 0 else
            (self.stats["correct"] / self.stats["total"]) * 100,
            "by_difficulty": {},
            "by_expression_length": {},
            "by_operator": {}
        }

        # 处理难度统计
        for diff, counts in self.stats["by_difficulty"].items():
            total = counts["total"]
            correct = counts["correct"]
            success_rate = 0 if total == 0 else (correct / total) * 100
            stats["by_difficulty"][diff] = {
                "total": total,
                "correct": correct,
                "success_rate": f"{success_rate:.2f}%"
            }

        # 处理表达式长度统计
        for length, counts in self.stats["by_expression_length"].items():
            total = counts["total"]
            correct = counts["correct"]
            success_rate = 0 if total == 0 else (correct / total) * 100
            stats["by_expression_length"][length] = {
                "total": total,
                "correct": correct,
                "success_rate": f"{success_rate:.2f}%"
            }

        # 处理运算符统计
        for op, counts in self.stats["by_operator"].items():
            total = counts["total"]
            correct = counts["correct"]
            success_rate = 0 if total == 0 else (correct / total) * 100
            stats["by_operator"][op] = {
                "total": total,
                "correct": correct,
                "success_rate": f"{success_rate:.2f}%"
            }

        return stats

    def reset_statistics(self) -> None:
        """重置统计信息"""
        self.stats = {
            "total": 0,
            "correct": 0,
            "by_difficulty": {
                "easy": {"total": 0, "correct": 0},
                "medium": {"total": 0, "correct": 0},
                "hard": {"total": 0, "correct": 0}
            },
            "by_operator": {},
            "by_expression_length": {
                "short": {"total": 0, "correct": 0},
                "medium": {"total": 0, "correct": 0},
                "long": {"total": 0, "correct": 0}
            }
        }

    def format_case(self, case: Dict, language: str = "en") -> str:
        """格式化案例为可读文本"""
        expression = case["expression"]
        if language == "en":
            return (
                f"Please evaluate the following arithmetic expression:\n\n"
                f"{expression}\n\n"
                f"The expression uses standard arithmetic operators (+, -, *, /) "
                f"and custom operators (><, ;, @, <>, [], #, !, ~, &, :, ][).\n"
                f"Please provide your answer as a decimal number."
            )
        else:  # Chinese
            return (
                f"请计算下面的算术表达式：\n\n"
                f"{expression}\n\n"
                f"表达式使用标准算术运算符 (+, -, *, /) "
                f"和自定义运算符 (><, ;, @, <>, [], #, !, ~, &, :, ][)。\n"
                f"请以小数形式提供你的答案。"
            )
