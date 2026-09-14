import random
import math
import logging
from typing import Dict, Tuple


class BBEHArithmeticGenerator:
    def __init__(self):
        self.operators = ['+', '-', '*', '/', '><', ';', '@', '<>', '[]', '#', '!', '~', '&', ':', '][']
        self.precedence = {
            '+': 1, '-': 1,
            '*': 2, '/': 2,
            '><': 3, ';': 3,
            '@': 4, '<>': 4, '[]': 4,
            '#': 5, '!': 5,
            '~': 6, '&': 6,
            ':': 7, '][': 7
        }
        self.number_words = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }
        self.logger = logging.getLogger(__name__)
        self.max_value = 1.7976931348623157e+308
        self.min_value = -1.7976931348623157e+308
        self.epsilon = 1e-10

    def generate_case(self, min_depth: int = 3, max_depth: int = 6,
                      max_length: int = 50, difficulty: str = "medium") -> Dict:
        """生成一个算术表达式案例"""
        # 根据难度调整参数
        if difficulty == "easy":
            min_depth = 2
            max_depth = 4
            max_length = 30
        elif difficulty == "hard":
            min_depth = 4
            max_depth = 8
            max_length = 70

        expression, answer = self._generate_expression(min_depth, max_depth, max_length)

        return {
            "expression": expression,
            "answer": answer,
            "difficulty": difficulty
        }

    def _generate_subexpression(self, depth: int, current_length: int,
                                max_length: int) -> Tuple[str, int]:
        """生成子表达式"""
        if depth == 0 or current_length >= max_length:
            if random.random() < 0.3:
                word = random.choice(list(self.number_words.keys()))
                return word, current_length + 1
            return str(random.randint(-100, 100)), current_length + 1

        choice = random.random()

        if choice < 0.4:
            op = random.choice(self.operators)
            left, left_length = self._generate_subexpression(depth - 1, current_length, max_length)
            right, right_length = self._generate_subexpression(depth - 1, left_length + 1, max_length)
            return f"({left} {op} {right})", right_length + 3
        else:
            op = random.choice(self.operators)
            left, left_length = self._generate_subexpression(depth - 1, current_length, max_length)
            right, right_length = self._generate_subexpression(depth - 1, left_length + 1, max_length)
            return f"({left} {op} {right})", right_length + 3

    def _generate_expression(self, min_depth: int, max_depth: int,
                             max_length: int) -> Tuple[str, float]:
        """生成完整表达式及其答案"""
        from libs.bbeh_multistep_arithmetic.bbeh_arithmetic_solver import BBEHArithmeticSolver
        solver = BBEHArithmeticSolver()

        while True:
            try:
                expression, _ = self._generate_subexpression(
                    random.randint(min_depth, max_depth), 0, max_length
                )
                answer = solver.solve(expression)

                # 验证答案是否有效
                if not math.isinf(answer) and not math.isnan(answer):
                    return expression, answer
            except Exception as e:
                self.logger.warning(f"Expression generation failed: {str(e)}")
                continue

    def _validate_expression(self, tokens):
        """验证表达式的有效性"""
        stack = []
        for token in tokens:
            if token == '(':
                stack.append(token)
            elif token == ')':
                if not stack:
                    return False
                stack.pop()
        return len(stack) == 0

    def _validate_expression_structure(self, tokens):
        """验证表达式的结构是否合法"""
        stack = []
        operand_count = 0
        operator_count = 0

        for token in tokens:
            if token == '(':
                stack.append(token)
            elif token == ')':
                if not stack:
                    return False
                stack.pop()
            elif token in self.operators:
                operator_count += 1
            else:
                operand_count += 1

        # 检查括号是否匹配
        if stack:
            return False

        # 检查操作数和操作符的数量关系
        return operand_count == operator_count + 1

    def _tokenize(self, expr):
        """将表达式转换为token列表"""
        try:
            tokens = []
            i = 0
            while i < len(expr):
                char = expr[i]

                # 处理空格
                if char.isspace():
                    i += 1
                    continue

                # 处理数字单词
                if char.isalpha() and (i == 0 or not expr[i - 1].isdigit()):
                    word = ''
                    while i < len(expr) and expr[i].isalpha():
                        word += expr[i]
                        i += 1
                    if word in self.number_words:
                        tokens.append(str(self.number_words[word]))
                    else:
                        raise ValueError(f"Unknown word: {word}")
                    continue

                # 处理数字（包括科学记数法）
                if char.isdigit() or (char == '-' and (not tokens or tokens[-1] in self.operators + ['('])):
                    num = char
                    i += 1
                    while i < len(expr) and (expr[i].isdigit() or expr[i] == '.'):
                        num += expr[i]
                        i += 1

                    if i < len(expr) and (expr[i] == 'e' or expr[i] == 'E'):
                        num += expr[i]
                        i += 1
                        if i < len(expr) and (expr[i] == '+' or expr[i] == '-'):
                            num += expr[i]
                            i += 1
                        while i < len(expr) and expr[i].isdigit():
                            num += expr[i]
                            i += 1

                    try:
                        float(num)
                        tokens.append(num)
                    except ValueError:
                        raise ValueError(f"Invalid number format: {num}")
                    continue

                # 处理括号
                if char in '()':
                    tokens.append(char)
                    i += 1
                    continue

                # 处理运算符
                if i < len(expr):
                    max_op_len = 3
                    matched = False
                    for length in range(max_op_len, 0, -1):
                        if i + length <= len(expr):
                            potential_op = expr[i:i + length]
                            if potential_op in self.operators:
                                tokens.append(potential_op)
                                i += length
                                matched = True
                                break
                    if not matched:
                        raise ValueError(f"Invalid character: {char}")

            if not self._validate_expression(tokens):
                raise ValueError("Invalid expression: Mismatched parentheses")

            return tokens
        except Exception as e:
            self.logger.error(f"Error in tokenization: {str(e)}")
            raise
