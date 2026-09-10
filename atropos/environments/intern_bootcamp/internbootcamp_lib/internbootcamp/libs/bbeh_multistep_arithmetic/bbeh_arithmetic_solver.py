import math
import logging
from typing import Dict, List, Union, Optional

class BBEHArithmeticSolver:
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
        self.epsilon = 1e-10
        self.max_value = 1.7976931348623157e+308
        self.min_value = -1.7976931348623157e+308

    def solve(self, expression: str) -> float:
        """求解算术表达式"""
        try:
            tokens = self._tokenize(expression)
            if not tokens:
                return float('inf')

            if not self._validate_expression_structure(tokens):
                return float('inf')

            return self._evaluate_expression(tokens)
        except Exception as e:
            self.logger.error(f"Error solving expression: {str(e)}")
            return float('inf')

    def _tokenize(self, expr: str) -> List[str]:
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

    def _validate_expression(self, tokens: List[str]) -> bool:
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

    def _validate_expression_structure(self, tokens: List[str]) -> bool:
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

        if stack:
            return False

        return operand_count == operator_count + 1

    def _safe_float_conversion(self, value: Union[str, int, float]) -> float:
        """安全地将值转换为浮点数"""
        try:
            if isinstance(value, (int, float)):
                if math.isinf(value) or math.isnan(value):
                    return float('inf') if value > 0 else float('-inf')
                return float(value)

            if isinstance(value, str):
                try:
                    result = float(value)
                    if math.isinf(result) or math.isnan(result):
                        return float('inf') if result > 0 else float('-inf')
                    if abs(result) > self.max_value:
                        return float('inf') if result > 0 else float('-inf')
                    if abs(result) != 0 and abs(result) < self.min_value:
                        return 0.0
                    return result
                except ValueError:
                    if value.lower() in self.number_words:
                        return float(self.number_words[value.lower()])
                    raise

            raise ValueError(f"Cannot convert {type(value)} to float")
        except (ValueError, TypeError) as e:
            self.logger.error(f"Error in safe_float_conversion: {str(e)}")
            return float('inf')

    def _perform_operation(self, op: str, a: float, b: float) -> float:
        """执行具体的运算"""
        try:
            if math.isinf(a) or math.isinf(b):
                if op in ['+', '*', '@', '<>', '[]', '#', '&']:
                    return float('inf') if (a > 0 or b > 0) else float('-inf')
                elif op in ['-', '/', '><', ';', '!', '~', ':', '][']:
                    return float('inf')

            if op == '/' and abs(b) < self.epsilon:
                return float('inf')

            def safe_operation(func):
                try:
                    result = func()
                    if math.isinf(result) or math.isnan(result):
                        return float('inf') if result > 0 else float('-inf')
                    if abs(result) > self.max_value:
                        return float('inf') if result > 0 else float('-inf')
                    if abs(result) != 0 and abs(result) < self.min_value:
                        return 0.0
                    return result
                except (OverflowError, ValueError, ZeroDivisionError):
                    return float('inf')

            if op == '+':
                result = safe_operation(lambda: a + b)
            elif op == '-':
                result = safe_operation(lambda: a - b)
            elif op == '*':
                result = safe_operation(lambda: a * b)
            elif op == '/':
                result = safe_operation(lambda: a / b if abs(b) > self.epsilon else float('inf'))
            elif op == '><':
                result = safe_operation(lambda: a - b if a * b > 0 else a + b)
            elif op == ';':
                result = safe_operation(lambda: (a - b) if a + b > 0 else a - b)
            elif op == '@':
                result = safe_operation(lambda: a * b if a > b else a + b)
            elif op == '<>':
                result = safe_operation(lambda: abs(a - b))
            elif op == '[]':
                result = safe_operation(lambda: max(a, b))
            elif op == '#':
                result = safe_operation(lambda: (a * b) if a > b else min(a, b))
            elif op == '!':
                result = safe_operation(lambda: (a - b) if a * b > 0 else (a + b))
            elif op == '~':
                result = safe_operation(lambda: (2 * b - a) if a > b else b)
            elif op == '&':
                result = safe_operation(lambda: (a + b) if a * b > 0 else (a - b))
            elif op == ':':
                result = safe_operation(lambda: (a * b) if a + b > 0 else (-a * -b))
            elif op == '][':
                try:
                    if abs(a) < 1e10 and abs(b) < 1e10:
                        gcd_value = self._gcd(abs(int(a)), abs(int(b)))
                        return safe_operation(lambda: (a - b) if gcd_value == 1 else float(gcd_value))
                    else:
                        return float('inf')
                except (OverflowError, ValueError):
                    return float('inf')
            else:
                raise ValueError(f"Unknown operator: {op}")

            return result
        except Exception as e:
            self.logger.error(f"Error in operation {op} with operands {a}, {b}: {str(e)}")
            return float('inf')

    def _gcd(self, a: int, b: int) -> int:
        """计算最大公约数"""
        while b:
            a, b = b, a % b
        return a

    def _evaluate_expression(self, tokens: List[str]) -> float:
        """计算表达式的值"""
        try:
            values = []
            operators = []

            for token in tokens:
                try:
                    if token == '(':
                        operators.append(token)
                    elif token == ')':
                        while operators and operators[-1] != '(':
                            if len(values) < 2:
                                raise ValueError(f"Not enough operands for operator {operators[-1]}")
                            op = operators.pop()
                            b = self._safe_float_conversion(values.pop())
                            a = self._safe_float_conversion(values.pop())
                            result = self._perform_operation(op, a, b)
                            values.append(result)
                        if operators:
                            operators.pop()
                    elif token in self.operators:
                        while (operators and operators[-1] != '(' and
                               self.precedence.get(operators[-1], 0) >= self.precedence.get(token, 0)):
                            if len(values) < 2:
                                raise ValueError(f"Not enough operands for operator {operators[-1]}")
                            op = operators.pop()
                            b = self._safe_float_conversion(values.pop())
                            a = self._safe_float_conversion(values.pop())
                            result = self._perform_operation(op, a, b)
                            values.append(result)
                        operators.append(token)
                    else:
                        try:
                            if isinstance(token, str):
                                token_lower = token.lower()
                                if token_lower in self.number_words:
                                    value = float(self.number_words[token_lower])
                                else:
                                    value = self._safe_float_conversion(token)
                                values.append(value)
                            else:
                                raise ValueError(f"Invalid token type: {type(token)}")
                        except ValueError as ve:
                            self.logger.error(f"Error converting number: {token}")
                            raise ValueError(f"Invalid number format: {token}")

                except Exception as e:
                    self.logger.error(f"Error processing token {token}: {str(e)}")
                    return float('inf')

            while operators:
                if len(values) < 2:
                    raise ValueError("Not enough operands for remaining operators")
                op = operators.pop()
                if op == '(':
                    raise ValueError("Mismatched parentheses")
                b = self._safe_float_conversion(values.pop())
                a = self._safe_float_conversion(values.pop())
                result = self._perform_operation(op, a, b)
                values.append(result)

            if len(values) != 1:
                raise ValueError("Invalid expression: too many values")

            return values[0]

        except Exception as e:
            self.logger.error(f"Error evaluating expression: {str(e)}")
            return float('inf')
