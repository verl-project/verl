import random
import json
import math
import re
import logging
import time

# 定义极限值 - 使用Python支持的最大范围
MAX_VALUE = 1.7976931348623157e+308  # 最大浮点数
MIN_VALUE = -1.7976931348623157e+308  # 最小浮点数
EPSILON = 1e-10  # 用于浮点数比较

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义更多操作符
OPERATORS = ['+', '-', '*', '/', '><', ';', '@', '<>', '[]', '#', '!', '~', '&', ':', '][']

# 定义优先级
PRECEDENCE = {
    '+': 1, '-': 1,
    '*': 2, '/': 2,
    '><': 3, ';': 3,
    '@': 4, '<>': 4, '[]': 4,
    '#': 5, '!': 5,
    '~': 6, '&': 6,
    ':': 7, '][': 7
}

# 添加数字单词映射
NUMBER_WORDS = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
}

# 定义极限值
MAX_VALUE = 1e100
MIN_VALUE = -1e100

def safe_pop(stack, error_msg="Empty stack"):
    """安全的pop操作，包含错误处理"""
    if not stack:
        raise ValueError(error_msg)
    return stack.pop()


def perform_operation(op, a, b):
    """执行操作，改进的错误处理和大数处理"""
    try:
        # 处理无穷大和非数值的情况
        if math.isinf(a) or math.isinf(b):
            if op in ['+', '*', '@', '<>', '[]', '#', '&']:
                return float('inf') if (a > 0 or b > 0) else float('-inf')
            elif op in ['-', '/', '><', ';', '!', '~', ':', '][']:
                return float('inf')

        # 处理零和接近零的情况
        if op == '/' and abs(b) < EPSILON:
            return float('inf')

        # 防止溢出的预检查
        def safe_operation(func):
            try:
                result = func()
                # 检查结果是否在有效范围内
                if math.isinf(result) or math.isnan(result):
                    return float('inf') if result > 0 else float('-inf')
                if abs(result) > MAX_VALUE:
                    return float('inf') if result > 0 else float('-inf')
                if abs(result) != 0 and abs(result) < MIN_VALUE:
                    return 0.0
                return result
            except (OverflowError, ValueError, ZeroDivisionError):
                return float('inf')

        # 使用安全操作来执行各种运算
        if op == '+':
            result = safe_operation(lambda: a + b)
        elif op == '-':
            result = safe_operation(lambda: a - b)
        elif op == '*':
            result = safe_operation(lambda: a * b)
        elif op == '/':
            result = safe_operation(lambda: a / b if abs(b) > EPSILON else float('inf'))
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
                    gcd_value = gcd(abs(int(a)), abs(int(b)))
                    return safe_operation(lambda: (a - b) if gcd_value == 1 else float(gcd_value))
                else:
                    return float('inf')
            except (OverflowError, ValueError):
                return float('inf')
        else:
            raise ValueError(f"Unknown operator: {op}")

        return result
    except Exception as e:
        logger.error(f"Error in operation {op} with operands {a}, {b}: {str(e)}")
        return float('inf')

def validate_expression(tokens):
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


def tokenize(expr):
    """改进的tokenize函数，支持科学记数法"""
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
            if char.isalpha() and (i == 0 or not expr[i - 1].isdigit()):  # 确保不是科学记数法中的'e'
                word = ''
                while i < len(expr) and expr[i].isalpha():
                    word += expr[i]
                    i += 1
                if word in NUMBER_WORDS:
                    tokens.append(str(NUMBER_WORDS[word]))
                else:
                    logger.warning(f"Unknown word encountered: {word}")
                    raise ValueError(f"Unknown word: {word}")
                continue

            # 处理数字（包括科学记数法）
            if char.isdigit() or (char == '-' and (not tokens or tokens[-1] in OPERATORS + ['('])):
                num = char
                i += 1
                # 处理整数和小数部分
                while i < len(expr) and (expr[i].isdigit() or expr[i] == '.'):
                    num += expr[i]
                    i += 1

                # 处理科学记数法
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
                    float(num)  # 验证数字格式是否正确
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
                # 尝试匹配最长的运算符
                max_op_len = 3
                matched = False
                for length in range(max_op_len, 0, -1):
                    if i + length <= len(expr):
                        potential_op = expr[i:i + length]
                        if potential_op in OPERATORS:
                            tokens.append(potential_op)
                            i += length
                            matched = True
                            break
                if not matched:
                    raise ValueError(f"Invalid character: {char}")

        if not validate_expression(tokens):
            raise ValueError("Invalid expression: Mismatched parentheses")

        return tokens
    except Exception as e:
        logger.error(f"Error in tokenization: {str(e)}")
        raise

def validate_expression_structure(tokens):
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
        elif token in OPERATORS:
            operator_count += 1
        else:
            operand_count += 1

    # 检查括号是否匹配
    if stack:
        return False

    # 检查操作数和操作符的数量关系
    # 对于二元运算符，操作数应该比操作符多1
    return operand_count == operator_count + 1


def evaluate_expression(expression):
    """改进的表达式求值函数"""
    try:
        tokens = tokenize(expression)
        if not tokens:
            logger.error(f"Empty expression: {expression}")
            return float('inf')

        # 验证表达式结构
        if not validate_expression_structure(tokens):
            logger.error(f"Invalid expression structure: {expression}")
            return float('inf')

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
                        b = safe_float_conversion(values.pop())
                        a = safe_float_conversion(values.pop())
                        result = perform_operation(op, a, b)
                        values.append(result)
                    if operators:
                        operators.pop()  # 移除 '('
                elif token in OPERATORS:
                    while (operators and operators[-1] != '(' and
                           PRECEDENCE.get(operators[-1], 0) >= PRECEDENCE.get(token, 0)):
                        if len(values) < 2:
                            raise ValueError(f"Not enough operands for operator {operators[-1]}")
                        op = operators.pop()
                        b = safe_float_conversion(values.pop())
                        a = safe_float_conversion(values.pop())
                        result = perform_operation(op, a, b)
                        values.append(result)
                    operators.append(token)
                else:
                    # 改进的数值转换部分
                    try:
                        if isinstance(token, str):
                            token_lower = token.lower()
                            if token_lower in NUMBER_WORDS:
                                # 处理数字单词
                                value = float(NUMBER_WORDS[token_lower])
                            else:
                                # 处理数字字符串（包括科学记数法）
                                value = safe_float_conversion(token)

                            values.append(value)
                        else:
                            raise ValueError(f"Invalid token type: {type(token)}")

                    except ValueError as ve:
                        logger.error(f"Error converting number: {token}")
                        raise ValueError(f"Invalid number format: {token}")

            except Exception as e:
                logger.error(f"Error processing token {token}: {str(e)}")
                return float('inf')

        # 处理剩余的操作符
        while operators:
            if len(values) < 2:
                raise ValueError("Not enough operands for remaining operators")
            op = operators.pop()
            if op == '(':
                raise ValueError("Mismatched parentheses")
            b = safe_float_conversion(values.pop())
            a = safe_float_conversion(values.pop())
            result = perform_operation(op, a, b)
            values.append(result)

        if len(values) != 1:
            raise ValueError("Invalid expression: too many values")

        return values[0]

    except Exception as e:
        logger.error(f"Error evaluating expression '{expression}': {str(e)}")
        return float('inf')


def safe_float_conversion(value):
    """安全地将值转换为浮点数，改进的版本"""
    try:
        if isinstance(value, (int, float)):
            if math.isinf(value) or math.isnan(value):
                return float('inf') if value > 0 else float('-inf')
            return float(value)

        if isinstance(value, str):
            try:
                result = float(value)
                # 处理特殊情况
                if math.isinf(result) or math.isnan(result):
                    return float('inf') if result > 0 else float('-inf')
                # 处理超大数和超小数
                if abs(result) > MAX_VALUE:
                    return float('inf') if result > 0 else float('-inf')
                if abs(result) != 0 and abs(result) < MIN_VALUE:
                    return 0.0
                return result
            except ValueError:
                # 处理数字单词
                if value.lower() in NUMBER_WORDS:
                    return float(NUMBER_WORDS[value.lower()])
                raise

        raise ValueError(f"Cannot convert {type(value)} to float")
    except (ValueError, TypeError) as e:
        logger.error(f"Error in safe_float_conversion: {str(e)}")
        return float('inf')

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def generate_expression(min_depth=3, max_depth=6, max_length=50):
    """改进的表达式生成函数"""
    def generate_subexpression(depth, current_length):
        if depth == 0 or current_length >= max_length:
            if random.random() < 0.3:
                return random.choice(list(NUMBER_WORDS.keys())), current_length + 1
            return str(random.randint(-1000000, 1000000)), current_length + 1

        choice = random.random()

        if choice < 0.4:
            # 生成简单的二元运算表达式
            op = random.choice(OPERATORS)
            left, left_length = generate_subexpression(depth - 1, current_length)
            right, right_length = generate_subexpression(depth - 1, left_length + 1)
            return f"({left} {op} {right})", right_length + 3

        else:
            # 生成带括号的表达式
            op = random.choice(OPERATORS)
            left, left_length = generate_subexpression(depth - 1, current_length)
            right, right_length = generate_subexpression(depth - 1, left_length + 1)
            return f"({left} {op} {right})", right_length + 3

    while True:
        try:
            expression, _ = generate_subexpression(random.randint(min_depth, max_depth), 0)
            # 验证生成的表达式
            tokens = tokenize(expression)
            if validate_expression_structure(tokens):
                answer = evaluate_expression(expression)
                if not math.isinf(answer):
                    return expression, answer
        except:
            continue

def generate_dataset(num_samples=100):
    dataset = []
    for _ in range(num_samples):
        expression, answer = generate_expression()
        dataset.append({"expression": expression, "answer": answer})
    return dataset

def solve_expression(expression):
    return evaluate_expression(expression)

def validate_solution(expression, expected_answer, calculated_answer):
    """改进的解决方案验证函数"""
    # 处理无穷大的情况
    if math.isinf(expected_answer) and math.isinf(calculated_answer):
        return expected_answer * calculated_answer > 0  # 确保符号相同

    # 处理零附近的值
    if abs(expected_answer) < EPSILON and abs(calculated_answer) < EPSILON:
        return True

    # 处理普通情况
    if abs(expected_answer) > EPSILON:
        relative_error = abs((expected_answer - calculated_answer) / expected_answer)
        return relative_error < EPSILON

    return abs(expected_answer - calculated_answer) < EPSILON

def performance_test(num_expressions=1000):
    start_time = time.time()
    for _ in range(num_expressions):
        expression, _ = generate_expression()
        evaluate_expression(expression)
    end_time = time.time()
    avg_time = (end_time - start_time) / num_expressions
    print(f"Average time per expression: {avg_time:.6f} seconds")

def test_system():
    # 从文件加载数据集
    with open("bbeh_arithmetic_dataset.json", "r") as f:
        dataset = json.load(f)

    total_tests = len(dataset)
    passed_tests = 0

    for item in dataset:
        expression = item["expression"]
        expected_answer = item["answer"]

        calculated_answer = solve_expression(expression)
        is_correct = validate_solution(expression, expected_answer, calculated_answer)

        if is_correct:
            passed_tests += 1
        else:
            print(f"Failed test: {expression}")
            print(f"Expected: {expected_answer}, Calculated: {calculated_answer}")

    success_rate = passed_tests / total_tests * 100
    print(f"Passed {passed_tests} out of {total_tests} tests.")
    print(f"Success rate: {success_rate:.2f}%")

    # 添加边界测试
    edge_cases = [
        # 基本运算的边界情况
        "1e308 + 1",  # 接近最大值
        "-1e308 - 1",  # 接近最小值
        "1e-307 * 1e-307",  # 接近最小正数
        "1e307 * 2",  # 溢出测试
        "1 / 1e-307",  # 除法边界
        # 自定义运算符的边界情况
        "1e100 >< -1e100",
        "1e100 ; 1e100",
        "1e100 + 1e100",
        "-1e100 - 1e100",
        "1e-100 * 1e-100",
        "1 / 1e-100",
        "1e100 >< -1e100",
        "1e100 ; 1e100",
        "1e100 @ 1e100",
        "1e100 <> -1e100",
        "1e100 [] -1e100",
        "1e100 # -1e100",
        "1e100 ! -1e100",
        "1e100 ~ -1e100",
        "1e100 & -1e100",
        "1e100 : -1e100",
        "1e100 ][ -1e100"
    ]

    print("\nTesting edge cases:")
    for case in edge_cases:
        result = evaluate_expression(case)
        print(f"{case} = {result}")

    # 运行性能测试
    print("\nRunning performance test:")
    performance_test()

# 生成数据集并保存到文件
dataset = generate_dataset()
with open("bbeh_arithmetic_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

# 运行测试
test_system()
