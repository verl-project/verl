import logging
import re
import time
from typing import Dict, Any, Optional, Union
from internbootcamp.bootcamp.base import Basebootcamp
from internbootcamp.libs.bbeh_multistep_arithmetic.bbeh_arithmetic_generator import BBEHArithmeticGenerator
from internbootcamp.libs.bbeh_multistep_arithmetic.bbeh_arithmetic_solver import BBEHArithmeticSolver
from internbootcamp.libs.bbeh_multistep_arithmetic.bbeh_arithmetic_validor import BBEHArithmeticVerifier

def print_section(title: str, char: str = "=") -> None:
    """打印带有分隔线的章节标题"""
    width = 80
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}\n")

def format_statistics(stats: Dict) -> str:
    """格式化统计信息"""
    output = []
    output.append("总体统计:")
    output.append(f"  总测试案例: {stats['total_cases']}")
    output.append(f"  正确答案数: {stats['correct_answers']}")
    output.append(f"  总体成功率: {stats['success_rate']}%")

    output.append("\n按难度分类:")
    for diff in ['easy', 'medium', 'hard']:
        diff_stats = stats['by_difficulty'][diff]
        output.append(
            f"  {diff.capitalize()}: {diff_stats['correct']}/{diff_stats['total']} ({diff_stats['success_rate']})")

    output.append("\n按表达式长度分类:")
    for length in ['short', 'medium', 'long']:
        length_stats = stats['by_expression_length'][length]
        output.append(
            f"  {length.capitalize()}: {length_stats['correct']}/{length_stats['total']} ({length_stats['success_rate']})")

    output.append("\n运算符使用统计:")
    for op, op_stats in stats['by_operator'].items():
        output.append(f"  {op}: {op_stats['correct']}/{op_stats['total']} ({op_stats['success_rate']})")

    return "\n".join(output)


def format_statistics(stats: Dict) -> str:
    """格式化统计信息"""
    output = []
    output.append("总体统计:")
    output.append(f"  总测试案例: {stats['total_cases']}")
    output.append(f"  正确答案数: {stats['correct_answers']}")
    output.append(f"  总体成功率: {stats['success_rate']}%")

    output.append("\n按难度分类:")
    for diff in ['easy', 'medium', 'hard']:
        diff_stats = stats['by_difficulty'][diff]
        output.append(
            f"  {diff.capitalize()}: {diff_stats['correct']}/{diff_stats['total']} ({diff_stats['success_rate']})")

    output.append("\n按表达式长度分类:")
    for length in ['short', 'medium', 'long']:
        length_stats = stats['by_expression_length'][length]
        output.append(
            f"  {length.capitalize()}: {length_stats['correct']}/{length_stats['total']} ({length_stats['success_rate']})")

    output.append("\n运算符使用统计:")
    for op, op_stats in stats['by_operator'].items():
        output.append(f"  {op}: {op_stats['correct']}/{op_stats['total']} ({op_stats['success_rate']})")

    return "\n".join(output)

class BBEHMultistepArithmeticV2bootcamp(Basebootcamp):  # 继承Basebootcamp类以保持一致
    verifier = BBEHArithmeticVerifier()
    def __init__(self, difficulty: str = "medium", timeout: int = 30, language: str = "zh"):
        """
        初始化BBEH算术训练场系统

        Args:
            difficulty: 难度级别 ("easy", "medium", "hard")
            timeout: 求解超时时间（秒）
            language: 语言选择 ("en", "zh")
        """
        self.generator = BBEHArithmeticGenerator()
        self.solver = BBEHArithmeticSolver()
        self.difficulty = difficulty
        self.timeout = timeout
        self.language = language
        self.logger = logging.getLogger(__name__)
        self.verification_details = {}  # 添加验证详情存储

    def case_generator(self, max_attempts: int = 5) -> Dict:
        """生成一个新的算术表达式案例"""
        # print(f"[开始] 生成{self.difficulty}难度的算术表达式")
        # print("-" * 40)

        for attempt in range(max_attempts):
            try:
                # print(f"[尝试 {attempt + 1}/{max_attempts}]")
                start_time = time.time()

                case = self.generator.generate_case(difficulty=self.difficulty)

                # 设置超时保护
                solution = None
                while time.time() - start_time < self.timeout:
                    solution = self.solver.solve(case["expression"])
                    if solution is not None:
                        break
                    time.sleep(0.1)

                if solution is None:
                    # print("⚠️ 求解超时，重试中...")
                    continue

                # 构建完整的案例
                case["solution"] = solution
                case["language"] = self.language

                # generation_time = time.time() - start_time
                # print(f"✓ 成功生成表达式 (用时: {generation_time:.2f}s)")
                # print("-" * 40)

                return case

            except Exception as e:
                # print(f"❌ 错误: {str(e)}")
                # print("重试中...")
                continue

        # print("⚠️ 达到最大尝试次数，使用备用表达式")
        return self._generate_fallback_case()

    def prompt_func(self, identity: Dict) -> str:
        """生成提示文本"""
        if self.language == "zh":
            # 中文提示
            statements = [
                f"""你是一位精通算术的智能助手。请计算以下算术表达式：

{identity['expression']}

表达式使用标准算术运算符 (+, -, *, /) 和自定义运算符 (><, ;, @, <>, [], #, !, ~, &, :, ][)。

自定义运算符说明：
- >< 表示取最大值，例如：a >< b = max(a, b)
- ; 表示连接，例如：a ; b = a * 10^(数字b的位数) + b
- @ 表示平均值，例如：a @ b = (a + b) / 2
- <> 表示交换，例如：a <> b = b * 10^(数字a的位数) + a
- [] 表示绝对差，例如：a [] b = |a - b|
- # 表示取模，例如：a # b = a % b
- ! 表示阶乘，例如：a ! = a的阶乘
- ~ 表示取反，例如：a ~ b = -a + b
- & 表示数字和，例如：a & b = 各位数字之和
- : 表示乘方，例如：a : b = a^b
- ][ 表示最小公倍数，例如：a ][ b = lcm(a, b)

其中，one=1, two=2, three=3, four=4, five=5, six=6, seven=7, eight=8, nine=9, ten=10

请提供准确的计算结果，以小数形式表示。

请仔细思考每一步计算，确保结果的精确性。"""
            ]
        else:
            # 英文提示
            statements = [
                f"""You are an intelligent assistant specialized in arithmetic. Please calculate the following arithmetic expression:

{identity['expression']}

The expression uses standard arithmetic operators (+, -, *, /) and custom operators (><, ;, @, <>, [], #, !, ~, &, :, ][).

Custom operators explanation:
- >< means maximum, e.g.: a >< b = max(a, b)
- ; means concatenation, e.g.: a ; b = a * 10^(number of digits in b) + b
- @ means average, e.g.: a @ b = (a + b) / 2
- <> means swap, e.g.: a <> b = b * 10^(number of digits in a) + a
- [] means absolute difference, e.g.: a [] b = |a - b|
- # means modulo, e.g.: a # b = a % b
- ! means factorial, e.g.: a ! = factorial of a
- ~ means negation, e.g.: a ~ b = -a + b
- & means digit sum, e.g.: a & b = sum of all digits
- : means power, e.g.: a : b = a^b
- ][ means least common multiple, e.g.: a ][ b = lcm(a, b)

Where one=1, two=2, three=3, four=4, five=5, six=6, seven=7, eight=8, nine=9, ten=10

Please provide the exact calculation result in decimal form.

Think through each step carefully to ensure accuracy."""
            ]

        instruction_following = """\nLet's think step by step and output the final answer with the following format: 
Final-answer: ```json
42.5
```"""

        return statements[0] + instruction_following

    @classmethod
    def _verify_correction(cls, output: int, identity: Dict) -> float:
        """验证答案并评分"""
        try:
            if output is None:
                # print("❌ 错误: 无法从输出中提取答案")
                return 0.0

            # 验证答案
            # expected_answer = identity.get('solution', identity.get('answer'))
            is_correct = cls.verifier.verify_answer(identity, output)

            return is_correct
        except Exception as e:
            # print(f"❌ 错误: 验证过程中出现异常: {str(e)}")
            return 0.0

    @classmethod
    def extract_output(cls, output: str) -> Optional[float]:
        """从输出中提取答案"""
        try:
            # 查找Python代码块
            pattern = r"```json\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*```"
            match = re.search(pattern, output)
            if not match:
                # 尝试查找任何数字
                pattern = r"Final-answer:.*?([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
                match = re.search(pattern, output)
                if not match:
                    return None

            # 转换为浮点数
            return float(match.group(1))

        except (ValueError, AttributeError) as e:
            return None

    def _generate_fallback_case(self) -> Dict:
        """生成一个简单的后备案例"""
        expression = "(2 + 3) * 4"  # 简单且保证可解的表达式
        answer = 20.0

        return {
            "expression": expression,
            "answer": answer,
            "solution": answer,
            "difficulty": "easy",
            "language": self.language,
            "is_fallback": True
        }

    def _count_operators(self, expression: str) -> Dict[str, int]:
        """统计表达式中的运算符使用情况"""
        operators = {
            '+': 0, '-': 0, '*': 0, '/': 0, '><': 0, ';': 0,
            '@': 0, '<>': 0, '[]': 0, '#': 0, '!': 0, '~': 0,
            '&': 0, ':': 0, '][': 0
        }

        i = 0
        while i < len(expression):
            # 检查两字符运算符
            if i + 1 < len(expression):
                two_char = expression[i:i + 2]
                if two_char in operators:
                    operators[two_char] += 1
                    i += 2
                    continue

            # 检查单字符运算符
            if expression[i] in operators:
                operators[expression[i]] += 1

            i += 1

        return {op: count for op, count in operators.items() if count > 0}

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.verifier.get_statistics()

    def reset_statistics(self) -> None:
        """重置统计信息"""
        self.verifier.reset_statistics()

    def set_language(self, language: str) -> None:
        """设置语言"""
        if language in ["en", "zh"]:
            self.language = language
        else:
            raise ValueError("不支持的语言。请使用 'en' 或 'zh'。")

    def set_difficulty(self, difficulty: str) -> None:
        """设置难度级别"""
        if difficulty in ["easy", "medium", "hard"]:
            self.difficulty = difficulty
        else:
            raise ValueError("不支持的难度级别。请使用 'easy', 'medium', 或 'hard'。")

    def set_timeout(self, timeout: int) -> None:
        """设置超时时间"""
        if timeout > 0:
            self.timeout = timeout
        else:
            raise ValueError("超时时间必须为正数。")


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        print_section("BBEH算术表达式求解器")

        # 创建训练场实例
        print_section("初始化系统", "-")
        bootcamp = BBEHMultistepArithmeticV2bootcamp(
            difficulty="medium",
            timeout=30,
            language="zh"
        )
        print("✓ 系统初始化完成")

        # 生成测试案例
        print_section("生成测试案例", "-")
        case = bootcamp.case_generator()
        print("生成的表达式:")
        print(f"  难度: {case['difficulty'].upper()}")
        print(f"  表达式: {case['expression']}")
        print(f"  预期答案: {case['solution']}")

        # 获取提示文本
        print_section("生成提示文本", "-")
        prompt = bootcamp.prompt_func(case)
        print(prompt)

        # 测试答案验证
        print_section("答案验证测试", "-")

        # 测试正确答案
        print("[测试1] 验证正确答案")
        correct_output = f"Final-answer: ```json\n{case['solution']}\n```"
        score = bootcamp.verify_score(correct_output, case, short_penalty=False)
        print(f"验证结果: {'✓ 通过' if score == 1.0 else '✗ 失败'}")
        print(f"得分: {score}\n")

        # 测试错误答案
        print("[测试2] 验证错误答案")
        wrong_output = f"Final-answer: ```json\n{case['solution']+1}\n```"
        score = bootcamp.verify_score(wrong_output, case, short_penalty=False)
        print(f"验证结果: {'✓ 通过' if score == 0.0 else '✗ 失败'}")
        print(f"得分: {score}")

        # 测试不同难度
        print_section("不同难度测试", "-")
        for difficulty in ["easy", "medium", "hard"]:
            print(f"\n[{difficulty.upper()}]")
            bootcamp.set_difficulty(difficulty)
            case = bootcamp.case_generator()
            print(f"表达式: {case['expression']}")
            print(f"答案: {case['solution']}")

        # 获取统计信息
        print_section("统计信息", "-")
        stats = bootcamp.get_statistics()
        print(format_statistics(stats))

        # 测试总结
        print_section("测试完成", "=")
        print(f"总测试用例数: {stats['total_cases']}")
        print(f"成功用例数: {stats['correct_answers']}")
        print(f"总体成功率: {stats['success_rate']}%")

    except Exception as e:
        logger.error(f"测试过程中出现错误: {str(e)}")


