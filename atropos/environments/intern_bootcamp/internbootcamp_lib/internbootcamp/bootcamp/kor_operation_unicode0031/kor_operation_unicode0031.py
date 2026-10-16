"""# 谜题训练场开发任务

## 任务概述
你是一位资深程序员，我需要你帮我实现一个特定谜题的训练场环境类。这个类继承自`Basebootcamp`，用于生成谜题实例并验证解答。

## 背景说明
我正在开发一系列谜题训练场，每个训练场对应一个特定类型的谜题。训练场类命名为`{PuzzleName}bootcamp`，其中`PuzzleName`是谜题的名称。

每个训练场类主要提供两个核心功能：
1. 生成该谜题类型的问题实例
2. 验证用户对问题的回答是否正确

## 技术接口规范

### 类方法实现要求

```python
from bootcamp import Basebootcamp

class {PuzzleName}bootcamp(Basebootcamp):
    def __init__(self, **params):
        \"\"\"
        请你自定义params，以保存该puzzle相关的参数，例如网格大小等，参数配有默认值
        \"\"\"
        pass
    
    def case_generator(self):
        \"\"\"
        生成谜题实例，提示：为保证谜题有解，可以先生成结果再对结果处理得到谜题
        返回：一个可JSON序列化的字典（避免包含set等无法通过json.dumps处理的数据结构）
        \"\"\"
        pass
    
    @staticmethod
    def prompt_func(question_case) -> str:
        \"\"\"
        将case_generator生成的谜题实例转换为文本形式的问题，问题中包含问题背景、对谜题规则的介绍、具体要解决的谜题实例、期望最终答案的格式，
        例如：你是xxxx，请你解答yyyy，规则如下：yyyy，最终答案放置在：zzzzz
        注意：请参照提供的谜题描述进行复述，规则应当描述详细，包括任务背景、具体任务操作规则、对题目格式和答案格式的含义介绍等，

        参数:
            question_case: 由case_generator生成的谜题实例
            
        返回:
            str: 格式化的问题字符串
            
        注意:
            1. 需考虑问题的格式，以便后续能正确提取
            2. 问题描述中应包含期望的答案格式说明，以便后续能正确提取，为了避免抽取时匹配出干扰项，请要求模型将答案放在特定标签（如双括号）内，例如[[your answer here]]
        \"\"\"
        pass
    
    @staticmethod
    def extract_output(output):
        \"\"\"
        从LLM的回复中提取符合格式要求的答案，如有多个，请抽取最后一个，避免使用re.search等只抽取第一个结果的方式。
        
        参数:
            output: LLM的完整输出（包含原始问题和回答）
            
        返回:
            提取的答案，若未找到符合格式的答案则返回None
        \"\"\"
        pass
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        \"\"\"
        验证提取的答案是否正确，注意一个问题可以能有多个解，按照谜题规则进行检验，不要直接匹配可能的答案。
        
        参数:
            solution: extract_output提取的答案
            identity: case_generator生成的谜题实例
            
        返回:
            bool: 答案是否正确
        \"\"\"
        pass
```

### 验证评分方法（基类已实现）

```python
@classmethod
def verify_score(cls, model_output, identity:dict, format_score=0.1) -> float:
    \"\"\"
    验证输出结果并评分。
    
    参数:
        model_output: 模型的完整输出
        identity: 谜题实例（由case_generator生成）
        format_score: 答案格式正确时的基础分数
    
    返回:
        float: 评分结果（0-1之间）
    \"\"\"
    score = 0. 
    try:
        extract_solution = cls.extract_output(model_output)
        if extract_solution is None:
            return score
        else:
            score = format_score # 格式正确时的基础分数
        if cls._verify_correction(extract_solution, identity):
            score = 1.  # 答案完全正确时的满分
    except Exception as e:
        # 处理异常情况
        pass
    return score
```

### 使用示例

```python
# 初始化谜题训练场
bootcamp = Puzzlebootcamp()

# 生成谜题实例
case = bootcamp.case_generator()

# 将谜题转换为文本问题
prompt = Puzzlebootcamp.prompt_func(case)

# 获取LLM对问题的解答
response = get_response(prompt, \"LLM\")

# 从完整对话中提取答案
extracted_output = Puzzlebootcamp.extract_output(prompt + response)

# 验证答案并评分
score = Puzzlebootcamp.verify_score(extracted_output, case)
```

## 你的任务
请根据以下谜题描述（谜题描述可能不完整，请先结合你的知识澄清规则），实现一个完整的谜题训练场类：

### 谜题描述
operation § means select the larger of the two numbers.
operation $ means select the smaller of the two numbers.Example questions are as follows:

<example 0>
Compute (3§5) + (2$4).
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 0>

<example 1>
Compute (7$4)×(6§3).
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 1>

<example 2>
Compute (2§8)-(5$1).
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 2>

<example 3>
Compute (9$6)+(4§4).
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 3>

<example 4>
Compute (3§7)/(8$2).
The answer should only be given as a number.
If the answer is a fraction, write it in 'a/b' text format.Decimals are not allowed.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 4>

<example 5>
If (X § 4) + (3 $ 2) = 10, find X.
The answer should be in the form of an inequality. For example, X ≥ 5 or X ≤ 10.
Use only the symbols \"≥\" or \"≤\". No other symbols are allowed.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 5>

<example 6>
If (6 $ X) + (9 § 2) = 15, find X.
The answer should be in the form of an inequality. For example, X ≥ 5 or X ≤ 10.
Use only the symbols \"≥\" or \"≤\". No other symbols are allowed.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 6>

<example 7>
If (X § 3) - (1 $ 4) = 2, find X.
The answer should be in the form of an inequality. For example, X ≥ 5 or X ≤ 10.
Use only the symbols \"≥\" or \"≤\". No other symbols are allowed.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 7>

<example 8>
If (X § 10) / (4 $ 2) = 5, find X.
The answer should be in the form of an inequality. For example, X ≥ 5 or X ≤ 10.
Use only the symbols \"≥\" or \"≤\". No other symbols are allowed.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 8>

<example 9>
If (5 $ X)×(2 § 6) = 30, find X.
The answer should be in the form of an inequality. For example, X ≥ 5 or X ≤ 10.
Use only the symbols \"≥\" or \"≤\". No other symbols are allowed.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from fractions import Fraction
import re
from math import gcd
from bootcamp import Basebootcamp

class KorOperationUnicode0031bootcamp(Basebootcamp):
    def __init__(self, equation_probability=0.5, max_num=10):
        self.equation_probability = equation_probability
        self.max_num = max_num
        super().__init__()
    
    def _simplify_fraction(self, num, den):
        """将分数转换为最简形式"""
        common_divisor = gcd(num, den)
        return num//common_divisor, den//common_divisor

    def case_generator(self):
        if random.random() < self.equation_probability:
            # 生成包含多种结构的方程题
            equation_types = [
                ("X§A op B$C", "leq"),  # 示例5结构
                ("A$X op B§C", "geq"),  # 示例9结构
                ("X$A op B§C", "geq"),  # 示例6结构
                ("A§X op B$C", "leq")   # 新增变种
            ]
            eq_type, sol_type = random.choice(equation_types)
            
            A = random.randint(1, self.max_num)
            B = random.randint(1, self.max_num)
            C = random.randint(1, self.max_num)
            op = random.choice(["+", "-", "*", "/"])
            
            # 根据方程结构计算目标值和解决方案
            if eq_type.startswith("X§"):
                left = ("§", A)
                compare = max(B, C)
            elif eq_type.startswith("A§"):
                left = ("§", A)
                compare = min(B, C)
            elif eq_type.startswith("X$"):
                left = ("$", A)
                compare = max(B, C)
            else:
                left = ("$", A)
                compare = min(B, C)

            # 计算目标值并处理除法异常
            try:
                if op == "+":
                    target = left[1] + compare
                elif op == "-":
                    target = left[1] - compare
                elif op == "*":
                    target = left[1] * compare
                else:
                    if compare == 0:
                        compare = 1
                    target = Fraction(left[1], compare)
            except ZeroDivisionError:
                compare = 1
                target = Fraction(left[1], compare)

            return {
                "type": "equation",
                "structure": (eq_type, A, op, B, C),
                "solution": (sol_type, left[1]),
                "target": target
            }
        else:
            # 增强计算题生成逻辑
            def gen_operand():
                op = random.choice(["§", "$"])
                a, b = random.choices(range(1, self.max_num+1), k=2)
                return (op, a, b)

            expr1 = gen_operand()
            expr2 = gen_operand()
            operator = random.choice(['+', '-', '*', '/'])
            
            val1 = max(expr1[1], expr1[2]) if expr1[0]=="§" else min(expr1[1], expr1[2])
            val2 = max(expr2[1], expr2[2]) if expr2[0]=="§" else min(expr2[1], expr2[2])
            
            # 处理除法分母为零
            if operator == '/' and val2 == 0:
                val2 = 1
            
            # 计算结果
            try:
                if operator == '+':
                    answer = val1 + val2
                elif operator == '-':
                    answer = val1 - val2
                elif operator == '*':
                    answer = val1 * val2
                else:
                    answer = Fraction(val1, val2)
            except ZeroDivisionError:
                answer = Fraction(val1, 1)

            return {
                "type": "computation",
                "expression": [expr1, expr2],
                "operator": operator,
                "answer": answer
            }
    
    @staticmethod
    def prompt_func(question_case):
        definition = "operation § means select the larger of the two numbers.\noperation $ means select the smaller of the two numbers.\n"
        if question_case["type"] == "computation":
            expr1 = question_case["expression"][0]
            expr2 = question_case["expression"][1]
            op = question_case["operator"]
            
            prompt = f"Compute ({expr1[1]}{expr1[0]}{expr1[2]}) {op} ({expr2[1]}{expr2[0]}{expr2[2]})"
            if op == '/' and not isinstance(question_case["answer"], int):
                prompt += "\nExpress the answer as a simplified fraction (e.g., 3/4)"
            prompt += "\nPut your final answer within [[ ]]."
            return definition + prompt
        else:
            struct = question_case["structure"]
            eq_pattern = {
                "X§A op B$C": f"(X§{struct[1]}) {struct[2]} ({struct[3]}${struct[4]})",
                "A$X op B§C": f"({struct[1]}${struct[0][2:]}) {struct[2]} ({struct[3]}§{struct[4]})",
                "X$A op B§C": f"(X${struct[1]}) {struct[2]} ({struct[3]}§{struct[4]})",
                "A§X op B$C": f"({struct[1]}§X) {struct[2]} ({struct[3]}${struct[4]})"
            }[struct[0]]
            return definition + f"If {eq_pattern} = {question_case['target']}, find X.\n" \
                   "Use inequalities with ≥ or ≤ (e.g., [[X≥5]]).\n" \
                   "Final answer within [[ ]]."

    @staticmethod
    def extract_output(output):
        # 增强提取逻辑，匹配最后出现的合法答案
        matches = re.findall(r'\[\[([Xx]\s*[≥≤]\s*\d+|[\d/]+)\]\]', output)
        return matches[-1].upper().replace(" ", "") if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            if identity["type"] == "computation":
                # 处理分数验证
                expected = identity["answer"]
                if isinstance(expected, Fraction):
                    # 允许任意等效分数形式
                    if '/' in solution:
                        num, den = map(int, solution.split('/'))
                        simplified = cls._simplify_fraction(num, den)
                        return simplified == (expected.numerator, expected.denominator)
                    else:
                        return expected.denominator == 1 and int(solution) == expected.numerator
                else:
                    return float(solution) == float(expected)
            else:
                # 强化不等式验证
                expected_op, boundary = identity["solution"]
                pattern = r'X\s*([≥≤])\s*(\d+)'
                match = re.search(pattern, solution, re.IGNORECASE)
                if not match:
                    return False
                
                actual_op = match.group(1)
                actual_val = int(match.group(2))
                return (actual_op == ("≥" if expected_op == "geq" else "≤")) and (actual_val == boundary)
        except:
            return False
