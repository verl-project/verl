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
a￠b=\log_{b}{a}+\log_{a}{b}.
a and b are positive integers.Example questions are as follows:

<example 0>
Compute 2￠8.
If the answer is a fraction, write it in 'a/b' text format.Decimals are not allowed.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 0>

<example 1>
Compute 3￠9.
If the answer is a fraction, write it in 'a/b' text format.Decimals are not allowed.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 1>

<example 2>
Compute 2￠2￠4.
If the answer is a fraction, write it in 'a/b' text format.Decimals are not allowed.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 2>

<example 3>
Compute 5￠8.
If the answer cannot be reduced to an integer or fraction then retain the form \log_{b}{a}+\log_{a}{b}.
Please provide your answer in LaTeX format. Please wrap the answer in double square brackets, like this: [[your answer]].
</example 3>

<example 4>
Compute 16￠256.
If the answer is a fraction, write it in 'a/b' text format.Decimals are not allowed.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 4>

<example 5>
Compute 9￠243.
If the answer is a fraction, write it in 'a/b' text format.Decimals are not allowed.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 5>

<example 6>
If 512￠X=82/9, find X.
The answer should only be given as a number.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 6>

<example 7>
If 3￠X=17/4, find X.
The answer should only be given as a number.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 7>

<example 8>
If X￠256=73/24, find X.
The answer should only be given as a number.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 8>

<example 9>
If 3￠3￠X=5/2, find X.
The answer should only be given as a number.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import math
from fractions import Fraction
import re
from bootcamp import Basebootcamp

class KorOperationUnicodeffe0bootcamp(Basebootcamp):
    def __init__(self, min_value=2, max_value=256, problem_types=None, allow_multilevel=True):
        self.min = min_value
        self.max = max_value
        self.problem_types = problem_types or ['compute', 'solve']
        self.allow_multilevel = allow_multilevel

    @staticmethod
    def is_power(a, b):
        """判断两个数是否互为整数幂"""
        if a == 1 or b == 1:
            return a == b
        try:
            exponent = math.log(a, b)
            return abs(exponent - round(exponent)) < 1e-10 and b**round(exponent) == a
        except ValueError:
            pass
        try:
            exponent = math.log(b, a)
            return abs(exponent - round(exponent)) < 1e-10 and a**round(exponent) == b
        except ValueError:
            return False

    def case_generator(self):
        problem_type = random.choice(self.problem_types)
        
        def generate_expressible():
            k = random.randint(1,4)
            b = random.randint(self.min, self.max)
            a = b ** k
            while a > self.max:
                k = random.randint(1,4)
                b = random.randint(self.min, self.max)
                a = b ** k
            return a, b, Fraction(k**2 +1, k)

        if problem_type == 'compute':
            if random.random() < 0.3:  
                a = random.randint(self.min, self.max)
                b = random.randint(self.min, self.max)
                while self.is_power(a, b) or a == b:
                    a = random.randint(self.min, self.max)
                    b = random.randint(self.min, self.max)
                return {
                    "type": "compute",
                    "expression": [a, b],
                    "log_expr": f"\\log_{{{b}}}{{{a}}} + \\log_{{{a}}}{{{b}}}",
                    "is_expressible": False
                }
            
            if self.allow_multilevel and random.random() < 0.5:
                a, base, frac = generate_expressible()
                m = random.randint(1,4)
                c = base ** m
                return {
                    "type": "compute",
                    "expression": [a, base, c],
                    "target": str(Fraction(m**2 +1, m)),
                    "is_expressible": True
                }
            else:
                a, b, frac = generate_expressible()
                return {
                    "type": "compute",
                    "expression": [a, b],
                    "target": f"{frac.numerator}/{frac.denominator}",
                    "is_expressible": True
                }
                
        else:  # solve类型完整实现
            # 生成单层求解案例
            position = random.choice([0, 1])
            X = random.randint(self.min, self.max)
            k = random.randint(1,4)
            if position == 0:
                b = X ** k
                equation = ['X', b]
            else:
                a = X ** k
                equation = [a, 'X']
            target = Fraction(k**2 +1, k)
            return {
                "type": "solve",
                "equation": equation,
                "target": f"{target.numerator}/{target.denominator}",
                "solution": X
            }

    @staticmethod
    def prompt_func(question_case):
        definition = """a￠b=\log_{b}{a}+\log_{a}{b}.
a and b are positive integers.
"""
        if question_case['type'] == 'compute':
            expr = '￠'.join(map(str, question_case['expression']))
            prompt = f"Compute {expr}.\n"
            if not question_case.get('is_expressible', True):
                prompt += "If the answer cannot be reduced to an integer or fraction then retain the form \\log_{b}{a}+\\log_{a}{b}.\n"
                prompt += "Please provide your answer in LaTeX format. "
            else:
                prompt += "If the answer is a fraction, write it in 'a/b' text format. Decimals are not allowed.\n"
            prompt += "Please wrap the answer in double square brackets, like this: [[your answer]]."
            return definition + prompt
        else:
            equation_str = '￠'.join(
                [str(x) if x != 'X' else 'X' for x in question_case['equation']]
            )
            prompt = f"If {equation_str} = {question_case['target']}, find X.\n"
            prompt += "The answer should only be given as a number.\n"
            prompt += "Please wrap the answer in double square brackets, like this: [[your answer]]."
            return definition + prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[\[(.*?)\]\]', output)
        return matches[-1] if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            if identity['type'] == 'compute':
                if identity.get('is_expressible', True):
                    return Fraction(solution) == Fraction(identity['target'])
                else:
                    given = re.sub(r'\s+', '', solution)
                    expected = re.sub(r'\s+', '', identity['log_expr'])
                    return given == expected
            else:
                return int(solution) == identity['solution']
        except:
            return False
