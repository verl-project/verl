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
a①b=\sqrt{a}+b^2.
a②b=\sqrt{a}×b.Example questions are as follows:

<example 0>
Compute 9②(4①2).
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 0>

<example 1>
Compute 49①(9②2).
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 1>

<example 2>
Compute (4①3)②2.
Please provide your answer in LaTeX format. 
If the answer is a fraction, write it as \frac{a}{b}. 
If it contains a root sign, use \sqrt{x} where x is the number under the root. 
Wrap the final answer in double square brackets, like this: [[your answer]].
</example 2>

<example 3>
Compute (16①2)②3.
Please provide your answer in LaTeX format. 
If the answer is a fraction, write it as \frac{a}{b}. 
If it contains a root sign, use \sqrt{x} where x is the number under the root. 
Wrap the final answer in double square brackets, like this: [[your answer]].
</example 3>

<example 4>
If (X①3)②2=10, find X.
The answer should only be given as a number.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 4>

<example 5>
If 4②(X①2)=20, find X.
The answer should only be given as a number.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 5>

<example 6>
If 6①(X②4)=40, find X.
Please provide your answer in LaTeX format. 
If the answer is a fraction, write it as \frac{a}{b}. 
If it contains a root sign, use \sqrt{x} where x is the number under the root. 
Wrap the final answer in double square brackets, like this: [[your answer]].
</example 6>

<example 7>
If (X①2)②5=35, find X.
The answer should only be given as a number.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 7>

<example 8>
If (X①2)②3=9, find X.
The answer should only be given as a number.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 8>

<example 9>
If 4②(X①1)=12, find X.
The answer should only be given as a number.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import re
import random
from typing import Optional

class KorOperationUnicode0033bootcamp(Basebootcamp):
    def __init__(self, compute_prob=0.5, max_square=10, operand_range=(1,5), max_depth=2):
        self.compute_prob = compute_prob
        self.max_square = max_square
        self.operand_range = operand_range
        self.max_depth = max_depth  # 控制嵌套层级
    
    def case_generator(self):
        if random.random() < self.compute_prob:
            # 生成带随机嵌套结构的计算题
            def build_expression(depth=0):
                # 基础情况：返回数字或创建新表达式
                if depth >= self.max_depth:
                    return random.randint(1, self.max_square) ** 2
                
                # 随机决定是否创建嵌套结构
                if random.random() < 0.5:
                    # 创建新的运算符节点
                    op = random.choice(['①', '②'])
                    left = build_expression(depth + 1)
                    right = build_expression(depth + 1)
                    return {'operator': op, 'left': left, 'right': right}
                else:
                    # 返回基本数字
                    return random.randint(1, self.max_square) ** 2
            
            return {
                'type': 'compute',
                'expression': build_expression()
            }
        else:
            # 生成解方程题（保持原逻辑）
            a = random.randint(1, self.operand_range[1])
            m = random.randint(a+1, a+5)
            X = (m**2 - a**2) ** 2
            b = random.randint(1, self.operand_range[1])
            c = m * b
            return {
                'type': 'solve',
                'equation': {'a': a, 'b': b, 'c': c},
                'X': X
            }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        definition = """a①b=\sqrt{a}+b^2.
a②b=\sqrt{a}×b.
"""
        if question_case['type'] == 'compute':
            expr_str = KorOperationUnicode0033bootcamp.expression_to_str(question_case['expression'])
            latex_rules = ("Please provide your answer in LaTeX format. "
                           "Use \\frac{a}{b} for fractions and \\sqrt{x} for roots. "
                           "Put your final answer within [[ ]].")
            return definition + f"Compute {expr_str}.\n{latex_rules}"
        else:
            eq = question_case['equation']
            return definition + (f"If (X①{eq['a']})②{eq['b']} = {eq['c']}, find X.\n"
                    "Provide only a numeric answer within [[ ]].")

    @staticmethod
    def expression_to_str(expr) -> str:
        if isinstance(expr, dict):
            left = KorOperationUnicode0033bootcamp.expression_to_str(expr['left'])
            right = KorOperationUnicode0033bootcamp.expression_to_str(expr['right'])
            return f"({left}{expr['operator']}{right})"
        return str(expr)

    @staticmethod
    def extract_output(output: str) -> Optional[str]:
        matches = re.findall(r'\[\[(.*?)\]\]', output)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution: str, identity: dict) -> bool:
        try:
            if identity['type'] == 'compute':
                actual = cls.parse_solution(solution)
                expected = cls.compute_expression(identity['expression'])
                return math.isclose(actual, expected, rel_tol=1e-9)
            else:
                X = int(solution)
                a = identity['equation']['a']
                b = identity['equation']['b']
                c = identity['equation']['c']
                return math.isclose(math.sqrt(math.sqrt(X) + a**2) * b, c, rel_tol=1e-9)
        except:
            return False

    @staticmethod
    def compute_expression(expr) -> float:
        if isinstance(expr, dict):
            left = KorOperationUnicode0033bootcamp.compute_expression(expr['left'])
            right = KorOperationUnicode0033bootcamp.compute_expression(expr['right'])
            if expr['operator'] == '①':
                return math.sqrt(left) + right**2
            return math.sqrt(left) * right
        return float(expr)

    @staticmethod
    def parse_solution(solution: str) -> float:
        solution = solution.replace(' ', '')
        # 处理分数
        frac_match = re.match(r'\\frac\{(-?\d+)\}\{(\d+)\}', solution)
        if frac_match:
            return float(frac_match[1]) / float(frac_match[2])
        
        # 处理根号表达式（支持系数）
        sqrt_match = re.match(r'(-?)(\d*)\\sqrt\{(\d+)\}', solution)
        if sqrt_match:
            sign = -1 if sqrt_match[1] else 1
            coeff = float(sqrt_match[2] or 1) * sign
            return coeff * math.sqrt(float(sqrt_match[3]))
        
        # 处理纯根号
        if solution.startswith('\\sqrt'):
            return math.sqrt(float(re.search(r'\d+', solution).group()))
        
        return float(solution)
