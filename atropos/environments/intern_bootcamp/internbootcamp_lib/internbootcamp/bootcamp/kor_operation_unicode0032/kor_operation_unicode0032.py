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
a♀b=(a+b)/2
a♂b=a×4+bExample questions are as follows:

<example 0>
Compute (3♀5)♂2.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 0>

<example 1>
Compute 7♂(6♀2)=32.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 1>

<example 2>
Compute (4♀8)♂3.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 2>

<example 3>
Compute 5♂(3♀7).
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 3>

<example 4>
If (X♀3)♂2=22, find X.
The answer should only be given as a number.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 4>

<example 5>
If 4♂(X♀2)=30, find X.
The answer should only be given as a number.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 5>

<example 6>
If (X♀4)♂5=37, find X.
The answer should only be given as a number.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 6>

<example 7>
If 6♂(X♀3)=42, find X.
The answer should only be given as a number.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 7>

<example 8>
Now ignoring the previous rule.
Given that a♂b=a×4+b and 4♀3=3.5, compute 2♂(4♀3).
If the answer is a fraction, write it in 'a/b' text format.Decimals are not allowed.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 8>

<example 9>
Now ignoring the previous rule.
Given that a♀b=(a+b)/2 and 5♂6=26, compute 5♂(4♀8).
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random  # 新增缺失的模块导入
from fractions import Fraction
from bootcamp import Basebootcamp

class KorOperationUnicode0032bootcamp(Basebootcamp):
    def __init__(self, min_val=1, max_val=20, equation_prob=0.5):
        self.min_val = min_val
        self.max_val = max_val
        self.equation_prob = equation_prob

    def case_generator(self):
        if random.random() < self.equation_prob:
            structure = random.choice(['struct1', 'struct2'])
            A = random.randint(self.min_val, self.max_val)
            B = random.randint(self.min_val, self.max_val)
            target_result = random.randint(self.min_val*2, self.max_val*2)

            if structure == 'struct1':
                x_val = Fraction(target_result - B, 2) - A
            else:
                fem_part = Fraction(A + B, 2)
                x_val = Fraction(target_result - fem_part, 4)
            
            return {
                'type': 'equation',
                'structure': structure,
                'A': A, 'B': B,
                'result': target_result,
                'correct_answer': {
                    'numerator': x_val.numerator,
                    'denominator': x_val.denominator
                }
            }
        else:
            structure = random.choice(['struct1', 'struct2'])
            a = random.randint(self.min_val, self.max_val)
            b = random.randint(self.min_val, self.max_val)
            c = random.randint(self.min_val, self.max_val)

            if structure == 'struct1':
                answer = 2*(a + b) + c
            else:
                answer = 4*a + Fraction(b + c, 2)
            
            return {
                'type': 'compute',
                'structure': structure,
                'a': a, 'b': b, 'c': c,
                'correct_answer': {
                    'numerator': answer.numerator,
                    'denominator': answer.denominator
                }
            }

    @staticmethod
    def prompt_func(question_case):
        operator_desc = (
            "You are given two custom operators: ♀ and ♂. The operator a♀b is defined as (a + b)/2, which calculates the average of a and b. "
            "The operator a♂b is defined as 4*a + b, which multiplies a by 4 and then adds b."
        )
        format_instruction = "Please wrap the answer in double square brackets like [[answer]]. Use a/b format for fractions."

        if question_case['type'] == 'compute':
            a, b, c = question_case['a'], question_case['b'], question_case['c']
            expr = f"({a}♀{b})♂{c}" if question_case['structure'] == 'struct1' else f"{a}♂({b}♀{c})"
            problem = f"Compute {expr}."
        else:
            A, B = question_case['A'], question_case['B']
            eq = f"(X♀{A})♂{B} = {question_case['result']}" if question_case['structure'] == 'struct1' else f"X♂({A}♀{B}) = {question_case['result']}"
            problem = f"If {eq}, find X."

        return f"{operator_desc}\n{problem}\n{format_instruction}"

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[\[(.*?)\]\]', output)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            # Parse solution
            if '/' in solution:
                num, den = map(int, solution.split('/'))
                user_ans = Fraction(num, den)
            else:
                user_ans = Fraction(int(solution))
            
            # Get correct answer
            ans = identity['correct_answer']
            correct_ans = Fraction(ans['numerator'], ans['denominator'])
            
            return user_ans == correct_ans
        except (ValueError, ZeroDivisionError, KeyError):
            return False
