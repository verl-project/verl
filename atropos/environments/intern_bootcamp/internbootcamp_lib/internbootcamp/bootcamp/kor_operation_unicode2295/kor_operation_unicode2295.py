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
a⊕b=a+bi.Example questions are as follows:

<example 0>
Compute (3⊕4)+(2⊕1).
If the answer is a complex number, write it in the form x + yi.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 0>

<example 1>
Compute (5⊕2)−(3⊕1)
If the answer is a complex number, write it in the form x + yi.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 1>

<example 2>
Compute (2⊕3)×(1⊕4).
If the answer is a complex number, write it in the form x + yi.
The answer may be negative, if so write it in a format such as '-5'.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 2>

<example 3>
Compute (4⊕2)/(2⊕1).
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 3>

<example 4>
Compute (6⊕3)+(1⊕2)×2.
If the answer is a complex number, write it in the form x + yi.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 4>

<example 5>
Compute 3×(2⊕1)−(1⊕3).
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 5>

<example 6>
If (X⊕2)+(1⊕3)=4+5i, find X.
The answer should only be given as a number.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 6>

<example 7>
If (3⊕Y)−(2⊕1)=1+3i Find Y
The answer should only be given as a number.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 7>

<example 8>
If (2⊕3)×(1⊕X)=−10+11i, find X.
The answer should only be given as a number.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 8>

<example 9>
If (6⊕3)+(X⊕2)×2=10+11i, find X.
The answer should only be given as a number.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class KorOperationUnicode2295bootcamp(Basebootcamp):
    def __init__(self, max_operand=10, equation_prob=0.5, allow_division=True):
        self.max_operand = max_operand
        self.equation_prob = equation_prob
        self.allow_division = allow_division
    
    def case_generator(self):
        if random.random() < self.equation_prob:
            return self._generate_equation_case()
        else:
            return self._generate_compute_case()
    
    def _generate_equation_case(self):
        operators = ['+', '-', '*']
        if self.allow_division:
            operators.append('/')
            
        for _ in range(100):
            x = random.uniform(-self.max_operand, self.max_operand)
            x = round(x, 1)  # 允许一位小数
            operand_index = random.choice([0, 1])
            part = random.choice(['a', 'b'])
            
            left_a = random.randint(-self.max_operand, self.max_operand)
            left_b = random.randint(-self.max_operand, self.max_operand)
            right_a = random.randint(-self.max_operand, self.max_operand)
            right_b = random.randint(-self.max_operand, self.max_operand)
            operator = random.choice(operators)
            
            if operand_index == 0:
                left_operand = {'a': 'X' if part == 'a' else left_a, 'b': 'X' if part == 'b' else left_b}
                right_operand = {'a': right_a, 'b': right_b}
                a1 = x if part == 'a' else left_a
                b1 = x if part == 'b' else left_b
                a2, b2 = right_a, right_b
            else:
                left_operand = {'a': left_a, 'b': left_b}
                right_operand = {'a': 'X' if part == 'a' else right_a, 'b': 'X' if part == 'b' else right_b}
                a1, b1 = left_a, left_b
                a2 = x if part == 'a' else right_a
                b2 = x if part == 'b' else right_b
            
            # 处理分母有效性
            if operator == '/':
                if (a2 == 0 and b2 == 0):
                    continue
                denominator = a2**2 + b2**2
                if denominator == 0:
                    continue
            
            try:
                if operator == '+':
                    target_real = a1 + a2
                    target_imag = b1 + b2
                elif operator == '-':
                    target_real = a1 - a2
                    target_imag = b1 - b2
                elif operator == '*':
                    target_real = a1 * a2 - b1 * b2
                    target_imag = a1 * b2 + b1 * a2
                else:
                    denominator = a2**2 + b2**2
                    target_real = (a1 * a2 + b1 * b2) / denominator
                    target_imag = (b1 * a2 - a1 * b2) / denominator
                
                # 允许浮点结果，保留两位小数
                target_real = round(target_real, 2)
                target_imag = round(target_imag, 2)
                
                return {
                    'type': 'equation',
                    'left_operands': [left_operand, right_operand],
                    'operator': operator,
                    'target_real': target_real,
                    'target_imag': target_imag,
                    'unknown': {'operand_index': operand_index, 'part': part},
                    'solution': round(x, 2)
                }
            except:
                continue
        return self._generate_compute_case()
    
    def _generate_compute_case(self):
        operators = ['+', '-', '*']
        if self.allow_division:
            operators.append('/')
            
        operator = random.choice(operators)
        
        for _ in range(100):
            a = random.randint(-self.max_operand, self.max_operand)
            b = random.randint(-self.max_operand, self.max_operand)
            c = random.randint(-self.max_operand, self.max_operand)
            d = random.randint(-self.max_operand, self.max_operand)
            
            if operator == '/' and (c == 0 and d == 0):
                continue
                
            if operator == '+':
                real = a + c
                imag = b + d
            elif operator == '-':
                real = a - c
                imag = b - d
            elif operator == '*':
                real = a * c - b * d
                imag = a * d + b * c
            else:
                denominator = c**2 + d**2
                real = (a * c + b * d) / denominator
                imag = (b * c - a * d) / denominator
            
            # 保留两位小数
            real = round(real, 2)
            imag = round(imag, 2)
            
            return {
                'type': 'compute',
                'operator': operator,
                'left_a': a,
                'left_b': b,
                'right_a': c,
                'right_b': d,
                'solution_real': real,
                'solution_imag': imag
            }
        
        return {
            'type': 'compute',
            'operator': '+',
            'left_a': random.randint(-self.max_operand, self.max_operand),
            'left_b': random.randint(-self.max_operand, self.max_operand),
            'right_a': random.randint(-self.max_operand, self.max_operand),
            'right_b': random.randint(-self.max_operand, self.max_operand),
            'solution_real': 0,
            'solution_imag': 0
        }
    
    @staticmethod
    def prompt_func(question_case):
        definition = "a⊕b=a+bi.\n"
        if question_case['type'] == 'compute':
            left = f"({question_case['left_a']}⊕{question_case['left_b']})"
            right = f"({question_case['right_a']}⊕{question_case['right_b']})"
            expr = f"{left} {question_case['operator']} {right}"
            return definition + f"Compute {expr}. If the answer is a complex number, write it in the form x + yi. Please wrap your answer in double square brackets, like this: [[answer]]."
        else:
            left_operand = question_case['left_operands'][0]
            right_operand = question_case['left_operands'][1]
            left = f"({left_operand['a']}⊕{left_operand['b']})"
            right = f"({right_operand['a']}⊕{right_operand['b']})"
            expr = f"{left} {question_case['operator']} {right}"
            target_real = question_case['target_real']
            target_imag = question_case['target_imag']
            
            # 显示优化
            if isinstance(target_real, float) and target_real.is_integer():
                target_real = int(target_real)
            if isinstance(target_imag, float) and target_imag.is_integer():
                target_imag = int(target_imag)
            
            if target_imag == 0:
                target_str = f"{target_real}"
            else:
                imag_abs = abs(target_imag)
                imag_sign = '+' if target_imag > 0 else '-'
                target_str = f"{target_real} {imag_sign} {imag_abs}i"
            
            return definition + f"If {expr} = {target_str}, find X. The answer should only be given as a number. Please wrap your answer in double square brackets, like this: [[answer]]."
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[\[(.*?)\]\]', output)
        if not matches:
            return None
        last_match = matches[-1].strip()
        return re.sub(r'\s+', '', last_match)  # 移除所有空格
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        def parse_complex(s):
            s = s.replace(' ', '').lower().replace('i', 'j')
            try:
                c = complex(s)
                return (round(c.real, 2), round(c.imag, 2))
            except:
                return (None, None)
        
        if identity['type'] == 'equation':
            try:
                user_value = round(float(solution), 2)
                return user_value == identity['solution']
            except:
                return False
        else:
            real, imag = parse_complex(solution)
            if real is None or imag is None:
                return False
            target_real = round(identity['solution_real'], 2)
            target_imag = round(identity['solution_imag'], 2)
            return (real == target_real) and (imag == target_imag)
