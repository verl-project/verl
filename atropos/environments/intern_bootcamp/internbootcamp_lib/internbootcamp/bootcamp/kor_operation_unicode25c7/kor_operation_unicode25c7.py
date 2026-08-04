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
a◇b= a^b.
a and b are positive integers.Example questions are as follows:

<example 0>
Compute 5◇1.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 0>

<example 1>
Compute 2◇4.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 1>

<example 2>
Compute 2◇3◇2.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 2>

<example 3>
Compute 4◇2◇3.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 3>

<example 4>
If X◇2=64, find X.
The answer should only be given as a number.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 4>

<example 5>
If 3◇X=27, find X.
The answer should only be given as a number.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 5>

<example 6>
If X◇3=125, find X.
The answer should only be given as a number.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 6>

<example 7>
If X◇3◇3=512, find X.
The answer should only be given as a number.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 7>

<example 8>
If 3◇X=243, find X.
The answer should only be given as a number.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 8>

<example 9>
If 5◇2◇X=625, find X.
The answer should only be given as a number.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from functools import reduce
from operator import mul
from bootcamp import Basebootcamp

class KorOperationUnicode25c7bootcamp(Basebootcamp):
    def __init__(self, min_value=1, max_value=10, min_operands=2, max_operands=3, solve_x_ratio=0.5):
        # 参数有效性校验
        if min_value < 1:
            raise ValueError("min_value must be at least 1")
        if max_value < min_value:
            raise ValueError("max_value must be >= min_value")
        if min_operands < 2:
            raise ValueError("min_operands must be at least 2")
        if max_operands < min_operands:
            raise ValueError("max_operands must be >= min_operands")
        if not (0 <= solve_x_ratio <= 1):
            raise ValueError("solve_x_ratio must be between 0 and 1")
        
        self.min_value = min_value
        self.max_value = max_value
        self.min_operands = min_operands
        self.max_operands = max_operands
        self.solve_x_ratio = solve_x_ratio

    def case_generator(self):
        if random.random() < self.solve_x_ratio:
            return self._generate_solve_x_case()
        return self._generate_compute_case()

    def _generate_compute_case(self):
        # 确保操作数数量在有效范围内
        n_operands = random.randint(self.min_operands, self.max_operands)
        operands = [random.randint(self.min_value, self.max_value) for _ in range(n_operands)]
        
        # 右结合幂运算计算
        try:
            result = reduce(lambda a, b: b ** a, reversed(operands))
        except OverflowError:
            # 处理极大数值异常，重新生成合理数值
            return self._generate_compute_case()
        
        return {
            'type': 'compute',
            'operands': operands,
            'result': result
        }

    def _generate_solve_x_case(self):
        # 确保至少有一个其他操作数
        n_other_operands = random.randint(1, self.max_operands-1)
        other_operands = [random.randint(self.min_value, self.max_value) for _ in range(n_other_operands)]
        exponent = reduce(mul, other_operands, 1)
        
        # 生成有效解X
        x_val = random.randint(self.min_value, self.max_value)
        result = x_val ** exponent
        
        return {
            'type': 'solve_x',
            'other_operands': other_operands,
            'x_value': x_val,
            'result': result,
            'exponent': exponent
        }

    @staticmethod
    def prompt_func(question_case):
        prefix = """a◇b means a raised to the power of b (a^b).

The diamond operator is right-associative, meaning:
a◇b◇c = a◇(b◇c) = a^(b^c)

"""
        if question_case['type'] == 'compute':
            expression = '◇'.join(map(str, question_case['operands']))
            return prefix + f"计算表达式 {expression} 的值。答案必须为单独的数字并用双方括号包裹，例如：[[答案]]。"
        else:
            other_ops = '◇'.join(map(str, question_case['other_operands']))
            return prefix + f"若 X◇{other_ops} = {question_case['result']}，求 X 的值。答案必须为单独的数字并用双方括号包裹，例如：[[答案]]。"

    @staticmethod
    def extract_output(output):
        # 匹配所有可能的答案格式，包括中英文括号
        matches = re.findall(r'\[\[(\d+)\]\]|【【(\d+)】】', output)
        # 优先取最后一个匹配的数字，支持多语言格式
        last_match = matches[-1] if matches else None
        return last_match[0] or last_match[1] if last_match else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            # 支持字符串和数字类型的比较
            solution_num = int(solution) if isinstance(solution, str) else solution
        except (ValueError, TypeError):
            return False
        
        # 验证数据结构完整性
        if identity['type'] not in ('compute', 'solve_x'):
            return False
            
        if identity['type'] == 'compute':
            return solution_num == identity.get('result', None)
        else:
            return solution_num == identity.get('x_value', None)
