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
a∞b=a^2+b^2.
a and b are positive integers.Example questions are as follows:

<example 0>
Compute 2∞3.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 0>

<example 1>
Compute 4∞5.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 1>

<example 2>
Compute 1∞7.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 2>

<example 3>
Compute 3∞6.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 3>

<example 4>
Compute 5∞8.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 4>

<example 5>
If X∞4=20, find X.
The answer should only be given as a number.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 5>

<example 6>
If 3∞X=18, find X.
The answer should only be given as a number.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 6>

<example 7>
If X∞5=41, find X.
The answer should only be given as a number.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 7>

<example 8>
If X∞8=80, find X.
The answer should only be given as a number.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 8>

<example 9>
If X∞6=45, find X.
The answer should only be given as a number.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class KorOperationUnicode221ebootcamp(Basebootcamp):
    def __init__(self, max_num=10, solve_ratio=0.5):
        self.max_num = max_num
        self.solve_ratio = solve_ratio
    
    def case_generator(self):
        """增强案例生成逻辑，保证数值有效性"""
        if random.random() < self.solve_ratio:
            var_type = random.choice(['x', 'y'])
            if var_type == 'x':
                while True:
                    b = random.randint(1, self.max_num)
                    x = random.randint(1, self.max_num)
                    c = x**2 + b**2
                    # 确保解在允许范围内
                    valid_solutions = [
                        i for i in range(1, self.max_num+1)
                        if i**2 + b**2 == c
                    ]
                    if len(valid_solutions) == 1:
                        return {"problem_type": "solve_x", "b": b, "c": c}
            else:
                while True:
                    a = random.randint(1, self.max_num)
                    y = random.randint(1, self.max_num)
                    c = a**2 + y**2
                    valid_solutions = [
                        i for i in range(1, self.max_num+1)
                        if a**2 + i**2 == c
                    ]
                    if len(valid_solutions) == 1:
                        return {"problem_type": "solve_y", "a": a, "c": c}
        else:
            # 允许生成相同的a和b
            a = random.randint(1, self.max_num)
            b = random.randint(1, self.max_num)
            return {"problem_type": "compute", "a": a, "b": b}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        problem_templates = {
            "compute": "计算 {a}∞{b} 的结果。答案格式示例：[[13]]",
            "solve_x": "解方程 X∞{b} = {c}。答案格式示例：[[2]]",
            "solve_y": "解方程 {a}∞Y = {c}。答案格式示例：[[3]]"
        }
        template = """请根据以下规则解决问题：
- 运算符定义：a∞b = a² + b²
- 所有变量均为正整数

当前问题：
{problem_statement}

请将最终答案放在双括号内，如：[[答案]]"""
        
        return template.format(
            problem_statement=problem_templates[question_case["problem_type"]].format(**question_case)
        )
    
    @staticmethod
    def extract_output(output):
        """增强正则匹配模式"""
        matches = re.findall(r'\[\[\s*(\d+)\s*\]\]', output)
        return matches[-1] if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        """增强数值范围验证"""
        try:
            sol = int(solution)
            if sol < 1:
                return False
        except (ValueError, TypeError):
            return False
        
        if identity["problem_type"] == "compute":
            a, b = identity["a"], identity["b"]
            return sol == a**2 + b**2
        
        if identity["problem_type"] == "solve_x":
            b_val = identity["b"]
            c_val = identity["c"]
            return sol**2 + b_val**2 == c_val and sol <= identity.get("max_num", 10)
        
        if identity["problem_type"] == "solve_y":
            a_val = identity["a"]
            return a_val**2 + sol**2 == identity["c"] and sol <= identity.get("max_num", 10)
        
        return False
