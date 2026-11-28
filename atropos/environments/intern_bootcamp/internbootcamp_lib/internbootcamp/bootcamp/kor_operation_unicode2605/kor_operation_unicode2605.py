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
a★b=\int_{a}^{b} 2x \, dxExample questions are as follows:

<example 0>
Compute 1★3.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 0>

<example 1>
Compute 0★2.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 1>

<example 2>
Compute -1★1.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 2>

<example 3>
Compute 2★5.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 3>

<example 4>
Compute -2★2.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 4>

<example 5>
Compute 3★6.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 5>

<example 6>
Compute 4★7.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 6>

<example 7>
If a★3=8, find a.
The answer may be negative, if so write it in a format such as '-5'.
If there is more than one answer, please separate them with 'or', e.g.[[1or2]].
Please wrap the final answer in double square brackets, like this: [[your answer]].
</example 7>

<example 8>
If 0★b=b, find b.
The answer may be negative, if so write it in a format such as '-5'.
If there is more than one answer, please separate them with 'or', e.g.[[1or2]].
Please wrap the final answer in double square brackets, like this: [[your answer]].
</example 8>

<example 9>
If a★5=21, find a.
The answer may be negative, if so write it in a format such as '-5'.
If there is more than one answer, please separate them with 'or', e.g.[[1or2]].
Please wrap the final answer in double square brackets, like this: [[your answer]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class KorOperationUnicode2605bootcamp(Basebootcamp):
    def __init__(self, compute_prob=0.7, equation_type2_prob=0.2, min_val=-10, max_val=10):
        super().__init__()
        if not 0 <= compute_prob <= 1 or not 0 <= equation_type2_prob <= 1:
            raise ValueError("Probability parameters must be between 0 and 1")
        if min_val >= max_val:
            raise ValueError("min_val must be less than max_val")
        
        self.compute_prob = compute_prob
        self.equation_type2_prob = equation_type2_prob
        self.min_val = min_val
        self.max_val = max_val

    def case_generator(self):
        if random.random() < self.compute_prob:
            a = random.randint(self.min_val, self.max_val - 1)  # 确保b > a的情况
            b = random.randint(a + 1, self.max_val)
            return {
                "type": "compute",
                "a": a,
                "b": b,
                "expected": b**2 - a**2
            }
        else:
            if random.random() < self.equation_type2_prob:
                return {
                    "type": "equation_type2",
                    "expected": [0, 1]
                }
            else:
                result = random.randint(1, 20)  # 生成正整数结果
                return {
                    "type": "equation_type1",
                    "solve_var": random.choice(['a', 'b']),
                    "result": result,
                    "expected": self._gen_equation_case(result)
                }

    def _gen_equation_case(self, result):
        """生成方程的有效整数解"""
        factors = [i for i in range(1, result+1) if result % i == 0]
        pairs = [(d, result//d) for d in factors]
        valid_solutions = list({x for a, b in pairs for x in (a, -a, b, -b)})
        return sorted(valid_solutions)

    @staticmethod
    def prompt_func(question_case):
        integral_rule = (
            "The operation a★b is defined as the definite integral of 2x from a to b.\n"
            "Mathematically: a★b = ∫ₐᵇ 2x dx = b² - a².\n\n"
        )
        
        if question_case['type'] == 'compute':
            problem = f"Compute {question_case['a']}★{question_case['b']}"
        elif question_case['type'] == 'equation_type1':
            if question_case['solve_var'] == 'a':
                problem = f"Find integer a such that a★b = {question_case['result']}"
            else:
                problem = f"Find integer b such that a★b = {question_case['result']}"
        elif question_case['type'] == 'equation_type2':
            problem = "Solve 0★b = b"
        else:
            raise ValueError("Invalid question type")
        
        format_instruction = (
            "\n\nPresent answer as integer(s) in [[ ]] brackets. "
            "For multiple answers use [[1or-2]] format."
        )
        return integral_rule + problem + format_instruction

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[\[(.*?)\]\]', output)
        if not matches:
            return None
        
        solutions = []
        for match in matches[-1].split('or'):
            try:
                solutions.append(int(match.strip()))
            except ValueError:
                continue
        return solutions or None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        expected = identity['expected']
        return sorted(solution) == sorted(expected)
