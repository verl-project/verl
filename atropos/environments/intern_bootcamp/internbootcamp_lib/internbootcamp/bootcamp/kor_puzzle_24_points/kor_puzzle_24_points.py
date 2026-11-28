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
1.Randomly select four numbers from 1-13, the numbers can have repetition.
2.Using the four basic operation symbols of addition (+), subtraction (-), multiplication (×), and division (÷).
3.Can also use parentheses to change the order of operations.
4.The result is equal to 24.
5.Each number must be used and can be used only once.Example questions are as follows:

<example 0>
The four randomly selected numbers are:
9 5 2 2.
Your answer should be in the form of a calculation expression, like this: a + b / c - d, giving one answer is sufficient.
Wrap your final answer in double square brackets, like this: [[a + b / c - d]].
</example 0>

<example 1>
The four randomly selected numbers are:
9 8 7 6.
Your answer should be in the form of a calculation expression, like this: a + b / c - d, giving one answer is sufficient.
Wrap your final answer in double square brackets, like this: [[a + b / c - d]].
</example 1>

<example 2>
The four randomly selected numbers are:
9 5 2 7.
Your answer should be in the form of a calculation expression, like this: a + b / c - d, giving one answer is sufficient.
Wrap your final answer in double square brackets, like this: [[a + b / c - d]].
</example 2>

<example 3>
The four randomly selected numbers are:
5 7 7 2.
Your answer should be in the form of a calculation expression, like this: a + b / c - d, giving one answer is sufficient.
Wrap your final answer in double square brackets, like this: [[a + b / c - d]].
</example 3>

<example 4>
The four randomly selected numbers are:
6 5 1 7.
Your answer should be in the form of a calculation expression, like this: a + b / c - d, giving one answer is sufficient.
Wrap your final answer in double square brackets, like this: [[a + b / c - d]].
</example 4>

<example 5>
The four randomly selected numbers are:
1 5 4 9
Your answer should be in the form of a calculation expression, like this: a + b / c - d, giving one answer is sufficient.
Wrap your final answer in double square brackets, like this: [[a + b / c - d]].
</example 5>

<example 6>
The four randomly selected numbers are:
7 8 3 8
Your answer should be in the form of a calculation expression, like this: a + b / c - d, giving one answer is sufficient.
Wrap your final answer in double square brackets, like this: [[a + b / c - d]].
</example 6>

<example 7>
The four randomly selected numbers are:
2 3 1 3
Your answer should be in the form of a calculation expression, like this: a + b / c - d, giving one answer is sufficient.
Wrap your final answer in double square brackets, like this: [[a + b / c - d]].
</example 7>

<example 8>
The four randomly selected numbers are:
1 3 7 10
Your answer should be in the form of a calculation expression, like this: a + b / c - d, giving one answer is sufficient.
Wrap your final answer in double square brackets, like this: [[a + b / c - d]].
</example 8>

<example 9>
The four randomly selected numbers are:
8 2 8 2
Your answer should be in the form of a calculation expression, like this: a + b / c - d, giving one answer is sufficient.
Wrap your final answer in double square brackets, like this: [[a + b / c - d]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class KorPuzzle24Pointsbootcamp(Basebootcamp):
    def __init__(self, min_num=1, max_num=13, allow_repeats=True):
        # 参数有效性校验
        if not allow_repeats and (max_num - min_num + 1) < 4:
            raise ValueError("When allow_repeats=False, min_num and max_num must span at least 4 numbers")
        
        self.min_num = min_num
        self.max_num = max_num
        self.allow_repeats = allow_repeats
    
    def case_generator(self):
        MAX_ATTEMPTS = 1000
        for _ in range(MAX_ATTEMPTS):
            # 根据参数生成不同特性的数字组合
            if self.allow_repeats:
                numbers = [random.randint(self.min_num, self.max_num) for _ in range(4)]
            else:
                numbers = random.sample(range(self.min_num, self.max_num+1), 4)
            
            if self._has_solution(numbers):
                return {'numbers': sorted(numbers)}  # 排序便于后续验证
            
        raise RuntimeError("Failed to generate valid case after maximum attempts")
    
    @staticmethod
    def prompt_func(question_case) -> str:
        nums = question_case['numbers']
        example = "9 5 2 7 -> (9 - 5) × (7 - 2)" if 9 in nums else "8 2 8 2 -> 8 × (2 + 2) - 8"
        return f"""
You are a 24-point puzzle solver. Using these numbers exactly once: {', '.join(map(str, nums))},
combine them with +, -, ×, ÷ and parentheses to make 24.

Rules:
1. Use each number exactly once
2. Standard order of operations applies
3. Final result must be exactly 24

Examples:
{example}

Put your final expression within double square brackets. Example: [[(a × b) + (c ÷ d)]]
"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[\[(.*?)\]\]', output)
        return matches[-1].strip() if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            # 提取并验证数字使用
            input_nums = sorted(identity['numbers'])
            used_nums = sorted(map(int, re.findall(r'\b\d+\b', solution)))
            if input_nums != used_nums:
                return False
            
            # 数学表达式验证
            expr = solution.replace('×', '*').replace('÷', '/').replace(' ', '')
            return abs(eval(expr) - 24) < 1e-6
        except:
            return False
    
    @classmethod
    def _has_solution(cls, numbers):
        def dfs(nums):
            if len(nums) == 1:
                return abs(nums[0] - 24) < 1e-6
            for i, a in enumerate(nums):
                for j, b in enumerate(nums):
                    if i == j:
                        continue
                    remaining = [n for idx, n in enumerate(nums) if idx not in (i,j)]
                    for op in ['+', '-', '*', '/']:
                        if op == '/' and b == 0:
                            continue
                        try:
                            res = eval(f"{a}{op}{b}") 
                            if dfs(remaining + [res]):
                                return True
                        except ZeroDivisionError:
                            continue
            return False
        
        return dfs(numbers)
