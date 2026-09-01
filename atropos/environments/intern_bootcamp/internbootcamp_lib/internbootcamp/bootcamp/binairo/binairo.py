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

        参数:
            question_case: 由case_generator生成的谜题实例
            
        返回:
            str: 格式化的问题字符串
            
        注意:
            1. 需考虑问题的格式，以便后续能正确提取
            2. 问题描述中应包含期望的答案格式说明，以便后续能正确提取，为了避免抽取时匹配出干扰项，请要求模型将答案放在特定标签，如[answer] [/answer]内
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

Binairo is a logic puzzle played on a rectangular grid (typically square and even-sized, e.g., 6x6, 8x8). The goal is to fill the grid with **0s and 1s** while adhering to these rules:

1. **Balance**:  
   Each row and column must contain an **equal number of 0s and 1s** (e.g., in an 8x8 grid, every row/column has four 0s and four 1s).

2. **No Triples**:  
   No three identical digits (0 or 1) can appear consecutively **in a row or column**. For example, \"000\" or \"111\" is invalid.

3. **Uniqueness**:  
   Each row must be **unique**, and each column must also be **unique**. Duplicate rows or columns are forbidden.

4. **Solution Uniqueness**:  
   The entire grid must have **exactly one valid solution** determined by logical deduction, with no guesswork required.

The puzzle starts with some cells pre-filled as clues, and players use elimination and pattern-matching to deduce the remaining cells.


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re

class Binairobootcamp(Basebootcamp):
    def __init__(self, size=6, clue_density=0.5):
        if size % 2 != 0:
            raise ValueError("Size must be even for Binairo puzzle.")
        self.size = size
        self.clue_density = clue_density

    def generate_solution(self):
        n = self.size
        possible_rows = self.generate_all_possible_rows(n)
        random.shuffle(possible_rows)
        
        for _ in range(1000):
            try:
                selected = random.sample(possible_rows, n)
            except ValueError:
                continue
            
            if len({tuple(r) for r in selected}) != n:
                continue
            
            if self.check_columns(selected, n):
                return selected
        
        # Fallback example for 4x4
        return [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1]
        ]

    def generate_all_possible_rows(self, n):
        return self.backtrack_row([], n, n//2, n//2)

    def backtrack_row(self, current, n, zeros, ones):
        if len(current) == n:
            return [current.copy()] if zeros == 0 and ones == 0 else []
        
        solutions = []
        for bit in [0, 1]:
            if (bit == 0 and zeros == 0) or (bit == 1 and ones == 0):
                continue
            
            if len(current) >= 2 and current[-1] == bit and current[-2] == bit:
                continue
            
            new_current = current.copy()
            new_current.append(bit)
            new_zeros = zeros - 1 if bit == 0 else zeros
            new_ones = ones - 1 if bit == 1 else ones
            solutions += self.backtrack_row(new_current, n, new_zeros, new_ones)
        
        return solutions

    def check_columns(self, grid, n):
        columns = list(zip(*grid))
        for col in columns:
            if col.count(0) != n//2 or col.count(1) != n//2:
                return False
            for i in range(len(col)-2):
                if col[i] == col[i+1] == col[i+2]:
                    return False
        return len(set(columns)) == len(columns)

    def case_generator(self):
        solution = self.generate_solution()
        puzzle = [
            [
                cell if random.random() < self.clue_density else None 
                for cell in row
            ]
            for row in solution
        ]
        return {'puzzle': puzzle, 'solution': solution}

    @staticmethod
    def prompt_func(question_case):
        puzzle = question_case['puzzle']
        size = len(puzzle)
        rows = []
        for i, row in enumerate(puzzle, 1):
            cells = ['_' if c is None else str(c) for c in row]
            rows.append(f"Row {i}: {' '.join(cells)}")
        return f"""Solve this Binairo puzzle (size {size}x{size}):

Rules:
1. Equal 0s/1s in each row/column
2. No three consecutive identical digits
3. All rows/columns must be unique
4. Exactly one valid solution

Puzzle:
{chr(10).join(rows)}

Place your final answer between [answer] and [/answer] tags as:

[answer]
1 0 1 0
0 1 0 1
...[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        
        try:
            solution = []
            for line in matches[-1].strip().split('\n'):
                solution.append([int(c) for c in line.split()])
            return solution
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity['solution']
        return solution == expected
