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

The Aquarium puzzle is solved by determining water levels for each aquarium region in a grid, adhering to the following rules:

1. **Grid Structure**: The grid is divided into contiguous regions (aquariums) by thick borders. Each cell belongs to exactly one aquarium.

2. **Water Levels**: Each aquarium must be filled with water up to a consistent horizontal level. This level is a specific row number chosen such that:
   - Every column within the aquarium contains cells up to at least this row (i.e., the level cannot exceed the shortest column height in the aquarium).
   - All cells in the aquarium’s columns from the bottom row up to the chosen level are filled. Cells above this level in the aquarium remain empty.

3. **Row and Column Clues**: 
   - Numbers on the right side of each row indicate the total number of filled cells required in that row across all aquariums.
   - Numbers at the bottom/top of each column indicate the total number of filled cells required in that column across all aquariums.

4. **Objective**: Fill cells to satisfy all row/column numerical clues while ensuring each aquarium’s water level is uniformly applied to its columns.


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from typing import Dict, List, Optional

class Aquariumbootcamp(Basebootcamp):
    def __init__(self, grid_rows: int = 5, grid_cols: int = 5):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
    
    def case_generator(self) -> Dict:
        # Generate regions where each column is a separate aquarium
        cols = self.grid_cols
        rows = self.grid_rows
        regions = []
        for r in range(rows):
            regions.append([c for c in range(cols)])
        
        # Generate water levels for each column (aquarium)
        k = [random.randint(0, rows - 1) for _ in range(cols)]
        
        # Compute row clues: number of filled cells per row
        row_clues = []
        for r in range(rows):
            count = sum(1 for c in range(cols) if k[c] >= r)
            row_clues.append(count)
        
        # Column clues are k[i] + 1
        col_clues = [ki + 1 for ki in k]
        
        return {
            'regions': regions,
            'row_clues': row_clues,
            'col_clues': col_clues,
        }
    
    @staticmethod
    def prompt_func(question_case: Dict) -> str:
        rows = len(question_case['regions'])
        cols = len(question_case['regions'][0]) if rows > 0 else 0
        regions_table = '\n'.join([f"Row {i}: {' '.join(map(str, row))}" for i, row in enumerate(question_case['regions'])])
        row_clues = question_case['row_clues']
        col_clues = question_case['col_clues']
        
        prompt = f"""You are to solve an Aquarium puzzle. The puzzle is played on a grid divided into aquarium regions. Each aquarium must be filled up to a horizontal level such that all its columns are filled to the same level. Here are the details:

- The grid has {rows} rows and {cols} columns.

- Aquarium regions are as follows (each number represents the aquarium ID for that cell):
{regions_table}

- Each row has a clue on the right indicating the total filled cells in that row. The row clues are: {row_clues}.

- Each column has a clue at the bottom indicating the total filled cells in that column. The column clues are: {col_clues}.

Your task is to determine the water level for each aquarium. The water level is the highest row number filled (0-based from the bottom). Each aquarium's water level must be such that all its columns are filled up to this level.

Provide your answer as a list of integers in column order (from left to right), where each integer is the water level for the corresponding column's aquarium. Enclose your answer within [answer] and [/answer]. For example, if the solution is levels 2, 1, 0 for columns 0, 1, 2, write:
[answer]2 1 0[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output: str) -> Optional[List[int]]:
        # Find all answer blocks
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        # Take the last match
        last_match = matches[-1].strip()
        try:
            solution = list(map(int, last_match.split()))
            return solution
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution: List[int], identity: Dict) -> bool:
        cols = len(identity['col_clues'])
        rows = len(identity['row_clues'])
        # Check solution length matches columns
        if len(solution) != cols:
            return False
        # Check each column's solution matches column clue
        for c in range(cols):
            if solution[c] + 1 != identity['col_clues'][c]:
                return False
        # Check each row's filled count matches row clue
        for r in range(rows):
            expected = identity['row_clues'][r]
            actual = sum(1 for c in range(cols) if solution[c] >= r)
            if actual != expected:
                return False
        return True
