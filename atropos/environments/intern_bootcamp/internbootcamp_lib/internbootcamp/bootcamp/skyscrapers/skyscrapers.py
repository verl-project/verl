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

**Skyscrapers Puzzle Rules (General Form):**

1. **Grid Structure**:  
   - The puzzle is played on an N×N grid (e.g., 5×5, 6×6).  
   - Each cell must contain a number from 1 to N, representing the height of a \"skyscraper.\"  

2. **Core Rules**:  
   - **Unique Heights**: Each row and column must contain every number from 1 to N exactly once (similar to Sudoku).  
   - **Visibility Clues**: Numbers are provided on the edges of the grid, indicating how many skyscrapers are visible from that direction.  

3. **Visibility Definition**:  
   - A skyscraper is \"visible\" if it is taller than all buildings between it and the edge of the grid.  
   - Example: In a row with heights [3, 1, 4, 2], looking from the left, you see 3 (blocks 1) and 4 (blocks 2). The clue here would be **2**.  

4. **Clue Placement**:  
   - **Edge Clues**: Numbers outside the grid correspond to the count of visible skyscrapers when looking inward:  
     - **Top/Bottom**: Clues for columns (viewed top-to-bottom or bottom-to-top).  
     - **Left/Right**: Clues for rows (viewed left-to-right or right-to-left).  

5. **Objective**:  
   - Fill the grid so that all row/column uniqueness constraints are satisfied, and the visibility counts match the provided clues.  

**Key Idea**: Taller buildings block shorter ones behind them, and clues enforce how many \"peaks\" are observable from each edge.  


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re

class Skyscrapersbootcamp(Basebootcamp):
    def __init__(self, n=4):
        self.n = n
    
    def case_generator(self):
        n = self.n
        square = self.generate_latin_square(n)
        clues = {
            'left': [],
            'right': [],
            'top': [],
            'bottom': []
        }
        
        for row in square:
            clues['left'].append(self.compute_view(row))
            clues['right'].append(self.compute_view(row[::-1]))
        
        for j in range(n):
            column = [square[i][j] for i in range(n)]
            clues['top'].append(self.compute_view(column))
            clues['bottom'].append(self.compute_view(column[::-1]))
        
        return {'n': n, 'clues': clues}
    
    @staticmethod
    def generate_latin_square(n):
        square = []
        for i in range(n):
            row = [(i + j) % n + 1 for j in range(n)]
            square.append(row)
        random.shuffle(square)
        square = list(map(list, zip(*square)))
        random.shuffle(square)
        square = list(map(list, zip(*square)))
        return square
    
    @staticmethod
    def compute_view(view):
        max_h = -1
        count = 0
        for h in view:
            if h > max_h:
                count += 1
                max_h = h
        return count
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        clues = question_case['clues']
        example = "\n".join([" ".join(['1'] * n)] * n)
        prompt = (
            "你正在解决一个数织谜题（Skyscrapers Puzzle）。规则如下：\n"
            "1. 在{}×{}网格中填入1至{}，每行每列数字不重复。\n"
            "2. 周围数字表示从该方向能看到的摩天大楼数量（较高建筑会遮挡后面较矮的）。\n\n"
            "谜题线索：\n"
            "- 网格大小：{}×{}\n"
            "- 顶部线索（各列从上至下可见数）：{}\n"
            "- 底部线索（各列从下至上可见数）：{}\n"
            "- 左侧线索（各行从左至右可见数）：{}\n"
            "- 右侧线索（各行从右至左可见数）：{}\n\n"
            "请填入符合要求的网格，并将答案放在[answer]和[/answer]之间。格式示例：\n"
            "[answer]\n{}[/answer]"
        ).format(
            n, n, n, n, n,
            ' '.join(map(str, clues['top'])),
            ' '.join(map(str, clues['bottom'])),
            ' '.join(map(str, clues['left'])),
            ' '.join(map(str, clues['right'])),
            example
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        solution_str = matches[-1].strip()
        solution = []
        for line in solution_str.split('\n'):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if not all(part.isdigit() for part in parts):
                return None
            solution.append([int(part) for part in parts])
        return solution
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        n = identity['n']
        clues = identity['clues']
        
        if len(solution) != n or any(len(row) != n for row in solution):
            return False
        
        for row in solution:
            if sorted(row) != list(range(1, n+1)):
                return False
        
        for col in range(n):
            column = [solution[row][col] for row in range(n)]
            if sorted(column) != list(range(1, n+1)):
                return False
        
        for i in range(n):
            row = solution[i]
            if (cls.compute_view(row) != clues['left'][i] or
                cls.compute_view(row[::-1]) != clues['right'][i]):
                return False
        
        for j in range(n):
            col = [solution[i][j] for i in range(n)]
            if (cls.compute_view(col) != clues['top'][j] or
                cls.compute_view(col[::-1]) != clues['bottom'][j]):
                return False
        
        return True
