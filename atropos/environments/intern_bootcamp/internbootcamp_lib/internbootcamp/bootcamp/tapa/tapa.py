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

**Tapa Puzzle Rules**  

**Objective**: Blacken cells on a grid to satisfy all clues while adhering to connectivity and area constraints.  

1. **Clues**:  
   - Each white cell containing a clue has one or more numbers (e.g., \"3\", \"1 2\").  
   - Numbers represent the lengths of **orthogonally connected black cell groups** in the 8 surrounding cells (up, down, left, right, and diagonals).  
   - Multiple numbers (e.g., \"2 1\") indicate **separate groups**, each isolated by at least one white cell.  

2. **Group Formation**:  
   - A \"group\" is a set of black cells connected orthogonally (horizontally/vertically), **not diagonally**.  
   - Example: A clue \"3\" requires three orthogonally connected black cells in its perimeter. A clue \"1 2\" requires one isolated black cell and a separate pair of orthogonally connected black cells.  

3. **Global Constraints**:  
   - **Single Connected Region**: All black cells must form one connected area (diagonals allowed for connectivity).  
   - **No 2×2 Black Blocks**: No 2×2 square can be entirely black.  
   - **No Isolation**: White cells must not be fully enclosed by black cells (i.e., all white cells must be reachable from the grid’s edges).  

**Key Notes**:  
- Clue cells themselves remain **white**.  
- Numbers in clues can appear in any order (e.g., \"1 2\" and \"2 1\" are equivalent).  
- A clue cell with \"0\" means none of its 8 surrounding cells are black.  


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
from collections import deque
import json

class Tapabootcamp(Basebootcamp):
    def __init__(self, rows=5, cols=5):
        super().__init__()
        self.rows = rows
        self.cols = cols
    
    def case_generator(self):
        # 示例解：中间为黑，其他为白
        solution = [[False for _ in range(self.cols)] for _ in range(self.rows)]
        center_r, center_c = self.rows//2, self.cols//2
        solution[center_r][center_c] = True
        
        clues = {}
        for r in range(self.rows):
            for c in range(self.cols):
                if not solution[r][c]:
                    groups = self.get_clue_numbers(solution, r, c, self.rows, self.cols)
                    if groups != [0]:
                        clues[f"{r},{c}"] = groups
        
        return {
            'rows': self.rows,
            'cols': self.cols,
            'clues': clues
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        rows = question_case['rows']
        cols = question_case['cols']
        clues = question_case['clues']
        prompt = "You are a solver for Tapa puzzles. Your task is to blacken cells in a grid according to the given clues and rules.\n\n"
        prompt += "**Rules**:\n"
        prompt += "- Each clue is a white cell with numbers indicating the lengths of orthogonally connected black cell groups in the surrounding 8 cells.\n"
        prompt += "- Multiple numbers indicate separate groups, each isolated by at least one white cell.\n"
        prompt += "- All black cells must form a single connected region (diagonally allowed).\n"
        prompt += "- No 2×2 area can be entirely black.\n"
        prompt += "- White cells must not be enclosed by black cells; they must be reachable from the grid's edge.\n\n"
        prompt += f"The puzzle grid is {rows}x{cols}. The clues are as follows:\n"
        for r_c, numbers in clues.items():
            row, col = map(int, r_c.split(','))
            nums_str = ' '.join(map(str, numbers))
            prompt += f"- Cell at row {row}, column {col}: {nums_str}\n"
        prompt += "\nYour answer should be a 2D list where each element is True (black) or False (white), enclosed within [answer] tags. For example:\n[answer]\n[[False, True], [True, False]]\n[/answer]"
        return prompt
    
    @staticmethod
    def extract_output(output):
        pattern = r'\[answer\](.*?)\[/answer\]'
        matches = re.findall(pattern, output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            solution = eval(last_match)
            if isinstance(solution, list) and all(isinstance(row, list) for row in solution):
                return solution
            return None
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        rows = identity['rows']
        cols = identity['cols']
        clues = identity['clues']
        
        if not isinstance(solution, list) or len(solution) != rows:
            return False
        for row in solution:
            if not isinstance(row, list) or len(row) != cols:
                return False
        
        for r_c, numbers in clues.items():
            row, col = map(int, r_c.split(','))
            if solution[row][col]:
                return False
            computed = cls.get_clue_numbers(solution, row, col, rows, cols)
            if sorted(numbers) != sorted(computed):
                return False
        
        black_cells = [(r, c) for r in range(rows) for c in range(cols) if solution[r][c]]
        if black_cells:
            if not cls.is_connected(black_cells, rows, cols):
                return False
        
        for r in range(rows - 1):
            for c in range(cols - 1):
                if solution[r][c] and solution[r][c+1] and solution[r+1][c] and solution[r+1][c+1]:
                    return False
        
        white_cells = [(r, c) for r in range(rows) for c in range(cols) if not solution[r][c]]
        if not white_cells:
            return False
        
        visited = set()
        queue = deque()
        for (r, c) in white_cells:
            if r == 0 or r == rows-1 or c == 0 or c == cols-1:
                if (r, c) not in visited:
                    queue.append((r, c))
                    visited.add((r, c))
        
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1,0), (1,0), (0,1), (0,-1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if not solution[nr][nc] and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        
        if any((r, c) not in visited for (r, c) in white_cells):
            return False
        
        return True
    
    @staticmethod
    def get_clue_numbers(solution, row, col, rows, cols):
        if solution[row][col]:
            return []
        
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1),  (1, 0), (1, 1)]
        adjacent = []
        for dr, dc in directions:
            r = row + dr
            c = col + dc
            if 0 <= r < rows and 0 <= c < cols:
                adjacent.append((r, c))
        
        black_cells = [(r, c) for (r, c) in adjacent if solution[r][c]]
        if not black_cells:
            return [0]
        
        visited = set()
        groups = []
        for (r, c) in black_cells:
            if (r, c) not in visited:
                queue = deque([(r, c)])
                visited.add((r, c))
                size = 1
                while queue:
                    x, y = queue.popleft()
                    for dx, dy in [(-1,0), (1,0), (0,1), (0,-1)]:
                        nx, ny = x + dx, y + dy
                        if (nx, ny) in black_cells and (nx, ny) not in visited:
                            visited.add((nx, ny))
                            queue.append((nx, ny))
                            size += 1
                groups.append(size)
        
        groups.sort()
        return groups if groups else [0]
    
    @classmethod
    def is_connected(cls, cells, rows, cols):
        if not cells:
            return True
        
        start = cells[0]
        visited = set([start])
        queue = deque([start])
        
        while queue:
            r, c = queue.popleft()
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in cells and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        
        return len(visited) == len(cells)
