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

**Tents Puzzle Rules:**

1. **Grid Structure**: The puzzle is played on a rectangular grid where some cells contain trees, and others are empty. The goal is to place tents in empty cells according to specific rules.

2. **Tree-Tent Pairing**:  
   - Every tree must have **exactly one tent** placed in an orthogonally adjacent cell (up, down, left, or right). Diagonal adjacency does not count.  
   - Conversely, every tent must be adjacent to **exactly one tree** (no shared tents between trees).

3. **Tent Placement Restrictions**:  
   - Tents cannot be adjacent to each other in **any direction**, including diagonally. A tent must be isolated by at least one empty cell from all other tents.  
   - Tents can only occupy **empty cells** (never on trees or other tents).

4. **Row/Column Clues**:  
   - Numbers on the **right side** of the grid indicate how many tents must be placed in each row.  
   - Numbers on the **bottom/top** of the grid indicate how many tents must be placed in each column.  
   - These clues must be satisfied exactly (no more, no fewer tents in a row/column).

5. **Key Logic**:  
   - All tents must be \"paired\" with trees via adjacency, and all trees must have exactly one tent.  
   - Use the row/column numbers and adjacency constraints to deduce valid tent placements through elimination.


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
import ast
from typing import List, Tuple

class Tentsbootcamp(Basebootcamp):
    def __init__(self, rows=5, cols=5):
        self.rows = rows
        self.cols = cols

    def case_generator(self) -> dict:
        while True:
            # Generate valid tent positions
            tent_positions = self._generate_tent_positions()
            if not tent_positions:
                continue
            
            # Generate corresponding tree positions
            grid, tree_positions = self._place_trees(tent_positions)
            if not grid:
                continue
            
            # Verify tree-tent mapping
            if not self._validate_tree_tents(grid, tent_positions, tree_positions):
                continue
            
            # Generate row and column clues
            row_clues = [sum(1 for x, y in tent_positions if x == i) for i in range(self.rows)]
            col_clues = [sum(1 for x, y in tent_positions if y == j) for j in range(self.cols)]
            
            # Convert grid to 0/1 matrix
            grid_matrix = [[1 if (i, j) in tree_positions else 0 for j in range(self.cols)] 
                          for i in range(self.rows)]
            
            return {
                'grid': grid_matrix,
                'row_clues': row_clues,
                'col_clues': col_clues,
                'solution': tent_positions
            }

    def _generate_tent_positions(self) -> List[Tuple[int, int]]:
        available = [[True for _ in range(self.cols)] for _ in range(self.rows)]
        tents = []
        positions = [(i, j) for i in range(self.rows) for j in range(self.cols)]
        random.shuffle(positions)
        
        for x, y in positions:
            if available[x][y]:
                tents.append((x, y))
                # Mark surrounding cells as unavailable
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nx, ny = x+dx, y+dy
                        if 0 <= nx < self.rows and 0 <= ny < self.cols:
                            available[nx][ny] = False
        return tents

    def _place_trees(self, tent_positions) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
        grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        tree_positions = []
        
        for x, y in tent_positions:
            directions = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
            random.shuffle(directions)
            placed = False
            for dx, dy in directions:
                if 0 <= dx < self.rows and 0 <= dy < self.cols:
                    if grid[dx][dy] == 0 and (dx, dy) not in tent_positions:
                        grid[dx][dy] = 1
                        tree_positions.append((dx, dy))
                        placed = True
                        break
            if not placed:
                return None, None
        return grid, tree_positions

    def _validate_tree_tents(self, grid, tents, trees) -> bool:
        # Check tent adjacency
        for i in range(len(tents)):
            for j in range(i+1, len(tents)):
                x1, y1 = tents[i]
                x2, y2 = tents[j]
                if abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1:
                    return False
        
        # Check tree-tent mapping
        tree_counts = {(i,j):0 for i in range(self.rows) for j in range(self.cols) if grid[i][j]}
        for x, y in tents:
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < self.rows and 0 <= ny < self.cols:
                    if grid[nx][ny]:
                        tree_counts[(nx, ny)] += 1
        return all(c == 1 for c in tree_counts.values())

    @staticmethod
    def prompt_func(question_case) -> str:
        grid = question_case['grid']
        row_clues = question_case['row_clues']
        col_clues = question_case['col_clues']
        
        # Build grid visualization
        grid_str = "   " + " ".join(str(i+1) for i in range(len(grid[0]))) + "\n"
        for idx, row in enumerate(grid):
            cells = ["T" if cell else "." for cell in row]
            grid_str += f"{idx+1:2} {' '.join(cells)} {row_clues[idx]}\n"
        grid_str += f"   {' '.join(map(str, col_clues))}"
        
        return f"""你是一个帐篷谜题专家，请根据以下规则布置帐篷：

规则：
1. 每个帐篷必须与一棵树正交相邻
2. 每棵树必须对应恰好一个帐篷
3. 帐篷之间不能相邻（包括对角线）
4. 行列数字表示对应行/列的帐篷数量

谜题网格（行末和底部为数量提示）：
{grid_str}

请将答案用[answer]标签包裹，例如：[answer] [(1,2), (3,4)] [/answer]。坐标采用(行号,列号)格式，从1开始计数。"""

    @staticmethod
    def extract_output(output: str) -> List[Tuple[int, int]]:
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            last_match = matches[-1].strip()
            solution = ast.literal_eval(last_match)
            if isinstance(solution, list) and all(isinstance(t, tuple) and len(t)==2 for t in solution):
                return solution
        except:
            pass
        return None

    @classmethod
    def _verify_correction(cls, solution, identity) -> bool:
        if not isinstance(solution, list):
            return False
        
        try:
            user_coords = [(x-1, y-1) for (x, y) in solution]
        except:
            return False
        
        if len(user_coords) != len(set(user_coords)):
            return False
        
        grid = identity['grid']
        rows, cols = len(grid), len(grid[0])
        row_clues = identity['row_clues']
        col_clues = identity['col_clues']
        
        # Coordinate validation
        for x, y in user_coords:
            if x < 0 or y < 0 or x >= rows or y >= cols:
                return False
            if grid[x][y] == 1:
                return False
        
        # Tent adjacency check
        tents = set(user_coords)
        for (x1, y1) in tents:
            for (x2, y2) in tents:
                if (x1, y1) == (x2, y2):
                    continue
                if abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1:
                    return False
        
        # Tree-tent mapping validation
        tree_counts = {(i,j):0 for i in range(rows) for j in range(cols) if grid[i][j]}
        for x, y in tents:
            adjacent_tree = False
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    if grid[nx][ny]:
                        adjacent_tree = True
                        tree_counts[(nx, ny)] += 1
            if not adjacent_tree:
                return False
        
        if any(cnt != 1 for cnt in tree_counts.values()):
            return False
        
        # Clues validation
        actual_rows = [0]*rows
        actual_cols = [0]*cols
        for x, y in tents:
            actual_rows[x] += 1
            actual_cols[y] += 1
        return actual_rows == row_clues and actual_cols == col_clues
