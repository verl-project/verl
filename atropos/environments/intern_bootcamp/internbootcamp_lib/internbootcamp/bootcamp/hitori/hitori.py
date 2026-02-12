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

Hitori is a logic puzzle played on a square grid where each cell contains a number. The objective is to shade (blacken) cells according to the following rules:

1. **No Duplicates in Rows/Columns**: 
   After shading, all *unshaded* numbers in every row and column must be unique. This means duplicates in the original grid must be resolved by shading some occurrences, ensuring no repeats in the final unshaded cells.

2. **Shaded Cells Cannot Be Adjacent**:
   No two shaded cells may be directly adjacent to each other horizontally or vertically (diagonally is allowed).

3. **Unshaded Cells Must Be Connected**:
   All unshaded cells must form a single continuous region connected orthogonally (horizontally or vertically). Unshaded cells cannot be isolated from the rest by shaded cells.

To solve the puzzle, shade cells strategically to eliminate duplicate numbers while adhering to adjacency and connectivity constraints.


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
import ast

class Hitoribootcamp(Basebootcamp):
    def __init__(self, size=5, number_range=None):
        super().__init__()
        self.size = size
        self.number_range = number_range if number_range else (1, size)
    
    def case_generator(self):
        size = self.size
        grid = self._generate_latin_square(size)
        shaded_cells = self._generate_valid_shaded_cells(size)
        
        for i, j in shaded_cells:
            unshaded_row = [(i, c) for c in range(size) if (i, c) not in shaded_cells and c != j]
            unshaded_col = [(r, j) for r in range(size) if (r, j) not in shaded_cells and r != i]
            if unshaded_row:
                sample_cell = random.choice(unshaded_row)
                grid[i][j] = grid[sample_cell[0]][sample_cell[1]]
            elif unshaded_col:
                sample_cell = random.choice(unshaded_col)
                grid[i][j] = grid[sample_cell[0]][sample_cell[1]]
        
        return {"grid": grid}
    
    def _generate_valid_shaded_cells(self, size):
        shaded = set()
        candidates = [(i, j) for i in range(size) for j in range(size)]
        random.shuffle(candidates)
        
        for cell in candidates:
            i, j = cell
            adjacent = any((i + di, j + dj) in shaded for di, dj in [(-1,0), (1,0), (0,-1), (0,1)] if 0 <= i + di < size and 0 <= j + dj < size)
            if adjacent:
                continue
            shaded.add(cell)
            if not self._is_connected(size, shaded):
                shaded.remove(cell)
        return shaded
    
    def _is_connected(self, size, shaded):
        unshaded = [(i, j) for i in range(size) for j in range(size) if (i, j) not in shaded]
        if not unshaded:
            return False
        visited = set()
        queue = [unshaded[0]]
        while queue:
            cell = queue.pop(0)
            if cell in visited:
                continue
            visited.add(cell)
            i, j = cell
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < size and 0 <= nj < size and (ni, nj) not in shaded and (ni, nj) not in visited:
                    queue.append((ni, nj))
        return len(visited) == len(unshaded)
    
    @staticmethod
    def _generate_latin_square(size):
        return [[(i + j) % size + 1 for j in range(size)] for i in range(size)]
    
    @staticmethod
    def prompt_func(question_case):
        grid = question_case["grid"]
        grid_str = "\n".join([" ".join(map(str, row)) for row in grid])
        return f"""你是一名Hitori谜题专家，请根据以下规则解决谜题：

规则：
1. 每行每列未涂黑数字必须唯一。
2. 涂黑单元格不能相邻（上下左右）。
3. 所有未涂黑单元格必须连通。

题目网格：
{grid_str}

请将答案的单元格坐标列表（从0开始）放在[answer]和[/answer]之间，例如：[answer][(0,1), (2,3)][/answer]"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            result = ast.literal_eval(matches[-1].strip())
            if isinstance(result, list) and all(isinstance(cell, tuple) and len(cell) == 2 for cell in result):
                return result
        except:
            pass
        return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not isinstance(solution, list):
            return False
        grid = identity["grid"]
        size = len(grid)
        shaded = set(solution)
        
        if len(shaded) != len(solution):
            return False
        
        for i, j in shaded:
            if not (0 <= i < size and 0 <= j < size):
                return False
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                if (i + di, j + dj) in shaded:
                    return False
        
        unshaded = [(i, j) for i in range(size) for j in range(size) if (i, j) not in shaded]
        for i in range(size):
            row = [grid[r][c] for r, c in unshaded if r == i]
            if len(row) != len(set(row)):
                return False
            col = [grid[r][c] for r, c in unshaded if c == i]
            if len(col) != len(set(col)):
                return False
        
        if not unshaded:
            return False
        visited = set()
        queue = [unshaded[0]]
        while queue:
            cell = queue.pop(0)
            if cell in visited:
                continue
            visited.add(cell)
            i, j = cell
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i + di, j + dj
                if (ni, nj) in unshaded and (ni, nj) not in visited:
                    queue.append((ni, nj))
        return len(visited) == len(unshaded)
