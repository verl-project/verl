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

**Objective**: Clear a rectangular grid of hidden cells without detonating any mines. Cells contain either mines or numbers indicating adjacent mines.

**Grid Setup**:
1. The grid consists of hidden cells, some containing mines (randomly placed).
2. Non-mine cells reveal a number when uncovered, representing the total mines in the 8 adjacent cells (vertically, horizontally, and diagonally).

**Gameplay**:
1. **Uncover a Cell**: Click/select a cell to reveal its content:
   - If it contains a mine, the game ends (loss).
   - If it shows a number, use this to deduce nearby mine locations.
   - If it shows **0** (no adjacent mines), all adjacent cells automatically uncover recursively until numbered cells are reached.

2. **Flagging Mines**: Right-click/mark a cell to flag it as a suspected mine (prevents accidental uncovering). Flags help track potential mines but do not affect gameplay logic otherwise.

**Win Condition**:
- All non-mine cells are uncovered, and all mines are correctly flagged.

**Logic Rules**:
- Numbers on the grid are **static hints**, not live updates. Flagging a mine does not change existing numbers.
- Use the numbers to infer mine positions: e.g., a cell labeled \"3\" must have exactly 3 mines in its 8 neighboring cells.

**Loss Condition**:
- Uncovering any cell containing a mine.


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
import ast

class Minesweeperbootcamp(Basebootcamp):
    def __init__(self, rows=8, cols=8, mines_count=10):
        if mines_count > rows * cols:
            raise ValueError("Number of mines cannot exceed grid size.")
        self.rows = rows
        self.cols = cols
        self.mines_count = mines_count
    
    def case_generator(self):
        all_cells = [(i, j) for i in range(self.rows) for j in range(self.cols)]
        if self.mines_count > len(all_cells):
            raise ValueError("mines_count exceeds valid cells count.")
        mines = random.sample(all_cells, self.mines_count)
        mines_list = [list(coord) for coord in mines]
        return {
            'rows': self.rows,
            'cols': self.cols,
            'mines': mines_list
        }
    
    @staticmethod
    def prompt_func(question_case):
        rows = question_case['rows']
        cols = question_case['cols']
        mines_count = len(question_case['mines'])
        mines_set = set(tuple(coord) for coord in question_case['mines'])
        
        grid_info = []
        for i in range(rows):
            for j in range(cols):
                count = 0
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        x, y = i + dx, j + dy
                        if 0 <= x < rows and 0 <= y < cols and (x, y) in mines_set:
                            count += 1
                grid_info.append((i, j, count))
        
        prompt = (
            f"You are playing Minesweeper on a {rows}x{cols} grid with {mines_count} mines.\n"
            "Each number below represents the count of adjacent mines for a cell. "
            "Find all the mine locations and provide them in the specified format.\n\n"
            "Revealed cells (format: row, column: count):\n"
        )
        for i, j, num in grid_info:
            prompt += f"- ({i}, {j}): {num}\n"
        prompt += (
            "\nYour answer must be a list of mine coordinates formatted as [[row1, col1], [row2, col2], ...]. "
            "Place your final answer between [answer] and [/answer]."
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        pattern = r'\[answer\](.*?)\[/answer\]'
        matches = re.findall(pattern, output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            solution = ast.literal_eval(last_match)
            if (isinstance(solution, list) and 
                all(isinstance(coord, list) and len(coord) == 2 for coord in solution)):
                return solution
            return None
        except (SyntaxError, ValueError):
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not isinstance(solution, list):
            return False
        try:
            solution_set = {tuple(coord) for coord in solution}
        except TypeError:
            return False
        mines = identity.get('mines', [])
        mines_set = {tuple(mine) for mine in mines}
        if len(solution_set) != len(mines_set):
            return False
        rows, cols = identity['rows'], identity['cols']
        for (r, c) in solution_set:
            if not (0 <= r < rows and 0 <= c < cols):
                return False
        return solution_set == mines_set
