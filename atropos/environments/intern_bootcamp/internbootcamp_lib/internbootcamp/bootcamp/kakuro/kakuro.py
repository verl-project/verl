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

**Kakuro Puzzle Rules:**

1. **Grid Structure**:  
   - The puzzle is played on a grid of white (empty) and black (clue) cells.  
   - **Clue cells** (black) contain hints for solving adjacent white cells. Each clue has two components:  
     - **Rightward (→)**: Sum of digits in the horizontal sequence of white cells to its right.  
     - **Downward (↓)**: Sum of digits in the vertical sequence of white cells below it.  

2. **Digit Placement**:  
   - Fill white cells with digits **1–9**.  
   - A digit **cannot repeat** within the same horizontal or vertical sequence (referred to as a \"run\").  

3. **Run Constraints**:  
   - Each run is defined by a clue cell. For example, a rightward clue of \"12 in 3 cells\" means the three adjacent horizontal cells must sum to 12, with no repeated digits.  
   - A white cell can belong to both a horizontal and vertical run simultaneously. Its digit must satisfy **both clues**.  

4. **Key Principles**:  
   - **Uniqueness**: All digits in a single run must be distinct.  
   - **No Zeros**: Digits must be between 1 and 9.  
   - **Interconnected Solutions**: Solving one run provides constraints for intersecting runs.  

**Objective**: Fill all white cells to satisfy all horizontal and vertical clues without violating the rules.  


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from ast import literal_eval

class Kakurobootcamp(Basebootcamp):
    def __init__(self, rows=3, cols=3):
        self.rows = rows
        self.cols = cols
    
    def case_generator(self):
        # 生成横向序列的数对
        a, b = self._generate_unique_pair()
        sum_r = a + b
        
        # 生成纵向序列的数对
        c, d = self._generate_unique_pair()
        sum_d = c + d
        
        # 构建网格结构
        grid = [[{'type': 'black', 'right': (sum_r, 2), 'down': (sum_d, 2)} if (row == 0 and col == 0) else
                {'type': 'white'} if ((row == 0 and col in (1, 2)) or (col == 0 and row in (1, 2))) else
                {'type': 'black'} for col in range(self.cols)] for row in range(self.rows)]
        
        solution = {
            "(0, 1)": a,
            "(0, 2)": b,
            "(1, 0)": c,
            "(2, 0)": d
        }
        
        return {
            'grid': grid,
            'solution': solution
        }
    
    def _generate_unique_pair(self):
        while True:
            a = random.randint(1, 9)
            b = random.randint(1, 9)
            if a != b:
                return a, b
    
    @staticmethod
    def prompt_func(question_case) -> str:
        clues = []
        grid = question_case['grid']
        for row_idx, row in enumerate(grid):
            for col_idx, cell in enumerate(row):
                if cell['type'] == 'black':
                    parts = []
                    if 'right' in cell:
                        sum_r, len_r = cell['right']
                        parts.append(f"右侧的 {len_r} 个白色格子之和为 {sum_r}")
                    if 'down' in cell:
                        sum_d, len_d = cell['down']
                        parts.append(f"下方的 {len_d} 个白色格子之和为 {sum_d}")
                    if parts:
                        clues.append(f"位于 ({row_idx}, {col_idx}) 的黑色格子：" + "，".join(parts))
        clues_text = "\n".join(clues)
        
        white_coords = []
        for row_idx, row in enumerate(grid):
            for col_idx, cell in enumerate(row):
                if cell['type'] == 'white':
                    white_coords.append(f"({row_idx}, {col_idx})")
        white_coords_text = ", ".join(white_coords)
        
        prompt = f"""你是Kakuro谜题解答者，请根据以下线索填充所有白色格子，确保每个横向或纵向的序列满足和的条件，且同一序列中的数字不重复。每个格子只能填1-9的整数。

谜题线索：
{clues_text}

需要填充的白色格子位于以下坐标：{white_coords_text}。

请将你的答案以字典形式放在[answer]和[/answer]之间，键为坐标字符串，如"(行,列)"，值为对应的整数。例如：
[answer]
{{"(0,1)": 3, "(0,2)": 4, "(1,0)":5, "(2,0)":2}}
[/answer]
请确保所有白色格子都被正确填写，且没有多余或缺少的项。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        last_block = answer_blocks[-1].strip()
        try:
            answer_dict = literal_eval(last_block)
            if not isinstance(answer_dict, dict):
                return None
            converted = {}
            for coord_str, value in answer_dict.items():
                coord_str = coord_str.strip('()')
                row, col = map(int, coord_str.split(','))
                converted[(row, col)] = value
            return converted
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        grid = identity['grid']
        solution = solution.copy()
        
        # Check all coordinates in solution are valid white cells
        for coord in solution:
            row, col = coord
            if row < 0 or col < 0 or row >= len(grid) or col >= len(grid[0]):
                return False
            cell = grid[row][col]
            if cell.get('type') != 'white':
                return False
            value = solution[coord]
            if not (1 <= value <= 9):
                return False
        
        # Check all clues
        for row_idx in range(len(grid)):
            for col_idx in range(len(grid[row_idx])):
                cell = grid[row_idx][col_idx]
                if cell.get('type') != 'black':
                    continue
                # Check right clue
                if 'right' in cell:
                    sum_r, len_r = cell['right']
                    run_coords = []
                    current_col = col_idx + 1
                    while current_col < len(grid[row_idx]) and grid[row_idx][current_col].get('type') == 'white':
                        run_coords.append((row_idx, current_col))
                        current_col += 1
                    if len(run_coords) != len_r:
                        return False
                    # Check all coords are in solution
                    for coord in run_coords:
                        if coord not in solution:
                            return False
                    values = [solution[coord] for coord in run_coords]
                    if sum(values) != sum_r or len(set(values)) != len_r:
                        return False
                # Check down clue
                if 'down' in cell:
                    sum_d, len_d = cell['down']
                    run_coords = []
                    current_row = row_idx + 1
                    while current_row < len(grid) and grid[current_row][col_idx].get('type') == 'white':
                        run_coords.append((current_row, col_idx))
                        current_row += 1
                    if len(run_coords) != len_d:
                        return False
                    for coord in run_coords:
                        if coord not in solution:
                            return False
                    values = [solution[coord] for coord in run_coords]
                    if sum(values) != sum_d or len(set(values)) != len_d:
                        return False
        return True
