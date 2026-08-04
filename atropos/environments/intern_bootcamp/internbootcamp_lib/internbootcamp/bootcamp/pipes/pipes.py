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

The \"Pipes\" puzzle is a grid-based logic game where players connect colored endpoints with continuous, non-overlapping paths under the following rules:

1. **Grid Structure**: The puzzle is played on a rectangular grid. Each cell can either be empty, contain a pipe segment, or hold a colored endpoint.

2. **Endpoints**: 
   - Each color appears exactly twice on the grid, acting as an input (start) and output (end). 
   - All endpoints are pre-placed, and the goal is to connect matching colors.

3. **Pipe Path Rules**:
   - Paths must connect pairs of the same color, filling all cells between them.
   - Pipes can only run horizontally or vertically (no diagonal moves).
   - Paths must be continuous and cannot branch, cross other paths, or overlap. Each cell belongs to at most one pipe.

4. **Grid Completion**:
   - Every cell in the grid must be occupied by either a pipe segment or an endpoint. No empty cells are allowed in the solved state.

5. **Validity**:
   - A valid solution ensures all colored pairs are connected, and the entire grid is filled without rule violations.


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import json
import re
from typing import Dict, List, Tuple

class Pipesbootcamp(Basebootcamp):
    def __init__(self, rows=5, cols=5, num_colors=2):
        self.rows = rows
        self.cols = cols
        self.num_colors = num_colors
        self.colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange'][:num_colors]

    def case_generator(self) -> dict:
        col_split = []
        base_cols = self.cols // self.num_colors
        remainder = self.cols % self.num_colors
        current = 0
        for i in range(self.num_colors):
            add = 1 if i < remainder else 0
            col_width = base_cols + add
            col_split.append((current, current + col_width))
            current += col_width

        endpoints = {}
        solution_paths = {}
        for i in range(self.num_colors):
            color = self.colors[i]
            start_col, end_col = col_split[i]
            path = []
            for row in range(self.rows):
                if row % 2 == 0:
                    cols_in_row = range(start_col, end_col)
                else:
                    cols_in_row = range(end_col-1, start_col-1, -1)
                for col in cols_in_row:
                    path.append((row, col))
            start = path[0]
            end = path[-1]
            endpoints[color] = [list(start), list(end)]
            solution_paths[color] = [list(coord) for coord in path]

        return {
            'grid_size': [self.rows, self.cols],
            'endpoints': endpoints,
            'solution_paths': solution_paths
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        grid_size = question_case['grid_size']
        endpoints = question_case['endpoints']
        prompt = (
            f"You are playing a 'Pipes' puzzle on a {grid_size[0]}x{grid_size[1]} grid. Connect each pair of colored endpoints "
            "with continuous, non-overlapping paths that fill all cells. Follow these rules:\n"
            "1. Paths must be straight lines (horizontal/vertical)\n"
            "2. All cells must be filled\n"
            "3. Paths cannot cross or overlap\n\n"
            "Endpoints positions:\n"
        )
        for color, points in endpoints.items():
            prompt += f"- {color}: Start at {points[0]}, End at {points[1]}\n"
        prompt += (
            "\nFormat your answer as a JSON dictionary where keys are colors and values are coordinate lists "
            "from start to end. Enclose your answer within [answer] and [/answer] tags.\n"
            "Example:\n[answer]\n{\n  \"red\": [[0,0], [0,1], [1,1]],\n  \"blue\": [[2,2], [2,3]]\n}\n[/answer]"
        )
        return prompt

    @staticmethod
    def extract_output(output: str):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            solution = json.loads(matches[-1].strip())
            converted = {}
            for color, path in solution.items():
                converted_path = []
                for coord in path:
                    if isinstance(coord, list) and len(coord) == 2:
                        converted_path.append(tuple(coord))
                    else:
                        return None
                converted[color] = converted_path
            return converted
        except json.JSONDecodeError:
            return None

    @classmethod
    def _verify_correction(cls, solution: Dict[str, List[Tuple[int, int]]], identity: dict) -> bool:
        endpoints = {color: [tuple(ep) for ep in eps] for color, eps in identity['endpoints'].items()}
        grid_size = tuple(identity['grid_size'])
        all_coords = set()

        if set(solution.keys()) != set(endpoints.keys()):
            return False

        for color, path in solution.items():
            if len(path) < 2:
                return False
            start, end = path[0], path[-1]
            expected = set(endpoints[color])
            if {start, end} != set(expected):
                return False

            prev = path[0]
            for coord in path[1:]:
                dx, dy = abs(coord[0]-prev[0]), abs(coord[1]-prev[1])
                if dx + dy != 1:
                    return False
                prev = coord

            for coord in path:
                if coord in all_coords:
                    return False
                all_coords.add(coord)

        expected_coords = {(i, j) for i in range(grid_size[0]) for j in range(grid_size[1])}
        if all_coords != expected_coords:
            return False

        for color, path in solution.items():
            other_eps = set()
            for other_color in endpoints:
                if other_color != color:
                    other_eps.update(tuple(ep) for ep in endpoints[other_color])
            for coord in path[1:-1]:
                if coord in other_eps:
                    return False

        return True
