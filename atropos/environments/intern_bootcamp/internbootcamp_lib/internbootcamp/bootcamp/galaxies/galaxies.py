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

**Objective**: Divide a grid into non-overlapping regions called \"galaxies,\" each containing exactly one central circle.  

**Key Rules**:  
1. **Symmetry**: Every galaxy must be rotationally symmetric around its central circle (180-degree rotation). This means each cell in the galaxy has a \"mirror\" cell positioned opposite the center.  
2. **Contiguity**: All cells in a galaxy must form a single connected region (adjacent horizontally or vertically).  
3. **Completeness**: Every grid cell must belong to exactly one galaxy.  

**How It Works**:  
- The grid contains pre-placed central circles. Your goal is to outline regions where each:  
  - Encloses exactly one center.  
  - Follows symmetry: If a cell is at position (x, y) relative to the center, there must be a corresponding cell at (-x, -y).  
  - Is connected and fills the grid without overlaps or gaps.  


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
from ast import literal_eval
from collections import deque

class Galaxiesbootcamp(Basebootcamp):
    def __init__(self, rows=5, cols=5):
        # Ensure odd dimensions for valid center placement
        self.rows = rows if rows % 2 != 0 else rows + 1
        self.cols = cols if cols % 2 != 0 else cols + 1

    def case_generator(self):
        """Generates a puzzle case with center(s) in a grid"""
        center = (self.rows // 2, self.cols // 2)
        return {
            'rows': self.rows,
            'cols': self.cols,
            'centers': [center]
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        """Generates problem description text with formatting instructions"""
        centers = question_case['centers']
        return f"""你是专业星系谜题解题专家，请根据以下信息划分星系：

**网格尺寸**: {question_case['rows']}x{question_case['cols']}
**中心位置**: {', '.join(f'({r},{c})' for r, c in centers)}

**规则要求**:
1. 每个星系必须包含且仅包含一个中心，且形状关于中心180度对称
2. 所有单元格必须属于且仅属于一个星系
3. 星系区域必须连通（上下左右相邻）

请按以下格式返回答案：
[answer]
[
    {{"center": (行坐标, 列坐标), "cells": [(坐标1), (坐标2), ...]}},
    ...
]
[/answer]

请确保：
1. 使用严格的Python列表和元组语法
2. 包含所有中心对应的星系
3. 每个坐标均为(row, column)格式"""

    @staticmethod
    def extract_output(output):
        """Extracts last valid answer block from LLM output"""
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return literal_eval(matches[-1].strip())
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """Validates solution against puzzle constraints"""
        try:
            # Validate solution structure
            if not cls._validate_structure(solution):
                return False

            # Check center consistency
            if not cls._check_centers(solution, identity['centers']):
                return False

            # Check grid coverage
            if not cls._check_coverage(solution, identity['rows'], identity['cols']):
                return False

            # Validate each galaxy
            for galaxy in solution:
                if not cls._validate_galaxy(galaxy):
                    return False
            return True
        except:
            return False

    @staticmethod
    def _validate_structure(solution):
        """Validate basic solution structure"""
        if not isinstance(solution, list):
            return False
        for g in solution:
            if not isinstance(g, dict) or 'center' not in g or 'cells' not in g:
                return False
            if not isinstance(g['cells'], list) or len(g['cells']) == 0:
                return False
        return True

    @staticmethod
    def _check_centers(solution, expected_centers):
        """Verify all expected centers are present"""
        solution_centers = {tuple(g['center']) for g in solution}
        expected_set = {tuple(c) for c in expected_centers}
        return solution_centers == expected_set

    @staticmethod
    def _check_coverage(solution, rows, cols):
        """Verify complete grid coverage without overlaps"""
        all_cells = []
        for g in solution:
            all_cells.extend(map(tuple, g['cells']))
        expected = {(r, c) for r in range(rows) for c in range(cols)}
        return len(all_cells) == len(expected) and set(all_cells) == expected

    @classmethod
    def _validate_galaxy(cls, galaxy):
        """Validate individual galaxy constraints"""
        cells = [tuple(c) for c in galaxy['cells']]
        center = tuple(galaxy['center'])
        
        # Check center presence
        if center not in cells:
            return False

        # Check symmetry
        cx, cy = center
        for (x, y) in cells:
            sym = (2*cx - x, 2*cy - y)
            if sym not in cells:
                return False

        # Check connectivity
        return cls._is_connected(cells)

    @staticmethod
    def _is_connected(cells):
        """BFS check for region connectivity"""
        if not cells:
            return False
        
        visited = set()
        q = deque([cells[0]])
        visited.add(cells[0])
        
        while q:
            x, y = q.popleft()
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                neighbor = (x+dx, y+dy)
                if neighbor in cells and neighbor not in visited:
                    visited.add(neighbor)
                    q.append(neighbor)
        
        return len(visited) == len(cells)
