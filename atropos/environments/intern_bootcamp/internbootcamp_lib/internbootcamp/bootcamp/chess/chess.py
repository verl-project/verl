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

The chess puzzle involves strategically placing a specified number of pieces (e.g., queens, knights) on an N×N grid, adhering to movement constraints inherent to each piece type. The goal is to arrange all pieces such that no two pieces violate predefined interaction rules (e.g., mutual non-attacking positions for queens, restricted adjacency for knights). The solution must satisfy these conditions across the entire grid, regardless of its size or the number of pieces. Each piece's movement capabilities (e.g., queens attacking along rows, columns, and diagonals; knights moving in L-shaped patterns) define the constraints. The puzzle is considered solved when a valid configuration is found for the generalized parameters (any grid size and piece count).


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import ast
import re

class ChessV2bootcamp(Basebootcamp):
    def __init__(self, grid_size=8, piece_type='queen', piece_count=None):
        super().__init__()
        self.grid_size = grid_size
        self.piece_type = piece_type
        
        if piece_count is None:
            if self.piece_type == 'queen':
                self.piece_count = grid_size
            else:
                self.piece_count = 1  # Default for other piece types
        else:
            self.piece_count = piece_count
        
        # Validate parameters
        if self.piece_type == 'queen' and self.piece_count != self.grid_size:
            raise ValueError("For queens, piece_count must equal grid_size")
        if self.grid_size < 4 and self.piece_type == 'queen':
            raise ValueError("Grid size must be at least 4 for queens")
    
    def case_generator(self):
        # Generate valid grid_size ensuring solvability for queens
        if self.piece_type == 'queen':
            valid_sizes = [4, 5, 6, 7, 8]
            grid_size = random.choice(valid_sizes)
            piece_count = grid_size
        else:
            grid_size = self.grid_size
            piece_count = self.piece_count
        
        return {
            'grid_size': grid_size,
            'piece_type': self.piece_type,
            'piece_count': piece_count
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        grid_size = question_case['grid_size']
        piece_type = question_case['piece_type']
        piece_count = question_case['piece_count']
        
        if piece_type == 'queen':
            rule_desc = "no two queens can share the same row, column, or diagonal"
        elif piece_type == 'knight':
            rule_desc = "no two knights can be positioned at a knight's move (L-shaped) from each other"
        else:
            rule_desc = "pieces cannot attack each other based on their movement rules"
        
        return f"""You are a chess puzzle solver. Place {piece_count} {piece_type}(s) on a {grid_size}x{grid_size} grid such that {rule_desc}.
The grid uses 1-based indexing (rows and columns range from 1 to {grid_size}).

Provide your answer as a list of coordinates in the format:
[answer]
[(row1, col1), (row2, col2), ..., (rowN, colN)]
[/answer]
Ensure all coordinates are integers between 1 and {grid_size}."""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        
        last_answer = matches[-1].strip()
        try:
            return ast.literal_eval(last_answer)
        except (SyntaxError, ValueError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        grid_size = identity['grid_size']
        piece_type = identity['piece_type']
        piece_count = identity['piece_count']
        
        # Basic format validation
        if not isinstance(solution, list) or len(solution) != piece_count:
            return False
        
        try:
            positions = [(int(r), int(c)) for r, c in solution]
        except (TypeError, ValueError):
            return False
        
        # Boundary check
        for r, c in positions:
            if not (1 <= r <= grid_size and 1 <= c <= grid_size):
                return False
        
        # Uniqueness check
        if len(set(positions)) != len(positions):
            return False
        
        # Piece-specific attack checks
        if piece_type == 'queen':
            for i in range(len(positions)):
                r1, c1 = positions[i]
                for j in range(i+1, len(positions)):
                    r2, c2 = positions[j]
                    if r1 == r2 or c1 == c2 or abs(r1-r2) == abs(c1-c2):
                        return False
            return True
        
        if piece_type == 'knight':
            for i in range(len(positions)):
                r1, c1 = positions[i]
                for j in range(i+1, len(positions)):
                    r2, c2 = positions[j]
                    dr = abs(r1 - r2)
                    dc = abs(c1 - c2)
                    if (dr == 2 and dc == 1) or (dr == 1 and dc == 2):
                        return False
            return True
        
        return False  # Unknown piece type
