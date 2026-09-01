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

Jigsaw Sudoku follows core Sudoku principles but replaces traditional fixed rectangular regions (\"boxes\") with irregularly shaped, contiguous regions (\"pieces\"). The rules are:

1. **Grid Structure**: The puzzle is played on an n×n grid, divided into n distinct regions. Each region must contain exactly n cells.

2. **Row and Column Rules**: Every row and column must contain all numbers from 1 to n exactly once (no duplicates, no omissions).

3. **Region Rule**: Each irregularly shaped region must also contain all numbers from 1 to n exactly once, with no repeats.

4. **Region Properties**: 
   - Regions can vary in shape (non-rectangular, \"jigsaw-like\").
   - All regions are contiguous (cells are connected edge-to-edge).
   - Regions do not overlap and fully partition the grid.

The challenge lies in satisfying all three constraints (rows, columns, and irregular regions) simultaneously. The grid size (n) is typically a perfect square (e.g., 4×4, 9×9), but the logic applies universally.


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re

class JigsawSudokuV2bootcamp(Basebootcamp):
    def __init__(self, n=4, keep_prob=0.5):
        self.n = n
        self.keep_prob = keep_prob  # 保留下来的已知数字的比例
    
    def case_generator(self):
        # 示例固定区域结构和解（以4x4为例）
        n = self.n
        regions = [
            [[0,0], [0,1], [1,0], [2,0]],
            [[0,2], [0,3], [1,3], [2,3]],
            [[1,1], [1,2], [2,1], [2,2]],
            [[3,0], [3,1], [3,2], [3,3]]
        ]
        solution = [
            [1, 3, 4, 2],
            [4, 2, 1, 3],
            [2, 4, 3, 1],
            [3, 1, 2, 4]
        ]
        # 生成谜题实例，挖空部分单元格
        puzzle = []
        for row in solution:
            puzzle_row = []
            for num in row:
                if random.random() > self.keep_prob:
                    puzzle_row.append(None)
                else:
                    puzzle_row.append(num)
            puzzle.append(puzzle_row)
        return {
            'n': n,
            'regions': regions,
            'puzzle': puzzle,
            'solution': solution
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        regions = question_case['regions']
        puzzle = question_case['puzzle']
        
        # 生成区域结构描述
        region_grid = [[-1]*n for _ in range(n)]
        for idx, cells in enumerate(regions):
            for (r, c) in cells:
                region_grid[r][c] = idx
        
        region_str = "区域结构（每个单元格显示所属区域的编号）：\n"
        region_str += '\n'.join([' '.join(map(str, row)) for row in region_grid])
        
        # 生成谜题表格
        puzzle_str = "初始谜题（空格用_表示）：\n"
        for row in puzzle:
            puzzle_row = ['_' if num is None else str(num) for num in row]
            puzzle_str += ' '.join(puzzle_row) + '\n'
        
        # 构建完整提示
        prompt = f"""你是一名Jigsaw Sudoku玩家，请根据以下信息解答谜题：

谜题规则：
1. 每一行必须包含数字1到{n}，每个数字恰好一次。
2. 每一列必须包含数字1到{n}，每个数字恰好一次。
3. 每个不规则区域（由下方区域结构定义）必须包含数字1到{n}，每个数字恰好一次。

{region_str}

{puzzle_str}

请将完整解答按行排列，每行数字以逗号分隔，放在[answer]和[/answer]之间。例如：
[answer]
1,3,4,2
4,2,1,3
2,4,3,1
3,1,2,4
[/answer]
"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        # 使用正则匹配最后一个答案块
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        
        last_answer = answer_blocks[-1].strip()
        solution = []
        for line in last_answer.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                solution.append([int(num) for num in line.split(',')])
            except:
                return None
        return solution if solution else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        puzzle = identity['puzzle']
        regions = identity['regions']
        
        # 检查已知数字是否匹配
        for r in range(n):
            for c in range(n):
                if puzzle[r][c] is not None and solution[r][c] != puzzle[r][c]:
                    return False
        
        # 验证行
        for row in solution:
            if sorted(row) != list(range(1, n+1)):
                return False
        
        # 验证列
        for c in range(n):
            if sorted(solution[r][c] for r in range(n)) != list(range(1, n+1)):
                return False
        
        # 验证区域
        for region in regions:
            region_numbers = [solution[r][c] for (r, c) in region]
            if sorted(region_numbers) != list(range(1, n+1)):
                return False
        
        return True
