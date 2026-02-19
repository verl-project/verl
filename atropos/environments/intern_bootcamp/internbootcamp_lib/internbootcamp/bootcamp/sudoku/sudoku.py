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

Sudoku is a logic-based number placement puzzle played on a square grid divided into smaller subgrids (called \"regions\" or \"blocks\"). The rules are as follows:

1. **Grid Structure**:  
   - The puzzle grid is a square of size N×N, where N is a perfect square (e.g., 9×9, 16×16).  
   - The grid is subdivided into N smaller rectangular regions, each of size √N×√N. For example, a 9×9 grid has nine 3×3 regions.

2. **Number Placement**:  
   - The grid is partially filled with numbers (or symbols) at the start.  
   - The solver must fill every empty cell with a number from **1 to N** (or symbols equivalent in count to N).

3. **Core Rules**:  
   - **Row Constraint**: Each number must appear **exactly once** in every row.  
   - **Column Constraint**: Each number must appear **exactly once** in every column.  
   - **Region Constraint**: Each number must appear **exactly once** in every subgrid/region.  

The puzzle is solved when all cells are filled while satisfying all three constraints. No arithmetic or guessing is required—only logical deduction.


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import random
from typing import List, Optional

class Sudokubootcamp(Basebootcamp):
    def __init__(self, size: int = 9):
        """
        初始化数独训练场参数
        
        参数:
            size: 数独尺寸（必须为完全平方数，默认9）
        """
        sqrt_n = math.isqrt(size)
        if sqrt_n * sqrt_n != size:
            raise ValueError("Size必须为完全平方数")
        self.size = size
        self.sqrt_n = sqrt_n
    
    def case_generator(self) -> dict:
        """
        生成数独谜题实例
        
        返回包含以下信息的字典:
        - puzzle: 数独初始网格（0表示空格）
        - size: 数独尺寸
        - region_rows: 子区域行数
        - region_cols: 子区域列数（同region_rows）
        """
        # 生成完整解
        solution = self._generate_full_sudoku()
        
        # 挖空50%的格子（可根据需求调整比例）
        puzzle = self._dig_holes(solution.copy(), dig_prob=0.5)
        
        return {
            "puzzle": [row.copy() for row in puzzle],
            "size": self.size,
            "region_rows": self.sqrt_n,
            "region_cols": self.sqrt_n
        }

    @staticmethod
    def prompt_func(question_case: dict) -> str:
        """
        将数独实例转换为自然语言描述的问题
        
        参数:
            question_case: case_generator生成的谜题实例
            
        返回:
            包含规则说明和当前谜题状态的格式化字符串
        """
        puzzle = question_case["puzzle"]
        size = question_case["size"]
        region_size = question_case["region_rows"]
        
        prompt = f"""你是一个专业数独玩家，请解决以下{size}x{size}的数独谜题。规则要求：
1. 每行必须包含1-{size}所有数字，无重复
2. 每列必须包含1-{size}所有数字，无重复
3. 每个{region_size}x{region_size}的子区域必须包含1-{size}所有数字，无重复

当前谜题状态（0表示空格）：
"""
        for i, row in enumerate(puzzle):
            prompt += f"第{i+1}行：" + " ".join(str(n) if n != 0 else "▢" for n in row) + "\n"

        prompt += "\n请将完整解答按如下格式放在[answer]标记之间：\n[answer]\n1 2 3 ...\n4 5 6 ...\n...\n[/answer]"
        return prompt

    @staticmethod
    def extract_output(output: str) -> Optional[List[List[int]]]:
        """
        从模型输出中提取最后一个数独解
        
        参数:
            output: 包含[answer]标记的完整输出文本
            
        返回:
            二维整数矩阵，解析失败时返回None
        """
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
            
        try:
            solution = []
            for line in matches[-1].strip().split('\n'):
                nums = list(map(int, line.strip().split()))
                if nums:
                    solution.append(nums)
            return solution
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution: List[List[int]], identity: dict) -> bool:
        """
        验证解的正确性
        
        参数:
            solution: 用户提交的解
            identity: 谜题实例信息
            
        返回:
            布尔值表示解的正确性
        """
        def is_valid_region(grid, row_start, col_start, size, region_size) -> bool:
            """验证子区域有效性"""
            nums = set()
            for i in range(row_start, row_start+region_size):
                for j in range(col_start, col_start+region_size):
                    num = grid[i][j]
                    if num < 1 or num > size or num in nums:
                        return False
                    nums.add(num)
            return True

        puzzle = identity["puzzle"]
        size = identity["size"]
        region_size = identity["region_rows"]
        
        # 基本维度检查
        if len(solution) != size or any(len(row) != size for row in solution):
            return False
        
        # 验证初始条件
        for i in range(size):
            for j in range(size):
                if puzzle[i][j] != 0 and solution[i][j] != puzzle[i][j]:
                    return False
        
        # 验证行、列、子区域
        valid_range = set(range(1, size+1))
        for i in range(size):
            if set(solution[i]) != valid_range:  # 行验证
                return False
            if set(solution[j][i] for j in range(size)) != valid_range:  # 列验证
                return False
            
        # 子区域验证
        for i in range(0, size, region_size):
            for j in range(0, size, region_size):
                if not is_valid_region(solution, i, j, size, region_size):
                    return False
        
        return True

    def _generate_full_sudoku(self) -> List[List[int]]:
        """生成有效完整数独的核心算法"""
        size = self.size
        region_size = self.sqrt_n
        grid = [[0]*size for _ in range(size)]
        
        # 填充对角线子区域
        for i in range(0, size, region_size):
            nums = list(range(1, size+1))
            random.shuffle(nums)
            for x in range(region_size):
                for y in range(region_size):
                    grid[i+x][i+y] = nums[x*region_size + y]
        
        # 解数独
        self._solve_sudoku(grid)
        return grid

    def _solve_sudoku(self, grid: List[List[int]]) -> bool:
        """回溯法解数独"""
        size = self.size
        region_size = self.sqrt_n
        empty = self._find_empty(grid)
        
        if not empty:
            return True
            
        row, col = empty
        for num in random.sample(range(1, size+1), size):  # 随机尝试增加多样性
            if self._is_safe(grid, row, col, num):
                grid[row][col] = num
                if self._solve_sudoku(grid):
                    return True
                grid[row][col] = 0
        return False

    def _find_empty(self, grid: List[List[int]]) -> Optional[tuple]:
        """寻找下一个空单元格"""
        for i in range(self.size):
            for j in range(self.size):
                if grid[i][j] == 0:
                    return (i, j)
        return None

    def _is_safe(self, grid: List[List[int]], row: int, col: int, num: int) -> bool:
        """检查数字是否可以安全填入"""
        size = self.size
        region_size = self.sqrt_n
        
        # 检查行和列
        if num in grid[row] or num in [grid[i][col] for i in range(size)]:
            return False
            
        # 检查子区域
        start_row, start_col = row - row%region_size, col - col%region_size
        for i in range(region_size):
            for j in range(region_size):
                if grid[start_row+i][start_col+j] == num:
                    return False
        return True

    def _dig_holes(self, grid: List[List[int]], dig_prob: float) -> List[List[int]]:
        """挖洞生成谜题（保证至少有一个解）"""
        size = self.size
        for i in range(size):
            for j in range(size):
                if random.random() < dig_prob:
                    grid[i][j] = 0
        return grid
