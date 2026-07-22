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
from bootcamp import Basebootcamp

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
        注意：请参照提供的谜题描述进行复述，规则应当描述详细，包括任务背景、具体任务操作规则、对题目格式和答案格式的含义介绍等，

        参数:
            question_case: 由case_generator生成的谜题实例
            
        返回:
            str: 格式化的问题字符串
            
        注意:
            1. 需考虑问题的格式，以便后续能正确提取
            2. 问题描述中应包含期望的答案格式说明，以便后续能正确提取，为了避免抽取时匹配出干扰项，请要求模型将答案放在特定标签（如双括号）内，例如[[your answer here]]
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
1.The game consists of a 9x9 grid which is subdivided into nine 3x3 subgrids.
2.The grid is pre-filled with numbers which act as hints to help the player solve the sudoku.
3.The player needs to fill the entire grid so that each row, column and 3x3 subgrid contains all the numbers from 1 to 9 without repetition.
4.The puzzle is given in the form of a matrix with blank squares filled with X, and the player should replace these areas with numbers.Example questions are as follows:

<example 0>
9 X X X 8 1 X X 3
X X 7 9 2 X X 5 X
X 8 X X X X 9 1 7
X 7 X 5 X 6 X X X
X X 4 X 7 X 5 9 2
X X 8 2 X X 6 7 X
4 X X X 5 X X 2 X
8 X X X 1 2 X 6 4
X X X 4 X X X X X
Please provide each element in order from left to right, and from top to bottom, with each element separated by a space and each row separated by a comma. Ensure that your final answer is wrapped in double square brackets.

For example, if the answer is:
A B C
D E F
G H I

please output [[A B C,D E F,G H I]].
</example 0>

<example 1>
X X X X X X X 6 9
X 3 X 5 X X X X 2
X X 6 4 8 X 1 3 7
9 X X 2 4 X X X 1
X 6 2 X 5 X 7 X X
X X X X 7 X X X X
5 X 3 X X X X X X
6 7 8 X 9 4 3 2 X
X 9 4 3 X 5 8 X X

Please provide each element in order from left to right, and from top to bottom, with each element separated by a space and each row separated by a comma. Ensure that your final answer is wrapped in double square brackets.

For example, if the answer is:
A B C
D E F
G H I

please output [[A B C,D E F,G H I]].
</example 1>

<example 2>
X 5 X 1 X 8 X X 9
2 X 4 7 5 X X 6 1
X 9 X 2 X X 4 X X
X X X X X X 7 9 X
X X 6 3 2 X X X X
X X 8 X 7 X X 5 6
X X X X X X X X 4
3 X 2 4 X 6 X 8 X
X 6 X X X X X X X
Please provide each element in order from left to right, and from top to bottom, with each element separated by a space and each row separated by a comma. Ensure that your final answer is wrapped in double square brackets.

For example, if the answer is:
A B C
D E F
G H I

please output [[A B C,D E F,G H I]].
</example 2>

<example 3>
X 5 X X X 1 X X 9
6 X 3 X X X 2 7 X
9 X X 7 8 X 6 4 X
7 X X X 1 X X X X
X 3 X X X X X X X
X X 1 X X 9 X X X
X X X X 5 X X X 6
8 4 X 6 X X X X X
X X X X X 2 X X X
Please provide each element in order from left to right, and from top to bottom, with each element separated by a space and each row separated by a comma. Ensure that your final answer is wrapped in double square brackets.

For example, if the answer is:
A B C
D E F
G H I

please output [[A B C,D E F,G H I]].
</example 3>

<example 4>
X 9 X X 6 X X 1 X
X X X X 7 X 9 X X
X 1 X 8 X X 4 X X
X X X X X 8 1 2 X
X X X X X X X X 5
X X 7 X 5 9 6 X X
4 X X 7 X X X X X
X X 2 X 8 5 X X X
6 X X X X 2 8 4 X
Please provide each element in order from left to right, and from top to bottom, with each element separated by a space and each row separated by a comma. Ensure that your final answer is wrapped in double square brackets.

For example, if the answer is:
A B C
D E F
G H I

please output [[A B C,D E F,G H I]].
</example 4>

<example 5>
2 X X 7 5 X X X X
X X 5 X X X X X X
X X X 3 X 4 X X X
5 2 X X X 6 4 X X
X 4 X X 3 X X 5 X
X X X X X 8 9 X 6
X X 4 X X X 2 X X
X X 7 X X X X 9 X
9 5 X X X X 6 7 3
Please provide each element in order from left to right, and from top to bottom, with each element separated by a space and each row separated by a comma. Ensure that your final answer is wrapped in double square brackets.

For example, if the answer is:
A B C
D E F
G H I

please output [[A B C,D E F,G H I]].
</example 5>

<example 6>
X X 4 X 6 X 9 X 1
9 X 6 X X X 3 X X
X 1 X 7 X X X X 6
X 8 X 5 X X X 4 X
2 X X X X X 1 X X
6 X X X X X X X X
X X X X 7 X X X 4
X X 8 1 X X X 2 X
X X X 6 4 3 X X 7
Please provide each element in order from left to right, and from top to bottom, with each element separated by a space and each row separated by a comma. Ensure that your final answer is wrapped in double square brackets.

For example, if the answer is:
A B C
D E F
G H I

please output [[A B C,D E F,G H I]].
</example 6>

<example 7>
3 2 X X 7 X 6 X X
X X X 8 X 9 X X X
X X X X X X X 4 5
X 7 X X 5 X X X 4
X X 3 6 X X 1 X X
X X X X 9 X X X X
9 X X 2 X 4 X X X
X 6 X X X X 4 X 1
2 3 X X X 7 X X 9
Please provide each element in order from left to right, and from top to bottom, with each element separated by a space and each row separated by a comma. Ensure that your final answer is wrapped in double square brackets.

For example, if the answer is:
A B C
D E F
G H I

please output [[A B C,D E F,G H I]].
</example 7>

<example 8>
X X X 7 X X X X X
7 X 9 X 2 X X 4 5
5 X 8 X 3 X X X X
X X 6 X X X X 9 X
X 8 X X 5 X 3 X X
2 X X X X X 5 X 1
X X X X X X X X 7
3 X X X 1 X X X X
9 X X 6 7 3 X X X
Please provide each element in order from left to right, and from top to bottom, with each element separated by a space and each row separated by a comma. Ensure that your final answer is wrapped in double square brackets.

For example, if the answer is:
A B C
D E F
G H I

please output [[A B C,D E F,G H I]].
</example 8>

<example 9>
X X X 1 4 X X X X
X 2 X X X X X X X
X X 7 3 X 9 X X X
9 X X 4 X X X X 7
7 X X X X X 2 X X
3 X 6 X X X X 5 X
X X X X X X X X 5
X X X 6 X 1 X X 4
X 1 X 7 2 3 6 X 9
Please provide each element in order from left to right, and from top to bottom, with each element separated by a space and each row separated by a comma. Ensure that your final answer is wrapped in double square brackets.

For example, if the answer is:
A B C
D E F
G H I

please output [[A B C,D E F,G H I]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
import copy
from bootcamp import Basebootcamp

class KorPuzzleSudokobootcamp(Basebootcamp):
    def __init__(self, holes=30):
        """
        初始化数独训练场环境，控制谜题难度（挖空数量）
        :param holes: 需要保留的已知数字数量，默认挖30个空
        """
        self.holes = min(max(holes, 17), 81)  # 数独至少需要17个已知数才有唯一解

    def case_generator(self):
        """
        生成包含唯一解的数独谜题及参考答案
        :return: 包含谜题矩阵和答案的字典
        """
        solution = self.generate_sudoku_solution()
        puzzle = [row.copy() for row in solution]
        positions = [(i, j) for i in range(9) for j in range(9)]
        random.shuffle(positions)
        holes_added = 0

        # 通过回溯算法确保每个挖空操作都保留唯一解
        for i, j in positions:
            if holes_added >= self.holes:
                break
            original = puzzle[i][j]
            puzzle[i][j] = 'X'
            if self.is_unique_solution(puzzle):
                holes_added += 1
            else:
                puzzle[i][j] = original  # 还原无法挖空的位置

        return {
            'puzzle': puzzle,
            'solution': solution
        }

    @staticmethod
    def generate_sudoku_solution():
        """
        生成合法的数独终盘（使用随机排列法保证多样性）
        """
        base = 3
        side = base * base
        
        def pattern(r, c):
            return (base * (r % base) + r // base + c) % side
        
        rng = random.Random()
        rows = [g * base + r for g in rng.sample(range(base), base) 
               for r in rng.sample(range(base), base)]
        cols = [g * base + c for g in rng.sample(range(base), base)
               for c in rng.sample(range(base), base)]
        nums = rng.sample(range(1, side + 1), side)
        
        return [[str(nums[pattern(r, c)]) for c in cols] for r in rows]

    def is_valid(self, grid, row, col, num):
        """
        验证数字填入的合法性（行、列、宫校验）
        """
        # 行检查
        if num in grid[row]:
            return False
        # 列检查
        if any(grid[i][col] == num for i in range(9)):
            return False
        # 宫检查
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if grid[i][j] == num:
                    return False
        return True

    def is_unique_solution(self, puzzle):
        """
        使用回溯算法验证谜题解的唯一性（找到第二个解立即终止）
        """
        grid = [[0 if c == 'X' else int(c) for c in row] for row in puzzle]
        solutions = []
        grid_backup = copy.deepcopy(grid)
        
        def backtrack():
            nonlocal solutions
            for i in range(9):
                for j in range(9):
                    if grid[i][j] == 0:
                        for num in random.sample(range(1,10),9):  # 随机顺序尝试加速发现不同解
                            if self.is_valid(grid, i, j, num):
                                grid[i][j] = num
                                if backtrack():
                                    return True
                                grid[i][j] = 0
                        return False  # 当前位置无解
            solutions.append(copy.deepcopy(grid))
            return len(solutions) < 2  # 找到两个解时终止
        
        backtrack()
        # 还原网格状态
        for i in range(9):
            for j in range(9):
                grid[i][j] = grid_backup[i][j]
        return len(solutions) == 1

    @staticmethod
    def prompt_func(question_case) -> str:
        """
        生成符合LLM输入的标准化问题描述
        """
        puzzle = question_case['puzzle']
        prompt = [
            "You are a KorPuzzleSudoko solving expert. Solve the following KorPuzzleSudoko puzzle "
            "where 'X' represents empty cells. Apply these rules:",
            "1. Each row must contain 1-9 without repetition",
            "2. Each column must contain 1-9 without repetition",
            "3. Each 3x3 subgrid must contain 1-9 without repetition",
            "\nPuzzle Grid:"
        ]
        prompt.append('\n'.join([' '.join(row) for row in puzzle]))
        prompt.append(
            "\nFormat your answer as 9 comma-separated rows with space-separated numbers, "
            "wrapped in double square brackets. Example:[[1 2 3 4 5 6 7 8 9,9 8 7 6 5 4 3 2 1,...]]"
        )
        return '\n'.join(prompt)

    @staticmethod
    def extract_output(output):
        """
        使用正则表达式从LLM输出中提取最后一个合规答案
        """
        matches = re.findall(r'\[\[(.*?)\]\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].replace('\n', ',').strip()
        # 清理多余的空格和分隔符
        cleaned = re.sub(r'\s+', ' ', last_match).replace(' ,', ',').replace(', ', ',')
        return cleaned

    @classmethod
    def _verify_correction(cls, solution, identity):
        """
        验证答案的正确性（同时验证格式和数独规则）
        """
        try:
            # 格式验证
            if not solution or solution.count(',') != 8:
                return False
            user_grid = []
            for row in solution.split(','):
                numbers = list(map(int, row.strip().split()))
                if len(numbers) != 9 or any(n < 1 or n > 9 for n in numbers):
                    return False
                user_grid.append(numbers)
            
            # 规则验证（不依赖参考答案）
            for i in range(9):
                # 行校验
                if len(set(user_grid[i])) != 9:
                    return False
                # 列校验
                if len(set(user_grid[j][i] for j in range(9))) != 9:
                    return False
                # 宫校验
                start_row, start_col = 3 * (i // 3), 3 * (i % 3)
                cells = []
                for x in range(start_row, start_row+3):
                    for y in range(start_col, start_col+3):
                        cells.append(user_grid[x][y])
                if len(set(cells)) != 9:
                    return False
            
            # 已知数字校验（防止生成错误解）
            original_puzzle = identity['puzzle']
            for i in range(9):
                for j in range(9):
                    if original_puzzle[i][j] != 'X' and user_grid[i][j] != int(original_puzzle[i][j]):
                        return False
            return True
        except:
            return False
