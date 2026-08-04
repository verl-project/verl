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
1.The game is played on an n*n grid, under each of which a mine may be hidden or empty.
2.Some squares show a number indicating the number of mines around them (8 squares including the diagonal).
3.Our puzzle uses X to fill the squares without numbers.
4.The player needs to find all the squares where mines are located and replace X with A.Example questions are as follows:

<example 0>
X	2	X	3	X
X	X	3	X	X
1	2	3	3	2
X	X	X	X	2
1	X	2	X	X
Ensure that your final answer is wrapped in double square brackets, like this: [[X X X,A 2 2,2 3 A]].
</example 0>

<example 1>
X   X	X	1	X
X	2	2	X	X
2	3	X	X	2
1	X	2	X	X
X	X	X	1	X
Ensure that your final answer is wrapped in double square brackets, like this: [[X X X,A 2 2,2 3 A]].
</example 1>

<example 2>
1	X	X	X	X
X	2	X	X	2
2	2	X	4	X
X	X	1	X	X
X	1	X	X	2
Ensure that your final answer is wrapped in double square brackets, like this: [[X X X,A 2 2,2 3 A]].
</example 2>

<example 3>
1	X	X	X	1
X	2	2	X	X
X	X	X	X	2
3	X	3	2	X
X	2	1	X	X
Ensure that your final answer is wrapped in double square brackets, like this: [[X X X,A 2 2,2 3 A]].
</example 3>

<example 4>
2	3	2	2	1	X	1
X	X	X	3	X	3	X
2	X	X	X	X	3	1
X	X	X	3	X	2	X
X	2	3	X	3	X	1
2	X	X	X	3	X	X
X	X	2	X	X	1	X
Ensure that your final answer is wrapped in double square brackets, like this: [[X X X,A 2 2,2 3 A]].
</example 4>

<example 5>
X	X	3	X	3	X	1
1	X	X	X	X	X	X
2	X	X	4	5	3	X
X	1	1	X	4	X	X
X	X	2	2	X	X	3
X	X	2	X	X	X	X
1	X	X	X	2	1	X
Ensure that your final answer is wrapped in double square brackets, like this: [[X X X,A 2 2,2 3 A]].
</example 5>

<example 6>
1	X	2	X	X	1	0
X	X	X	X	X	3	X
2	X	X	1	X	2	X
1	X	X	X	X	X	1
X	X	3	1	X	1	X
1	X	X	X	X	X	X
X	2	2	X	1	X	1
Ensure that your final answer is wrapped in double square brackets, like this: [[X X X,A 2 2,2 3 A]].
</example 6>

<example 7>
X	2	X	2	1	X	X
X	X	X	X	X	X	2
1	X	3	X	X	X	X
X	1	X	X	2	X	2
1	X	X	2	3	2	2
1	2	2	X	X	X	X
X	X	X	X	X	1	X
Ensure that your final answer is wrapped in double square brackets, like this: [[X X X,A 2 2,2 3 A]].
</example 7>

<example 8>
X	1	X	X	1
1	X	X	X	X
2	2	X	X	1
1	X	X	X	1
X	2	X	1	X
Ensure that your final answer is wrapped in double square brackets, like this: [[X X X,A 2 2,2 3 A]].
</example 8>

<example 9>
X	2	X	X	X
X	X	X	3	2
3	X	3	X	X
2	X	X	4	X
X	1	X	2	X
Ensure that your final answer is wrapped in double square brackets, like this: [[X X X,A 2 2,2 3 A]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from itertools import product
from bootcamp import Basebootcamp

class KorPuzzleMinesweeperbootcamp(Basebootcamp):
    def __init__(self, size=5, mine_ratio=0.15, min_hints=3):
        super().__init__()
        self.size = size
        self.mine_count = max(1, min(int(size*size*mine_ratio), size*size-2))
        self.min_hints = min_hints
        self.params = {
            'size': size,
            'mine_count': self.mine_count,
            'min_hints': min_hints
        }

    def case_generator(self):
        size = self.size
        while True:  # 确保生成有效谜题
            # 随机生成地雷布局
            mines = set()
            while len(mines) < self.mine_count:
                mines.add((random.randint(0, size-1), random.randint(0, size-1)))
            
            # 生成完整提示网格
            full_grid = []
            for r in range(size):
                row = []
                for c in range(size):
                    if (r,c) in mines:
                        row.append('X')
                    else:
                        count = sum(1 for dr,dc in product((-1,0,1), repeat=2)
                                   if (r+dr, c+dc) in mines and 0<=r+dr<size and 0<=c+dc<size)
                        row.append(str(count) if count>0 else '0')
                full_grid.append(row)
            
            # 选择需要保留的提示格（至少保留min_hints个非零提示）
            safe_cells = [(r,c) for r,c in product(range(size), repeat=2) 
                        if (r,c) not in mines and full_grid[r][c] != '0']
            if len(safe_cells) < self.min_hints:
                continue  # 重生成
            
            keep_hints = random.sample(safe_cells, k=random.randint(self.min_hints, len(safe_cells)))
            
            # 生成谜题网格
            puzzle_grid = []
            for r in range(size):
                puzzle_row = []
                for c in range(size):
                    if (r,c) in keep_hints or (r,c) in mines:
                        puzzle_row.append(full_grid[r][c])
                    else:
                        puzzle_row.append('X')
                puzzle_grid.append(puzzle_row)
            
            # 有效性检查：至少有一个可推断的确定位置
            if self.validate_puzzle(puzzle_grid, mines):
                return {
                    'grid': puzzle_grid,
                    'mines': list(mines),
                    'full_grid': full_grid
                }

    @staticmethod
    def validate_puzzle(puzzle_grid, mines):
        """至少存在一个可直接推断的确定位置"""
        size = len(puzzle_grid)
        for r in range(size):
            for c in range(size):
                if puzzle_grid[r][c] == 'X':
                    continue
                if puzzle_grid[r][c] == '0':
                    return True  # 0周围必无雷
                required = int(puzzle_grid[r][c])
                hidden = 0
                confirmed_mines = 0
                for dr, dc in product((-1,0,1), repeat=2):
                    nr, nc = r+dr, c+dc
                    if 0<=nr<size and 0<=nc<size:
                        if (nr, nc) in mines:
                            confirmed_mines +=1
                        elif puzzle_grid[nr][nc] == 'X':
                            hidden +=1
                if required == confirmed_mines and hidden >0:
                    return True  # 存在可标记的安全区域
        return False

    @staticmethod
    def prompt_func(question_case) -> str:
        grid = question_case['grid']
        grid_str = "\n".join(["\t".join(row) for row in grid])
        return f"""扫雷谜题规则：
1. 数字表示周围8格中地雷的总数
2. X需要判断是否为地雷：是地雷改为A，不是则保持X
3. 现有数字和X的位置不可修改，只能改动X→A

当前谜题（{len(grid)}x{len(grid)}）：
{grid_str}

最终答案格式示例（每行用空格分隔，行间用逗号分隔）：
[[A X 2,X A 3,2 3 A]]"""

    @staticmethod
    def extract_output(output):
        import re
        pattern = r'\[\[((?:[^[\]]|\[.*?\])*?)\]\]'  # 处理嵌套括号
        matches = re.findall(pattern, output, re.DOTALL)
        if not matches:
            return None
        
        last_match = matches[-1].replace('\n', ' ').replace('\t', ' ')
        rows = [r.strip() for r in last_match.split(',')]
        solution = []
        for row in rows:
            cells = list(filter(None, re.split(r'[\s,]+', row)))
            if cells:
                solution.append(cells)
        return solution

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            puzzle_grid = identity['grid']
            full_grid = identity['full_grid']
            mines = set(map(tuple, identity['mines']))
            
            # 结构校验
            if len(solution) != len(puzzle_grid):
                return False
            for r in range(len(puzzle_grid)):
                if len(solution[r]) != len(puzzle_grid[r]):
                    return False
            
            # 内容校验
            for r in range(len(puzzle_grid)):
                for c in range(len(puzzle_grid[r])):
                    puzzle_val = puzzle_grid[r][c]
                    ans_val = solution[r][c]
                    
                    if puzzle_val != 'X':  # 原始提示格必须保持一致
                        if ans_val != puzzle_val:
                            return False
                    else:  # X格需要验证地雷状态
                        is_mine = (r,c) in mines
                        if is_mine and ans_val != 'A':
                            return False
                        if not is_mine and ans_val != 'X':
                            return False
            return True
        except Exception as e:
            print(f"Verification error: {e}")
            return False
