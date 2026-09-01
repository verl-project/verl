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
1.The game is played on an n*m grid with trees at some locations in the grid.
2.To place tents on the grid.
3.Each tent must be orthogonally adjacent to a tree (i.e., above, below, to the left, or to the right of the tree).
4.Tents cannot be orthogonally or diagonally adjacent to other tents.
5.Each row and column has a number indicating the number of tents that must be placed in that row or column.
6.Each puzzle has and has only one unique solution.
7.The puzzle is given by a matrix in the form of T, which represents the position of the tree, and X, which represents the spaces,To the right and below the matrix are numerical constraints, and you need to replace X with C (for tents) for some spaces, and the answer is a matrix.Example questions are as follows:

<example 0>
T	X	X	X	X	X	T	X	T	X	3
X	X	X	X	X	X	X	X	X	X	2
X	X	X	T	X	X	T	X	X	T	1
X	X	T	X	T	X	X	X	X	X	3
X	X	X	T	X	X	X	T	X	X	1
0	2	0	2	0	2	0	2	1	1

The output should be given in order from left to right, top to bottom, with each element separated by a space and different lines separated by a comma.
Ensure that your final answer is wrapped in double square brackets like this: [[T C X X,X X X C,X X X T,X C T C]].
</example 0>

<example 1>
X	X	T	X	T	X	T	X	X	X	3
X	X	X	X	X	X	X	X	T	X	2
X	T	X	X	X	X	X	X	X	X	1
X	X	X	T	X	T	X	X	X	X	2
T	T	X	X	X	X	X	X	T	X	2
1	1	1	2	0	1	1	1	0	2

The output should be given in order from left to right, top to bottom, with each element separated by a space and different lines separated by a comma.
Ensure that your final answer is wrapped in double square brackets like this: [[T C X X,X X X C,X X X T,X C T C]].
</example 1>

<example 2>
X	X	T	X	X	X	X	X	X	X	2
X	T	X	X	X	X	X	X	T	X	1
X	X	X	X	X	T	X	T	X	X	2
T	X	X	X	X	X	X	X	X	X	1
X	X	T	T	X	X	X	T	X	T	4
2	0	1	1	1	0	2	0	3	0

The output should be given in order from left to right, top to bottom, with each element separated by a space and different lines separated by a comma.
Ensure that your final answer is wrapped in double square brackets like this: [[T C X X,X X X C,X X X T,X C T C]].
</example 2>

<example 3>
T	X	X	X	X	X	X	X	X	X	1
X	X	X	X	T	X	T	X	T	X	3
X	X	X	X	X	X	X	X	X	T	1
X	T	X	X	X	T	T	X	X	X	1
X	X	T	X	X	X	X	X	T	X	4
1	1	0	2	0	1	2	1	0	2
The output should be given in order from left to right, top to bottom, with each element separated by a space and different lines separated by a comma.
Ensure that your final answer is wrapped in double square brackets like this: [[T C X X,X X X C,X X X T,X C T C]].
</example 3>

<example 4>
T	X	X	X	X	X	X	T	X	X	2
X	X	X	X	X	X	X	X	X	X	2
X	X	X	T	T	T	T	X	X	T	2
X	X	X	T	X	X	X	X	X	X	2
T	X	X	X	X	X	X	X	X	T	2
1	1	1	1	1	1	1	1	1	1
The output should be given in order from left to right, top to bottom, with each element separated by a space and different lines separated by a comma.
Ensure that your final answer is wrapped in double square brackets like this: [[T C X X,X X X C,X X X T,X C T C]].
</example 4>

<example 5>
X	X	X	X	X	X	X	X	T	X	3
X	T	T	T	X	X	T	X	X	X	1
X	T	X	X	X	X	X	X	X	X	2
X	X	X	X	X	X	T	X	T	X	3
X	X	X	X	T	T	X	X	X	X	1
1	1	1	1	1	1	1	1	0	2
The output should be given in order from left to right, top to bottom, with each element separated by a space and different lines separated by a comma.
Ensure that your final answer is wrapped in double square brackets like this: [[T C X X,X X X C,X X X T,X C T C]].
</example 5>

<example 6>
X	X	X	X	X	X	X	X	T	X	2
T	X	X	X	X	T	X	X	X	X	1
X	T	X	X	X	X	X	T	X	X	3
X	X	T	X	X	T	X	X	X	X	1
X	T	T	X	X	X	X	X	X	X	3
X	X	X	X	T	X	X	X	X	T	1
X	T	X	X	T	X	X	T	X	X	3
X	X	X	T	T	X	X	X	X	X	1
X	X	X	X	X	T	X	X	X	X	4
T	T	X	X	X	X	X	X	T	X	1
5	0	4	0	3	1	2	0	3	2
The output should be given in order from left to right, top to bottom, with each element separated by a space and different lines separated by a comma.
Ensure that your final answer is wrapped in double square brackets like this: [[T C X X,X X X C,X X X T,X C T C]].
</example 6>

<example 7>
X	X	X	X	T	X	T	X	X	X	4
X	X	X	X	T	T	X	T	X	T	1
X	T	X	X	X	X	X	X	X	T	2
X	X	X	X	X	X	X	X	X	X	1
X	X	T	X	X	X	X	X	X	X	3
X	T	X	X	X	X	X	T	X	X	1
X	T	X	X	T	X	T	X	X	T	2
X	X	X	X	X	X	X	X	X	T	2
X	X	X	X	T	X	X	T	X	X	0
X	T	X	X	X	X	X	X	T	X	4
1	2	1	3	2	1	1	4	1	4
The output should be given in order from left to right, top to bottom, with each element separated by a space and different lines separated by a comma.
Ensure that your final answer is wrapped in double square brackets like this: [[T C X X,X X X C,X X X T,X C T C]].
</example 7>

<example 8>
X	T	X	T	X	X	X	X	T	X	3
X	X	X	X	X	X	X	T	X	X	1
X	X	X	T	T	X	X	X	T	X	2
X	X	X	X	X	X	X	X	X	X	2
T	T	X	X	X	T	X	X	X	T	3
T	X	X	X	X	T	X	X	X	X	1
X	X	X	T	X	T	X	X	X	X	3
X	X	X	X	X	X	X	T	X	X	1
X	T	X	X	X	X	X	T	X	T	1
X	X	X	X	X	T	X	X	X	X	3
3	1	4	0	2	1	3	1	2	3
The output should be given in order from left to right, top to bottom, with each element separated by a space and different lines separated by a comma.
Ensure that your final answer is wrapped in double square brackets like this: [[T C X X,X X X C,X X X T,X C T C]].
</example 8>

<example 9>
X	X	X	X	X	X	X	T	X	X	3
X	T	X	X	X	T	X	X	X	T	1
X	T	X	X	X	X	X	X	X	X	2
X	X	T	X	X	X	X	X	X	T	2
X	X	X	X	X	X	T	T	T	X	1
X	X	X	T	X	X	X	X	X	T	1
X	X	T	X	X	X	T	X	X	X	4
X	X	X	X	X	X	T	X	X	X	1
X	X	X	X	X	X	X	T	X	X	1
T	X	X	X	T	X	T	X	T	X	4
2	2	0	3	1	2	2	3	0	5
The output should be given in order from left to right, top to bottom, with each element separated by a space and different lines separated by a comma.
Ensure that your final answer is wrapped in double square brackets like this: [[T C X X,X X X C,X X X T,X C T C]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from itertools import product

class KorPuzzleCampsitebootcamp(Basebootcamp):
    def __init__(self, rows=5, cols=5):
        self.rows = rows
        self.cols = cols

    def case_generator(self):
        """生成具有唯一解的合法谜题案例"""
        n, m = self.rows, self.cols
        
        # 生成树和帐篷的合法布局
        while True:
            try:
                # 步骤1：生成随机树布局
                grid = [['X' for _ in range(m)] for _ in range(n)]
                for i in range(n):
                    j = random.choice(range(m))
                    grid[i][j] = 'T'
                
                # 步骤2：生成合法帐篷布局
                solution = [row.copy() for row in grid]
                tents = []
                # 为每个树尝试放置帐篷
                for i in range(n):
                    for j in range(m):
                        if grid[i][j] == 'T':
                            # 寻找可用相邻位置
                            adj_positions = []
                            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                                ni, nj = i+di, j+dj
                                if 0<=ni<n and 0<=nj<m and solution[ni][nj] == 'X':
                                    adj_positions.append((ni, nj))
                            if adj_positions:
                                # 随机选择一个位置放置帐篷
                                ni, nj = random.choice(adj_positions)
                                solution[ni][nj] = 'C'
                                tents.append((ni, nj))
                                # 标记周围禁止帐篷
                                for dx in [-1,0,1]:
                                    for dy in [-1,0,1]:
                                        x, y = ni+dx, nj+dy
                                        if 0<=x<n and 0<=y<m and solution[x][y] == 'X':
                                            solution[x][y] = '.'
                # 清理占位符
                for i in range(n):
                    for j in range(m):
                        if solution[i][j] == '.':
                            solution[i][j] = 'X'
                
                # 计算约束条件
                row_clues = [row.count('C') for row in solution]
                col_clues = [sum(solution[i][j] == 'C' for i in range(n)) for j in range(m)]
                
                # 确保至少有一个帐篷且约束有效
                if sum(row_clues) == 0:
                    continue
                
                # 生成谜题网格（隐藏帐篷）
                puzzle_grid = [row.copy() for row in grid]
                return {
                    'grid': puzzle_grid,
                    'row_clues': row_clues,
                    'col_clues': col_clues,
                    'solution': solution
                }
            except:
                continue

    @staticmethod
    def prompt_func(question_case) -> str:
        grid = question_case['grid']
        row_clues = question_case['row_clues']
        col_clues = question_case['col_clues']
        n = len(grid)
        m = len(col_clues)
        
        # 构建网格表示
        grid_lines = []
        for i in range(n):
            row = '\t'.join(grid[i]) + '\t' + str(row_clues[i])
            grid_lines.append(row)
        
        # 添加列约束行
        col_line = '\t'.join(map(str, col_clues))
        grid_lines.append(col_line)
        
        # 构建完整提示
        prompt = f"""根据以下规则在{n}x{m}网格中放置帐篷：
1. 每个帐篷(C)必须与至少一棵树(T)正交相邻（上下左右）
2. 帐篷之间不能相邻（包括对角）
3. 行末数字表示该行需要的帐篷数
4. 最后一行数字表示各列需要的帐篷数

网格布局（T=树，X=空地）：
"""
        prompt += '\n'.join(grid_lines)
        prompt += "\n\n用C替换需要放置帐篷的X，保持T不变。答案格式：[[行元素 用 空格 分隔，行间用 逗号 分隔]]"
        prompt += "\n示例：[[T C X, X X T]]"
        return prompt

    @staticmethod
    def extract_output(output):
        # 匹配最后一个[[...]]结构
        matches = re.findall(r'\[\[(.*?)\]\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        
        # 清理并分割行列
        rows = [row.strip() for row in last_match.split(',') if row.strip()]
        solution = []
        for row in rows:
            elements = re.split(r'\s+', row.strip())
            cleaned = [e.upper().replace('，', ',') for e in elements if e]
            solution.append(cleaned)
        return solution

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            grid = identity['grid']
            row_clues = identity['row_clues']
            col_clues = identity['col_clues']
            n, m = len(grid), len(grid[0])
            
            # 1. 检查解的结构有效性
            if len(solution) != n:
                return False
            if any(len(row) != m for row in solution):
                return False
            
            # 2. 验证原树位置保留
            for i in range(n):
                for j in range(m):
                    if grid[i][j] == 'T' and solution[i][j] != 'T':
                        return False
            
            # 3. 验证行约束
            for i in range(n):
                if solution[i].count('C') != row_clues[i]:
                    return False
            
            # 4. 验证列约束
            for j in range(m):
                col_count = sum(solution[i][j] == 'C' for i in range(n))
                if col_count != col_clues[j]:
                    return False
            
            # 5. 验证帐篷放置规则
            for i in range(n):
                for j in range(m):
                    if solution[i][j] != 'C':
                        continue
                    
                    # 5.1 检查与树相邻
                    has_tree = False
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        x, y = i+dx, j+dy
                        if 0 <= x < n and 0 <= y < m:
                            if solution[x][y] == 'T':
                                has_tree = True
                                break
                    if not has_tree:
                        return False
                    
                    # 5.2 检查帐篷相邻
                    for dx in [-1,0,1]:
                        for dy in [-1,0,1]:
                            if dx == 0 and dy == 0:
                                continue
                            x, y = i+dx, j+dy
                            if 0 <= x < n and 0 <= y < m:
                                if solution[x][y] == 'C':
                                    return False
            return True
        except Exception as e:
            print(f"验证异常：{str(e)}")
            return False
