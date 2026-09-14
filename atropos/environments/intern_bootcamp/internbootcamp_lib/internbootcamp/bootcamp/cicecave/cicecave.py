"""# 

### 谜题描述
You play a computer game. Your character stands on some level of a multilevel ice cave. In order to move on forward, you need to descend one level lower and the only way to do this is to fall through the ice.

The level of the cave where you are is a rectangular square grid of n rows and m columns. Each cell consists either from intact or from cracked ice. From each cell you can move to cells that are side-adjacent with yours (due to some limitations of the game engine you cannot make jumps on the same place, i.e. jump from a cell to itself). If you move to the cell with cracked ice, then your character falls down through it and if you move to the cell with intact ice, then the ice on this cell becomes cracked.

Let's number the rows with integers from 1 to n from top to bottom and the columns with integers from 1 to m from left to right. Let's denote a cell on the intersection of the r-th row and the c-th column as (r, c). 

You are staying in the cell (r1, c1) and this cell is cracked because you've just fallen here from a higher level. You need to fall down through the cell (r2, c2) since the exit to the next level is there. Can you do this?

Input

The first line contains two integers, n and m (1 ≤ n, m ≤ 500) — the number of rows and columns in the cave description.

Each of the next n lines describes the initial state of the level of the cave, each line consists of m characters \".\" (that is, intact ice) and \"X\" (cracked ice).

The next line contains two integers, r1 and c1 (1 ≤ r1 ≤ n, 1 ≤ c1 ≤ m) — your initial coordinates. It is guaranteed that the description of the cave contains character 'X' in cell (r1, c1), that is, the ice on the starting cell is initially cracked.

The next line contains two integers r2 and c2 (1 ≤ r2 ≤ n, 1 ≤ c2 ≤ m) — the coordinates of the cell through which you need to fall. The final cell may coincide with the starting one.

Output

If you can reach the destination, print 'YES', otherwise print 'NO'.

Examples

Input

4 6
X...XX
...XX.
.X..X.
......
1 6
2 2


Output

YES


Input

5 4
.X..
...X
X.X.
....
.XX.
5 3
1 1


Output

NO


Input

4 7
..X.XX.
.XX..X.
X...X..
X......
2 2
1 6


Output

YES

Note

In the first sample test one possible path is:

<image>

After the first visit of cell (2, 2) the ice on it cracks and when you step there for the second time, your character falls through the ice as intended.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def get_adjacent(x, y, graph):
    n = len(graph)
    m = len(graph[0])
    temp = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    adj = []
    for (x, y) in temp:
        if x >= 0 and x < n and y >= 0 and y < m:
            adj.append((x, y))
    return adj


def dfs(sx, sy, ex, ey, graph):
    n = len(graph)
    m = len(graph[0])
    is_visted = []
    for i in xrange(n):
        is_visted.append([False] * m)
    stack = [(sx, sy)]
    is_visted[sx][sy] = True
    while len(stack) > 0:
        (curx, cury) = stack.pop()
        if curx == ex and cury == ey:
            return True
        adj = get_adjacent(curx, cury, graph)
        for (x, y) in adj:
            if is_visted[x][y] is False and graph[x][y] == '.':
                stack.append((x, y))
                is_visted[x][y] = True
    return False


nums = map(int, raw_input('').split(' '))
[n, m] = [nums[0] - 1, nums[1] - 1]
graph = []
for i in xrange(n + 1):
    graph.append([char for char in raw_input('')])
nums = map(int, raw_input('').split(' '))
[sx, sy] = [nums[0] - 1, nums[1] - 1]
nums = map(int, raw_input('').split(' '))
[ex, ey] = [nums[0] - 1, nums[1] - 1]


is_already_cracked = False
if graph[ex][ey] == 'X':
    is_already_cracked = True


if sx == ex and sy == ey:
    adjacent = get_adjacent(ex, ey, graph)
    adjacent = [point for point in adjacent if graph[point[0]][point[1]] == '.']
    if len(adjacent) > 0:
        print 'YES'
    else:
        print 'NO'
else:
    graph[ex][ey] = '.'
    if is_already_cracked:
        is_possible = dfs(sx, sy, ex, ey, graph)
    else:
        adjacent = get_adjacent(ex, ey, graph)
        is_possible = False
        for (x, y) in adjacent:
            if graph[x][y] == '.':
                graph[x][y] = 'X'
                if dfs(sx, sy, ex, ey, graph) is True:
                    is_possible = True
                    break
                graph[x][y] = '.'
    if is_possible:
        print 'YES'
    else:
        print 'NO'
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
from random import randint, random

class Cicecavebootcamp(Basebootcamp):
    def __init__(self, n=5, m=5, p=0.3):
        """
        初始化冰洞谜题训练场参数。
        :param n: 网格行数，默认为5
        :param m: 网格列数，默认为5
        :param p: 其他单元格为'X'的概率，默认为0.3
        """
        self.n = n
        self.m = m
        self.p = p
    
    def case_generator(self):
        """
        生成冰洞谜题的实例。确保起点为'X'，其他单元格随机生成。
        返回包含网格大小、网格状态、起点和终点的字典。
        """
        # 初始化网格
        grid = [['.' for _ in range(self.m)] for _ in range(self.n)]
        # 随机选择起点，并确保为'X'
        r1 = randint(1, self.n)
        c1 = randint(1, self.m)
        grid[r1-1][c1-1] = 'X'
        
        # 随机填充其他单元格
        for i in range(self.n):
            for j in range(self.m):
                if (i, j) == (r1-1, c1-1):
                    continue
                if random() < self.p:
                    grid[i][j] = 'X'
        
        # 转换为字符串列表便于存储
        grid_str = [''.join(row) for row in grid]
        
        # 随机选择终点
        r2 = randint(1, self.n)
        c2 = randint(1, self.m)
        
        return {
            'n': self.n,
            'm': self.m,
            'grid': grid_str,
            'start': (r1, c1),
            'end': (r2, c2)
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        """将谜题实例转换为文本描述的问题，包含规则和具体实例。"""
        n = question_case['n']
        m = question_case['m']
        grid = '\n'.join(question_case['grid'])
        r1, c1 = question_case['start']
        r2, c2 = question_case['end']
        return f"""你正在玩一个电脑游戏。你的角色站在一个多层冰洞的某一层。要前进，你需要下降到更下一层，唯一的方法是掉落通过冰块。

当前层的洞穴是一个由{n}行{m}列组成的矩形网格。每个单元格由完整冰块（.）或裂缝冰块（X）组成。从每个单元格，你可以移动到相邻的四个方向（上下左右）。移动到裂缝冰块（X）时，你会掉下去。移动到完整冰块（.）时，该冰块会变成裂缝冰块（X）。

你的任务是判断：从起始位置（{r1}, {c1}）出发，能否找到一条路径，使得在移动到目标位置（{r2}, {c2}）的单元格时，该单元格为裂缝冰块（X），从而掉落下去。

初始时，起始位置的冰块已经破裂（X）。其他单元格的状态如下：

{grid}

请判断是否可以达成目标。如果可能，输出“YES”；否则输出“NO”。请将你的答案放置在[answer]标签内，例如：[answer]YES[/answer]或[answer]NO[/answer]。"""
    
    @staticmethod
    def extract_output(output):
        """从模型输出中提取最后一个[answer]标签内的内容。"""
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.IGNORECASE)
        if not matches:
            return None
        last_answer = matches[-1].strip().upper()
        return last_answer if last_answer in ['YES', 'NO'] else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        """验证答案是否正确。"""
        # 解析实例参数
        n = identity['n']
        m = identity['m']
        grid = [list(row) for row in identity['grid']]
        start = identity['start']
        end = identity['end']
        sx = start[0] - 1
        sy = start[1] - 1
        ex = end[0] - 1
        ey = end[1] - 1
        
        # 获取正确解
        def get_adjacent(x, y):
            directions = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            return [(nx, ny) for nx, ny in directions if 0 <= nx < n and 0 <= ny < m]
        
        def dfs(sx, sy, ex, ey, grid):
            visited = [[False]*m for _ in range(n)]
            stack = [(sx, sy)]
            visited[sx][sy] = True
            while stack:
                x, y = stack.pop()
                if x == ex and y == ey:
                    return True
                for nx, ny in get_adjacent(x, y):
                    if not visited[nx][ny] and grid[nx][ny] == '.':
                        visited[nx][ny] = True
                        stack.append((nx, ny))
            return False
        
        # 处理起点和终点相同的情况
        if (sx, sy) == (ex, ey):
            adjacent = get_adjacent(sx, sy)
            if any(grid[x][y] == '.' for x, y in adjacent):
                correct = 'YES'
            else:
                correct = 'NO'
        else:
            original_end = grid[ex][ey]
            is_already_cracked = (original_end == 'X')
            grid[ex][ey] = '.'
            
            if is_already_cracked:
                possible = dfs(sx, sy, ex, ey, grid)
                correct = 'YES' if possible else 'NO'
            else:
                adjacent = get_adjacent(ex, ey)
                possible = False
                for x, y in adjacent:
                    if grid[x][y] == '.':
                        grid[x][y] = 'X'
                        if dfs(sx, sy, ex, ey, grid):
                            possible = True
                            grid[x][y] = '.'
                            break
                        grid[x][y] = '.'
                correct = 'YES' if possible else 'NO'
            grid[ex][ey] = original_end  # 恢复现场
        
        return solution == correct
