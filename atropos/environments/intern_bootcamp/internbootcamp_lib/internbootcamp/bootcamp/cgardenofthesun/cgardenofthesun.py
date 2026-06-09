"""# 

### 谜题描述
There are many sunflowers in the Garden of the Sun.

Garden of the Sun is a rectangular table with n rows and m columns, where the cells of the table are farmlands. All of the cells grow a sunflower on it. Unfortunately, one night, the lightning stroke some (possibly zero) cells, and sunflowers on those cells were burned into ashes. In other words, those cells struck by the lightning became empty. Magically, any two empty cells have no common points (neither edges nor corners).

Now the owner wants to remove some (possibly zero) sunflowers to reach the following two goals: 

  * When you are on an empty cell, you can walk to any other empty cell. In other words, those empty cells are connected. 
  * There is exactly one simple path between any two empty cells. In other words, there is no cycle among the empty cells. 



You can walk from an empty cell to another if they share a common edge.

Could you please give the owner a solution that meets all her requirements?

Note that you are not allowed to plant sunflowers. You don't need to minimize the number of sunflowers you remove. It can be shown that the answer always exists.

Input

The input consists of multiple test cases. The first line contains a single integer t (1≤ t≤ 10^4) — the number of test cases. The description of the test cases follows.

The first line contains two integers n, m (1 ≤ n,m ≤ 500) — the number of rows and columns. 

Each of the next n lines contains m characters. Each character is either 'X' or '.', representing an empty cell and a cell that grows a sunflower, respectively.

It is guaranteed that the sum of n ⋅ m for all test cases does not exceed 250 000.

Output

For each test case, print n lines. Each should contain m characters, representing one row of the table. Each character should be either 'X' or '.', representing an empty cell and a cell with a sunflower, respectively.

If there are multiple answers, you can print any. It can be shown that the answer always exists.

Example

Input


5
3 3
X.X
...
X.X
4 4
....
.X.X
....
.X.X
5 5
.X...
....X
.X...
.....
X.X.X
1 10
....X.X.X.
2 2
..
..


Output


XXX
..X
XXX
XXXX
.X.X
.X..
.XXX
.X...
.XXXX
.X...
.X...
XXXXX
XXXXXXXXXX
..
..

Note

Let's use (x,y) to describe the cell on x-th row and y-th column.

In the following pictures white, yellow, and blue cells stand for the cells that grow a sunflower, the cells lightning stroke, and the cells sunflower on which are removed, respectively.

<image>

In the first test case, one possible solution is to remove sunflowers on (1,2), (2,3) and (3 ,2). 

<image>

Another acceptable solution is to remove sunflowers on (1,2), (2,2) and (3,2). 

<image>

This output is considered wrong because there are 2 simple paths between any pair of cells (there is a cycle). For example, there are 2 simple paths between (1,1) and (3,3).

  1. (1,1)→ (1,2)→ (1,3)→ (2,3)→ (3,3)

  2. (1,1)→ (2,1)→ (3,1)→ (3,2)→ (3,3) 



<image>

This output is considered wrong because you can't walk from (1,1) to (3,3).

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division,print_function
from heapq import*
import sys
le = sys.__stdin__.read().split(\"\n\")[::-1]
sys.setrecursionlimit(5)
af=[]
#vive l'union find
def pe(pt):#compression de chemin mais pas d'équilibrage des tailles donc n log
    print(pt,p[pt])
    if p[pt]==pt:
        return pt
    else:
        p[pt]=pe(p[pt])
        return p[pt]
def v(pt):
    global n,m
    l=[]
    if pt[0]!=0:
        l.append((pt[0]-1,pt[1]))
    if pt[1]!=0:
        l.append((pt[0],pt[1]-1))
    if pt[0]!=n-1:
        l.append((pt[0]+1,pt[1]))
    if pt[1]!=m-1:
        l.append((pt[0],pt[1]+1))
    return l
for zorg in range(int(le.pop())):
    n,m = list(map(int,le.pop().split()))
    gr = [[False]*m for k in range(n)]
    p = {}
    for k in range(n):
        s=le.pop()
        for i in range(m):
            gr[k][i]=(s[i]==\"X\")
            #p[k,i]=(k,i)
    if m==1:
        for k in range(n):
            gr[k][i]=True
    else:
        for k in range(0,n,3):
            for i in range(m):
                gr[k][i]=True
            if k+2<n:
                if gr[k+1][1] or gr[k+2][1]:
                    gr[k+1][1],gr[k+2][1]=True,True
                else:
                    gr[k+1][0],gr[k+2][0]=True,True
        if n%3==0:
            for i in range(m):
                if gr[-1][i]:
                    gr[-2][i]=True
    for k in range(n):
        af.append(\"\".join([\"X\" if gr[k][i] else \".\" for i in range(m)]))   
print(\"\n\".join(map(str,af)))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import deque
import re

class Cgardenofthesunbootcamp(Basebootcamp):
    def __init__(self, n_range=(3, 10), m_range=(3, 10)):
        self.n_range = n_range
        self.m_range = m_range

    def case_generator(self):
        n = random.randint(*self.n_range)
        m = random.randint(*self.m_range)
        
        # 确保至少有一个初始X
        while True:
            solution = self.generate_tree(n, m)
            initial_x = self.create_valid_initial_x(solution)
            if initial_x:
                break
        
        initial_grid = self.create_initial_grid(n, m, initial_x)
        
        return {
            'n': n,
            'm': m,
            'initial_grid': initial_grid,
            'expected_solution': solution
        }

    def generate_tree(self, n, m):
        """使用Prim算法生成生成树结构"""
        grid = [[False for _ in range(m)] for _ in range(n)]
        directions = [(-1,0), (0,1), (1,0), (0,-1)]
        
        # 随机选择起点
        start = (random.randint(0, n-1), random.randint(0, m-1))
        grid[start[0]][start[1]] = True
        frontier = []
        
        # 初始化边界
        for dx, dy in directions:
            nx, ny = start[0]+dx, start[1]+dy
            if 0 <= nx < n and 0 <= ny < m:
                frontier.append((nx, ny))
        
        while frontier:
            # 随机选择边界点
            idx = random.randint(0, len(frontier)-1)
            x, y = frontier.pop(idx)
            
            # 寻找相邻的已选节点
            neighbors = []
            for dx, dy in directions:
                nx, ny = x+dx, y+dy
                if 0 <= nx < n and 0 <= ny < m and grid[nx][ny]:
                    neighbors.append((nx, ny))
            
            if neighbors:
                # 随机选择一个邻居连接
                parent = random.choice(neighbors)
                grid[x][y] = True
                
                # 添加新边界
                for dx, dy in directions:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < n and 0 <= ny < m and not grid[nx][ny]:
                        if (nx, ny) not in frontier:
                            frontier.append((nx, ny))
        
        return grid

    def create_valid_initial_x(self, solution):
        """生成满足条件的初始X集合"""
        n, m = len(solution), len(solution[0])
        candidates = [(i,j) for i in range(n) for j in range(m) if solution[i][j]]
        initial = set()
        banned = set()
        
        # 随机打乱候选顺序
        random.shuffle(candidates)
        
        for x, y in candidates:
            # 检查8邻域是否冲突
            conflict = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if (x+dx, y+dy) in initial:
                        conflict = True
                        break
                if conflict:
                    break
            if not conflict:
                initial.add((x, y))
                # 将周围8格标记为禁止区
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        banned.add((x+dx, y+dy))
        
        return initial

    def create_initial_grid(self, n, m, initial_x):
        grid = [['.' for _ in range(m)] for _ in range(n)]
        for x, y in initial_x:
            grid[x][y] = 'X'
        return [''.join(row) for row in grid]

    @staticmethod
    def prompt_func(case) -> str:
        grid = '\n'.join(case['initial_grid'])
        return f"""你需要解决一个花园迷宫问题。现有网格如下（X表示空单元格，.表示有向日葵）：
{grid}

要求：
1. 最终所有X必须连通（四方向移动）
2. X之间构成无环树结构
3. 必须保留所有原始X
4. 只能通过添加新的X（移除向日葵）来满足条件

将答案放在[answer]标签内，每行表示网格。示例：
[answer]
XXX
.X.
XXX
[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        return [line.strip() for line in last_match.split('\n') if line.strip()]

    @classmethod
    def _verify_correction(cls, solution, case):
        try:
            # 基础格式验证
            if not solution or len(solution) != case['n']:
                return False
            if any(len(line) != case['m'] for line in solution):
                return False
            if any(c not in ('X', '.') for line in solution for c in line):
                return False
            
            # 保留原始X验证
            initial_x = {(i,j) for i, line in enumerate(case['initial_grid']) 
                        for j, c in enumerate(line) if c == 'X'}
            for i, j in initial_x:
                if solution[i][j] != 'X':
                    return False
            
            # 收集所有X坐标
            x_cells = [(i,j) for i, line in enumerate(solution)
                      for j, c in enumerate(line) if c == 'X']
            
            # 空验证
            if not x_cells:
                return False
            
            # 连通性验证
            visited = set()
            queue = deque([x_cells[0]])
            visited.add(x_cells[0])
            directions = [(-1,0), (0,1), (1,0), (0,-1)]
            
            while queue:
                x, y = queue.popleft()
                for dx, dy in directions:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < case['n'] and 0 <= ny < case['m']:
                        if solution[nx][ny] == 'X' and (nx, ny) not in visited:
                            visited.add((nx, ny))
                            queue.append((nx, ny))
            
            if len(visited) != len(x_cells):
                return False
            
            # 树结构验证（边数 = 节点数 - 1）
            edge_count = 0
            for x, y in x_cells:
                for dx, dy in directions:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < case['n'] and 0 <= ny < case['m']:
                        if solution[nx][ny] == 'X':
                            edge_count += 1
            return (edge_count // 2) == (len(x_cells) - 1)
        
        except Exception as e:
            print(f"Verification error: {e}")
            return False
