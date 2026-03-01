"""# 

### 谜题描述
Vivek has encountered a problem. He has a maze that can be represented as an n × m grid. Each of the grid cells may represent the following:

  * Empty — '.' 
  * Wall — '#' 
  * Good person — 'G' 
  * Bad person — 'B' 



The only escape from the maze is at cell (n, m).

A person can move to a cell only if it shares a side with their current cell and does not contain a wall. Vivek wants to block some of the empty cells by replacing them with walls in such a way, that all the good people are able to escape, while none of the bad people are able to. A cell that initially contains 'G' or 'B' cannot be blocked and can be travelled through.

Help him determine if there exists a way to replace some (zero or more) empty cells with walls to satisfy the above conditions.

It is guaranteed that the cell (n,m) is empty. Vivek can also block this cell.

Input

The first line contains one integer t (1 ≤ t ≤ 100) — the number of test cases. The description of the test cases follows.

The first line of each test case contains two integers n, m (1 ≤ n, m ≤ 50) — the number of rows and columns in the maze.

Each of the next n lines contain m characters. They describe the layout of the maze. If a character on a line equals '.', the corresponding cell is empty. If it equals '#', the cell has a wall. 'G' corresponds to a good person and 'B' corresponds to a bad person.

Output

For each test case, print \"Yes\" if there exists a way to replace some empty cells with walls to satisfy the given conditions. Otherwise print \"No\"

You may print every letter in any case (upper or lower).

Example

Input


6
1 1
.
1 2
G.
2 2
#B
G.
2 3
G.#
B#.
3 3
#B.
#..
GG.
2 2
#B
B.


Output


Yes
Yes
No
No
Yes
Yes

Note

For the first and second test cases, all conditions are already satisfied.

For the third test case, there is only one empty cell (2,2), and if it is replaced with a wall then the good person at (1,2) will not be able to escape.

For the fourth test case, the good person at (1,1) cannot escape.

For the fifth test case, Vivek can block the cells (2,3) and (2,2).

For the last test case, Vivek can block the destination cell (2, 2).

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def getAdj(x, y, N, M) :
  moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
  adj = []
  for dx, dy in moves :
    if x+dx < 0 or x+dx >= M : continue
    if y+dy < 0 or y+dy >= N : continue
    adj.append((x+dx,y+dy))
  return adj

T = input()
for t in range(T) :
  N, M = map(int,raw_input().split())

  good, bad = set(), []
  grid = []
  for n in range(N) :
    row = list(raw_input())
    grid.append(row)
    for m in range(M) :
      if row[m] == 'G' : good.add((n,m))
      if row[m] == 'B' : bad.append((n,m))
  
  valid = True
  for n, m in bad :
    if N-n + M-m <= 3 : valid = False
    adj = getAdj(m,n,N,M)

    for x, y in adj :
      if grid[y][x] == '#' : continue
      elif grid[y][x] == '.' : grid[y][x] = '#'

    if not valid : break
  
  if not valid and len(good) == 0 : print \"Yes\"
  elif not valid and len(good) > 0 : print \"No\"
  else :

    marked = [[False]*M for n in range(N)]
    q = [(N-1,M-1)]
    marked[N-1][M-1] = True

    while q :
      curY, curX = q.pop()
      if not valid: break

      for x, y in getAdj(curX, curY, N, M) :
        if grid[y][x] == '#' : continue
        if grid[y][x] == 'B' : valid = False
        if marked[y][x] : continue
        marked[y][x] = True
        good.discard((y,x))
        q.append((y,x))

    if len(good) > 0 : valid = False
    print \"Yes\" if valid else \"No\"
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import deque
from bootcamp import Basebootcamp

def get_adj(x, y, n_rows, m_cols):
    return [(nx, ny) for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)] 
            if 0<=(nx:=x+dx)<m_cols and 0<=(ny:=y+dy)<n_rows]

def solve_maze(n, m, original_grid):
    grid = [row.copy() for row in original_grid]
    good = set()
    bad = []
    
    # 收集所有好人坏人位置
    for y in range(n):
        for x in range(m):
            if grid[y][x] == 'G':
                good.add((y,x))
            elif grid[y][x] == 'B':
                bad.append((y,x))
    
    # 处理坏人周围的墙
    valid = True
    for y, x in bad:
        # 检查坏人是否离出口太近（曼哈顿距离）
        if (n-1 - y) + (m-1 - x) <= 1:
            valid = False
        
        # 将坏人周围的空地变为墙
        for ax, ay in get_adj(x, y, n, m):
            if grid[ay][ax] == '.':
                grid[ay][ax] = '#'
        
        if not valid: break
    
    # 提前终止条件
    if not valid:
        return "Yes" if len(good) == 0 else "No"
    
    # 出口被墙阻挡的情况
    if grid[n-1][m-1] == '#':
        return "Yes" if len(good) == 0 else "No"
    
    # BFS检查可达性
    marked = [[False]*m for _ in range(n)]
    queue = deque([(n-1, m-1)])
    marked[n-1][m-1] = True
    valid = True
    
    while queue:
        y, x = queue.popleft()
        
        # 遇到坏人直接失败
        if grid[y][x] == 'B':
            valid = False
            break
        
        # 处理相邻单元格
        for ax, ay in get_adj(x, y, n, m):
            if not marked[ay][ax] and grid[ay][ax] != '#':
                marked[ay][ax] = True
                queue.append((ay, ax))
                if (ay, ax) in good:
                    good.remove((ay, ax))
    
    return "Yes" if valid and not good else "No"

class Dsolvethemazebootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=50, min_m=1, max_m=50):
        self.min_n = min_n
        self.max_n = max_n
        self.min_m = min_m
        self.max_m = max_m
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        m = random.randint(self.min_m, self.max_m)
        grid = []
        for y in range(n):
            row = []
            for x in range(m):
                # 确保出口单元格（n-1,m-1）初始为空
                if y == n-1 and x == m-1:
                    row.append('.')
                else:
                    row.append(random.choice(['.', '#', 'G', 'B']))
            grid.append(row)
        return {'n':n, 'm':m, 'grid':grid}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        grid_str = '\n'.join(''.join(row) for row in question_case['grid'])
        return f"""Given a {question_case['n']}x{question_case['m']} maze grid where:
- '.' = empty cell
- '#' = wall
- 'G' = good person
- 'B' = bad person

Exit is at bottom-right cell ({question_case['n']}, {question_case['m']}). Can we block cells (turn '.' to '#') such that:
1. All G can reach exit
2. All B cannot reach exit

Maze layout:
{grid_str}

Answer with [answer]Yes[/answer] or [answer]No[/answer]."""

    @staticmethod 
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](yes|no)\[/answer\]', output, re.I)
        return matches[-1].title() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == solve_maze(identity['n'], identity['m'], identity['grid'])
