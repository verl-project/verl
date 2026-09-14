"""# 

### 谜题描述
The Robot is in a rectangular maze of size n × m. Each cell of the maze is either empty or occupied by an obstacle. The Robot can move between neighboring cells on the side left (the symbol \"L\"), right (the symbol \"R\"), up (the symbol \"U\") or down (the symbol \"D\"). The Robot can move to the cell only if it is empty. Initially, the Robot is in the empty cell.

Your task is to find lexicographically minimal Robot's cycle with length exactly k, which begins and ends in the cell where the Robot was initially. It is allowed to the Robot to visit any cell many times (including starting).

Consider that Robot's way is given as a line which consists of symbols \"L\", \"R\", \"U\" and \"D\". For example, if firstly the Robot goes down, then left, then right and up, it means that his way is written as \"DLRU\".

In this task you don't need to minimize the length of the way. Find the minimum lexicographical (in alphabet order as in the dictionary) line which satisfies requirements above.

Input

The first line contains three integers n, m and k (1 ≤ n, m ≤ 1000, 1 ≤ k ≤ 106) — the size of the maze and the length of the cycle. 

Each of the following n lines contains m symbols — the description of the maze. If the symbol equals to \".\" the current cell is empty. If the symbol equals to \"*\" the current cell is occupied by an obstacle. If the symbol equals to \"X\" then initially the Robot is in this cell and it is empty. It is guaranteed that the symbol \"X\" is found in the maze exactly once. 

Output

Print the lexicographically minimum Robot's way with the length exactly k, which starts and ends in the cell where initially Robot is. If there is no such way, print \"IMPOSSIBLE\"(without quotes).

Examples

Input

2 3 2
.**
X..


Output

RL


Input

5 6 14
..***.
*...X.
..*...
..*.**
....*.


Output

DLDDLLLRRRUURU


Input

3 3 4
***
*X*
***


Output

IMPOSSIBLE

Note

In the first sample two cyclic ways for the Robot with the length 2 exist — \"UD\" and \"RL\". The second cycle is lexicographically less. 

In the second sample the Robot should move in the following way: down, left, down, down, left, left, left, right, right, right, up, up, right, up. 

In the third sample the Robot can't move to the neighboring cells, because they are occupied by obstacles.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin
from collections import *

rstr = lambda: stdin.readline().strip()
rstr_2d = lambda n: [rstr() for _ in range(n)]
valid = lambda x, y: -1 < x < n and -1 < y < m and grid[x][y] != '*'
dx, dy = [1, 0, 0, -1], [0, -1, 1, 0]
dir, inv = ('D', 'L', 'R', 'U'), {'U': 'D', 'D': 'U', 'L': 'R', 'R': 'L'}

n, m, k = map(int, stdin.readline().split())
grid, ans, x, y, dist = rstr_2d(n), [], -1, -1, [float('inf')] * (n * m)

if k & 1:
    print('IMPOSSIBLE')
    exit()

for i in range(n):
    for j in range(m):
        if grid[i][j] == 'X':
            x, y = i, j
            break
    if x != -1:
        break

que, dist[x * m + y] = deque([(x, y, 0)]), 0
while que:
    x1, y1, d = que.popleft()
    for i in range(4):
        nx, ny = x1 + dx[i], y1 + dy[i]
        if valid(nx, ny) and dist[nx * m + ny] == float('inf'):
            que.append((nx, ny, d + 1))
            dist[nx * m + ny] = d + 1

for i in range(k):
    for j in range(4):
        nx, ny = x + dx[j], y + dy[j]
        if valid(nx, ny) and dist[nx * m + ny] <= k - i - 1:
            ans.append(dir[j])
            x, y = nx, ny
            break
    else:
        print('IMPOSSIBLE')
        exit()

print(''.join(ans))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from collections import deque
import random
import re
from bootcamp import Basebootcamp

class Ccycleinmazebootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = params.copy()
        # 调整默认参数确保迷宫可解性
        self.params.setdefault('max_n', 5)
        self.params.setdefault('max_m', 5)
        self.params.setdefault('min_k', 2)
        self.params.setdefault('max_k', 20)
        self.params.setdefault('obstacle_prob', 0.3)
        # 确保生成的k是偶数概率更高
        self.params.setdefault('even_k_prob', 0.8)
    
    def case_generator(self):
        # 参数设定优化
        max_n = self.params['max_n']
        max_m = self.params['max_m']
        min_k = self.params['min_k']
        max_k = self.params['max_k']
        obstacle_prob = self.params['obstacle_prob']
        even_k_prob = self.params['even_k_prob']
        
        # 生成k值，控制偶数概率
        if random.random() < even_k_prob:
            k = 2 * random.randint(min_k//2, (max_k+1)//2)
            if k == 0:  # 确保最小k为2
                k = 2
        else:
            k = random.randint(min_k, max_k)
        
        # 生成初始位置和迷宫
        while True:
            n = random.randint(1, max_n)
            m = random.randint(1, max_m)
            x, y = random.randint(0, n-1), random.randint(0, m-1)
            
            # 生成迷宫时确保至少一个可移动方向
            grid = []
            valid_directions = 0
            for i in range(n):
                row = []
                for j in range(m):
                    if i == x and j == y:
                        row.append('X')
                    else:
                        if (abs(i-x) + abs(j-y)) == 1:  # 邻接位置减少障碍概率
                            prob = obstacle_prob / 2
                        else:
                            prob = obstacle_prob
                        cell = '*' if random.random() < prob else '.'
                        row.append(cell)
                        # 统计初始可移动方向
                        if cell == '.' and (abs(i-x) + abs(j-y)) == 1:
                            valid_directions += 1
                grid.append(''.join(row))
            
            # 如果初始位置完全被阻塞且k>0，重新生成
            if valid_directions == 0 and k > 0:
                continue
            else:
                break
        
        # 生成正确答案
        correct_answer = self._generate_solution(n, m, k, grid)
        return {
            'n': n,
            'm': m,
            'k': k,
            'grid': grid,
            'correct_answer': correct_answer
        }
    
    @staticmethod
    def _generate_solution(n, m, k, grid):
        """ 使用BFS生成正确答案 """
        if k % 2 != 0:
            return "IMPOSSIBLE"
        
        # 查找起始点
        start_x, start_y = -1, -1
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 'X':
                    start_x, start_y = i, j
                    break
            if start_x != -1:
                break
        
        dx = [1, 0, 0, -1]  # D, L, R, U
        dy = [0, -1, 1, 0]
        dirs = ['D', 'L', 'R', 'U']
        size = n * m
        dist = [float('inf')] * size
        q = deque([(start_x, start_y, 0)])
        dist[start_x * m + start_y] = 0
        
        # BFS计算最短路径
        while q:
            x, y, d = q.popleft()
            for i in range(4):
                nx, ny = x + dx[i], y + dy[i]
                if 0 <= nx < n and 0 <= ny < m and grid[nx][ny] != '*':
                    pos = nx * m + ny
                    if dist[pos] > d + 1:
                        dist[pos] = d + 1
                        q.append((nx, ny, d + 1))
        
        path = []
        x, y = start_x, start_y
        for step in range(k):
            found = False
            for i in range(4):  # 按字典序选择方向
                nx = x + dx[i]
                ny = y + dy[i]
                if 0 <= nx < n and 0 <= ny < m and grid[nx][ny] != '*':
                    pos = nx * m + ny
                    remaining = k - step - 1
                    if dist[pos] <= remaining:
                        path.append(dirs[i])
                        x, y = nx, ny
                        found = True
                        break
            if not found:
                return "IMPOSSIBLE"
        
        # 最终必须回到起点
        return ''.join(path) if (x, y) == (start_x, start_y) else "IMPOSSIBLE"
    
    @staticmethod
    def prompt_func(question_case):
        grid = "\n".join(question_case['grid'])
        return f"""You control a robot in a {question_case['n']}x{question_case['m']} maze. The robot starts at 'X' and must return after exactly {question_case['k']} moves. Find the lexicographically smallest path using moves D (down), L (left), R (right), U (up). 

Maze Layout:
{grid}

Rules:
1. Moves are ordered by lex priority: D < L < R < U
2. Each move must be to an empty cell (.) or back to start
3. Path length must be exactly {question_case['k']}
4. If impossible, respond with IMPOSSIBLE

Format your answer between [answer] and [/answer]. Example: [answer]DLRU[/answer]"""

    @staticmethod
    def extract_output(output):
        # 提取最后一个答案并标准化格式
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL|re.IGNORECASE)
        if not matches:
            return None
        raw = matches[-1].strip().upper().replace(' ', '')
        # 过滤无效字符
        cleaned = ''.join([c for c in raw if c in {'D','L','R','U'}])
        return cleaned if (cleaned and len(cleaned) == len(raw)) else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 处理特殊case
        if solution == "IMPOSSIBLE":
            return identity['correct_answer'] == "IMPOSSIBLE"
        
        # 验证路径合法性
        k = identity['k']
        if len(solution) != k or k % 2 != 0:
            return False
        
        n, m = identity['n'], identity['m']
        grid = identity['grid']
        dir_map = {'D': (1,0), 'L': (0,-1), 'R': (0,1), 'U': (-1,0)}
        
        # 定位起点
        start_x, start_y = -1, -1
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 'X':
                    start_x, start_y = i, j
                    break
        
        x, y = start_x, start_y
        for move in solution:
            dx, dy = dir_map.get(move, (0,0))
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= n or ny < 0 or ny >= m or grid[nx][ny] == '*':
                return False
            x, y = nx, ny
        
        # 必须回到起点
        return (x, y) == (start_x, start_y)
