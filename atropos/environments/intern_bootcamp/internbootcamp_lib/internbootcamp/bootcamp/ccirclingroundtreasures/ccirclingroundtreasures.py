"""# 

### 谜题描述
You have a map as a rectangle table. Each cell of the table is either an obstacle, or a treasure with a certain price, or a bomb, or an empty cell. Your initial position is also given to you.

You can go from one cell of the map to a side-adjacent one. At that, you are not allowed to go beyond the borders of the map, enter the cells with treasures, obstacles and bombs. To pick the treasures, you need to build a closed path (starting and ending in the starting cell). The closed path mustn't contain any cells with bombs inside. Let's assume that the sum of the treasures' values that are located inside the closed path equals v, and besides, you've made k single moves (from one cell to another) while you were going through the path, then such path brings you the profit of v - k rubles.

Your task is to build a closed path that doesn't contain any bombs and brings maximum profit.

Note that the path can have self-intersections. In order to determine if a cell lies inside a path or not, use the following algorithm:

  1. Assume that the table cells are points on the plane (the table cell on the intersection of the i-th column and the j-th row is point (i, j)). And the given path is a closed polyline that goes through these points. 
  2. You need to find out if the point p of the table that is not crossed by the polyline lies inside the polyline. 
  3. Let's draw a ray that starts from point p and does not intersect other points of the table (such ray must exist). 
  4. Let's count the number of segments of the polyline that intersect the painted ray. If this number is odd, we assume that point p (and consequently, the table cell) lie inside the polyline (path). Otherwise, we assume that it lies outside. 

Input

The first line contains two integers n and m (1 ≤ n, m ≤ 20) — the sizes of the table. Next n lines each contains m characters — the description of the table. The description means the following:

  * character \"B\" is a cell with a bomb; 
  * character \"S\" is the starting cell, you can assume that it's empty; 
  * digit c (1-8) is treasure with index c; 
  * character \".\" is an empty cell; 
  * character \"#\" is an obstacle. 



Assume that the map has t treasures. Next t lines contain the prices of the treasures. The i-th line contains the price of the treasure with index i, vi ( - 200 ≤ vi ≤ 200). It is guaranteed that the treasures are numbered from 1 to t. It is guaranteed that the map has not more than 8 objects in total. Objects are bombs and treasures. It is guaranteed that the map has exactly one character \"S\".

Output

Print a single integer — the maximum possible profit you can get.

Examples

Input

4 4
....
.S1.
....
....
10


Output

2


Input

7 7
.......
.1###2.
.#...#.
.#.B.#.
.3...4.
..##...
......S
100
100
100
100


Output

364


Input

7 8
........
........
....1B..
.S......
....2...
3.......
........
100
-100
100


Output

0


Input

1 1
S


Output

0

Note

In the first example the answer will look as follows.

<image>

In the second example the answer will look as follows.

<image>

In the third example you cannot get profit.

In the fourth example you cannot get profit as you cannot construct a closed path with more than one cell.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from collections import deque
def main():
    n, m = map(int, raw_input().split())
    b = [list(raw_input().strip()) for _ in xrange(n)]
    def inboard(y, x):
        return 0 <= y < n and 0 <= x < m
    c = 0
    for i in xrange(n):
        for j in xrange(m):
            if b[i][j] in '123456789':
                c += 1
                b[i][j] = chr(ord(b[i][j]) - 1)
            elif b[i][j] == 'S':
                sy, sx = i, j
    v = [int(raw_input()) for _ in xrange(c)]
    inf = 1000000
    up = [[0] * m for i in xrange(n)]
    down = [[0] * m for i in xrange(n)]
    for i in xrange(n):
        for j in xrange(m):
            if b[i][j] in '0123456789':
                for l in xrange(j+1, m):
                    up[i][l] |= 1 << int(b[i][j])
                    down[i][l] |= 1 << int(b[i][j])
            elif b[i][j] == 'B':
                for l in xrange(j+1, m):
                    up[i][l] |= 1 << c
                    down[i][l] |= 1 << c
                v.append(-inf)
                c += 1
    dp = [[[inf] * (1 << c) for j in xrange(m)] for i in xrange(n)]
    dp[sy][sx][0] = 0
    d = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    q = deque()
    q.append((sy, sx, 0, 0))
    while q:
        y, x, mask, t = q.popleft()
        if t > dp[y][x][mask]: continue
        nt = t + 1
        for dy, dx in d:
            ny, nx, nmask = dy + y, dx + x, mask
            if not inboard(ny, nx) or b[ny][nx] not in '.S': continue
            if dx == 0:
                if dy == 1:
                    nmask ^= down[ny][nx]
                else:
                    nmask ^= down[y][x]
            if dp[ny][nx][nmask] > nt:
                dp[ny][nx][nmask] = nt
                q.append((ny, nx, nmask, nt))
    print max(sum(v[j] for j in xrange(c) if ((i >> j) & 1))-dp[sy][sx][i] for i in xrange(1 << c))
main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import deque
from io import StringIO
import sys

class Ccirclingroundtreasuresbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = {
            'min_n': 1,
            'max_n': 8,
            'min_m': 1,
            'max_m': 8,
            'max_objects': 8,
            'obstacle_prob': 0.2,
        }
        self.params.update(params)
        self.params['max_objects'] = min(self.params['max_objects'], 8)

    def case_generator(self):
        n = random.randint(self.params['min_n'], self.params['max_n'])
        m = random.randint(self.params['min_m'], self.params['max_m'])
        
        # 生成唯一S坐标
        sy, sx = random.randint(0,n-1), random.randint(0,m-1)
        grid = [['.' for _ in range(m)] for _ in range(n)]
        grid[sy][sx] = 'S'

        # 动态计算最大可放置对象
        available_cells = n*m - 1
        max_objs = min(self.params['max_objects'], available_cells)
        t_objects = random.randint(0, max_objs)
        t_treasure = random.randint(0, t_objects)
        t_bomb = t_objects - t_treasure

        # 生成候选位置
        available_pos = [(i,j) for i in range(n) for j in range(m) if (i,j)!=(sy,sx)]
        random.shuffle(available_pos)
        selected_pos = available_pos[:t_objects]

        # 放置对象
        treasure_ids = list(range(1, t_treasure+1)) if t_treasure else []
        for idx, (i,j) in enumerate(selected_pos):
            grid[i][j] = str(treasure_ids[idx]) if idx < t_treasure else 'B'

        # 添加障碍物
        for i in range(n):
            for j in range(m):
                if grid[i][j] == '.' and random.random() < self.params['obstacle_prob']:
                    grid[i][j] = '#'

        # 生成价值
        prices = [random.randint(-200,200) for _ in range(t_treasure)]

        # 转换为输入
        grid_str = [''.join(row) for row in grid]
        input_lines = [f"{n} {m}"] + grid_str
        if t_treasure:
            input_lines.extend(map(str, prices))
        
        # 求解正确答案
        try:
            correct_answer = self.solve_input('\n'.join(input_lines))
        except:
            correct_answer = 0

        return {
            'n':n, 'm':m,
            'grid':grid_str,
            'prices':prices,
            'correct_answer':correct_answer
        }

    def solve_input(self, input_str):
        sys.stdin = StringIO(input_str)
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            self.main()
            return int(sys.stdout.getvalue().strip())
        except Exception as e:
            return 0
        finally:
            sys.stdout = old_stdout

    def main(self):
        # 参考实现（Python3适配版）
        n, m = map(int, input().split())
        b = [list(input().strip()) for _ in range(n)]
        
        def inboard(y, x): return 0<=y<n and 0<=x<m
        
        c = 0
        for i in range(n):
            for j in range(m):
                if b[i][j] in '123456789':
                    c += 1
                    b[i][j] = str(int(b[i][j])-1)
                elif b[i][j] == 'S':
                    sy, sx = i, j
        
        v = [int(input()) for _ in range(c)] if c else []
        
        INF = 10**18
        up = [[0]*m for _ in range(n)]
        down = [[0]*m for _ in range(n)]
        
        for i in range(n):
            for j in range(m):
                if b[i][j] in '0123456789':
                    idx = int(b[i][j])
                    for l in range(j+1, m):
                        up[i][l] |= 1<<idx
                        down[i][l] |= 1<<idx
                elif b[i][j] == 'B':
                    for l in range(j+1, m):
                        up[i][l] |= 1<<c
                        down[i][l] |= 1<<c
                    v.append(-INF)
                    c += 1
        
        dp = [[[INF]*(1<<c) for _ in range(m)] for __ in range(n)]
        dp[sy][sx][0] = 0
        dq = deque([(sy, sx, 0, 0)])
        
        dirs = [(-1,0),(1,0),(0,1),(0,-1)]
        while dq:
            y, x, mask, t = dq.popleft()
            if t > dp[y][x][mask]: continue
            for dy, dx in dirs:
                ny, nx = y+dy, x+dx
                if not inboard(ny,nx) or b[ny][nx] not in '.S': continue
                nmask = mask
                if dy == 1 and dx == 0: nmask ^= down[y][x]
                elif dy == -1 and dx == 0: nmask ^= up[y][x]
                if dp[ny][nx][nmask] > t+1:
                    dp[ny][nx][nmask] = t+1
                    dq.append((ny,nx,nmask,t+1))
        
        max_profit = max( (sum(v[j] for j in range(len(v)) if (i>>j)&1) - dp[sy][sx][i]) 
                        for i in range(1<<c) if dp[sy][sx][i]<INF )
        print(max(max_profit,0))

    @staticmethod 
    def prompt_func(case):
        grid = '\n'.join(case['grid'])
        prices = '\n'.join(map(str, case['prices'])) if case['prices'] else "无宝藏"
        return f"""## 寻宝路径规划

**地图尺寸**: {case['n']}行×{case['m']}列

**地图布局**:
{grid}

**宝藏价值**:
{prices}

**要求**:
1. 从S出发形成闭合路径
2. 路径包围区域内不能有Bomb
3. 最大利润=宝藏价值总和-路径步数

输出格式必须为：[answer]整数答案[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL|re.I)
        if not matches: return None
        try: return int(matches[-1].strip())
        except: return None

    @classmethod
    def _verify_correction(cls, sol, case):
        return sol == case['correct_answer']
