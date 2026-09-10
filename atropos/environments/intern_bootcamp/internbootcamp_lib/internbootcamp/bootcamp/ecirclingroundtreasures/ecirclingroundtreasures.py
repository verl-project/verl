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
#include <bits/stdc++.h>
using namespace std;
int n, m, ans, cnt, flag, tot;
int sx, sy;
int dx[] = {-1, 1, 0, 0};
int dy[] = {0, 0, -1, 1};
int dp[25][25][1 << 8];
int val[8], w[1 << 8];
int gx[8], gy[8];
char mp[25][25];
char s[25];
struct Node {
  int x, y, k;
} cur, now;
bool issur(int nx, int ny, int tx, int ty, int j) {
  if (nx == gx[j] && ny < gy[j]) {
    if (tx < gx[j]) return true;
  } else if (tx == gx[j] && ty < gy[j]) {
    if (nx < gx[j]) return true;
  }
  return false;
}
void bfs() {
  int i, j, t, k, nx, ny, tx, ty, nk;
  queue<Node> q;
  memset(dp, -1, sizeof(dp));
  ans = -0x3f3f3f3f;
  cur.x = sx;
  cur.y = sy;
  cur.k = 0;
  dp[sx][sy][0] = 0;
  q.push(cur);
  while (!q.empty()) {
    now = q.front();
    q.pop();
    nx = now.x;
    ny = now.y;
    nk = now.k;
    if (nx == sx && ny == sy) ans = max(ans, w[nk] - dp[nx][ny][nk]);
    for (i = 0; i < 4; i++) {
      tx = nx + dx[i];
      ty = ny + dy[i];
      k = nk;
      if (tx < 1 || tx > n || ty < 1 || ty > m) continue;
      if (!(mp[tx][ty] == 'S' || mp[tx][ty] == '.')) continue;
      for (j = 0; j <= cnt; j++) {
        if (issur(nx, ny, tx, ty, j)) k ^= (1 << j);
      }
      if (dp[tx][ty][k] != -1 && dp[tx][ty][k] <= dp[nx][ny][nk] + 1) continue;
      cur.x = tx;
      cur.y = ty;
      cur.k = k;
      dp[tx][ty][k] = dp[nx][ny][nk] + 1;
      q.push(cur);
    }
  }
}
int main() {
  int i, j, t;
  while (~scanf(\"%d%d\", &n, &m)) {
    cnt = -1;
    for (i = 1; i <= n; i++) {
      scanf(\"%s\", s);
      for (j = 1; j <= m; j++) {
        mp[i][j] = s[j - 1];
        if (mp[i][j] == 'S') sx = i, sy = j;
        if (mp[i][j] >= '1' && mp[i][j] <= '8') {
          t = mp[i][j] - '1';
          cnt++;
          gx[t] = i, gy[t] = j;
        }
      }
    }
    for (i = 0; i <= cnt; i++) {
      scanf(\"%d\", &val[i]);
    }
    for (i = 1; i <= n; i++) {
      for (j = 1; j <= m; j++) {
        if (mp[i][j] == 'B') gx[++cnt] = i, gy[cnt] = j, val[cnt] = -10000;
      }
    }
    memset(w, 0, sizeof(w));
    tot = 1 << (cnt + 1);
    for (i = 0; i < tot; i++) {
      for (j = 0; j <= cnt; j++) {
        if (i & (1 << j)) w[i] += val[j];
      }
    }
    bfs();
    printf(\"%d\n\", ans);
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import deque
from bootcamp import Basebootcamp

class Ecirclingroundtreasuresbootcamp(Basebootcamp):
    def __init__(self, max_grid_size=8, max_objects=8):
        self.max_grid_size = max_grid_size
        self.max_objects = max_objects

    def case_generator(self):
        for _ in range(100):
            n = random.randint(1, self.max_grid_size)
            m = random.randint(1, self.max_grid_size)
            grid = [['.' for _ in range(m)] for __ in range(n)]
            sx, sy = random.randint(0, n-1), random.randint(0, m-1)
            grid[sx][sy] = 'S'
            total_objects = random.randint(0, self.max_objects)
            num_treasures = random.randint(0, total_objects)
            num_bombs = total_objects - num_treasures
            available = []
            for i in range(n):
                for j in range(m):
                    if i != sx or j != sy:
                        available.append((i, j))
            if len(available) < total_objects:
                continue
            selected = random.sample(available, total_objects)
            for idx in range(num_treasures):
                i, j = selected[idx]
                grid[i][j] = str(idx + 1)
            for idx in range(num_treasures, total_objects):
                i, j = selected[idx]
                grid[i][j] = 'B'
            treasures_in_grid = [int(c) for row in grid for c in row if c.isdigit()]
            if num_treasures:
                sorted_t = sorted(treasures_in_grid)
                if sorted_t[0] != 1 or sorted_t[-1] != num_treasures or len(set(sorted_t)) != num_treasures:
                    continue
            treasure_values = [random.randint(-200, 200) for _ in range(num_treasures)]
            identity = {
                'n': n,
                'm': m,
                'grid': [''.join(row) for row in grid],
                'treasure_values': treasure_values,
                'correct_answer': None
            }
            try:
                identity['correct_answer'] = self.compute_max_profit(identity)
                if identity['correct_answer'] is not None:
                    return identity
            except:
                continue
        return {
            'n': 1,
            'm': 1,
            'grid': ['S'],
            'treasure_values': [],
            'correct_answer': 0
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        grid = question_case['grid']
        treasure_values = question_case['treasure_values']
        input_lines = [f"{n} {m}"] + grid + list(map(str, treasure_values))
        input_str = '\n'.join(input_lines)
        prompt = f"""You are a treasure hunter tasked with finding the maximum profit from a closed path on a map. The map is a grid where each cell can be an obstacle ('#'), a bomb ('B'), a treasure (digits '1'-'8'), your starting position ('S'), or empty ('.'). 

The goal is to construct a closed path starting and ending at 'S' that does not enclose any bombs. The profit is calculated as the sum of the values of treasures inside the path minus the number of moves in the path. 

Rules:
- The path must be a closed loop starting and ending at 'S'.
- Moves are allowed to adjacent cells (up, down, left, right).
- You cannot enter cells containing treasure, bombs, or obstacles.
- Bombs inside the enclosed area invalidate the path.
- Paths can self-intersect. Use the even-odd rule for determining enclosed cells.

Input:
{input_str}

Output a single integer: the maximum profit. If no valid path exists, output 0.

Please provide your answer within [answer] and [/answer], e.g., [answer]5[/answer]."""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        correct = identity.get('correct_answer', None)
        return solution == correct if correct is not None else False

    @staticmethod
    def compute_max_profit(identity):
        n, m = identity['n'], identity['m']
        grid = identity['grid']
        treasure_values = identity['treasure_values']
        sx = sy = None
        treasures = []
        bombs = []
        for i in range(n):
            for j in range(m):
                c = grid[i][j]
                if c == 'S':
                    sx, sy = i+1, j+1
                elif c.isdigit():
                    treasures.append((int(c), i+1, j+1))
                elif c == 'B':
                    bombs.append((i+1, j+1))
        treasures.sort()
        gx, gy, val = [], [], []
        for num, x, y in treasures:
            gx.append(x)
            gy.append(y)
            val.append(treasure_values[num-1])
        for x, y in bombs:
            gx.append(x)
            gy.append(y)
            val.append(-10000)
        m_objects = len(gx)
        tot = 1 << m_objects
        w = [0] * tot
        for mask in range(tot):
            total = 0
            for j in range(m_objects):
                if mask & (1 << j):
                    total += val[j]
            w[mask] = total
        INF = float('inf')
        dp = [[[INF]*tot for _ in range(m+2)] for __ in range(n+2)]
        dp[sx][sy][0] = 0
        q = deque([(sx, sy, 0)])
        max_profit = -INF
        dx = [-1, 1, 0, 0]
        dy = [0, 0, -1, 1]
        while q:
            x, y, mask = q.popleft()
            if x == sx and y == sy:
                current_profit = w[mask] - dp[x][y][mask]
                max_profit = max(max_profit, current_profit)
            for i in range(4):
                tx, ty = x + dx[i], y + dy[i]
                if tx < 1 or tx > n or ty < 1 or ty > m:
                    continue
                cell = grid[tx-1][ty-1]
                if cell not in ('.', 'S'):
                    continue
                new_mask = mask
                for j in range(m_objects):
                    nx, ny = x, y
                    obj_x, obj_y = gx[j], gy[j]
                    if nx == obj_x and ny < obj_y:
                        if tx < obj_x:
                            new_mask ^= (1 << j)
                    elif tx == obj_x and ty < obj_y:
                        if nx < obj_x:
                            new_mask ^= (1 << j)
                if dp[tx][ty][new_mask] > dp[x][y][mask] + 1:
                    dp[tx][ty][new_mask] = dp[x][y][mask] + 1
                    q.append((tx, ty, new_mask))
        return max(max_profit, 0)
