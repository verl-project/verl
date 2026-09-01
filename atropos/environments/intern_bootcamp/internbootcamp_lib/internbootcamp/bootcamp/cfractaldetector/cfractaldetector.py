"""# 

### 谜题描述
Little Vasya likes painting fractals very much.

He does it like this. First the boy cuts out a 2 × 2-cell square out of squared paper. Then he paints some cells black. The boy calls the cut out square a fractal pattern. Then he takes a clean square sheet of paper and paints a fractal by the following algorithm:

  1. He divides the sheet into four identical squares. A part of them is painted black according to the fractal pattern. 
  2. Each square that remained white, is split into 4 lesser white squares, some of them are painted according to the fractal pattern. Each square that remained black, is split into 4 lesser black squares. 



In each of the following steps step 2 repeats. To draw a fractal, the boy can make an arbitrary positive number of steps of the algorithm. But he need to make at least two steps. In other words step 2 of the algorithm must be done at least once. The resulting picture (the square with painted cells) will be a fractal. The figure below shows drawing a fractal (here boy made three steps of the algorithm).

<image>

One evening Vasya got very tired, so he didn't paint the fractal, he just took a sheet of paper, painted a n × m-cell field. Then Vasya paint some cells black. 

Now he wonders, how many squares are on the field, such that there is a fractal, which can be obtained as described above, and which is equal to that square. Square is considered equal to some fractal if they consist of the same amount of elementary not divided cells and for each elementary cell of the square corresponding elementary cell of the fractal have the same color.

Input

The first line contains two space-separated integers n, m (2 ≤ n, m ≤ 500) — the number of rows and columns of the field, correspondingly. 

Next n lines contain m characters each — the description of the field, painted by Vasya. Character \".\" represents a white cell, character \"*\" represents a black cell.

It is guaranteed that the field description doesn't contain other characters than \".\" and \"*\".

Output

On a single line print a single integer — the number of squares on the field, such that these squares contain a drawn fractal, which can be obtained as described above.

Examples

Input

6 11
......*.***
*.*.*....**
.***....*.*
..***.*....
.*.*.....**
......*.*..


Output

3


Input

4 4
..**
..**
....
....


Output

0

Note

The answer for the first sample is shown on the picture below. Fractals are outlined by red, blue and green squares.

<image>

The answer for the second sample is 0. There is no fractal, equal to the given picture.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int N = 500, K = 16;
const int b[4] = {8, 4, 2, 1};
const int kx[4] = {0, 0, 1, 1};
const int ky[4] = {0, 1, 0, 1};
int n, m;
bool dp[10][K][N][N];
int sum[N][N];
char c;
bool a[N][N];
int getsum(int x, int y) { return ((x >= 0 && y >= 0) ? sum[x][y] : 0); }
bool is_black(int lx, int ly, int rx, int ry) {
  int a = getsum(rx, ry);
  int b = getsum(lx - 1, ry);
  int c = getsum(rx, ly - 1);
  int d = getsum(lx - 1, ly - 1);
  int sum2 = (rx - lx + 1) * (ry - ly + 1);
  return (a - b - c + d) == sum2;
}
int main() {
  cin >> n >> m;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++) {
      cin >> c;
      a[i][j] = (c == '*');
    }
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
      sum[i][j] = (i > 0 ? sum[i - 1][j] : 0) + (j > 0 ? sum[i][j - 1] : 0) -
                  ((i > 0 && j > 0) ? sum[i - 1][j - 1] : 0) + a[i][j];
  for (int mask = 0; mask < K; mask++)
    for (int i = 0; i < n; i++)
      for (int j = 0; j < m; j++) dp[0][mask][i][j] = (a[i][j] == 0);
  for (int st = 1; st < 10; st++) {
    int w = (1 << (st - 1));
    if (2 * w > min(n, m)) continue;
    for (int mask = 0; mask < K; mask++)
      for (int i = 0; i + 2 * w <= n; i++)
        for (int j = 0; j + 2 * w <= m; j++) {
          bool f = true;
          for (int q = 0; q < 4; q++) {
            int tx = i + kx[q] * w, ty = j + ky[q] * w;
            if (mask & b[q])
              f &= is_black(tx, ty, tx + w - 1, ty + w - 1);
            else
              f &= dp[st - 1][mask][tx][ty];
          }
          dp[st][mask][i][j] = f;
        }
  }
  int ans = 0;
  for (int st = 2; st < 10; st++)
    for (int mask = 0; mask < K; mask++)
      for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
          if (dp[st][mask][i][j]) {
            ans += dp[st][mask][i][j];
          }
  cout << ans;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cfractaldetectorbootcamp(Basebootcamp):
    def __init__(self, max_steps=4, max_size=64):
        self.max_steps = max_steps
        self.max_size = max_size

    def case_generator(self):
        mask = random.randint(0, 15)
        possible_steps = []
        for steps in range(2, self.max_steps + 1):
            size = 2 ** (steps + 1)
            if size <= self.max_size:
                possible_steps.append(steps)
        if not possible_steps:
            steps = 2
        else:
            steps = random.choice(possible_steps)
        grid = self.generate_fractal(mask, steps)
        n = len(grid)
        m = len(grid[0]) if n else 0
        return {
            'n': n,
            'm': m,
            'grid': [''.join(row) for row in grid],
            'mask': mask,
            'steps': steps
        }

    @staticmethod
    def generate_fractal(mask, steps):
        size = 2 ** (steps + 1)
        grid = [['.' for _ in range(size)] for _ in range(size)]
        kx = [0, 0, 1, 1]
        ky = [0, 1, 0, 1]

        def fill(x, y, block_size, current_step, is_black):
            if current_step > steps:
                for i in range(x, x + block_size):
                    for j in range(y, y + block_size):
                        grid[i][j] = '*' if is_black else '.'
                return
            
            new_size = block_size // 2
            for q in range(4):
                dx = kx[q] * new_size
                dy = ky[q] * new_size
                nx = x + dx
                ny = y + dy
                if is_black:
                    fill(nx, ny, new_size, current_step + 1, True)
                else:
                    bit = (mask >> (3 - q)) & 1
                    sub_black = bit == 1
                    fill(nx, ny, new_size, current_step + 1, sub_black)

        fill(0, 0, size, 0, False)
        return grid

    @staticmethod
    def prompt_func(question_case):
        return f"""Analyze the {question_case['n']}x{question_case['m']} grid to find valid fractal patterns:
        
Cfractaldetector Rules:
1. Starts with 2x2 pattern (Step 0)
2. Each step:
   - Divide white areas into 4 quadrants
   - Paint using original pattern (black areas stay solid)
3. Minimum 2 steps required (total size ≥8x8)

Grid:
{question_case['n']} {question_case['m']}
""" + '\n'.join(question_case['grid']) + """

Output the count of valid fractal squares as [answer]N[/answer] where N is the number."""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            grid = [list(row) for row in identity['grid']]
            return cls.count_valid_fractals(grid) == solution
        except:
            return False

    @staticmethod
    def count_valid_fractals(grid):
        n, m = len(grid), len(grid[0]) if grid else 0
        if n < 8 or m < 8:
            return 0

        # 修正二维前缀和计算
        sum_ = [[0]*(m+1) for _ in range(n+1)]
        for i in range(1, n+1):
            for j in range(1, m+1):
                sum_[i][j] = sum_[i-1][j] + sum_[i][j-1] - sum_[i-1][j-1] + (1 if grid[i-1][j-1] == '*' else 0)

        MAX_ST = 10
        K = 16
        # 调整DP数组维度顺序
        dp = [[[[False]*m for _ in range(n)] for __ in range(K)] for ___ in range(MAX_ST)]

        # 初始化st=0的状态
        for i in range(n):
            for j in range(m):
                for mask in range(K):
                    dp[0][mask][i][j] = (grid[i][j] == '.')

        # 动态规划状态转移
        for st in range(1, MAX_ST):
            w = 1 << (st-1)
            if 2*w > min(n, m):
                continue
            for mask in range(K):
                for i in range(n - 2*w +1):
                    for j in range(m - 2*w +1):
                        valid = True
                        for q in range(4):
                            x = i + (q//2)*w
                            y = j + (q%2)*w
                            if (mask >> (3-q)) & 1:  # 检查当前象限是否需要全黑
                                # 计算区域全黑的正确公式
                                a, b = x+1, y+1
                                c, d = x + w, y + w
                                total = sum_[c][d] - sum_[a-1][d] - sum_[c][b-1] + sum_[a-1][b-1]
                                if total != w*w:
                                    valid = False
                                    break
                            else:
                                if x >=n or y >=m or not dp[st-1][mask][x][y]:
                                    valid = False
                                    break
                        dp[st][mask][i][j] = valid

        # 统计有效解
        count = 0
        for st in range(2, MAX_ST):
            w = 1 << st
            if 2*w > min(n, m):
                continue
            for i in range(n - 2*w +1):
                for j in range(m - 2*w +1):
                    for mask in range(K):
                        if dp[st][mask][i][j]:
                            count +=1
        return count
