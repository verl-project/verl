"""# 

### 谜题描述
There is a square grid of size n × n. Some cells are colored in black, all others are colored in white. In one operation you can select some rectangle and color all its cells in white. It costs max(h, w) to color a rectangle of size h × w. You are to make all cells white for minimum total cost.

Input

The first line contains a single integer n (1 ≤ n ≤ 50) — the size of the square grid.

Each of the next n lines contains a string of length n, consisting of characters '.' and '#'. The j-th character of the i-th line is '#' if the cell with coordinates (i, j) is black, otherwise it is white.

Output

Print a single integer — the minimum total cost to paint all cells in white.

Examples

Input


3
###
#.#
###


Output


3


Input


3
...
...
...


Output


0


Input


4
#...
....
....
#...


Output


2


Input


5
#...#
.#.#.
.....
.#...
#....


Output


5

Note

The examples and some of optimal solutions are shown on the pictures below.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int k, i, n, j, ii, jj, f[51][51], d[51][51][51][51], val;
char c[51][51];
int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);
  cin >> n;
  for (i = 1; i <= n; i++)
    for (j = 1; j <= n; j++) cin >> c[i][j];
  for (i = 1; i <= n; i++)
    for (j = 1; j <= n; j++)
      f[i][j] = f[i - 1][j] + f[i][j - 1] - f[i - 1][j - 1] + (c[i][j] == '#');
  auto check = [&](int x, int y, int xx, int yy) {
    return f[xx][yy] - f[x - 1][yy] - f[xx][y - 1] + f[x - 1][y - 1];
  };
  for (i = n; i >= 1; i--)
    for (j = n; j >= 1; j--)
      for (ii = i; ii <= n; ii++)
        for (jj = j; jj <= n; jj++) {
          if (check(i, j, ii, jj) == 0) {
            d[i][j][ii][jj] = 0;
            continue;
          }
          val = max(jj - j + 1, ii - i + 1);
          for (k = j; k < jj; k++)
            val = min(val, d[i][j][ii][k] + d[i][k + 1][ii][jj]);
          for (k = i; k < ii; k++)
            val = min(val, d[i][j][k][jj] + d[k + 1][j][ii][jj]);
          d[i][j][ii][jj] = val;
        }
  cout << d[1][1][n][n];
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Drectanglepainting1bootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=5, black_prob=0.3):
        self.min_n = min_n
        self.max_n = max_n
        self.black_prob = black_prob
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        max_attempts = 100
        
        for _ in range(max_attempts):
            grid = [
                ''.join('#' if random.random() < self.black_prob else '.' 
                       for _ in range(n))
                for _ in range(n)
            ]
            black_count = sum(row.count('#') for row in grid)
            
            # 允许生成全白或包含黑块的case
            if black_count == 0 or any('#' in row for row in grid):
                break
        else:  # 多次尝试失败后生成全白网格
            grid = ['.'*n for _ in range(n)]
        
        min_cost = self.compute_min_cost(n, grid)
        return {'n': n, 'grid': grid, 'min_cost': min_cost}
    
    @staticmethod
    def compute_min_cost(n, grid):
        # 初始化前缀和数组（从1开始索引）
        f = [[0]*(n+2) for _ in range(n+2)]
        for i in range(1, n+1):
            for j in range(1, n+1):
                cell_value = 1 if grid[i-1][j-1] == '#' else 0
                f[i][j] = f[i-1][j] + f[i][j-1] - f[i-1][j-1] + cell_value
        
        # 初始化四维DP数组
        d = [[[[0]*(n+2) for _ in range(n+2)] 
             for __ in range(n+2)] 
             for ___ in range(n+2)]
        
        # 动态规划计算
        for i in range(n, 0, -1):
            for j in range(n, 0, -1):
                for ii in range(i, n+1):
                    for jj in range(j, n+1):
                        # 计算当前区域的黑块总数
                        total = f[ii][jj] - f[i-1][jj] - f[ii][j-1] + f[i-1][j-1]
                        
                        if total == 0:
                            d[i][j][ii][jj] = 0
                            continue
                        
                        # 初始值为区域的最大边长
                        h = ii - i + 1
                        w = jj - j + 1
                        val = max(h, w)
                        
                        # 垂直切分尝试
                        for k in range(j, jj):
                            val = min(val, d[i][j][ii][k] + d[i][k+1][ii][jj])
                        
                        # 水平切分尝试
                        for k in range(i, ii):
                            val = min(val, d[i][j][k][jj] + d[k+1][j][ii][jj])
                        
                        d[i][j][ii][jj] = val
        
        return d[1][1][n][n]
    
    @staticmethod
    def prompt_func(question_case):
        problem_desc = [
            "你将得到一个n×n的网格，其中#表示黑色单元格，.表示白色单元格。",
            "每次操作可以选择任意矩形区域将其全部变为白色，费用为该矩形的高度和宽度中的较大值。",
            "请计算将所有黑色单元格变为白色的最小总费用。",
            "",
            "输入格式：",
            f"第一行：{question_case['n']}",
            "接下来n行：每行包含n个字符",
            "",
            "当前问题：",
            str(question_case['n'])
        ]
        problem_desc.extend(question_case['grid'])
        problem_desc.append(
            "\n请将最终答案放在[answer]和[/answer]标记之间，例如：[answer]5[/answer]"
        )
        return '\n'.join(problem_desc)
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except (ValueError, TypeError):
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity["min_cost"]
