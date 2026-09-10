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
int n;
char s[51][51];
int ans[51][51][51][51];
int solve(int r1, int c1, int r2, int c2) {
  if (r1 == r2 && c1 == c2) ans[r1][c1][r2][c2] = (s[r1][c1] == '#');
  if (ans[r1][c1][r2][c2] != -1) return ans[r1][c1][r2][c2];
  ans[r1][c1][r2][c2] = max(r2 - r1 + 1, c2 - c1 + 1);
  int cans = ans[r1][c1][r2][c2];
  for (int i = r1 + 1; i <= r2; i++) {
    cans = min(cans, solve(r1, c1, i - 1, c2) + solve(i, c1, r2, c2));
  }
  for (int i = c1 + 1; i <= c2; i++) {
    cans = min(cans, solve(r1, c1, r2, i - 1) + solve(r1, i, r2, c2));
  }
  ans[r1][c1][r2][c2] = cans;
  return ans[r1][c1][r2][c2];
}
int main() {
  scanf(\"%d\", &n);
  for (int i = 0; i < n; i++) scanf(\"%s\", s[i]);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++)
        for (int l = 0; l < n; l++) ans[i][j][k][l] = -1;
  int ans = solve(0, 0, n - 1, n - 1);
  printf(\"%d\n\", ans);
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Frectanglepainting1bootcamp(Basebootcamp):
    def __init__(self, n_min=3, n_max=8, black_prob=0.4):
        self.n_min = max(1, n_min)
        self.n_max = min(50, n_max)  # 根据题目约束调整范围
        self.black_prob = black_prob
    
    def case_generator(self):
        while True:
            n = random.randint(self.n_min, self.n_max)
            grid = [
                ['#' if random.random() < self.black_prob else '.' 
                 for _ in range(n)]
                for _ in range(n)
            ]
            
            # 确保至少有一个黑格或全白
            black_count = sum(row.count('#') for row in grid)
            if black_count == 0:
                correct_answer = 0
                break
                
            try:
                correct_answer = self.calculate_min_cost([row[:] for row in grid])
                break
            except RecursionError:
                continue  # 处理极大网格时的递归深度问题

        return {
            'n': n,
            'grid': [''.join(row) for row in grid],
            'correct_answer': correct_answer
        }

    @staticmethod
    def prompt_func(question_case):
        input_lines = [str(question_case['n'])] + question_case['grid']
        problem_instance = '\n'.join(input_lines)
        return f"""你正在解决一个网格涂白优化问题。给定一个n×n网格，每次可选择任意矩形区域涂白，费用为矩形高度和宽度的较大值。请计算将所有黑色格子涂白的最小总费用。

输入格式：
第一行为整数n
随后n行，每行n个字符（.表示白色，#表示黑色）

当前问题：
{problem_instance}

请将最终答案放在[answer]标签内，例如：[answer]5[/answer]。答案应为整数。"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output, re.IGNORECASE)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']

    @staticmethod
    def calculate_min_cost(grid):
        n = len(grid)
        memo = {}
        
        def solve(r1, c1, r2, c2):
            if r1 > r2 or c1 > c2:
                return 0
            key = (r1, c1, r2, c2)
            if key in memo:
                return memo[key]
            
            # 优化全白判断
            has_black = False
            for i in range(r1, r2+1):
                if '#' in grid[i][c1:c2+1]:
                    has_black = True
                    break
            if not has_black:
                memo[key] = 0
                return 0
            
            min_cost = max(r2-r1+1, c2-c1+1)
            
            # 水平分割优化
            for i in range(r1, r2):
                cost = solve(r1, c1, i, c2) + solve(i+1, c1, r2, c2)
                if cost < min_cost:
                    min_cost = cost
                    if min_cost == 1:  # 提前终止
                        break
            
            # 垂直分割优化
            for j in range(c1, c2):
                cost = solve(r1, c1, r2, j) + solve(r1, j+1, r2, c2)
                if cost < min_cost:
                    min_cost = cost
                    if min_cost == 1:
                        break
            
            memo[key] = min_cost
            return min_cost
        
        return solve(0, 0, n-1, n-1)
