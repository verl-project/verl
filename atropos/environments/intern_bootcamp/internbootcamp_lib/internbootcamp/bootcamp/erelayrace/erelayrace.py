"""# 

### 谜题描述
Furik and Rubik take part in a relay race. The race will be set up on a large square with the side of n meters. The given square is split into n × n cells (represented as unit squares), each cell has some number.

At the beginning of the race Furik stands in a cell with coordinates (1, 1), and Rubik stands in a cell with coordinates (n, n). Right after the start Furik runs towards Rubik, besides, if Furik stands at a cell with coordinates (i, j), then he can move to cell (i + 1, j) or (i, j + 1). After Furik reaches Rubik, Rubik starts running from cell with coordinates (n, n) to cell with coordinates (1, 1). If Rubik stands in cell (i, j), then he can move to cell (i - 1, j) or (i, j - 1). Neither Furik, nor Rubik are allowed to go beyond the boundaries of the field; if a player goes beyond the boundaries, he will be disqualified. 

To win the race, Furik and Rubik must earn as many points as possible. The number of points is the sum of numbers from the cells Furik and Rubik visited. Each cell counts only once in the sum.

Print the maximum number of points Furik and Rubik can earn on the relay race.

Input

The first line contains a single integer (1 ≤ n ≤ 300). The next n lines contain n integers each: the j-th number on the i-th line ai, j ( - 1000 ≤ ai, j ≤ 1000) is the number written in the cell with coordinates (i, j).

Output

On a single line print a single number — the answer to the problem. 

Examples

Input

1
5


Output

5


Input

2
11 14
16 12


Output

53


Input

3
25 16 25
12 18 19
11 13 8


Output

136

Note

Comments to the second sample: The profitable path for Furik is: (1, 1), (1, 2), (2, 2), and for Rubik: (2, 2), (2, 1), (1, 1). 

Comments to the third sample: The optimal path for Furik is: (1, 1), (1, 2), (1, 3), (2, 3), (3, 3), and for Rubik: (3, 3), (3, 2), (2, 2), (2, 1), (1, 1). The figure to the sample: 

<image> Furik's path is marked with yellow, and Rubik's path is marked with pink.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int MAXN = 3e2 + 10;
const int INF = 1e8 + 100;
int dp[MAXN][MAXN][MAXN];
bool vis[MAXN][MAXN][MAXN];
int a[MAXN][MAXN];
int memo(int r, int d, int R) {
  if (vis[r][d][R]) return dp[r][d][R];
  vis[r][d][R] = true;
  int D = r + d - R;
  int &res = dp[r][d][R];
  res = -INF;
  if (r + d == 0) {
    res = a[0][0];
    return res;
  }
  int temp = a[r][d] + a[R][D];
  if (R == r) temp = a[r][d];
  if (r && R) res = max(res, temp + memo(r - 1, d, R - 1));
  if (r && D) res = max(res, temp + memo(r - 1, d, R));
  if (d && R) res = max(res, temp + memo(r, d - 1, R - 1));
  if (d && D) res = max(res, temp + memo(r, d - 1, R));
  return res;
}
int main() {
  int n;
  cin >> n;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) cin >> a[i][j];
  cout << memo(n - 1, n - 1, n - 1);
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re
from functools import lru_cache
import sys  # Added to adjust recursion limit

class Erelayracebootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=5):
        self.min_n = min_n
        self.max_n = max_n

    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        grid = []
        for _ in range(n):
            row = [random.randint(-1000, 1000) for _ in range(n)]
            grid.append(row)
        return {'n': n, 'grid': grid}

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        grid = question_case['grid']
        grid_str = '\n'.join(' '.join(map(str, row)) for row in grid)
        prompt = f"""You are participating in a programming competition and need to solve the following problem.

Furik and Rubik are participating in a relay race on an n×n grid. The goal is to maximize the total points collected from the cells they visit. Each cell is counted only once, even if both visit it.

Furik starts at cell (1,1) (top-left) and moves to (n,n) (bottom-right), moving only right or down. Rubik then starts at (n,n) and moves to (1,1), moving only left or up. The total points are the sum of all unique cells visited by either.

Input:
The first line has integer n (1 ≤ n ≤ 300). The next n lines each have n integers.

Output:
The maximum points.

Input for your task:
{n}
{grid_str}

Format your answer as a single integer within [answer] and [/answer]. For example: [answer]53[/answer].

Your answer:"""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        grid = identity['grid']
        n = identity['n']
        if n != len(grid) or any(len(row)!=n for row in grid):
            return False
        try:
            correct = calculate_max_score(grid)
            return solution == correct
        except Exception:
            return False


def calculate_max_score(grid):
    sys.setrecursionlimit(1 << 25)  # Prevent stack overflow for large n
    n = len(grid)
    if n == 0:
        return 0
    a = grid
    INF = -10**18

    @lru_cache(maxsize=None)
    def memo(r, d, R):
        D = r + d - R
        # Check all indices are within valid range
        if not (0 <= r < n and 0 <= d < n and 0 <= R < n and 0 <= D < n):
            return INF
        if r + d != R + D:  # Path steps must be consistent
            return INF
        if r + d == 0:
            return a[r][d]
        current_sum = a[r][d] + a[R][D] if (r != R or d != D) else a[r][d]
        max_val = INF
        # Explore all valid previous states
        if r > 0 and R > 0:
            val = memo(r-1, d, R-1)
            if val != INF:
                max_val = max(max_val, current_sum + val)
        if r > 0 and D > 0:
            val = memo(r-1, d, R)
            if val != INF:
                max_val = max(max_val, current_sum + val)
        if d > 0 and R > 0:
            val = memo(r, d-1, R-1)
            if val != INF:
                max_val = max(max_val, current_sum + val)
        if d > 0 and D > 0:
            val = memo(r, d-1, R)
            if val != INF:
                max_val = max(max_val, current_sum + val)
        return max_val

    result = memo(n-1, n-1, n-1)
    return result if result != INF else 0
