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
const long long MOD = 1e9 + 7;
const long double E = 1e-17;
char ccc;
inline void read(int &n) {
  n = 0;
  while (true) {
    ccc = getchar();
    if (ccc == ' ' || ccc == '\n') break;
    n = n * 10 + ccc - '0';
  }
}
template <typename T>
inline T sqr(T t) {
  return t * t;
}
int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  ;
  srand(time(NULL));
  cout.precision(15);
  cout << fixed;
  int n;
  cin >> n;
  int ar[n][n];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      cin >> ar[i][j];
    }
  }
  int dp[2][n + n][n + n];
  memset(dp, 0, sizeof(dp));
  dp[0][0][0] = ar[0][0];
  vector<int> vec[n + n];
  vector<pair<long long, long long> > v[n + n];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      vec[i + j].push_back(ar[i][j]);
      v[i + j].push_back(make_pair(i, j));
    }
  }
  for (int i = 1; i <= (n - 1) * 2; i++) {
    int w = i & 1;
    int q = 1 - w;
    for (int a = 0; a < n + n; a++) {
      for (int b = 0; b < n + n; b++) {
        dp[w][a][b] = -1e9;
      }
    }
    if (vec[i].size() > vec[i - 1].size()) {
      int prev_size = (int)vec[i - 1].size();
      for (int a = 0; a < (int)vec[i].size(); a++) {
        for (int b = 0; b < (int)vec[i].size(); b++) {
          for (int c = max(0, a - 1); c <= min(prev_size - 1, a); c++) {
            for (int d = max(0, b - 1); d <= min(prev_size - 1, b); d++) {
              pair<long long, long long> f = v[i][a], g = v[i][b];
              dp[w][a][b] =
                  max(dp[w][a][b], dp[q][c][d] + ar[f.first][f.second] +
                                       (a != b ? ar[g.first][g.second] : 0));
            }
          }
        }
      }
    } else {
      int prev_size = (int)vec[i - 1].size();
      for (int a = 0; a < (int)vec[i].size(); a++) {
        for (int b = 0; b < (int)vec[i].size(); b++) {
          for (int c = a; c < min(a + 2, prev_size); c++) {
            for (int d = b; d < min(b + 2, prev_size); d++) {
              pair<long long, long long> f = v[i][a], g = v[i][b];
              dp[w][a][b] =
                  max(dp[w][a][b], dp[q][c][d] + ar[f.first][f.second] +
                                       (a != b ? ar[g.first][g.second] : 0));
            }
          }
        }
      }
    }
    if (0)
      for (int i = 0; i < n + n; i++) {
        for (int j = 0; j < n + n; j++) {
          cout << dp[w][i][j] << \"\t\";
        }
        cout << \"\n\";
      }
  }
  cout << dp[0][0][0] << \"\n\";
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Crelayracebootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = params
        self.params.setdefault('n', 3)
        self.params.setdefault('min_val', -1000)
        self.params.setdefault('max_val', 1000)
        
        # 确保参数合法性
        self.params['n'] = max(1, min(300, self.params['n']))
        self.params['min_val'] = min(self.params['min_val'], self.params['max_val'])
        self.params['max_val'] = max(self.params['min_val'], self.params['max_val'])
    
    def case_generator(self):
        n = self.params['n']
        min_val = self.params['min_val']
        max_val = self.params['max_val']
        grid = []
        for _ in range(n):
            row = [random.randint(min_val, max_val) for _ in range(n)]
            grid.append(row)
        try:
            answer = self.calculate_max_score(grid)
        except Exception as e:
            raise RuntimeError(f"Solution calculation failed: {str(e)}")
        return {
            'n': n,
            'grid': grid,
            'answer': answer
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        grid = question_case['grid']
        grid_lines = []
        for row in grid:
            grid_lines.append(' '.join(map(str, row)))
        problem_text = (
            "Furik和Rubik参加接力赛跑。比赛场地是一个边长为n米的正方形，被划分为n×n的单元格。每个单元格中的数字表示得分。\n\n"
            "规则说明：\n"
            "1. Furik从（1,1）出发，每次只能向右或向下移动，到达（n,n）\n"
            "2. Rubik随后从（n,n）出发，每次只能向左或向上移动，返回（1,1）\n"
            "3. 单元格得分只能计算一次，求最大总得分\n\n"
            "输入格式：\n"
            "第一行：整数n（1≤n≤300）\n"
            "随后n行：每行n个整数（-1000≤a_i,j≤1000）\n\n"
            "输出格式：\n"
            "一个整数，表示最大得分\n\n"
            "当前题目：\n"
            f"{n}\n" + '\n'.join(grid_lines) + "\n\n"
            "请将答案用[answer]标签包裹，例如：[answer]42[/answer]"
        )
        return problem_text
    
    @staticmethod
    def extract_output(output):
        # 增强正则表达式匹配能力
        matches = re.findall(
            r'\[answer\]\s*(-?[0-9,]+)\s*\[/answer\]', 
            output.replace(',', ''),  # 兼容数字中的逗号分隔
            flags=re.IGNORECASE
        )
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except (ValueError, IndexError):
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['answer']
    
    @classmethod
    def calculate_max_score(cls, grid):
        n = len(grid)
        if n == 0:
            return 0
        if n == 1:
            return grid[0][0]
        
        # 动态规划优化版本
        dp = [[[-float('inf')]*(n) for _ in range(n)] for __ in range(n)]
        dp[0][0][0] = grid[0][0]
        
        for step in range(1, 2*n - 1):
            new_dp = [[[-float('inf')]*(n) for _ in range(n)] for __ in range(n)]
            for x1 in range(n):
                for y1 in range(n):
                    if x1 + y1 != step:
                        continue
                    for x2 in range(n):
                        y2 = step - x2
                        if not (0 <= y2 < n):
                            continue
                        prev_step = step - 1
                        max_prev = -float('inf')
                        for px1 in [x1, x1-1]:
                            for px2 in [x2, x2-1]:
                                if px1 >=0 and px2 >=0 and prev_step >=0:
                                    if px1 + (prev_step - px1) == prev_step and \
                                       px2 + (prev_step - px2) == prev_step:
                                        max_prev = max(max_prev, dp[prev_step%2][px1][px2])
                        current_sum = grid[x1][y1]
                        if (x1, y1) != (x2, y2):
                            current_sum += grid[x2][y2]
                        new_dp[step%2][x1][x2] = max_prev + current_sum
            dp = new_dp
        
        return dp[(2*n-2)%2][n-1][n-1]
