"""# 

### 谜题描述
You have a rectangular n × m-cell board. Some cells are already painted some of k colors. You need to paint each uncolored cell one of the k colors so that any path from the upper left square to the lower right one doesn't contain any two cells of the same color. The path can go only along side-adjacent cells and can only go down or right.

Print the number of possible paintings modulo 1000000007 (109 + 7).

Input

The first line contains three integers n, m, k (1 ≤ n, m ≤ 1000, 1 ≤ k ≤ 10). The next n lines contain m integers each — the board. The first of them contains m uppermost cells of the board from the left to the right and the second one contains m cells from the second uppermost row and so on. If a number in a line equals 0, then the corresponding cell isn't painted. Otherwise, this number represents the initial color of the board cell — an integer from 1 to k.

Consider all colors numbered from 1 to k in some manner.

Output

Print the number of possible paintings modulo 1000000007 (109 + 7).

Examples

Input

2 2 4
0 0
0 0


Output

48


Input

2 2 4
1 2
2 1


Output

0


Input

5 6 10
0 0 0 0 0 0
0 0 0 0 0 0
0 0 0 0 0 0
0 0 0 0 0 0
0 0 0 0 0 0


Output

3628800


Input

2 6 10
1 2 3 4 5 6
0 0 0 0 0 0


Output

4096

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
long long n, m, k, i, j, lim[9][9], lim2[9][9], a[9][9], s, mod = 1e9 + 7;
vector<long long> v;
long long dfs(long long x, long long y, long long cnt) {
  if (x > n) return 1;
  if (y > m) return dfs(x + 1, 1, cnt);
  long long ans = 0, i;
  lim[x][y] = lim[x - 1][y] | lim[x][y - 1];
  for (i = 0; i < k; i++) {
    if (a[x][y] != -1 && a[x][y] != i) continue;
    if (lim[x][y] & (1 << i)) continue;
    if (lim2[x][y] & (1 << i)) continue;
    if (!((1 << i) & s)) {
      if (a[x][y] == -1 && v[cnt] < i) continue;
    }
    lim[x][y] ^= (1 << i);
    long long s =
        dfs(x, y + 1, min((long long)((int)v.size() - 1), cnt + (v[cnt] == i)));
    if (i == v[cnt] && a[x][y] == -1)
      (ans += s * (v.size() - cnt)) %= mod;
    else
      (ans += s) %= mod;
    lim[x][y] ^= (1 << i);
  }
  return ans;
}
int main() {
  scanf(\"%I64d%I64d%I64d\", &n, &m, &k);
  if (n + m > 11 || n + m - 1 > k) {
    cout << 0;
    return 0;
  }
  for (i = 1; i <= n; i++) {
    for (j = 1; j <= m; j++) {
      scanf(\"%I64d\", &a[i][j]);
      a[i][j]--;
      if (a[i][j] >= 0) s |= (1 << a[i][j]);
      if (a[i][j] >= 0) lim2[i - 1][j] |= (1 << a[i][j]);
      if (a[i][j] >= 0) lim2[i][j - 1] |= (1 << a[i][j]);
    }
  }
  for (i = n; i >= 1; i--) {
    for (j = m; j >= 1; j--) {
      if (a[i][j] >= 0 && ((lim2[i][j] >> a[i][j]) & 1)) {
        cout << 0;
        return 0;
      }
      lim2[i][j] |= lim2[i + 1][j] | lim2[i][j + 1];
    }
  }
  for (i = 0; i < k; i++)
    if ((s >> i) % 2 == 0) v.push_back(i);
  if (v.size() == 0) {
    cout << 1;
    return 0;
  }
  cout << dfs(1, 1, 0);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

MOD = 10**9 + 7

class Ddistinctpathsbootcamp(Basebootcamp):
    def __init__(self, max_n=5, max_m=5, max_k=10):
        self.max_n = max_n
        self.max_m = max_m
        self.max_k = max_k

    def generate_valid_full_grid(self, n, m, k):
        grid = [[0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                used = set()
                if i > 0:
                    used.add(grid[i-1][j])
                if j > 0:
                    used.add(grid[i][j-1])
                available = [c for c in range(1, k+1) if c not in used]
                if not available:
                    return None
                grid[i][j] = min(available)
        return grid

    def case_generator(self):
        n, m = 0, 0
        for _ in range(100):  # Retry to get valid n, m
            n = random.randint(1, self.max_n)
            m = random.randint(1, self.max_m)
            if n + m <= 11:
                break
        k = random.randint(max(n + m -1, 1), self.max_k)
        grid_full = self.generate_valid_full_grid(n, m, k)
        if not grid_full:
            grid_initial = [[0]*m for _ in range(n)]
        else:
            grid_initial = []
            for row in grid_full:
                new_row = []
                for val in row:
                    if random.random() < 0.3:  # 30% chance to retain color
                        new_row.append(val)
                    else:
                        new_row.append(0)
                grid_initial.append(new_row)
        # Compute expected solution
        expected = self.compute_solution(n, m, k, grid_initial)
        return {
            'n': n,
            'm': m,
            'k': k,
            'grid': grid_initial,
            'expected': expected
        }

    @staticmethod
    def compute_solution(n, m, k, grid):
        if n + m > 11 or (n + m - 1) > k:
            return 0
        grid = [[cell-1 if cell !=0 else -1 for cell in row] for row in grid]
        a = [[-1]*(m+2) for _ in range(n+2)]
        for i in range(n):
            for j in range(m):
                a[i+1][j+1] = grid[i][j] if grid[i][j] != -1 else -1
        lim2 = [[0]*(m+2) for _ in range(n+2)]
        s = 0
        for i in range(1, n+1):
            for j in range(1, m+1):
                if a[i][j] != -1:
                    s |= 1 << a[i][j]
                    if i < n:
                        lim2[i][j] |= 1 << a[i][j]
                    if j < m:
                        lim2[i][j] |= 1 << a[i][j]
        for i in range(n, 0, -1):
            for j in range(m, 0, -1):
                lim2[i][j] |= lim2[i+1][j] | lim2[i][j+1]
                if a[i][j] != -1 and (lim2[i][j] & (1 << a[i][j])):
                    return 0
        v = []
        for color in range(k):
            if not (s & (1 << color)):
                v.append(color)
        if not v:
            return 1
        # DFS to compute answer
        memo = {}
        def dfs(x, y, cnt, lim):
            if x > n:
                return 1
            if y > m:
                return dfs(x+1, 1, cnt, lim)
            key = (x, y, cnt, tuple(map(tuple, lim)))
            if key in memo:
                return memo[key]
            current_lim = lim[x-1][y] | lim[x][y-1]
            total = 0
            for color in range(k):
                if a[x][y] != -1 and color != a[x][y]:
                    continue
                if current_lim & (1 << color):
                    continue
                if lim2[x][y] & (1 << color):
                    continue
                if not (s & (1 << color)):
                    if a[x][y] == -1 and (cnt >= len(v) or color > v[cnt]):
                        continue
                new_lim = [row[:] for row in lim]
                new_lim[x][y] = current_lim | (1 << color)
                new_cnt = cnt
                if a[x][y] == -1 and not (s & (1 << color)):
                    if color == v[cnt]:
                        new_cnt = min(len(v)-1, cnt + 1)
                res = dfs(x, y+1, new_cnt, new_lim)
                if a[x][y] == -1 and color in v and cnt < len(v) and color == v[cnt]:
                    total = (total + res * (len(v) - cnt)) % MOD
                else:
                    total = (total + res) % MOD
            memo[key] = total
            return total
        lim_init = [[0]*(m+2) for _ in range(n+2)]
        result = dfs(1, 1, 0, lim_init)
        return result

    @staticmethod
    def prompt_func(question_case):
        grid = question_case['grid']
        n = question_case['n']
        m = question_case['m']
        k = question_case['k']
        grid_str = '\n'.join(' '.join(map(str, row)) for row in grid)
        return f"""You are given a {n}x{m} grid. Some cells are colored (1 to {k}) and others are 0 (uncolored). Your task is to count the number of ways to color all 0-cells such that every path from the top-left to the bottom-right (moving only right or down) has all distinct colors. Output the count modulo 1e9+7.

Input:
{n} {m} {k}
{grid_str}

The output must be the number of valid ways modulo 1e9+7. Place your final answer within [answer] tags, e.g., [answer]12345[/answer]."""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return int(matches[-1].strip()) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected']
