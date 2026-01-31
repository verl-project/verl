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
const int NS = 1111;
const int MOD = 1000000007;
int n, m, k;
int v[11], u[NS];
int s[NS][NS], a[NS][NS];
int one_num(int z) {
  int cnt = 0;
  for (; z > 0; z = z & (z - 1)) cnt++;
  return cnt;
}
long long dfs(int x, int y) {
  if (y > m) return dfs(x + 1, 1);
  if (x > n) return 1;
  int cur = s[x - 1][y] | s[x][y - 1];
  if (n + m - x - y >= k - __builtin_popcount(cur)) return 0;
  long long ans = 0, tmp = -1;
  for (int t = ((~cur) & ((1 << k) - 1)); t > 0; t -= t & (-t)) {
    int i = u[t & (-t)] + 1;
    if (!(cur & (1 << (i - 1))) && (a[x][y] == i || !a[x][y])) {
      s[x][y] = cur | (1 << (i - 1));
      if (v[i])
        v[i]++, ans += dfs(x, y + 1);
      else {
        v[i]++;
        if (tmp == -1) tmp = dfs(x, y + 1);
        ans += tmp;
      }
      v[i]--;
      ans %= MOD;
    }
  }
  return ans;
}
int main() {
  while (~scanf(\"%d%d%d\", &n, &m, &k)) {
    memset(v, 0, sizeof(v));
    for (int i = 0; i <= k; i++) u[1 << i] = i;
    for (int i = 1; i <= n; i++)
      for (int j = 1; j <= m; j++) {
        scanf(\"%d\", &a[i][j]);
        v[a[i][j]]++;
      }
    if ((n + m - 1) > k) {
      printf(\"0\n\");
      continue;
    }
    long long ans = dfs(1, 1);
    printf(\"%I64d\n\", ans);
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

MOD = 10**9 + 7

class Bdistinctpathsbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', 2)
        self.m = params.get('m', 2)
        self.k = params.get('k', 4)
    
    def case_generator(self):
        # 生成n和m，确保required_k <= 10
        while True:
            n = random.randint(1, 1000)
            m = random.randint(1, 1000)
            required_k = n + m - 1
            if required_k <= 10:
                break
        k = random.randint(required_k, 10)
        board = [[0 for _ in range(m)] for _ in range(n)]
        # 确保至少有1个颜色已经涂色
        for i in range(random.randint(0, 5)):
            x = random.randint(0, n-1)
            y = random.randint(0, m-1)
            color = random.randint(1, k)
            board[x][y] = color
        return {
            'n': n,
            'm': m,
            'k': k,
            'board': board
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        k = question_case['k']
        board = question_case['board']
        board_str = '\n'.join([' '.join(map(str, row)) for row in board])
        prompt = f"你有一个{n}×{m}的棋盘，每个格子可能已经涂色或未涂色。你需要将所有未涂色的格子涂上颜色（颜色编号为1到{k}），满足以下规则：\n\n"
        prompt += "规则：任何一条从左上角到右下角的路径只能向右或向下移动，并且路径上的所有格子的颜色必须互不相同。\n\n"
        prompt += "棋盘的初始状态如下：\n\n"
        prompt += board_str + "\n\n"
        prompt += "请计算满足条件的涂色方案数，并将答案放在[answer]标签内，格式为：[answer]数字[/answer]。"
        return prompt
    
    @staticmethod
    def extract_output(output):
        start = output.rfind('[answer]')
        if start == -1:
            return None
        end = output.find('[/answer]', start)
        if end == -1:
            return None
        answer_str = output[start + 8:end].strip()
        if not answer_str:
            return None
        if not answer_str.isdigit():
            return None
        return int(answer_str)
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        m = identity['m']
        k = identity['k']
        board = identity['board']
        
        # 检查初始棋盘是否有冲突
        for i in range(n):
            for j in range(m):
                current = board[i][j]
                if current == 0:
                    continue
                if (i > 0 and board[i-1][j] == current) or (j > 0 and board[i][j-1] == current):
                    return solution == 0
        
        memo = {}
        def dfs(x, y, mask):
            if (x, y, mask) in memo:
                return memo[(x, y, mask)]
            if x == n and y == m:
                return 1
            res = 0
            for color in range(1, k+1):
                if not (mask & (1 << (color - 1))):
                    if board[x-1][y-1] == 0 or board[x-1][y-1] == color:
                        new_mask = mask | (1 << (color - 1))
                        if y < m:
                            res += dfs(x, y+1, new_mask)
                        else:
                            res += dfs(x+1, 1, new_mask)
            memo[(x, y, mask)] = res % MOD
            return memo[(x, y, mask)]
        
        total = dfs(1, 1, 0) % MOD
        return solution == total

