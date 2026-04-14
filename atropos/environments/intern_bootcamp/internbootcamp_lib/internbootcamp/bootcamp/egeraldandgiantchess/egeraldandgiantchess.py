"""# 

### 谜题描述
Giant chess is quite common in Geraldion. We will not delve into the rules of the game, we'll just say that the game takes place on an h × w field, and it is painted in two colors, but not like in chess. Almost all cells of the field are white and only some of them are black. Currently Gerald is finishing a game of giant chess against his friend Pollard. Gerald has almost won, and the only thing he needs to win is to bring the pawn from the upper left corner of the board, where it is now standing, to the lower right corner. Gerald is so confident of victory that he became interested, in how many ways can he win?

The pawn, which Gerald has got left can go in two ways: one cell down or one cell to the right. In addition, it can not go to the black cells, otherwise the Gerald still loses. There are no other pawns or pieces left on the field, so that, according to the rules of giant chess Gerald moves his pawn until the game is over, and Pollard is just watching this process.

Input

The first line of the input contains three integers: h, w, n — the sides of the board and the number of black cells (1 ≤ h, w ≤ 105, 1 ≤ n ≤ 2000). 

Next n lines contain the description of black cells. The i-th of these lines contains numbers ri, ci (1 ≤ ri ≤ h, 1 ≤ ci ≤ w) — the number of the row and column of the i-th cell.

It is guaranteed that the upper left and lower right cell are white and all cells in the description are distinct.

Output

Print a single line — the remainder of the number of ways to move Gerald's pawn from the upper left to the lower right corner modulo 109 + 7.

Examples

Input

3 4 2
2 2
2 3


Output

2


Input

100 100 3
15 16
16 15
99 88


Output

545732279

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int maxv = 2045;
const long long mod = 1e9 + 7;
const int maxn = 1e6 + 40;
long long fac[maxn], inv[maxn];
long long qpow(long long a, long long p) {
  long long ans = 1;
  long long xx = a;
  while (p > 0) {
    if (p & 1) ans = (xx * ans) % mod;
    xx = (xx * xx) % mod;
    p >>= 1;
  }
  return ans;
}
void init() {
  fac[0] = 1;
  inv[0] = 1;
  for (long long i = 1; i < maxn; i++) {
    fac[i] = (fac[i - 1] * i) % mod;
    inv[i] = inv[i - 1] * qpow(i, mod - 2) % mod;
  }
}
int h, w, n;
pair<long long, long long> a[maxv];
long long dp[maxv];
long long culC(long long a, long long b) {
  return fac[a] * inv[a - b] % mod * inv[b] % mod;
}
long long path(long long sx, long long sy, long long tx, long long ty) {
  return culC(ty - sy + tx - sx, tx - sx);
}
void solve() {
  for (int i = 0; i <= n; i++) {
    long long ans = 0;
    for (int j = 0; j < i; j++) {
      if (a[j].second <= a[i].second)
        ans += path(a[j].first, a[j].second, a[i].first, a[i].second) * dp[j] %
               mod,
            ans %= mod;
    }
    dp[i] = (path(1, 1, a[i].first, a[i].second) - ans) % mod + mod,
    dp[i] %= mod;
  }
}
int main() {
  init();
  cin >> h >> w >> n;
  for (int i = 0; i < n; i++) {
    int c, r;
    scanf(\"%d%d\", &r, &c);
    a[i].first = r;
    a[i].second = c;
  }
  sort(a, a + n);
  a[n] = pair<long long, long long>(h, w);
  solve();
  cout << dp[n] << endl;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
from bootcamp import Basebootcamp

# Global variables for precomputing factorials and their inverses
global_fac = [1]
global_inv = [1]
mod_value = 10**9 + 7

def init_global_fac_inv(maxn):
    global global_fac, global_inv, mod_value
    if maxn < len(global_fac):
        return
    current_len = len(global_fac)
    for i in range(current_len, maxn + 1):
        global_fac.append((global_fac[-1] * i) % mod_value)
        inv_i = pow(i, mod_value - 2, mod_value)
        new_inv = (global_inv[-1] * inv_i) % mod_value
        global_inv.append(new_inv)

def culC(a, b):
    if a < 0 or b < 0 or a < b:
        return 0
    init_global_fac_inv(a)
    return global_fac[a] * global_inv[b] % mod_value * global_inv[a - b] % mod_value

def path(sx, sy, tx, ty):
    dx = tx - sx
    dy = ty - sy
    if dx < 0 or dy < 0:
        return 0
    return culC(dx + dy, dx)

def compute_solution(h, w, blocks):
    mod = 10**9 + 7
    blocks_sorted = sorted(blocks, key=lambda x: (x[0], x[1]))
    blocks_sorted.append((h, w))
    n = len(blocks_sorted)
    dp = [0] * n

    for i in range(n):
        r, c = blocks_sorted[i]
        total = path(1, 1, r, c)
        for j in range(i):
            pr, pc = blocks_sorted[j]
            if pr <= r and pc <= c:
                ways = path(pr, pc, r, c) * dp[j]
                total = (total - ways) % mod
        dp[i] = total % mod
    return dp[-1]

class Egeraldandgiantchessbootcamp(Basebootcamp):
    def __init__(self, h_range=(1, 100), w_range=(1, 100), n_max=2000):
        self.h_range = h_range
        self.w_range = w_range
        self.n_max = n_max
    
    def case_generator(self):
        import random
        while True:
            h = random.randint(self.h_range[0], self.h_range[1])
            w = random.randint(self.w_range[0], self.w_range[1])
            if h * w >= 3:
                break
        
        max_black = h * w - 2
        n = random.randint(1, min(self.n_max, max_black))
        
        available = []
        for r in range(1, h + 1):
            for c in range(1, w + 1):
                if (r, c) != (1, 1) and (r, c) != (h, w):
                    available.append((r, c))
        
        selected = random.sample(available, n)
        selected_sorted = sorted(selected, key=lambda x: (x[0], x[1]))
        
        correct_answer = compute_solution(h, w, selected_sorted)
        
        return {
            'h': h,
            'w': w,
            'n': n,
            'black_cells': selected_sorted,
            'correct_answer': correct_answer
        }
    
    @staticmethod
    def prompt_func(question_case):
        h = question_case['h']
        w = question_case['w']
        n = question_case['n']
        black_cells = question_case['black_cells']
        
        input_lines = [f"{h} {w} {n}"] + [f"{r} {c}" for r, c in black_cells]
        input_str = '\n'.join(input_lines)
        
        example_output = "2"
        
        prompt = f"""You are playing Giant Chess and need to calculate the number of valid paths from the start to the end. The rules are as follows:

- The chessboard is {h} rows (h) by {w} columns (w). 
- Start at the top-left corner (1,1) and end at the bottom-right corner ({h},{w}).
- You can only move right or down one cell at a time.
- Black cells are blocked. The start and end cells are always white.
- Output the number of valid paths modulo 1e9+7.

Input format:
The first line contains three integers h, w, n. The next n lines each contain two integers r and c, representing the position of a black cell.

Given the following input data:

{input_str}

Please compute the correct answer. Ensure your final answer is enclosed within [answer] and [/answer] tags. For example: [answer]{example_output}[/answer]."""
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
        return solution == identity['correct_answer']
