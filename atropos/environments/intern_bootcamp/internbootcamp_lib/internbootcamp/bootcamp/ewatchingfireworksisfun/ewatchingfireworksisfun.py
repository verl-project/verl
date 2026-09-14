"""# 

### 谜题描述
A festival will be held in a town's main street. There are n sections in the main street. The sections are numbered 1 through n from left to right. The distance between each adjacent sections is 1.

In the festival m fireworks will be launched. The i-th (1 ≤ i ≤ m) launching is on time ti at section ai. If you are at section x (1 ≤ x ≤ n) at the time of i-th launching, you'll gain happiness value bi - |ai - x| (note that the happiness value might be a negative value).

You can move up to d length units in a unit time interval, but it's prohibited to go out of the main street. Also you can be in an arbitrary section at initial time moment (time equals to 1), and want to maximize the sum of happiness that can be gained from watching fireworks. Find the maximum total happiness.

Note that two or more fireworks can be launched at the same time.

Input

The first line contains three integers n, m, d (1 ≤ n ≤ 150000; 1 ≤ m ≤ 300; 1 ≤ d ≤ n).

Each of the next m lines contains integers ai, bi, ti (1 ≤ ai ≤ n; 1 ≤ bi ≤ 109; 1 ≤ ti ≤ 109). The i-th line contains description of the i-th launching.

It is guaranteed that the condition ti ≤ ti + 1 (1 ≤ i < m) will be satisfied.

Output

Print a single integer — the maximum sum of happiness that you can gain from watching all the fireworks.

Please, do not write the %lld specifier to read or write 64-bit integers in C++. It is preferred to use the cin, cout streams or the %I64d specifier.

Examples

Input

50 3 1
49 1 1
26 1 4
6 1 10


Output

-31


Input

10 2 1
1 1000 4
9 1000 4


Output

1992

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int a[301], t[301];
int dp[2][150001];
int sp[18][150001];
int n, m, d, ans = 1e9, tt;
long long vv = 0;
void build(int x) {
  for (int i = 1; i <= n; i++) sp[0][i] = dp[x][i];
  int temp = log(n + 0.0) / log(2.0);
  for (int i = 1; i <= temp; i++)
    for (int j = 1; j + (1 << i) - 1 <= n; j++)
      sp[i][j] = min(sp[i - 1][j], sp[i - 1][j + (1 << (i - 1))]);
}
int f2(int x, int y) {
  int temp = log(y - x + 1.0) / log(2.0);
  return min(sp[temp][x], sp[temp][y - (1 << temp) + 1]);
}
int main() {
  scanf(\"%d %d %d\", &n, &m, &d);
  for (int i = 1; i <= m; i++) scanf(\"%d %d %d\", &a[i], &tt, &t[i]), vv += tt;
  for (int i = 1; i <= n; i++) dp[1][i] = abs(a[1] - i);
  for (int i = 2; i <= m; i++) {
    build((i - 1) & 1);
    int tt = min((long long)n, (long long)d * (t[i] - t[i - 1]));
    for (int j = 1; j <= n; j++)
      dp[i & 1][j] = f2(max(1, j - tt), min(n, j + tt)) + abs(a[i] - j);
  }
  printf(\"%I64d\n\", vv - *min_element(dp[m & 1] + 1, dp[m & 1] + n + 1));
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import math
from bootcamp import Basebootcamp

def build_sparse_table(arr, n):
    log_table = [0] * (n + 1)
    for i in range(2, n + 1):
        log_table[i] = log_table[i // 2] + 1
    k_max = log_table[n] + 1
    st = [[0] * (n + 1) for _ in range(k_max)]
    for i in range(1, n + 1):
        st[0][i] = arr[i]
    for j in range(1, k_max):
        for i in range(1, n + 1 - (1 << j) + 1):
            st[j][i] = min(st[j-1][i], st[j-1][i + (1 << (j-1))])
    return st, log_table

def query_min(st, log_table, l, r):
    length = r - l + 1
    k = log_table[length]
    return min(st[k][l], st[k][r - (1 << k) + 1])

def calculate_answer(n, m, d, fireworks):
    sum_bi = sum(b for a, b, t in fireworks)
    a_list = [a for a, b, t in fireworks]
    t_list = [t for a, b, t in fireworks]
    
    prev_dp = [0] * (n + 2)
    a1 = a_list[0]
    for j in range(1, n + 1):
        prev_dp[j] = abs(a1 - j)
    
    for i in range(1, m):
        ai = a_list[i]
        ti = t_list[i]
        delta_t = ti - t_list[i-1]
        tt = d * delta_t
        tt = min(tt, n)
        
        st, log_table = build_sparse_table(prev_dp, n)
        curr_dp = [0] * (n + 2)
        
        for j in range(1, n + 1):
            left = max(1, j - tt)
            right = min(n, j + tt)
            if left > right:
                curr_dp[j] = float('inf')
            else:
                min_prev = query_min(st, log_table, left, right)
                curr_dp[j] = min_prev + abs(ai - j)
        
        prev_dp, curr_dp = curr_dp, prev_dp
    
    min_final = min(prev_dp[j] for j in range(1, n + 1))
    return sum_bi - min_final

class Ewatchingfireworksisfunbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = params
        self.n = params.get('n', 50)
        self.m = params.get('m', 3)
        self.d = params.get('d', 1)
    
    def case_generator(self):
        n = random.randint(10, 50)
        m = random.randint(2, 5)
        d = random.randint(1, 5)
        fireworks = []
        t = 1
        for _ in range(m):
            ai = random.randint(1, n)
            bi = random.randint(1, 100)
            increment = random.randint(0, 5)
            t += increment
            fireworks.append((ai, bi, t))
        fireworks.sort(key=lambda x: x[2])  # Ensure ti is non-decreasing
        
        try:
            correct_answer = calculate_answer(n, m, d, fireworks)
        except:
            return self.case_generator()
        
        return {
            'n': n,
            'm': m,
            'd': d,
            'fireworks': fireworks,
            'correct_answer': correct_answer
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        d = question_case['d']
        fireworks = question_case['fireworks']
        input_lines = [f"{n} {m} {d}"]
        for a, b, t in fireworks:
            input_lines.append(f"{a} {b} {t}")
        input_str = '\n'.join(input_lines)
        
        return f"""You are participating in a festival on a main street with {n} sections. Ewatchingfireworksisfun will be launched at specific times and sections. Your goal is to maximize the total happiness by positioning yourself optimally.

Rules:
1. You can move up to {d} units per time unit.
2. At each firework launch (time t_i), your happiness is calculated as (b_i - |a_i - x|), where x is your current section.
3. You start at any section at time 1.

Input format:
{input_str}

Calculate the maximum possible total happiness. Output the answer inside [answer] and [/answer] tags. For example: [answer]42[/answer]."""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity.get('correct_answer')
