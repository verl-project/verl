"""# 

### 谜题描述
Levko loves strings of length n, consisting of lowercase English letters, very much. He has one such string s. For each string t of length n, Levko defines its beauty relative to s as the number of pairs of indexes i, j (1 ≤ i ≤ j ≤ n), such that substring t[i..j] is lexicographically larger than substring s[i..j].

The boy wondered how many strings t are there, such that their beauty relative to s equals exactly k. Help him, find the remainder after division this number by 1000000007 (109 + 7).

A substring s[i..j] of string s = s1s2... sn is string sisi + 1... sj.

String x = x1x2... xp is lexicographically larger than string y = y1y2... yp, if there is such number r (r < p), that x1 = y1, x2 = y2, ... , xr = yr and xr + 1 > yr + 1. The string characters are compared by their ASCII codes.

Input

The first line contains two integers n and k (1 ≤ n ≤ 2000, 0 ≤ k ≤ 2000).

The second line contains a non-empty string s of length n. String s consists only of lowercase English letters. 

Output

Print a single number — the answer to the problem modulo 1000000007 (109 + 7).

Examples

Input

2 2
yz


Output

26


Input

2 3
yx


Output

2


Input

4 7
abcd


Output

21962

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
inline int read() {
  int x = 0, f = 1;
  char ch = getchar();
  for (; !isdigit(ch); ch = getchar())
    if (ch == '-') f = -1;
  for (; isdigit(ch); ch = getchar()) x = x * 10 + ch - '0';
  return x * f;
}
const int N = 2005, mod = 1e9 + 7;
int dp[N][N], sum[N];
char s[N];
void add(int &x, int y) {
  x += y;
  if (x >= mod) x -= mod;
}
int main() {
  int n = read(), k = read();
  scanf(\"%s\", s + 1);
  sum[0] = dp[0][0] = 1;
  for (int i = 1; i <= n; ++i) {
    for (int j = 0; j <= k; ++j) {
      for (int l = i - 1; l >= 0 && (i - l) * (n - i + 1) <= j; --l)
        add(dp[i][j], dp[l][j - (i - l) * (n - i + 1)]);
      dp[i][j] = 1ll * ('z' - s[i]) * dp[i][j] % mod;
      add(dp[i][j], 1ll * sum[j] * (s[i] - 'a') % mod);
      add(sum[j], dp[i][j]);
    }
  }
  int ans = 0;
  for (int i = 0; i <= n; ++i) add(ans, dp[i][k]);
  cout << ans;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp

MOD = 10**9 + 7

def compute_answer(n, k, s):
    # 保持原有高效实现不变
    dp = [[0] * (k+1) for _ in range(n+1)]
    sum_ = [0] * (k+1)
    dp[0][0] = 1
    sum_[0] = 1
    
    for i in range(1, n+1):
        new_dp = [0] * (k+1)
        current_char = ord(s[i-1])
        
        for j in range(k+1):
            # 计算第一个部分
            part1 = 0
            for l in range(i-1, -1, -1):
                cost = (i - l) * (n - i + 1)
                if cost > j:
                    break
                part1 = (part1 + dp[l][j - cost]) % MOD
            
            # 计算第二个部分
            part1 = part1 * (ord('z') - current_char) % MOD
            part2 = sum_[j] * (current_char - ord('a')) % MOD
            
            new_dp[j] = (part1 + part2) % MOD
        
        # 更新前缀和数组
        for j in range(k+1):
            sum_[j] = (sum_[j] + new_dp[j]) % MOD
            dp[i][j] = new_dp[j]
    
    return sum(dp[i][k] for i in range(n+1)) % MOD

class Elevkoandstringsbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n_min = params.get('n_min', 1)
        self.n_max = params.get('n_max', 5)  # 调整默认范围
        self.k_max = params.get('k_max', 2000)
        
    def case_generator(self):
        import random
        
        # 生成有效字符串参数
        n = random.randint(self.n_min, self.n_max)
        s = ''.join(
            random.choice('abcdefghijklmnopqrstuvwxyz')
            for _ in range(n)
        )
        
        # 计算最大有效k值
        max_possible_k = n * (n + 1) // 2
        k_max = min(max_possible_k, self.k_max)
        k = random.randint(0, k_max)
        
        return {
            'n': n,
            'k': k,
            's': s,
            'correct_answer': compute_answer(n, k, s)
        }
    
    @staticmethod
    def prompt_func(question_case):
        return (
            "Elevkoandstrings's String Puzzle\n"
            "Given a string s of length n, find the number of strings t with the same length such that:\n"
            "- The beauty of t relative to s equals exactly k\n"
            "- Beauty is defined as the number of pairs (i,j) where 1 ≤ i ≤ j ≤ n and t[i..j] > s[i..j] lexicographically\n\n"
            "Input Format:\n"
            "First line: n k\n"
            "Second line: s\n\n"
            "Problem Instance:\n"
            f"{question_case['n']} {question_case['k']}\n"
            f"{question_case['s']}\n\n"
            "Compute the answer modulo 1e9+7 and put within [answer][/answer] tags."
        )
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(
            r'\[answer\][\s]*?(?P<answer>\d+)[\s]*?\[/answer\]',
            output,
            re.DOTALL | re.IGNORECASE
        )
        
        if matches:
            try:
                return int(matches[-1]) % MOD
            except ValueError:
                pass
        return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
