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
template <class T>
string tostring(T x) {
  ostringstream out;
  out << x;
  return out.str();
}
long long toint(string s) {
  istringstream in(s);
  long long x;
  in >> x;
  return x;
}
int dx[8] = {0, 0, 1, -1, 1, 1, -1, -1};
int dy[8] = {1, -1, 0, 0, -1, 1, -1, 1};
int kx[8] = {1, 1, -1, -1, 2, 2, -2, -2};
int ky[8] = {2, -2, 2, -2, 1, -1, 1, -1};
long long dp[2222][2222];
char buf[2222];
string s;
long long sum1[2222];
long long sum2[2222][2222];
bool used[2222];
int main() {
  int n, k;
  scanf(\"%d%d\", &n, &k);
  scanf(\"%s\", buf);
  s = string(buf);
  dp[n][0] = 1;
  for (int i = n - 1; i >= 0; i--) {
    for (int j = 0; j <= k; j++) {
      memset(used, false, sizeof(used));
      dp[i][j] = (s[i] - 'a') * dp[i + 1][j];
      if (j - (n - i) >= 0) dp[i][j] += ('z' - s[i]) * dp[i + 1][j - (n - i)];
      for (int l = n - 1; l > i; l--) {
        used[l] = true;
        if ((n - l) * (l - i + 1) <= j)
          dp[i][j] += ('z' - s[l]) * dp[l + 1][j - (n - l) * (l - i + 1)];
        else
          break;
      }
      for (int l = i + 1; l < n; l++) {
        if (used[l]) break;
        if ((n - l) * (l - i + 1) <= j)
          dp[i][j] += ('z' - s[l]) * dp[l + 1][j - (n - l) * (l - i + 1)];
        else
          break;
      }
      dp[i][j] += sum1[j];
      if (j == 0) dp[i][j]++;
      dp[i][j] %= 1000000007;
      sum1[j] = (sum1[j] + (s[i] - 'a') * dp[i + 1][j]) % 1000000007;
    }
  }
  cout << dp[0][k] << endl;
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

class Clevkoandstringsbootcamp(Basebootcamp):
    def __init__(self, max_n=8, max_k=50):
        self.max_n = max_n
        self.max_k = max_k  # 允许生成更大的k值
    
    def case_generator(self):
        n = random.randint(1, self.max_n)
        max_possible = n * (n + 1) // 2
        # 允许k值覆盖更多边界条件
        k = random.choice([
            0,
            random.randint(1, min(self.max_k, max_possible)),
            min(self.max_k, max_possible)
        ])
        s = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=n))
        return {
            'n': n,
            'k': k,
            's': s,
            'correct_answer': self.compute_answer(n, k, s)
        }
    
    @staticmethod
    def compute_answer(n, k, s):
        if k > n*(n+1)//2 or k < 0:
            return 0
        MAX_K = 2000
        k = min(k, MAX_K)
        
        dp = [[0]*(MAX_K+1) for _ in range(n+1)]
        sum1 = [0]*(MAX_K+1)
        dp[n][0] = 1
        
        for i in range(n-1, -1, -1):
            new_sum1 = [0]*(MAX_K+1)
            for j in range(MAX_K, -1, -1):
                current = 0
                
                # Case 1: t[i] < s[i]
                if j <= MAX_K:
                    current += (ord(s[i]) - ord('a')) * dp[i+1][j]
                
                # Case 2: t[i] > s[i]
                delta = n - i
                if delta <= j <= MAX_K:
                    current += (ord('z') - ord(s[i])) * dp[i+1][j - delta]
                
                # Case 3: Find first differing position
                used = [False]*(n+1)
                # 处理降序
                for l in range(n-1, i, -1):
                    used[l] = True
                    cnt = (n - l) * (l - i + 1)
                    if cnt > j:
                        break
                    rem = j - cnt
                    if 0 <= rem <= MAX_K:
                        current += (ord('z') - ord(s[l])) * dp[l+1][rem]
                
                # 处理升序
                for l in range(i+1, n):
                    if used[l]:
                        break
                    cnt = (n - l) * (l - i + 1)
                    if cnt > j:
                        break
                    rem = j - cnt
                    if 0 <= rem <= MAX_K:
                        current += (ord('z') - ord(s[l])) * dp[l+1][rem]
                
                # Add sum from previous steps
                current += sum1[j]
                if j == 0:
                    current += 1  # 全匹配的情况
                
                dp[i][j] = current % MOD
                # 更新sum1
                new_sum1[j] = (sum1[j] + (ord(s[i]) - ord('a')) * dp[i+1][j]) % MOD
            
            sum1 = new_sum1
        
        return dp[0][k] % MOD

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k = question_case['k']
        s = question_case['s']
        return f"""解决以下编程问题，将答案放入[answer]标签：

给定n={n}, k={k}，字符串s="{s}"，计算满足美丽值为k的字符串t的个数（模10^9+7）。

规则：
1. 美丽值定义为满足t[i..j] > s[i..j]的(i,j)对的数量
2. 比较按字典序进行
3. 结果需要对1000000007取模

示例：
输入：
2 2
yz
输出：
26

你的答案应为一个整数，放在[answer][/answer]标签中。例如：[answer]26[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
