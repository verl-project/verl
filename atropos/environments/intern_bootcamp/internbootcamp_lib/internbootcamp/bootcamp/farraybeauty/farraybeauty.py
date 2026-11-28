"""# 

### 谜题描述
Let's call beauty of an array b_1, b_2, …, b_n (n > 1) — min_{1 ≤ i < j ≤ n} |b_i - b_j|.

You're given an array a_1, a_2, … a_n and a number k. Calculate the sum of beauty over all subsequences of the array of length exactly k. As this number can be very large, output it modulo 998244353.

A sequence a is a subsequence of an array b if a can be obtained from b by deletion of several (possibly, zero or all) elements.

Input

The first line contains integers n, k (2 ≤ k ≤ n ≤ 1000).

The second line contains n integers a_1, a_2, …, a_n (0 ≤ a_i ≤ 10^5).

Output

Output one integer — the sum of beauty over all subsequences of the array of length exactly k. As this number can be very large, output it modulo 998244353.

Examples

Input


4 3
1 7 3 5


Output


8

Input


5 5
1 10 100 1000 10000


Output


9

Note

In the first example, there are 4 subsequences of length 3 — [1, 7, 3], [1, 3, 5], [7, 3, 5], [1, 7, 5], each of which has beauty 2, so answer is 8.

In the second example, there is only one subsequence of length 5 — the whole array, which has the beauty equal to |10-1| = 9.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
template <typename T>
ostream& operator<<(ostream& os, const vector<T>& v) {
  for (auto i : v) os << i << \" \";
  return os;
}
template <typename T>
ostream& operator<<(ostream& os, const set<T>& v) {
  for (auto i : v) os << i << \" \";
  return os;
}
const int M = 1e5 + 5;
const int N = 1000 + 80;
const int MOD = 998244353;
const int INF = 1e9 + 7;
int dp[N][N], Prev[N][N];
int a[N], l[N];
int n, k;
int mod(int x) {
  while (x < 0) x += MOD;
  while (x >= MOD) x -= MOD;
  return x;
}
void output() {
  printf(\"dp::\n\");
  for (int i = 1; i <= k; ++i) {
    for (int j = 1; j <= n; ++j) printf(\"%d \", dp[i][j]);
    printf(\"\n\");
  }
  printf(\"\nPrev::\n\");
  for (int i = 1; i <= k; ++i) {
    for (int j = 1; j <= n; ++j) printf(\"%d \", Prev[i][j]);
    printf(\"\n\");
  }
  printf(\"\n\");
}
int main() {
  scanf(\"%d%d\", &n, &k);
  a[0] = -INF;
  for (int i = 1; i <= n; ++i) scanf(\"%d\", a + i);
  sort(a, a + n + 1);
  int max_a = a[n];
  int max_x = max_a / (k - 1);
  int ans = 0;
  for (int x = 1; x * (k - 1) < M; ++x) {
    for (int i = 0; i <= k; ++i)
      for (int j = 0; j <= n; ++j) dp[i][j] = 0;
    l[0] = 0;
    for (int i = 1; i <= n; ++i) {
      l[i] = l[i - 1];
      while (a[i] - a[l[i] + 1] >= x) ++l[i];
    }
    dp[0][0] = 1;
    for (int i = 0; i + 1 <= k; ++i) {
      Prev[i][0] = dp[i][0];
      for (int j = 1; j <= n; ++j)
        Prev[i][j] = (Prev[i][j - 1] + dp[i][j]) % MOD;
      for (int j = 1; j <= n; ++j) dp[i + 1][j] = Prev[i][l[j]];
    }
    int res = 0;
    for (int i = 1; i <= n; ++i) res = (res + dp[k][i]) % MOD;
    ans += res;
    ans %= MOD;
  }
  printf(\"%d\n\", ans);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bisect import bisect_right

MOD = 998244353
INF = 10**18

class Farraybeautybootcamp(Basebootcamp):
    def __init__(self, min_n=5, max_n=10, min_k=3, min_a=0, max_a=100):
        """
        调整默认参数以确保验证效率：
        - 限制数组长度范围5-10
        - 限制元素数值范围0-100
        - 要求k至少为3以控制循环次数
        """
        self.min_n = min_n
        self.max_n = max_n
        self.min_k = min_k
        self.min_a = min_a
        self.max_a = max_a
        
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        k = random.randint(max(self.min_k, 3), n)  # 保证k至少为3减少循环次数
        a = [random.randint(self.min_a, self.max_a) for _ in range(n)]
        return {
            "n": n,
            "k": k,
            "a": a.copy()
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        problem = f"""You are given an array of integers and need to calculate the sum of beauty values for all subsequences of length exactly k. The result must be modulo 998244353.

Problem Description:
- Original array elements: {', '.join(map(str, question_case['a']))}
- Array length (n): {question_case['n']}
- Subsequence length (k): {question_case['k']}

Rules:
1. A subsequence is formed by deleting elements while maintaining order
2. Beauty is the minimum absolute difference between any two elements in the subsequence
3. Consider ALL possible subsequences of length EXACTLY k
4. Output the sum modulo 998244353

Note: The array elements may not be sorted. The minimal valid k is 3.

Format your final numerical answer within [answer] and [/answer] tags."""

        return problem
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return solution == cls.compute_answer(
                identity['n'],
                identity['k'],
                identity['a']
            )
        except:
            return False
    
    @staticmethod
    def compute_answer(n, k, original_a):
        """优化后的计算逻辑，添加了提前终止条件和范围优化"""
        if k < 2:
            return 0
        
        sorted_a = sorted(original_a)
        max_diff = sorted_a[-1] - sorted_a[0]
        max_x = max_diff // (k-1) if k > 1 else 0
        
        # 调整循环范围为实际可能的最小值
        M = min(10**5 + 5, max_x + 2) if max_x else 10**5 + 5
        a = [-INF] + sorted_a
        ans = 0

        for x in range(1, M + 1):
            if x * (k-1) > M:
                break
            
            # 预处理指针数组
            l = [0]*(n+1)
            for i in range(1, n+1):
                target = a[i] - x
                l[i] = bisect_right(a, target, 0, i) - 1
                l[i] = max(l[i], l[i-1])

            # 动态规划部分
            dp = [[0]*(n+1) for _ in range(k+1)]
            dp[0][0] = 1
            
            for i in range(k):
                prefix = [0]*(n+1)
                prefix[0] = dp[i][0]
                for j in range(1, n+1):
                    prefix[j] = (prefix[j-1] + dp[i][j]) % MOD
                
                for j in range(1, n+1):
                    if l[j] >= 0:
                        dp[i+1][j] = prefix[l[j]] % MOD

            res = sum(dp[k][j] for j in range(1, n+1)) % MOD
            ans = (ans + res) % MOD
        
        return ans
