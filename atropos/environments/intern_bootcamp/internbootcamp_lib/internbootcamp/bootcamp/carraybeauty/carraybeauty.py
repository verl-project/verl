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
const long long mod = 998244353;
const int N = 1005;
long long dp[N][N], tot[N][N], ans = 0, ret[N * N];
int n, a[N], k;
void solve(int x) {
  dp[0][0] = 1;
  tot[0][0] = 1;
  int now = 0;
  for (int i = 1; i <= n; ++i) {
    while ((a[i] - a[now]) >= x) ++now;
    for (int j = 0; j <= k; ++j) {
      if (j > 0) dp[i][j] = tot[now - 1][j - 1];
      tot[i][j] = (tot[i - 1][j] + dp[i][j]) % mod;
    }
    ret[x] = (ret[x] + dp[i][k]) % mod;
  }
}
int main() {
  scanf(\"%d%d\", &n, &k);
  for (int i = 1; i <= n; ++i) scanf(\"%d\", &a[i]);
  sort(a + 1, a + 1 + n);
  a[0] = -1000000000;
  for (int i = 1; i * (k - 1) <= a[n]; ++i) {
    solve(i);
    ans = (ans + ret[i]) % mod;
  }
  printf(\"%lld\n\", ans);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from itertools import combinations
from bootcamp import Basebootcamp

class Carraybeautybootcamp(Basebootcamp):
    def __init__(self, n_min=2, n_max=10, a_min=0, a_max=10**5):
        self.n_min = n_min
        self.n_max = n_max
        self.a_min = a_min
        self.a_max = a_max
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        k = random.randint(2, n)
        a = [random.randint(self.a_min, self.a_max) for _ in range(n)]
        a.sort()
        return {
            'n': n,
            'k': k,
            'a': a
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k = question_case['k']
        a = question_case['a']
        a_str = ' '.join(map(str, a))
        problem = f"""你需要解决以下谜题：

给定一个长度为{n}的数组，数组中的元素依次为{a_str}。同时给定一个整数k={k}。你的任务是计算所有长度为k的子序列的“美”之和，并将结果对998244353取模。

子序列的定义是通过删除原数组中的若干元素（可能不删除或全部删除）后得到的序列。注意，子序列中的元素必须保持其在原数组中的相对顺序。

一个数组的“美”定义为该数组中任意两个元素之间的最小绝对差。例如，数组[1, 3, 5]的“美”是2，因为3-1=2，5-3=2，而5-1=4，取最小值为2。

你的答案是所有长度为k的子序列的“美”的总和模998244353的值。请确保你的答案放在[answer]标签中，例如[answer]42[/answer]。"""
        return problem
    
    @staticmethod
    def extract_output(output):
        import re
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
        if solution is None:
            return False
        n = identity['n']
        k = identity['k']
        a = identity['a']
        mod = 998244353
        total = 0
        for indices in combinations(range(n), k):
            elements = [a[i] for i in indices]
            min_diff = float('inf')
            for i in range(k-1):
                diff = elements[i+1] - elements[i]
                if diff < min_diff:
                    min_diff = diff
            total = (total + min_diff) % mod
        return solution % mod == total % mod
