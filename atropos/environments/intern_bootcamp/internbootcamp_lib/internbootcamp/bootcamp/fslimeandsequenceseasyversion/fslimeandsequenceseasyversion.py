"""# 

### 谜题描述
Note that the only differences between easy and hard versions are the constraints on n and the time limit. You can make hacks only if all versions are solved.

Slime is interested in sequences. He defined good positive integer sequences p of length n as follows:

  * For each k>1 that presents in p, there should be at least one pair of indices i,j, such that 1 ≤ i < j ≤ n, p_i = k - 1 and p_j = k.



For the given integer n, the set of all good sequences of length n is s_n. For the fixed integer k and the sequence p, let f_p(k) be the number of times that k appears in p. For each k from 1 to n, Slime wants to know the following value:

$$$\left(∑_{p∈ s_n} f_p(k)\right)\ mod\ 998 244 353$$$

Input

The first line contains one integer n\ (1≤ n≤ 5000).

Output

Print n integers, the i-th of them should be equal to \left(∑_{p∈ s_n} f_p(i)\right)\ mod\ 998 244 353.

Examples

Input


2


Output


3 1 


Input


3


Output


10 7 1 


Input


1


Output


1 

Note

In the first example, s=\{[1,1],[1,2]\}.

In the second example, s=\{[1,1,1],[1,1,2],[1,2,1],[1,2,2],[2,1,2],[1,2,3]\}.

In the third example, s=\{[1]\}.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
inline bool cmin(int &x, int y) { return x > y ? x = y, 1 : 0; }
inline bool cmax(int &x, int y) { return x < y ? x = y, 1 : 0; }
const int _ = 5055, mod = 998244353;
int N, fac[_], inv[_];
int f[_][_];
int ans[_];
inline int dec(int x) { return x >= mod ? x - mod : x; }
inline int Pow(int x, int y = mod - 2) {
  int ans(1);
  for (; y; y >>= 1, x = (long long)x * x % mod)
    if (y & 1) ans = (long long)ans * x % mod;
  return ans;
}
int main() {
  cin >> N;
  fac[0] = fac[1] = inv[0] = 1;
  for (int i = 2, I = N; i <= I; ++i) fac[i] = 1ll * fac[i - 1] * i % mod;
  inv[N] = Pow(fac[N]);
  for (int i = N, I = 2; i >= I; --i) inv[i - 1] = 1ll * inv[i] * i % mod;
  f[0][0] = 1;
  for (int i = 1, I = N; i <= I; ++i)
    for (int j = 1, I = i; j <= I; ++j)
      f[i][j] =
          (1ll * f[i - 1][j] * j + 1ll * f[i - 1][j - 1] * (i - j + 1)) % mod;
  for (int i = 1, I = N; i <= I; ++i)
    for (int j = 1, I = N; j <= I; ++j)
      ans[j] = (ans[j] + 1ll * f[i][j] * inv[i]) % mod;
  for (int i = 1, I = N; i <= I; ++i)
    cout << 1ll * ans[i] * fac[N] % mod << \" \";
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
from bootcamp import Basebootcamp

MOD = 998244353

class Fslimeandsequenceseasyversionbootcamp(Basebootcamp):
    def __init__(self, max_n=100):  # 默认限制测试用例规模
        """
        限制默认max_n以保证生成效率，用户可自行设置更大值
        """
        self.max_n = min(max_n, 5000)  # 强制不超过题目要求的5000上限
    
    def case_generator(self):
        import random
        n = random.randint(1, self.max_n)
        expected = self.compute_answer(n)
        return {'n': n, 'expected': expected}
    
    @staticmethod
    def compute_answer(n):
        mod = MOD
        if n == 0:
            return []
        
        # 预计算阶乘和逆阶乘
        fac = [1]*(n+1)
        for i in range(1, n+1):
            fac[i] = fac[i-1] * i % mod
            
        inv_fac = [1]*(n+1)
        inv_fac[n] = pow(fac[n], mod-2, mod)
        for i in range(n-1, -1, -1):
            inv_fac[i] = inv_fac[i+1] * (i+1) % mod
        
        # 使用滚动数组优化空间
        ans = [0]*(n+2)
        prev = [0]*(n+2)
        prev[0] = 1
        
        for i in range(1, n+1):
            curr = [0]*(n+2)
            for j in range(1, i+1):
                term1 = prev[j] * j % mod
                term2 = prev[j-1] * (i-j+1) % mod
                curr[j] = (term1 + term2) % mod
            
            # 实时累加结果，避免存储全表
            for j in range(1, i+1):
                ans[j] = (ans[j] + curr[j] * inv_fac[i]) % mod
            
            prev = curr  # 滚动更新
        
        # 最终调整
        return [ans[j] * fac[n] % mod for j in range(1, n+1)]
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        return f"""Given n = {n}, compute the sum of occurrences for each k (1-{n}) in all valid sequences. Enclose your answer in [answer][/answer] tags.

Sample format:
n=2 → [answer]3 1[/answer]
n=3 → [answer]10 7 1[/answer]

Rules:
1. For any k>1 in sequence, must have earlier k-1
2. Output n space-separated values mod 998244353
3. Numbers must be in order from k=1 to k={n}

Your answer for n={n}:"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        
        last = matches[-1].strip()
        try:
            return list(map(int, last.split()))
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity['expected']
        return solution == expected

# 测试样例验证
if __name__ == "__main__":
    # 验证n=2的示例
    bootcamp = Fslimeandsequenceseasyversionbootcamp()
    test_case = {'n': 2}
    assert bootcamp.compute_answer(2) == [3, 1], "Sample 2 failed"
    
    # 验证n=3的示例
    test_case = {'n': 3}
    assert bootcamp.compute_answer(3) == [10, 7, 1], "Sample 3 failed"
    
    print("All sample tests passed!")
