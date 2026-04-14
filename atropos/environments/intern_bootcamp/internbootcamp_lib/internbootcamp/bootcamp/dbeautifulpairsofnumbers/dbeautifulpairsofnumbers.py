"""# 

### 谜题描述
The sequence of integer pairs (a1, b1), (a2, b2), ..., (ak, bk) is beautiful, if the following statements are fulfilled: 

  * 1 ≤ a1 ≤ b1 < a2 ≤ b2 < ... < ak ≤ bk ≤ n, where n is a given positive integer; 
  * all numbers b1 - a1, b2 - a2, ..., bk - ak are distinct. 



For the given number n find the number of beautiful sequences of length k. As the answer can be rather large, print the remainder after dividing it by 1000000007 (109 + 7).

Input

The first line contains integer t (1 ≤ t ≤ 2·105) — the number of the test data.

Each of the next t lines contains two integers n and k (1 ≤ k ≤ n ≤ 1000).

Output

For each test from the input print the answer to the problem modulo 1000000007 (109 + 7). Print the answers to the tests in the order in which the tests are given in the input.

Examples

Input

6
1 1
2 1
2 2
3 1
3 2
3 3


Output

1
3
0
6
2
0

Note

In the first test sample there is exactly one beautiful sequence: (1, 1).

In the second test sample, the following sequences are beautiful: 

  * (1, 1); 
  * (1, 2); 
  * (2, 2). 



In the fourth test sample, the following sequences are beautiful: 

  * (1, 1); 
  * (1, 2); 
  * (1, 3); 
  * (2, 2); 
  * (2, 3); 
  * (3, 3). 



In the fifth test sample, the following sequences are beautiful: 

  * (1, 1), (2, 3); 
  * (1, 2), (3, 3). 



In the third and sixth samples, there are no beautiful sequences.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const double pi = acos(-1);
const double eps = 1e-10;
const int inf = 0x3f3f3f3f;
const long long infLL = 0x3f3f3f3f3f3f3f3fLL;
const int maxn = 2000 + 5;
const long long mod = 1000000007;
long long fac[maxn], ifac[maxn];
void gcd(long long a, long long b, long long &d, long long &x0, long long &y0) {
  if (!b) {
    d = a;
    x0 = 1;
    y0 = 0;
  } else {
    gcd(b, a % b, d, y0, x0);
    y0 -= x0 * (a / b);
  }
}
long long inv(long long a, long long m = mod) {
  long long d, x, y;
  gcd(a, m, d, x, y);
  return d == 1 ? (x + m) % m : -1;
}
void mk_fac() {
  fac[0] = 1;
  for (int i = 1; i < maxn; ++i) fac[i] = fac[i - 1] * i % mod;
  for (int i = 0; i < maxn; ++i) ifac[i] = inv(fac[i]);
}
long long A(int n, int m) {
  if (!fac[0]) mk_fac();
  return fac[n] * ifac[n - m] % mod;
}
long long C(int n, int m) {
  if (!fac[0]) mk_fac();
  return fac[n] * ifac[n - m] % mod * ifac[m] % mod;
}
long long s[maxn];
long long f[maxn][maxn];
void init() {
  s[0] = 0;
  for (int i = 1; i < maxn; ++i) s[i] = s[i - 1] + i;
  for (int i = 1; i < maxn; ++i) f[i][1] = 1;
  for (int j = 2; j < maxn; ++j) {
    if (s[j] >= maxn) break;
    f[s[j]][j] = fac[j];
    for (int i = s[j] + 1; i < maxn; ++i) {
      f[i][j] += f[i - j][j] + f[i - j][j - 1] * j % mod;
      f[i][j] %= mod;
    }
  }
}
int main() {
  ios::sync_with_stdio(false);
  mk_fac();
  init();
  int T;
  cin >> T;
  while (T--) {
    int n, k;
    cin >> n >> k;
    --n;
    long long res = 0;
    for (int i = s[k - 1]; i <= n; ++i) {
      int t = n - i - (k - 1);
      if (t < 0) break;
      res += f[i + k][k] * C(k + t, t) % mod;
      res %= mod;
    }
    cout << res << endl;
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Dbeautifulpairsofnumbersbootcamp(Basebootcamp):
    mod = 10**9 + 7
    maxn = 2005
    fac = []
    ifac = []
    s = []
    f = []
    initialized = False

    def __init__(self, max_n=1000, min_n=1, **kwargs):
        super().__init__(**kwargs)
        self.max_n = max_n
        self.min_n = min_n
        self.initialize_data()

    @classmethod
    def initialize_data(cls):
        if cls.initialized:
            return
        # Precompute factorial and inverse factorial arrays
        cls.fac = [1] * cls.maxn
        for i in range(1, cls.maxn):
            cls.fac[i] = cls.fac[i-1] * i % cls.mod

        cls.ifac = [1] * cls.maxn
        cls.ifac[cls.maxn - 1] = pow(cls.fac[cls.maxn - 1], cls.mod - 2, cls.mod)
        for i in range(cls.maxn - 2, -1, -1):
            cls.ifac[i] = cls.ifac[i + 1] * (i + 1) % cls.mod

        # Precompute s array
        cls.s = [0] * cls.maxn
        for i in range(1, cls.maxn):
            cls.s[i] = cls.s[i-1] + i

        # Initialize f array using dynamic programming
        cls.f = [[0] * cls.maxn for _ in range(cls.maxn)]
        for i in range(1, cls.maxn):
            cls.f[i][1] = 1

        for j in range(2, cls.maxn):
            if cls.s[j] >= cls.maxn:
                break
            if cls.s[j] < cls.maxn:
                cls.f[cls.s[j]][j] = cls.fac[j] % cls.mod
            for i in range(cls.s[j] + 1, cls.maxn):
                prev_i = i - j
                if prev_i >= 0:
                    term1 = cls.f[prev_i][j]
                    term2 = (cls.f[prev_i][j-1] * j) % cls.mod
                    cls.f[i][j] = (term1 + term2) % cls.mod

        cls.initialized = True

    @classmethod
    def compute_answer(cls, n, k):
        if k < 1 or k > n:
            return 0
        new_n = n - 1
        res = 0
        s_k_1 = cls.s[k-1]
        for i in range(s_k_1, new_n + 1):
            t = new_n - i - (k - 1)
            if t < 0:
                break
            comb = cls.C(k + t, t)
            if (i + k) >= cls.maxn or k >= cls.maxn:
                f_val = 0
            else:
                f_val = cls.f[i + k][k]
            res = (res + f_val * comb) % cls.mod
        return res

    @classmethod
    def C(cls, n, m):
        if m < 0 or m > n:
            return 0
        return cls.fac[n] * cls.ifac[m] % cls.mod * cls.ifac[n - m] % cls.mod

    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        k = random.randint(1, n)
        correct_answer = self.compute_answer(n, k)
        return {
            'n': n,
            'k': k,
            'correct_answer': correct_answer
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k = question_case['k']
        prompt = f"""你是一个算法竞赛选手，现在需要解决一个数学谜题。请仔细阅读以下问题描述，并输出你的答案。

问题描述:
给定一个正整数n和k，计算满足条件的“美丽序列”的数量。答案需要对1e9+7取模。

美丽序列的定义:
- 序列由k个整数对组成：(a1, b1), (a2, b2), ..., (ak, bk)。
- 满足以下两个条件：
  1. 所有整数对严格递增且互不重叠，即1 ≤ a1 ≤ b1 < a2 ≤ b2 < ... < ak ≤ bk ≤ n。
  2. 每个整数对的差（即bi - ai）互不相同。

输入要求:
- n的值为{n}，k的值为{k}。

输出要求:
- 输出满足条件的美丽序列的数量模1000000007的结果。

请将最终答案放在[answer]和[/answer]的标签之间。例如：[answer]42[/answer]。"""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
