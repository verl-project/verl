"""# 

### 谜题描述
Once, during a lesson, Sasha got bored and decided to talk with his friends. Suddenly, he saw Kefa. Since we can talk endlessly about Kefa, we won't even start doing that. The conversation turned to graphs. Kefa promised Sasha to tell him about one interesting fact from graph theory if Sasha helps Kefa to count the number of beautiful trees. 

In this task, a tree is a weighted connected graph, consisting of n vertices and n-1 edges, and weights of edges are integers from 1 to m. Kefa determines the beauty of a tree as follows: he finds in the tree his two favorite vertices — vertices with numbers a and b, and counts the distance between them. The distance between two vertices x and y is the sum of weights of edges on the simple path from x to y. If the distance between two vertices a and b is equal to m, then the tree is beautiful.

Sasha likes graph theory, and even more, Sasha likes interesting facts, that's why he agreed to help Kefa. Luckily, Sasha is familiar with you the best programmer in Byteland. Help Sasha to count the number of beautiful trees for Kefa. Two trees are considered to be distinct if there is an edge that occurs in one of them and doesn't occur in the other one. Edge's weight matters.

Kefa warned Sasha, that there can be too many beautiful trees, so it will be enough to count the number modulo 10^9 + 7.

Input

The first line contains four integers n, m, a, b (2 ≤ n ≤ 10^6, 1 ≤ m ≤ 10^6, 1 ≤ a, b ≤ n, a ≠ b) — the number of vertices in the tree, the maximum weight of an edge and two Kefa's favorite vertices.

Output

Print one integer — the number of beautiful trees modulo 10^9+7.

Examples

Input


3 2 1 3


Output


5


Input


3 1 1 2


Output


2


Input


5 15 1 5


Output


345444

Note

There are 5 beautiful trees in the first example:

<image>

In the second example the following trees are beautiful:

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int maxn = 2e6 + 5;
const int mod = 1e9 + 7;
inline int add(int a, int b) {
  if ((a += b) >= mod) a -= mod;
  return a;
}
inline int mul(int a, int b) { return 1ll * a * b % mod; }
inline int qm(int a, int b) {
  int s = 1;
  while (b) {
    if (b & 1) s = mul(s, a);
    a = mul(a, a);
    b >>= 1;
  }
  return s;
}
int fn[maxn], fac[maxn], f[maxn], inv[maxn];
inline int C(int n, int m) { return mul(fac[n], mul(inv[m], inv[n - m])); }
inline int A(int n, int m) { return mul(fac[n], inv[n - m]); }
int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  int n, m, x;
  cin >> n >> m;
  cin >> x;
  cin >> x;
  int N = 1e6;
  fac[0] = 1;
  for (int i = (1); i < (N + 1); i++) fac[i] = mul(fac[i - 1], i);
  inv[N] = qm(fac[N], mod - 2);
  for (int i = N - 1; i >= 0; i--) inv[i] = mul(inv[i + 1], i + 1);
  int ans = 0;
  f[0] = 1;
  for (int i = (1); i < (N + 1); i++) f[i] = mul(f[i - 1], m);
  fn[0] = 1;
  for (int i = (1); i < (N + 1); i++) fn[i] = mul(fn[i - 1], n);
  for (int i = (1); i < (n); i++) {
    if (i > m) break;
    int s = mul(f[n - 1 - i], C(m - 1, i - 1));
    s = mul(s, A(n - 2, i - 1));
    if (i < n - 1) s = mul(s, mul(i + 1, fn[n - i - 2]));
    ans = add(ans, s);
  }
  cout << ans;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

MOD = 10**9 + 7

class Fsashaandinterestingfactfromgraphtheorybootcamp(Basebootcamp):
    def __init__(self, n_min=2, n_max=10, m_min=1, m_max=10):
        self.n_min = max(n_min, 2)  # n至少为2
        self.n_max = n_max
        self.m_min = max(m_min, 1)  # m至少为1
        self.m_max = m_max
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        m = random.randint(self.m_min, self.m_max)
        a = random.randint(1, n)
        b = a
        while b == a:
            b = random.randint(1, n)
        return {
            'n': n,
            'm': m,
            'a': a,
            'b': b
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        m = question_case['m']
        a = question_case['a']
        b = question_case['b']
        return f"""You are solving a graph theory problem. Compute the number of beautiful trees modulo {MOD}.

Problem Details:
- Tree properties: {n} vertices, edges have weights from 1 to {m}
- Beautiful tree condition: Distance between vertex {a} and {b} must be exactly {m}
- Distance is the sum of edge weights on the path between them

Output Requirements:
1. Answer must be an integer
2. Place your final answer between [answer] and [/answer] tags

Example Valid Response:
The number of beautiful trees is [answer]42[/answer]"""

    @staticmethod
    def extract_output(output):
        import re
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
        try:
            # 正确答案计算与a,b无关，源于参考代码特性
            n = identity['n']
            m = identity['m']
            correct = cls.compute_answer(n, m)
            return (int(solution) % MOD) == (correct % MOD)
        except:
            return False

    @staticmethod
    def compute_answer(n, m):
        mod = MOD
        if n < 2 or m < 1:
            return 0

        # 动态计算代替预先生成大数组
        def comb(n, k):
            if n < 0 or k < 0 or k > n:
                return 0
            numerator = 1
            for i in range(n, n-k, -1):
                numerator = numerator * i % mod
            denominator = 1
            for i in range(1, k+1):
                denominator = denominator * i % mod
            return numerator * pow(denominator, mod-2, mod) % mod

        def perm(n, k):
            if n < 0 or k < 0 or k > n:
                return 0
            res = 1
            for i in range(n, n-k, -1):
                res = res * i % mod
            return res

        ans = 0
        for i in range(1, n):
            if i > m:
                break

            c = comb(m-1, i-1)
            a_val = perm(n-2, i-1)
            f_val = pow(m, max(n-1-i, 0), mod)
            term = c * a_val % mod
            term = term * f_val % mod

            if i < n-1:
                term = term * (i+1) % mod
                term = term * pow(n, n-i-2, mod) % mod

            ans = (ans + term) % mod

        return ans
