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
const long long MOD = (long long)1e9 + 7;
const int N = 1e6 + 5, inf = 1e9 + 5;
long long add(long long x, long long y) {
  x += y;
  if (x >= MOD) return x - MOD;
  return x;
}
long long sub(long long x, long long y) {
  x -= y;
  if (x < 0) return x + MOD;
  return x;
}
long long mult(long long x, long long y) { return (x * y) % MOD; }
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
long long mod_pow(long long x, long long e) {
  long long ans = 1;
  while (e > 0) {
    if (e & 1) ans = mult(ans, x);
    x = mult(x, x);
    e >>= 1;
  }
  return ans;
}
long long fact[N], inv_fact[N];
void pre_fact() {
  fact[0] = inv_fact[0] = 1;
  for (long long i = 1; i <= N; ++i) {
    fact[i] = mult(fact[i - 1], i);
    inv_fact[i] = mod_pow(fact[i], MOD - 2);
  }
}
long long binom(long long n, long long k) {
  if (k == 0) return 1;
  return mult(fact[n], mult(inv_fact[k], inv_fact[n - k]));
}
long long cayley(long long n, long long k) {
  if (n - k - 1 < 0) return 1;
  return mult(k, mod_pow(n, n - k - 1));
}
int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  long long n, m, a, b;
  cin >> n >> m >> a >> b;
  long long ans = 0;
  pre_fact();
  for (long long k = 1; k < min(n, m + 1); ++k) {
    long long curr = 1;
    curr = mult(curr, binom(m - 1, k - 1));
    curr = mult(curr, mod_pow(m, n - 1 - k));
    curr = mult(curr, cayley(n, k + 1));
    curr = mult(curr, mult(fact[n - 2], inv_fact[n - 2 - (k - 1)]));
    ans = add(ans, curr);
  }
  cout << ans << endl;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

MOD = 10**9 + 7

class Dsashaandinterestingfactfromgraphtheorybootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_m=10):
        self.max_n = max_n
        self.max_m = max_m
    
    def case_generator(self):
        # 限制参数范围保证计算可行性
        n = random.randint(2, min(self.max_n, 100))
        m = random.randint(1, min(self.max_m, 100))
        a, b = random.sample(range(1, n+1), 2)
        
        return {
            'n': n,
            'm': m,
            'a': a,
            'b': b,
            'correct_answer': self._compute_answer(n, m)
        }
    
    def _compute_answer(self, n, m):
        """重构后的正确算法实现"""
        MOD = 10**9 + 7
        if m == 0:
            return 0
        
        # 预处理必要阶乘
        max_fact = max(n-2, m-1)
        fact = [1]*(max_fact+2)
        for i in range(1, max_fact+1):
            fact[i] = fact[i-1] * i % MOD
            
        total = 0
        for k in range(1, min(n, m+1)):
            # 组合数计算 C(m-1, k-1)
            comb = 1
            if k-1 > 0:
                comb = fact[m-1] * pow(fact[k-1]*fact[(m-1)-(k-1)], MOD-2, MOD) % MOD
            
            # 凯莱公式部分
            if k+1 > n:
                c = 0
            else:
                c = pow(n, n - (k+1) - 1, MOD) * (k+1) % MOD  # 修正凯莱公式
            
            # 幂次部分
            pow_part = pow(m, (n-1)-k, MOD)
            
            # 排列数 P(n-2, k-1)
            perm = 1
            if k-1 > 0:
                perm = fact[n-2] * pow(fact[(n-2)-(k-1)], MOD-2, MOD) % MOD
                
            total = (total + comb * pow_part % MOD * c % MOD * perm % MOD) % MOD
        
        return total
    
    @staticmethod
    def prompt_func(question_case) -> str:
        return f"""给定n={question_case['n']}个顶点，边权范围1到{question_case['m']}。计算顶点{question_case['a']}和{question_case['b']}间距离恰好为{question_case['m']}的树的数量（模1e9+7）。答案写在[answer][/answer]中。"""
    
    @staticmethod
    def extract_output(output):
        last_match = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return int(last_match[-1].strip()) if last_match else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']

# 测试样例验证增强
if __name__ == "__main__":
    # 验证第一个样例输入
    bootcamp = Dsashaandinterestingfactfromgraphtheorybootcamp()
    test_case1 = {'n':3, 'm':2, 'a':1, 'b':3}
    assert bootcamp._compute_answer(3,2) == 5, "Sample 1 Failed"
    
    # 验证第二个样例输入
    test_case2 = {'n':3, 'm':1, 'a':1, 'b':2}
    assert bootcamp._compute_answer(3,1) == 2, "Sample 2 Failed"
