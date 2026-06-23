"""# 

### 谜题描述
Vasya will fancy any number as long as it is an integer power of two. Petya, on the other hand, is very conservative and only likes a single integer p (which may be positive, negative, or zero). To combine their tastes, they invented p-binary numbers of the form 2^x + p, where x is a non-negative integer.

For example, some -9-binary (\"minus nine\" binary) numbers are: -8 (minus eight), 7 and 1015 (-8=2^0-9, 7=2^4-9, 1015=2^{10}-9).

The boys now use p-binary numbers to represent everything. They now face a problem: given a positive integer n, what's the smallest number of p-binary numbers (not necessarily distinct) they need to represent n as their sum? It may be possible that representation is impossible altogether. Help them solve this problem.

For example, if p=0 we can represent 7 as 2^0 + 2^1 + 2^2.

And if p=-9 we can represent 7 as one number (2^4-9).

Note that negative p-binary numbers are allowed to be in the sum (see the Notes section for an example).

Input

The only line contains two integers n and p (1 ≤ n ≤ 10^9, -1000 ≤ p ≤ 1000).

Output

If it is impossible to represent n as the sum of any number of p-binary numbers, print a single integer -1. Otherwise, print the smallest possible number of summands.

Examples

Input


24 0


Output


2


Input


24 1


Output


3


Input


24 -1


Output


4


Input


4 -7


Output


2


Input


1 1


Output


-1

Note

0-binary numbers are just regular binary powers, thus in the first sample case we can represent 24 = (2^4 + 0) + (2^3 + 0).

In the second sample case, we can represent 24 = (2^4 + 1) + (2^2 + 1) + (2^0 + 1).

In the third sample case, we can represent 24 = (2^4 - 1) + (2^2 - 1) + (2^2 - 1) + (2^2 - 1). Note that repeated summands are allowed.

In the fourth sample case, we can represent 4 = (2^4 - 7) + (2^1 - 7). Note that the second summand is negative, which is allowed.

In the fifth sample case, no representation is possible.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
const long long INF = 2000000005;
const long long BIG_INF = 2000000000000000005;
const long long mod = 1000000007;
const long long P = 31;
const long double PI = 3.141592653589793238462643;
const double eps = 1e-9;
using namespace std;
vector<pair<long long, long long> > dir = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
bool valid(long long x, long long y, long long n, long long m) {
  return x >= 0 && y >= 0 && x < n && y < m;
}
mt19937 rng(1999999973);
signed main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  ;
  long long n, p;
  cin >> n >> p;
  for (long long i = 1; i < 1000000; i++) {
    if (n - p * i > 0 && __builtin_popcountll(n - p * i) <= i &&
        n - p * i >= i) {
      cout << i;
      return 0;
    }
  }
  cout << -1;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Apbinarybootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = {
            'p_min': -1000,
            'p_max': 1000,
            'n_min': 1,
            'n_max': 10**9,
            'edge_case_prob': 0.2  # 20%生成边界案例
        }
        self.params.update(params)
    
    def case_generator(self):
        if random.random() < self.params['edge_case_prob']:
            return self._generate_edge_case()
        
        p = random.randint(self.params['p_min'], self.params['p_max'])
        n = random.randint(self.params['n_min'], self.params['n_max'])
        return {'n': n, 'p': p}
    
    def _generate_edge_case(self):  # 修正：添加正确的缩进
        """生成边界测试案例"""
        edge_types = [
            {'p': 0},  # 常规二进制
            {'p': -1000, 'n': 10**9},  # 最小p值
            {'p': 1000, 'n': 1},       # 无解情况
            {'n': 1, 'p': 1},          # 样例5
            {'p': -1, 'n': 2**20 + 1}  # 大数值案例
        ]
        case = random.choice(edge_types)
        p = case.get('p', random.randint(-1000, 1000))
        n = case.get('n', random.randint(1, 10**9))
        return {'n': n, 'p': p}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        p = question_case['p']
        prompt = f"""你是一个数学问题解决者，需要帮助解决一个关于p-binary数的特殊求和问题。给定两个整数n和p，找出能够组成n的最少数量的p-binary数，若不可能则返回-1。

**p-binary数**的定义为：2^x + p，其中x是非负整数。允许使用重复的p-binary数，并且允许和为负数的情况。

**输入参数**:
n = {n}
p = {p}

请确定所需的最少p-binary数的数量，并将答案放在[answer]标签内。例如：[answer]3[/answer]。

**示例**:
- 输入：n=24, p=0 → 输出：2（因为24=16+8=2^4+0 + 2^3+0）
- 输入：n=4, p=-7 → 输出：2（因为4= (16-7) + (2-7)）

请严格遵循输出格式，将最终答案放置在[answer]和[/answer]之间。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\-?\d+)\s*\[/answer\]', output, re.IGNORECASE)
        return int(matches[-1]) if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        p = identity['p']
        correct = cls.solve(n, p)
        return solution == correct
    
    @staticmethod
    def solve(n, p):
        if p == 0:  # 优化常规二进制情况
            if (n & (n-1)) == 0:
                return 1
            return bin(n).count('1')
        
        max_i = 10**6 if p < 0 else min(10**6, n//abs(p)+2)
        for i in range(1, max_i+1):
            s = n - p * i
            if s <= 0:
                continue
            if s.bit_length() > 60:  # 处理极大数值溢出
                continue
            if bin(s).count('1') <= i and s >= i:
                return i
        return -1
