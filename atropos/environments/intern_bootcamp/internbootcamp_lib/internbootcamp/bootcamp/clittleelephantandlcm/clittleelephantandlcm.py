"""# 

### 谜题描述
The Little Elephant loves the LCM (least common multiple) operation of a non-empty set of positive integers. The result of the LCM operation of k positive integers x1, x2, ..., xk is the minimum positive integer that is divisible by each of numbers xi.

Let's assume that there is a sequence of integers b1, b2, ..., bn. Let's denote their LCMs as lcm(b1, b2, ..., bn) and the maximum of them as max(b1, b2, ..., bn). The Little Elephant considers a sequence b good, if lcm(b1, b2, ..., bn) = max(b1, b2, ..., bn).

The Little Elephant has a sequence of integers a1, a2, ..., an. Help him find the number of good sequences of integers b1, b2, ..., bn, such that for all i (1 ≤ i ≤ n) the following condition fulfills: 1 ≤ bi ≤ ai. As the answer can be rather large, print the remainder from dividing it by 1000000007 (109 + 7).

Input

The first line contains a single positive integer n (1 ≤ n ≤ 105) — the number of integers in the sequence a. The second line contains n space-separated integers a1, a2, ..., an (1 ≤ ai ≤ 105) — sequence a.

Output

In the single line print a single integer — the answer to the problem modulo 1000000007 (109 + 7).

Examples

Input

4
1 4 3 2


Output

15


Input

2
6 3


Output

13

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int a[100010];
vector<int> f[100010];
long long mypow(long long x, long long y) {
  long long res = 1;
  while (y) {
    if (y & 1) {
      res *= x;
      res %= 1000000007;
    }
    x *= x;
    x %= 1000000007;
    y >>= 1;
  }
  return res;
}
int main() {
  int i, j, k, n;
  scanf(\"%d\", &n);
  for (i = 1; i <= n; i++) scanf(\"%d\", &a[i]);
  for (i = 1; i <= 100000; i++) {
    for (j = i; j <= 100000; j += i) {
      f[j].push_back(i);
    }
  }
  sort(a + 1, a + 1 + n);
  int now = 1;
  long long an = 1;
  for (i = 2; i <= a[n]; i++) {
    while (now <= n && i > a[now]) now++;
    long long res = mypow((int)f[i].size(), n - now + 1) -
                    mypow((int)f[i].size() - 1, n - now + 1);
    while (res < 0) res += 1000000007;
    int ll = 1;
    for (j = 1; j < f[i].size(); j++) {
      int rr = lower_bound(a + ll, a + n + 1, f[i][j]) - a;
      res = res * mypow(j, rr - ll) % 1000000007;
      ll = rr;
    }
    an += res;
  }
  an %= 1000000007;
  if (an < 0) an += 1000000007;
  cout << an << endl;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import bisect
import random
import re
from bootcamp import Basebootcamp

MOD = 10**9 + 7

class Clittleelephantandlcmbootcamp(Basebootcamp):
    def __init__(self, n_max=5, ai_max=10):
        self.n_max = n_max
        self.ai_max = ai_max
    
    def case_generator(self):
        n = random.randint(1, self.n_max)
        a = [random.randint(1, self.ai_max) for _ in range(n)]
        a_sorted = sorted(a)
        answer = self.calculate_answer(a_sorted)
        return {
            'n': n,
            'a': a,
            'expected_answer': answer
        }

    def calculate_answer(self, a_sorted):
        n = len(a_sorted)
        if n == 0:
            return 0
        
        max_a = a_sorted[-1]
        answer = 1  # 初始计数全1的情况

        for current_max in range(2, max_a + 1):
            # 动态生成因数列表（优化版）
            factors = [d for d in range(1, current_max+1) if current_max % d == 0]
            
            # 计算有效元素数量
            first_valid = bisect.bisect_left(a_sorted, current_max)
            valid_count = n - first_valid
            
            if valid_count == 0:
                continue

            # 计算基础项
            k = len(factors)
            term = (pow(k, valid_count, MOD) - pow(k-1, valid_count, MOD)) % MOD

            # 因子分割计算
            prev_idx = 0
            for idx, divisor in enumerate(factors[1:], 1):  # 跳过因子1
                split_point = bisect.bisect_left(a_sorted, divisor, prev_idx)
                term = (term * pow(idx, split_point - prev_idx, MOD)) % MOD
                prev_idx = split_point

            answer = (answer + term) % MOD

        return answer

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = question_case['a']
        return (
            f"Given sequence a (length {n}): {a}\n\n"
            f"Count good sequences where:\n"
            f"1. 1 ≤ b_i ≤ a_i\n"
            f"2. LCM(b) = max(b)\n\n"
            f"Output modulo 1e9+7. Format: [answer]{{number}}[/answer]"
        )

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](\d+)', output, re.IGNORECASE)
        return int(matches[-1]) % MOD if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected_answer']
