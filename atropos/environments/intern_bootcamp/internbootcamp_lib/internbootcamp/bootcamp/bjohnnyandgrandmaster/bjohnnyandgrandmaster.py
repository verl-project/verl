"""# 

### 谜题描述
Johnny has just found the new, great tutorial: \"How to become a grandmaster?\". The tutorial tells many strange and unexpected for Johnny things, such as you have to be patient or that very important is solving many harder and harder problems. 

The boy has found an online judge with tasks divided by topics they cover. He has picked p^{k_i} problems from i-th category (p is his favorite number). He wants to solve them in two weeks (the patience condition is too hard for Johnny, so for simplicity, he looks only at easy tasks, which can be solved in such a period). Now our future grandmaster has to decide which topics to cover first and which the second week. Help him assign topics in such a way, that workload is balanced.

Formally, given n numbers p^{k_i}, the boy wants to divide them into two disjoint sets, minimizing the absolute difference between sums of numbers in each set. Find the minimal absolute difference. Output the result modulo 10^{9}+7.

Input

Input consists of multiple test cases. The first line contains one integer t (1 ≤ t ≤ 10^5) — the number of test cases. Each test case is described as follows:

The first line contains two integers n and p (1 ≤ n, p ≤ 10^6). The second line contains n integers k_i (0 ≤ k_i ≤ 10^6).

The sum of n over all test cases doesn't exceed 10^6.

Output

Output one integer — the reminder of division the answer by 1 000 000 007.

Example

Input


4
5 2
2 3 4 4 3
3 1
2 10 1000
4 5
0 1 1 100
1 8
89


Output


4
1
146981438
747093407

Note

You have to minimize the difference, not it's remainder. For example, if the minimum difference is equal to 2, but there is also a distribution where the difference is 10^9 + 8, then the answer is 2, not 1.

In the first test case of the example, there're the following numbers: 4, 8, 16, 16, and 8. We can divide them into such two sets: {4, 8, 16} and {8, 16}. Then the difference between the sums of numbers in sets would be 4.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
long long k[1000005];
long long qkm(long long x, long long y, long long mod) {
  long long ans = 1;
  for (; y; y >>= 1, x = x * x % mod)
    if (y & 1) ans = ans * x % mod;
  return ans;
}
int main() {
  long long i, n, p, t, sum1, sum2;
  cin >> t;
  while (t--) {
    cin >> n >> p;
    for (i = 1; i <= n; i++) scanf(\"%lld\", &k[i]);
    if (p == 1) {
      cout << (n & 1) << endl;
      continue;
    }
    sort(k + 1, k + n + 1);
    sum1 = sum2 = 0;
    for (i = n; i >= 1; i--) {
      if (!sum1 && !sum2) {
        sum1 = qkm(p, k[i], 1000000007);
        sum2 = qkm(p, k[i], 1621836843);
      } else {
        sum1 = (sum1 - qkm(p, k[i], 1000000007) + 1000000007) % 1000000007;
        sum2 = (sum2 - qkm(p, k[i], 1621836843) + 1621836843) % 1621836843;
      }
    }
    cout << sum1 << endl;
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

MOD = 10**9 + 7

class Bjohnnyandgrandmasterbootcamp(Basebootcamp):
    def __init__(self, case_types='balanced'):
        """
        :param case_types: 'balanced'（混合） | 'special_p1'（强制p=1） | 'large_k'（大指数）
        """
        self.case_types = case_types
        
    def case_generator(self):
        # 生成三种核心测试场景
        if self.case_types == 'special_p1':
            p = 1
            n = random.randint(1, 10**3)
            k_list = [0]*n  # p=1时k没有意义
        elif self.case_types == 'large_k':
            p = random.choice([2,3,5])
            n = random.randint(100, 1000)
            k_list = [random.randint(10**5, 10**6) for _ in range(n)]
        else:  # 通用情况
            p = random.choices([1, random.randint(2,10**6)], weights=[0.3,0.7])[0]
            n = random.randint(1, 10**4)
            k_list = [random.randint(0, 10**6) for _ in range(n)]
        
        return {
            'n': n,
            'p': p,
            'k_list': k_list,
            'answer': self._compute_answer(n, p, k_list)
        }

    @staticmethod
    def _compute_answer(n, p, k_list):
        if p == 1:
            return n % 2  # 所有元素为1的拆分
        
        k_list.sort(reverse=True)
        balance = 0
        for k in k_list:
            if balance == 0:
                balance = pow(p, k, MOD)
            else:
                balance = (balance - pow(p, k, MOD)) % MOD
        return balance

    @staticmethod
    def prompt_func(case):
        problem_desc = f"""Split {case['n']} numbers (p={case['p']}, exponents=[{', '.join(map(str, case['k_list']))}]) into two subsets. The absolute difference of their sums must be minimized. Output the minimal difference modulo {MOD}."""
        format_instruction = "Put your final answer within [answer]...[/answer], like [answer]0[/answer] for difference 0."
        return f"{problem_desc}\n\n{format_instruction}"

    @staticmethod
    def extract_output(text):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', text)
        try:
            return int(matches[-1].strip()) % MOD if matches else None
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, case):
        expected = case['answer']
        return (solution % MOD) == (expected % MOD)
