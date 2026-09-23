"""# 

### 谜题描述
You are given a prime number p, n integers a_1, a_2, …, a_n, and an integer k. 

Find the number of pairs of indexes (i, j) (1 ≤ i < j ≤ n) for which (a_i + a_j)(a_i^2 + a_j^2) ≡ k mod p.

Input

The first line contains integers n, p, k (2 ≤ n ≤ 3 ⋅ 10^5, 2 ≤ p ≤ 10^9, 0 ≤ k ≤ p-1). p is guaranteed to be prime.

The second line contains n integers a_1, a_2, …, a_n (0 ≤ a_i ≤ p-1). It is guaranteed that all elements are different.

Output

Output a single integer — answer to the problem.

Examples

Input


3 3 0
0 1 2


Output


1

Input


6 7 2
1 2 3 4 5 6


Output


3

Note

In the first example:

(0+1)(0^2 + 1^2) = 1 ≡ 1 mod 3.

(0+2)(0^2 + 2^2) = 8 ≡ 2 mod 3.

(1+2)(1^2 + 2^2) = 15 ≡ 0 mod 3.

So only 1 pair satisfies the condition.

In the second example, there are 3 such pairs: (1, 5), (2, 3), (4, 6).

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, p, k = map(int, raw_input().strip().split())
a = map(int, raw_input().strip().split())

rems = {}
for i in xrange(n):
    x = (a[i] ** 4 - k * a[i]) % p
    try: rems[x] += 1
    except: rems[x] = 1

ans = 0
for r in rems:
    cnt = rems[r]
    ans += ((cnt * (cnt - 1)) // 2)
    
print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Bcountpairsbootcamp(Basebootcamp):
    def __init__(self, possible_p=None, n_min=2, n_max=10000, **kwargs):
        super().__init__(**kwargs)
        if possible_p is None:
            possible_p = [
                3, 5, 7, 11, 13, 17, 19, 23, 29,  # 小质数
                1009, 7919,  # 中等大小质数
                1000003, 1000000007  # 大质数示例
            ]
        self.possible_p = possible_p
        self.n_min = n_min
        self.n_max = n_max  # 默认上限调整为更接近题目范围

    def case_generator(self):
        p = random.choice(self.possible_p)
        k = random.randint(0, p - 1)
        max_n = min(p, self.n_max)
        n = random.randint(self.n_min, max_n)
        a = []
        seen = set()
        # 生成n个不同的a_i，避免大内存占用
        while len(a) < n:
            x = random.randint(0, p-1)
            if x not in seen:
                seen.add(x)
                a.append(x)
        # 计算正确答案
        rems = {}
        for x in a:
            x4_mod = pow(x, 4, p)
            kx_mod = (k * x) % p
            term = (x4_mod - kx_mod) % p
            rems[term] = rems.get(term, 0) + 1
        ans = sum(cnt * (cnt - 1) // 2 for cnt in rems.values())
        return {
            'n': n,
            'p': p,
            'k': k,
            'a': a,
            'ans': ans
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        p = question_case['p']
        k = question_case['k']
        a = question_case['a']
        a_str = ' '.join(map(str, a))
        prompt = f"""You are a programming expert. Solve the following problem.

Given a prime number p, a list of {n} distinct integers, and an integer k, your task is to find the number of pairs of indices (i, j) where 1 ≤ i < j ≤ n and the expression (a_i + a_j)(a_i² + a_j²) is congruent to k modulo p.

Input Format:
- The first line contains three integers n, p, k. Here, n is the number of integers (2 ≤ n ≤ 3e5), p is a prime number (2 ≤ p ≤ 1e9), and k is an integer (0 ≤ k ≤ p-1).
- The second line contains n distinct integers a_1, a_2, ..., a_n, each between 0 and p-1 inclusive.

Output Format:
Output a single integer, the number of valid pairs.

Example:
For input:
3 3 0
0 1 2
The correct output is 1, as only the pair (1, 3) satisfies the condition.

Now, solve the following input case:
Input:
{n} {p} {k}
{a_str}

Please provide your answer within the tags [answer] and [/answer]. Ensure your final answer is the last one within these tags."""
        return prompt

    @staticmethod
    def extract_output(output):
        # 匹配所有可能的答案块，提取最后一个的数字部分
        matches = re.findall(r'\[answer\][^\d]*(\d+)[^\d]*\[/answer\]', output, re.IGNORECASE)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['ans']
