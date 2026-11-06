"""# 

### 谜题描述
Gottfried learned about binary number representation. He then came up with this task and presented it to you.

You are given a collection of n non-negative integers a_1, …, a_n. You are allowed to perform the following operation: choose two distinct indices 1 ≤ i, j ≤ n. If before the operation a_i = x, a_j = y, then after the operation a_i = x~AND~y, a_j = x~OR~y, where AND and OR are bitwise AND and OR respectively (refer to the Notes section for formal description). The operation may be performed any number of times (possibly zero).

After all operations are done, compute ∑_{i=1}^n a_i^2 — the sum of squares of all a_i. What is the largest sum of squares you can achieve?

Input

The first line contains a single integer n (1 ≤ n ≤ 2 ⋅ 10^5).

The second line contains n integers a_1, …, a_n (0 ≤ a_i < 2^{20}).

Output

Print a single integer — the largest possible sum of squares that can be achieved after several (possibly zero) operations.

Examples

Input


1
123


Output


15129


Input


3
1 3 5


Output


51


Input


2
349525 699050


Output


1099509530625

Note

In the first sample no operation can be made, thus the answer is 123^2.

In the second sample we can obtain the collection 1, 1, 7, and 1^2 + 1^2 + 7^2 = 51.

If x and y are represented in binary with equal number of bits (possibly with leading zeros), then each bit of x~AND~y is set to 1 if and only if both corresponding bits of x and y are set to 1. Similarly, each bit of x~OR~y is set to 1 if and only if at least one of the corresponding bits of x and y are set to 1. For example, x = 3 and y = 5 are represented as 011_2 and 101_2 (highest bit first). Then, x~AND~y = 001_2 = 1, and x~OR~y = 111_2 = 7.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
range = xrange
input = raw_input

n = int(input())
A = [int(x) for x in input().split()]

count = [0] * 20
for a in A:
    for i in range(20):
        count[i] += (a >> i) & 1

sq = 0
for _ in range(n):
    a = 0
    for i in range(20):
        if count[i]:
            count[i] -= 1
            a |= 1 << i
    sq += a * a
print sq
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Dandorandsquaresumbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=100, max_bits=20):
        self.min_n = max(1, min_n)
        self.max_n = min(max_n, 200000)  # 安全限制
        self.max_bits = min(max_bits, 20)
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        a = [
            random.randint(0, (1 << self.max_bits) - 1)
            for _ in range(n)
        ]
        # 确保至少一个案例全零
        if random.random() < 0.05:
            a = [0]*n
        return {'n': n, 'a': a}
    
    @staticmethod
    def prompt_func(question_case):
        a_str = ' '.join(map(str, question_case['a']))
        return f"""给定n={question_case['n']}个整数：{a_str}。通过任意次操作（选择i,j，设x=a_i,y=a_j，a_i变x&y，a_j变x|y），求最大平方和。答案写在[answer]标签内。例：[answer]0[/answer]。"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        count = [0]*20
        for num in identity['a']:
            for i in range(20):
                count[i] += (num >> i) & 1
        
        sum_sq = 0
        bits = count.copy()
        for _ in range(identity['n']):
            a = 0
            for i in range(20):  # 严格按参考代码顺序（低位→高位）
                if bits[i]:
                    a |= (1 << i)
                    bits[i] -= 1
            sum_sq += a*a
        return solution == sum_sq
