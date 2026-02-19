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
n, p = map(int, raw_input().strip().split())

k = 0
x = n
while 1:
    if (x <= 0):
        print -1
        exit()
    
    cnt = bin(x).count('1')
    if cnt <= k:
        if x >= k:
            print k
            exit()

    k += 1
    x -= p
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Cpbinarybootcamp(Basebootcamp):
    def __init__(self, max_n=10**9, min_p=-1000, max_p=1000):
        self.max_n = max_n
        self.min_p = min_p
        self.max_p = max_p
    
    def case_generator(self):
        # 生成保证有解的案例
        max_attempts = 100
        for _ in range(max_attempts):
            n = random.randint(1, self.max_n)
            p = random.randint(self.min_p, self.max_p)
            # 检查是否有可能的k值
            k_found = None
            x = n
            for k_candidate in range(0, 1000):  # 限制k的范围以确保终止
                x_current = x - k_candidate * p
                if x_current <= 0:
                    break
                if bin(x_current).count('1') <= k_candidate and x_current >= k_candidate:
                    k_found = k_candidate
                    break
            if k_found is not None:
                return {'n': n, 'p': p}
        # 回退到p=0的合法案例
        n = random.randint(1, self.max_n)
        return {'n': n, 'p': 0}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        p = question_case['p']
        prompt = f"""Vasya和Petya定义了一种p-binary数，形式为2^x + p（x为非负整数）。给定n和p，求用最少数量的p-binary数相加得到n。如果无解，返回-1。

输入:
n = {n}
p = {p}

输出要求:
- 答案必须为整数，格式如[answer]答案[/answer]，例如[answer]3[/answer]或[answer]-1[/answer]。
- 确保答案为最小可能的数量。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        # 提取最后一个[answer]标签内容
        answers = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answers:
            return None
        last_answer = answers[-1].strip()
        try:
            return int(last_answer)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        MAX_K_CHECK = 10000  # 防御性设计防止无限循环
        
        n = identity['n']
        p = identity['p']
        
        if solution == -1:
            # 检查是否存在有效k ≤ MAX_K_CHECK
            for k in range(0, MAX_K_CHECK + 1):
                x = n - k * p
                if x <= 0:
                    continue
                if bin(x).count('1') <= k and x >= k:
                    return False  # 存在解但返回了-1，错误
            return True  # 未找到有效k
        else:
            x = n - solution * p
            if x <= 0:
                return False
            return bin(x).count('1') <= solution and x >= solution
