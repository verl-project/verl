"""# 

### 谜题描述
Neko loves divisors. During the latest number theory lesson, he got an interesting exercise from his math teacher.

Neko has two integers a and b. His goal is to find a non-negative integer k such that the least common multiple of a+k and b+k is the smallest possible. If there are multiple optimal integers k, he needs to choose the smallest one.

Given his mathematical talent, Neko had no trouble getting Wrong Answer on this problem. Can you help him solve it?

Input

The only line contains two integers a and b (1 ≤ a, b ≤ 10^9).

Output

Print the smallest non-negative integer k (k ≥ 0) such that the lowest common multiple of a+k and b+k is the smallest possible.

If there are many possible integers k giving the same value of the least common multiple, print the smallest one.

Examples

Input


6 10


Output


2

Input


21 31


Output


9

Input


5 10


Output


0

Note

In the first test, one should choose k = 2, as the least common multiple of 6 + 2 and 10 + 2 is 24, which is the smallest least common multiple possible.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import random
import time

seed = int(time.time())
random.seed(seed)

def gcd(a, b):
    if b > a:
        return gcd(b, a)

    while b > 0:
        a, b = b, a % b

    return a

def lcm(a, b):
    return (a*b)//gcd(a, b)

def solve_slow(a, b):
    best = lcm(a, b)
    best_k = 0
    for k in range(1, 20**2):
        cand = lcm(a+k, b+k)
        if cand < best:
            best = cand
            best_k = k
    return best_k

def solve_slow2(a, b):
    if a > b:
        return solve_slow2(b, a)

    best = lcm(a, b)
    best_k = 0
    for m in range(2, a+1):
        if (a-b) % m == 0:
            print 'm:', m
            break
    return 0

def solve(a, b):
    if a > b:
        return solve(b, a)
    elif a == b:
        return 0

    delta = b - a
    best = lcm(a, b)
    best_k = 0
    for fac in range(1, int(delta**.5)+1):
        if delta % fac != 0:
            continue

        k = fac - (a % fac)
        candidate = lcm(a+k, b+k)
        if candidate < best:
            best = candidate
            best_k = k

        invfac = delta//fac
        k = invfac - (a % invfac)
        candidate = lcm(a+k, b+k)
        if candidate < best:
            best = candidate
            best_k = k

    return best_k

def test():
    print solve(random.randint(1, 10**9), random.randint(1, 10**9))
    return
    for a in range(1, 100):
        for b in range(1, 100):
            actual = solve(a, b)
            expected = solve_slow(a, b)
            if actual != expected:
                print 'death on a=%d b=%d: expected %d, got %d' % (a, b, expected, actual)
                return

def main():
    xs = [int(x) for x in raw_input().strip().split()]
    print solve(*xs)

if '__main__'==__name__:
    main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Cnekodoesmathsbootcamp(Basebootcamp):
    def __init__(self, min_val=1, max_val=10**9):
        if min_val < 1:
            raise ValueError("min_val must be ≥1")
        if max_val < min_val:
            raise ValueError("max_val must be ≥ min_val")
        self.min_val = min_val
        self.max_val = max_val
    
    @staticmethod
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    @staticmethod
    def lcm(a, b):
        return a * b // Cnekodoesmathsbootcamp.gcd(a, b)
    
    @staticmethod
    def solve(a, b):
        if a == b:
            return 0
        if a > b:
            a, b = b, a
        delta = b - a
        best_k = 0
        best_lcm = Cnekodoesmathsbootcamp.lcm(a, b)
        
        factors = set()
        for fac in range(1, int(delta**0.5) + 1):
            if delta % fac == 0:
                factors.update({fac, delta//fac})
        
        for current_fac in sorted(factors, reverse=True):
            k = (current_fac - a % current_fac) % current_fac
            new_a = a + k
            new_b = b + k
            current_lcm = Cnekodoesmathsbootcamp.lcm(new_a, new_b)
            
            if (current_lcm < best_lcm) or (current_lcm == best_lcm and k < best_k):
                best_lcm = current_lcm
                best_k = k
        
        return best_k

    def case_generator(self):
        if random.random() < 0.2:
            case_type = random.choice([
                ('equal', 1, 1),
                ('small_diff', 5, 10),
                ('prime_diff', 2, 5),
                ('large_diff', 10**9-100, 10**9)
            ])
            a, b = {
                'equal': (lambda: (x:=random.randint(1,10**9), x)),
                'small_diff': lambda: (random.randint(1,100), random.randint(1,100)+5),
                'prime_diff': lambda: (random.choice([2,3,5,7,11]), random.choice([2,3,5,7,11])+2),
                'large_diff': lambda: (10**9 - random.randint(1,1000), 10**9)
            }[case_type[0]]()
        else:
            a = random.randint(self.min_val, self.max_val)
            b = random.randint(self.min_val, self.max_val)
        
        return {
            'a': a,
            'b': b,
            'correct_k': self.solve(a, b)
        }
    
    @staticmethod
    def prompt_func(question_case):
        a = question_case['a']
        b = question_case['b']
        return f"""Neko遇到了一个数论问题：给定两个正整数a和b，找到最小非负整数k，使得a+k和b+k的最小公倍数最小。

**问题参数**
- a = {a}
- b = {b}

**求解要求**
1. 如果存在多个k能得到相同的最小LCM，返回最小的k值
2. 答案必须是非负整数
3. 请将最终答案放在[answer]和[/answer]标签之间

**示例格式**
如果正确答案是0，应写：[answer]0[/answer]"""

    @staticmethod
    def extract_output(output):
        answers = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(answers[-1]) if answers else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_k']
