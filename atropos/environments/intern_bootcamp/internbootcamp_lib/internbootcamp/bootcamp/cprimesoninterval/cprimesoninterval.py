"""# 

### 谜题描述
You've decided to carry out a survey in the theory of prime numbers. Let us remind you that a prime number is a positive integer that has exactly two distinct positive integer divisors.

Consider positive integers a, a + 1, ..., b (a ≤ b). You want to find the minimum integer l (1 ≤ l ≤ b - a + 1) such that for any integer x (a ≤ x ≤ b - l + 1) among l integers x, x + 1, ..., x + l - 1 there are at least k prime numbers. 

Find and print the required minimum l. If no value l meets the described limitations, print -1.

Input

A single line contains three space-separated integers a, b, k (1 ≤ a, b, k ≤ 106; a ≤ b).

Output

In a single line print a single integer — the required minimum l. If there's no solution, print -1.

Examples

Input

2 4 2


Output

3


Input

6 13 1


Output

4


Input

1 4 3


Output

-1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def f(a, b):
    t = [1] * (b + 1)
    for i in range(3, int(b ** 0.5) + 1):
        if t[i]: t[i * i :: 2 * i] = [0] * ((b - i * i) // (2 * i) + 1)
    return [i for i in range(3, b + 1, 2) if t[i] and i > a]

a, b, k = map(int, raw_input().split())

p = f(a - 1, b)
if 3 > a and b > 1: p = [2] + p
 
if k > len(p): print(-1)
elif len(p) == k: print(max(p[k - 1] - a + 1, b - p[0] + 1))
else: print(max(p[k - 1] - a + 1, b - p[len(p) - k] + 1, max(p[i + k] - p[i] for i in range(len(p) - k))))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cprimesonintervalbootcamp(Basebootcamp):
    def __init__(self, a_min=1, a_max=10**6, b_max=10**6, k_max=10**6):
        self.a_min = a_min
        self.a_max = min(a_max, b_max)  # 确保a_max <= b_max
        self.b_max = b_max
        self.k_max = k_max
    
    def case_generator(self):
        a = random.randint(self.a_min, self.a_max)
        b = random.randint(a, self.b_max)
        k = random.randint(1, self.k_max)
        return {'a': a, 'b': b, 'k': k}
    
    @staticmethod
    def prompt_func(question_case):
        a = question_case['a']
        b = question_case['b']
        k = question_case['k']
        prompt = f"""You are conducting a survey on prime numbers. A prime number is a positive integer greater than 1 with exactly two distinct divisors: 1 and itself.

Your task is to find the minimum length l such that EVERY consecutive sequence of l numbers between {a} and {b} (inclusive) contains at least {k} prime numbers. If no such l exists, output -1.

Examples:
Input: 2 4 2 → Output: 3 (Primes [2,3] need 3-length window)
Input: 6 13 1 → Output: 4 (Primes [7,11,13] need 4-length window)
Input: 1 4 3 → Output: -1 (Only 2 primes exist)

Rules:
1. l must be the smallest integer satisfying: for ALL x where a ≤ x ≤ b-l+1,
   the window [x, x+l-1] contains ≥k primes
2. If total primes in [a,b] < k → output -1

Format your answer as:
[answer]你的答案[/answer]"""

        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        def optimized_solver(a, b, k):
            if a > b or k == 0:
                return -1

            # 生成区间内的素数列表
            def generate_primes(low, high):
                if high < 2:
                    return []
                sieve = [True] * (high + 1)
                sieve[0] = sieve[1] = False
                for i in range(3, int(high**0.5)+1, 2):
                    if sieve[i]:
                        sieve[i*i::2*i] = [False] * len(sieve[i*i::2*i])
                
                primes = []
                if low <= 2 <= high:
                    primes.append(2)
                primes += [i for i in range(3, high+1, 2) if sieve[i] and i >= low]
                return primes

            primes = generate_primes(a, b)
            
            if len(primes) < k:
                return -1
            
            if k == 1:
                max_gap = max(
                    primes[0] - a + 1,
                    b - primes[-1] + 1,
                    max(p2 - p1 for p1, p2 in zip(primes, primes[1:]))
                )
                return max_gap
            
            if len(primes) == k:
                return max(primes[-1] - a + 1, b - primes[0] + 1)
            
            return max(
                primes[k-1] - a + 1,
                b - primes[-k] + 1,
                max(primes[i+k] - primes[i] for i in range(len(primes)-k))
            )

        try:
            a = identity['a']
            b = identity['b']
            k = identity['k']
            user_answer = int(solution)
            
            # 特殊情况处理
            if user_answer == -1:
                actual_primes = generate_primes(a, b)
                return len(actual_primes) < k
            
            # 计算正确答案
            correct_answer = optimized_solver(a, b, k)
            return user_answer == correct_answer
        except:
            return False

def generate_primes(a, b):
    """独立的素数生成函数用于验证"""
    if b < 2:
        return []
    sieve = [True] * (b + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(b**0.5)+1):
        if sieve[i]:
            sieve[i*i::i] = [False]*len(sieve[i*i::i])
    return [p for p in range(a, b+1) if sieve[p]]
