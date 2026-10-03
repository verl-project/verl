"""# 

### 谜题描述
It is Professor R's last class of his teaching career. Every time Professor R taught a class, he gave a special problem for the students to solve. You being his favourite student, put your heart into solving it one last time.

You are given two polynomials f(x) = a_0 + a_1x + ... + a_{n-1}x^{n-1} and g(x) = b_0 + b_1x + ... + b_{m-1}x^{m-1}, with positive integral coefficients. It is guaranteed that the cumulative GCD of the coefficients is equal to 1 for both the given polynomials. In other words, gcd(a_0, a_1, ..., a_{n-1}) = gcd(b_0, b_1, ..., b_{m-1}) = 1. Let h(x) = f(x)⋅ g(x). Suppose that h(x) = c_0 + c_1x + ... + c_{n+m-2}x^{n+m-2}. 

You are also given a prime number p. Professor R challenges you to find any t such that c_t isn't divisible by p. He guarantees you that under these conditions such t always exists. If there are several such t, output any of them.

As the input is quite large, please use fast input reading methods.

Input

The first line of the input contains three integers, n, m and p (1 ≤ n, m ≤ 10^6, 2 ≤ p ≤ 10^9), — n and m are the number of terms in f(x) and g(x) respectively (one more than the degrees of the respective polynomials) and p is the given prime number.

It is guaranteed that p is prime.

The second line contains n integers a_0, a_1, ..., a_{n-1} (1 ≤ a_{i} ≤ 10^{9}) — a_i is the coefficient of x^{i} in f(x).

The third line contains m integers b_0, b_1, ..., b_{m-1} (1 ≤ b_{i} ≤ 10^{9}) — b_i is the coefficient of x^{i} in g(x).

Output

Print a single integer t (0≤ t ≤ n+m-2) — the appropriate power of x in h(x) whose coefficient isn't divisible by the given prime p. If there are multiple powers of x that satisfy the condition, print any.

Examples

Input


3 2 2
1 1 2
2 1


Output


1


Input


2 2 999999937
2 1
3 1


Output


2

Note

In the first test case, f(x) is 2x^2 + x + 1 and g(x) is x + 2, their product h(x) being 2x^3 + 5x^2 + 3x + 2, so the answer can be 1 or 2 as both 3 and 5 aren't divisible by 2.

In the second test case, f(x) is x + 2 and g(x) is x + 3, their product h(x) being x^2 + 5x + 6, so the answer can be any of the powers as no coefficient is divisible by the given prime.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n,m,p=map(int,raw_input().strip().split())
a=map(int,raw_input().strip().split())
b=map(int,raw_input().strip().split())
ans=0
for i in xrange(n):
    if a[i]%p!=0:
        ans+=i
        break
for i in xrange(m):
    if b[i]%p!=0:
        ans+=i
        break
print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import re
import random
import math

class Cprimitiveprimesbootcamp(Basebootcamp):
    def __init__(self, max_n=5, max_m=5, primes=None, **params):
        super().__init__(**params)
        self.max_n = max_n
        self.max_m = max_m
        self.primes = primes or [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    
    def case_generator(self):
        while True:
            n = random.randint(1, self.max_n)
            m = random.randint(1, self.max_m)
            p = random.choice(self.primes)
            
            # Ensure first non-divisible index exists
            k_a = random.randint(0, n-1)
            k_b = random.randint(0, m-1)

            # Generate a-coefficients
            a = []
            valid_a = False
            for i in range(n):
                if i < k_a or i > k_a:
                    a_val = p  # Ensure other coefficients are multiples of p
                else:
                    a_val = random.randint(1, 10)
                    while a_val % p == 0:
                        a_val = random.randint(1, 10)
                a.append(a_val)
            
            # Check GCD for a
            gcd_a = a[0]
            for num in a[1:]:
                gcd_a = math.gcd(gcd_a, num)
                if gcd_a == 1:
                    break
            if gcd_a != 1:
                continue
            
            # Generate b-coefficients
            b = []
            for j in range(m):
                if j < k_b or j > k_b:
                    b_val = p  # Ensure other coefficients are multiples of p
                else:
                    b_val = random.randint(1, 10)
                    while b_val % p == 0:
                        b_val = random.randint(1, 10)
                b.append(b_val)
            
            # Check GCD for b
            gcd_b = b[0]
            for num in b[1:]:
                gcd_b = math.gcd(gcd_b, num)
                if gcd_b == 1:
                    break
            if gcd_b != 1:
                continue
            
            return {
                'n': n,
                'm': m,
                'p': p,
                'a': a,
                'b': b
            }
    
    @staticmethod
    def prompt_func(question_case):
        a_coeffs = ', '.join(map(str, question_case['a']))
        b_coeffs = ', '.join(map(str, question_case['b']))
        prompt = f"""It is Professor R's last class of his teaching career. Every time Professor R taught a class, he gave a special problem for the students to solve. You, being his favorite student, put your heart into solving it one last time.

You are given two polynomials f(x) and g(x) with positive integral coefficients. The cumulative GCD of the coefficients of each polynomial is 1. You are also given a prime number p. Your task is to find any exponent t in the product polynomial h(x) = f(x)⋅g(x) such that the coefficient of x^t, c_t, is not divisible by p. If there are multiple such t, output any of them.

Input format:
- The first line contains three integers n, m, p: the number of terms in f(x), g(x), and the prime number p respectively.
- The second line contains n integers a_0, a_1, ..., a_{{n-1}} — the coefficients of f(x).
- The third line contains m integers b_0, b_1, ..., b_{{m-1}} — the coefficients of g(x).

Input values for this problem:
n = {question_case['n']}, m = {question_case['m']}, p = {question_case['p']}
a coefficients: {a_coeffs}
b coefficients: {b_coeffs}

Please determine the appropriate value of t and output it. Place your final answer within [answer] and [/answer] tags. For example, if your answer is 3, write [answer]3[/answer]."""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.IGNORECASE)
        if not matches:
            return None
        return matches[-1].strip()
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            t = int(solution)
        except:
            return False
        
        p = identity['p']
        a = identity['a']
        b = identity['b']
        
        # Find first non-divisible index in a
        i = next((idx for idx, coeff in enumerate(a) if coeff % p != 0), None)
        # Find first non-divisible index in b
        j = next((idx for idx, coeff in enumerate(b) if coeff % p != 0), None)
        
        return t == i + j
