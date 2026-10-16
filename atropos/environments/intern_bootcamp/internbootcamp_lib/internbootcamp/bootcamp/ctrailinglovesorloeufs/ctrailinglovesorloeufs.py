"""# 

### 谜题描述
The number \"zero\" is called \"love\" (or \"l'oeuf\" to be precise, literally means \"egg\" in French), for example when denoting the zero score in a game of tennis. 

Aki is fond of numbers, especially those with trailing zeros. For example, the number 9200 has two trailing zeros. Aki thinks the more trailing zero digits a number has, the prettier it is.

However, Aki believes, that the number of trailing zeros of a number is not static, but depends on the base (radix) it is represented in. Thus, he considers a few scenarios with some numbers and bases. And now, since the numbers he used become quite bizarre, he asks you to help him to calculate the beauty of these numbers.

Given two integers n and b (in decimal notation), your task is to calculate the number of trailing zero digits in the b-ary (in the base/radix of b) representation of n ! ([factorial](https://en.wikipedia.org/wiki/Factorial) of n). 

Input

The only line of the input contains two integers n and b (1 ≤ n ≤ 10^{18}, 2 ≤ b ≤ 10^{12}).

Output

Print an only integer — the number of trailing zero digits in the b-ary representation of n!

Examples

Input

6 9


Output

1


Input

38 11


Output

3


Input

5 2


Output

3


Input

5 10


Output

1

Note

In the first example, 6!_{(10)} = 720_{(10)} = 880_{(9)}.

In the third and fourth example, 5!_{(10)} = 120_{(10)} = 1111000_{(2)}.

The representation of the number x in the b-ary base is d_1, d_2, …, d_k if x = d_1 b^{k - 1} + d_2 b^{k - 2} + … + d_k b^0, where d_i are integers and 0 ≤ d_i ≤ b - 1. For example, the number 720 from the first example is represented as 880_{(9)} since 720 = 8 ⋅ 9^2 + 8 ⋅ 9 + 0 ⋅ 1.

You can read more about bases [here](https://en.wikipedia.org/wiki/Radix).

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
'''
Miller-rabin primality test
Valid for all 64-bit unsigned integers
Sometimes wrong for numbers greater
'''
def check_composite(n, s, d, a):
    '''
   check compositeness of n with witness a
   (n,s,d) should satisfy d*2^s = n-1 and d is odd
   '''
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return False
    for y in xrange(1, s):
        x = x * x % n
        if x == 1:
            return True
        if x == n - 1:
            return False
    return True
small_primes = set([3,5,7,11,13,17,19,23,29,31,37])
small_is_prime = [s in small_primes for s in xrange(38)]
small_is_prime[2] = True
 
# witnesses for different bounds (taken from http://miller-rabin.appspot.com/ )
witnesses_bounds = [
    (341531, [9345883071009581737]),
    (716169301, [15, 13393019396194701]),
    (154639673381, [15, 176006322, 4221622697]),
    (47636622961201, [2, 2570940, 211991001, 3749873356]),
    (3770579582154547, [2, 2570940, 880937, 610386380, 4130785767]),
]
# set of witnesses for < 2^64 (taken from http://miller-rabin.appspot.com/ )
i64_witnesses = [2, 325, 9375, 28178, 450775, 9780504, 1795265022]
def is_prime(n):
    '''
   returns True if n is probably prime, and False if n is definitely not prime.
   if n < 2^64, then is_prime(n) never returns a wrong answer.
   '''
    # if too small, check small_is_prime
    if n < 38:
        return small_is_prime[n]
    # check divisibility with small primes
    for s in small_primes:
        if n % s == 0:
            return False
    # find (d,s) such that d*2^s = n-1 with d odd
    d = n - 1
    s = 0
    while not d & 1:
        d >>= 1
        s += 1
    # find the best set of witnesses
    best_witnesses = i64_witnesses
    for bound, bound_ws in witnesses_bounds:
        if n < bound:
            best_witnesses = bound_ws
            break
    # check compositeness with the witnesses
    for a in best_witnesses:
        a %= n
        if a and check_composite(n, s, d, a):
            return False
    return True
#########################################################

n, b = map(int, raw_input().split())

prime_divisors = []
d = 1
while d*d <= b:
    if b%d == 0:
        if is_prime(d):
            prime_divisors.append(d)
        d2 = b/d
        if d2 != d and is_prime(d2):
            prime_divisors.append(d2)
    d += 1

factors = []
for p in prime_divisors:
    cnt = 0
    tmp = b
    while tmp%p == 0:
        cnt += 1
        tmp /= p
    factors.append((p, cnt))

res = 10**18
for p, req in factors:
    num_occ = 0
    tmp = n
    while tmp/p > 0:
        num_occ += tmp/p
        tmp /= p
    res = min(res, num_occ/req)
print res
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re
import math

class Ctrailinglovesorloeufsbootcamp(Basebootcamp):
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    witnesses_bounds = [
        (341531, [9345883071009581737]),
        (716169301, [15, 13393019396194701]),
        (154639673381, [15, 176006322, 4221622697]),
        (47636622961201, [2, 2570940, 211991001, 3749873356]),
        (3770579582154547, [2, 2570940, 880937, 610386380, 4130785767]),
    ]
    i64_witnesses = [2, 325, 9375, 28178, 450775, 9780504, 1795265022]

    def __init__(self, max_n=10**18, max_b=10**12):
        self.max_n = max_n
        self.max_b = max_b
        self.small_prime_set = set(self.small_primes)
        self.small_is_prime = [i in self.small_prime_set for i in range(38)]

    def check_composite(self, n, s, d, a):
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            return False
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                return False
        return True

    def is_prime(self, n):
        if n < 2: return False
        if n < 38: return self.small_is_prime[n]
        
        # Check small primes first
        for p in self.small_primes:
            if n % p == 0:
                return n == p
        
        # Miller-Rabin test
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
            
        for a in self.i64_witnesses:
            a %= n
            if a == 0:
                continue
            if self.check_composite(n, s, d, a):
                return False
        return True

    def case_generator(self):
        # Generate n with logarithmic distribution
        n = int(10**random.uniform(0, math.log10(self.max_n)))
        if n > self.max_n:
            n = self.max_n
        
        # Generate b with realistic distribution
        b = random.randint(2, min(self.max_b, 10**6))  # Balance between diversity and performance
        
        # Prime factorization with early termination
        prime_divisors = set()
        temp_b = b
        
        # Check small primes first
        for p in self.small_primes:
            if temp_b % p == 0:
                prime_divisors.add(p)
                while temp_b % p == 0:
                    temp_b //= p
            if temp_b == 1:
                break
        
        # Handle remaining factors
        if temp_b > 1:
            sqrt_b = int(math.isqrt(temp_b))
            for d in range(self.small_primes[-1] + 2, sqrt_b + 1, 2):
                if d > temp_b:
                    break
                if temp_b % d == 0 and self.is_prime(d):
                    prime_divisors.add(d)
                    while temp_b % d == 0:
                        temp_b //= d
                    sqrt_b = int(math.isqrt(temp_b))
                if temp_b == 1:
                    break
            
            if temp_b > 1 and self.is_prime(temp_b):
                prime_divisors.add(temp_b)
        
        # Handle base cases
        if not prime_divisors and self.is_prime(b):
            prime_divisors.add(b)
        
        # Calculate exponents
        factors = []
        for p in sorted(prime_divisors):
            cnt = 0
            tmp = b
            while tmp % p == 0:
                cnt += 1
                tmp //= p
            factors.append((p, cnt))
        
        return {'n': n, 'b': b, 'factors': factors}

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        b = question_case['b']
        return f"""Calculate the number of trailing zeros in the base-{b} representation of {n}!.
Provide your answer in [answer]...[/answer] tags. Example: [answer]3[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL | re.IGNORECASE)
        return int(matches[-1].strip()) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False
            
        n = identity['n']
        factors = identity['factors']
        if not factors:
            return solution == 0
        
        min_zeros = float('inf')
        for p, exp_in_base in factors:
            count = 0
            current = n
            while current > 0:
                current //= p
                count += current
            min_zeros = min(min_zeros, count // exp_in_base)
        
        return solution == min_zeros
