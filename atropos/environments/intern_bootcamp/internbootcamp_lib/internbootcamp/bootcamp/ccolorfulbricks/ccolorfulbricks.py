"""# 

### 谜题描述
On his free time, Chouti likes doing some housework. He has got one new task, paint some bricks in the yard.

There are n bricks lined in a row on the ground. Chouti has got m paint buckets of different colors at hand, so he painted each brick in one of those m colors.

Having finished painting all bricks, Chouti was satisfied. He stood back and decided to find something fun with these bricks. After some counting, he found there are k bricks with a color different from the color of the brick on its left (the first brick is not counted, for sure).

So as usual, he needs your help in counting how many ways could he paint the bricks. Two ways of painting bricks are different if there is at least one brick painted in different colors in these two ways. Because the answer might be quite big, you only need to output the number of ways modulo 998 244 353.

Input

The first and only line contains three integers n, m and k (1 ≤ n,m ≤ 2000, 0 ≤ k ≤ n-1) — the number of bricks, the number of colors, and the number of bricks, such that its color differs from the color of brick to the left of it.

Output

Print one integer — the number of ways to color bricks modulo 998 244 353.

Examples

Input


3 3 0


Output


3


Input


3 2 1


Output


4

Note

In the first example, since k=0, the color of every brick should be the same, so there will be exactly m=3 ways to color the bricks.

In the second example, suppose the two colors in the buckets are yellow and lime, the following image shows all 4 possible colorings.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#import resource
import sys
#resource.setrlimit(resource.RLIMIT_STACK, [0x100000000, resource.RLIM_INFINITY])
import threading
threading.stack_size(2**25)
sys.setrecursionlimit(10**6)
mod=998244353
fact=[1]
for i in range(1,3001):
    fact.append((fact[-1]*i)%mod)
ifact=[0]*3001
ifact[3000]=pow(fact[3000],mod-2,mod)
for i in range(3000,0,-1):
    ifact[i-1]=(i*ifact[i])%mod
from sys import stdin, stdout
from bisect import bisect_left as bl
from bisect import bisect_right as br
import itertools
import math
import heapq
#from random import randint as rn
from Queue import Queue as Q
def modinv(n,p):
    return pow(n,p-2,p)
def ncr(n,r,p):
    t=((fact[n])*((ifact[r]*ifact[n-r])%p))%p
    return t
def ain():
    return map(int,sin().split())
def sin():
    return stdin.readline().strip()
def GCD(x, y):
   while(y):
       x, y = y, x % y
   return x
def isprime(x):
    p=int(math.sqrt(x))+1
    if(x==1):
        return 0
    for i in range(2,p):
        if(x%p==0):
            return 0
    return 1
\"\"\"**************************************************************************\"\"\"
n,m,k=map(int,raw_input().split())
p=ncr(n-1,k,mod)
q=m
for i in range(k):
    q=(q*(m-1))%mod
p=(p*q)%mod
print p
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Ccolorfulbricksbootcamp(Basebootcamp):
    mod = 998244353
    max_fact = 3000  # Precompute factorials up to 3000 to handle n up to 2000

    # Precompute factorial and inverse factorial arrays
    fact = [1] * (max_fact + 1)
    for i in range(1, max_fact + 1):
        fact[i] = (fact[i-1] * i) % mod

    ifact = [0] * (max_fact + 1)
    ifact[max_fact] = pow(fact[max_fact], mod-2, mod)
    for i in range(max_fact-1, -1, -1):
        ifact[i] = (ifact[i+1] * (i+1)) % mod

    def __init__(self, **params):
        self.n_min = params.get('n_min', 1)
        self.n_max = params.get('n_max', 2000)
        self.m_min = params.get('m_min', 1)
        self.m_max = params.get('m_max', 2000)

    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        k = random.randint(0, n-1) if n > 1 else 0
        m = random.randint(self.m_min, self.m_max)
        return {'n': n, 'm': m, 'k': k}

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        m = question_case['m']
        k = question_case['k']
        return f"""Chouti needs to paint a row of {n} bricks using {m} different colors. Exactly {k} bricks must differ in color from their immediate left neighbor. 

Calculate the number of valid ways to paint the bricks, considering:
1. The first brick has no left neighbor
2. Two colorings are different if any brick differs
3. Answer modulo 998244353

Put your final answer within [answer] and [/answer] tags. For example: [answer]12345[/answer]"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except:
            return None

    @classmethod
    def compute_combination(cls, n, r):
        if r < 0 or r > n:
            return 0
        return (cls.fact[n] * cls.ifact[r] % cls.mod) * cls.ifact[n - r] % cls.mod

    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        m = identity['m']
        k = identity['k']
        mod = cls.mod

        if n == 0:
            return solution == 0

        # Calculate combination
        comb = cls.compute_combination(n-1, k)
        if comb == 0:
            return False

        # Calculate m*(m-1)^k mod mod
        term = m % mod
        if k > 0:
            term = (term * pow(m-1, k, mod)) % mod

        expected = (comb * term) % mod
        return solution == expected
