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
\"Template made by : https://codeforces.com/profile/c1729 , github repo : https://github.com/cheran-senthil/PyRival\"
from __future__ import division, print_function
import bisect
import math
import itertools
import sys
from atexit import register

if sys.version_info[0] < 3:
    from io import BytesIO as stream
else:
    from io import StringIO as stream


if sys.version_info[0] < 3:
    class dict(dict):
        \"\"\"dict() -> new empty dictionary\"\"\"
        def items(self):
            \"\"\"D.items() -> a set-like object providing a view on D's items\"\"\"
            return dict.iteritems(self)

        def keys(self):
            \"\"\"D.keys() -> a set-like object providing a view on D's keys\"\"\"
            return dict.iterkeys(self)

        def values(self):
            \"\"\"D.values() -> an object providing a view on D's values\"\"\"
            return dict.itervalues(self)

    input = raw_input
    range = xrange

    filter = itertools.ifilter
    map = itertools.imap
    zip = itertools.izip


def sync_with_stdio(sync=True):
    \"\"\"Set whether the standard Python streams are allowed to buffer their I/O.

    Args:
        sync (bool, optional): The new synchronization setting.

    \"\"\"
    global input, flush

    if sync:
        flush = sys.stdout.flush
    else:
        sys.stdin = stream(sys.stdin.read())
        input = lambda: sys.stdin.readline().rstrip('\r\n')

        sys.stdout = stream()
        register(lambda: sys.__stdout__.write(sys.stdout.getvalue()))

def main():
    n,p,k=map(int,input().split())
    l1=list(map(int,input().split()))
    d1={}
    for item in l1:
        f=(item**4-k*item)%p
        if f in d1:
            d1[f]+=1
        else :
            d1[f]=1
    ans=0
    for k in d1:
        ans+=(d1[k]*(d1[k]-1))//2
    print(ans)
if __name__ == '__main__':
    sync_with_stdio(False)
    main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def is_prime(n):
    """Miller-Rabin primality test"""
    if n < 2: return False
    for p in [2,3,5,7,11,13,17,19,23,29,31,37]:
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in [2,3,5,7,11,13,17,19,23,29,31,37]:
        if a >= n: continue
        x = pow(a, d, n)
        if x == 1 or x == n-1: continue
        for _ in range(s-1):
            x = pow(x, 2, n)
            if x == n-1: break
        else: return False
    return True

def generate_prime(p_min, p_max):
    for _ in range(100):
        candidate = random.randint(p_min, p_max)
        if is_prime(candidate):
            return candidate
    for n in range(p_max, p_min-1, -1):
        if is_prime(n): return n
    raise ValueError(f"No prime in [{p_min}, {p_max}]")

class Ecountpairsbootcamp(Basebootcamp):
    def __init__(self, n_min=2, n_max=10, p_min=2, p_max=1000):
        # 参数有效性验证
        if n_min < 2: raise ValueError("n_min must ≥2")
        if p_max < p_min: raise ValueError("p_max must ≥p_min")
        if n_max > p_max:
            raise ValueError("n_max cannot exceed p_max")
        self.n_min = max(n_min, 2)
        self.n_max = n_max
        self.p_min = max(p_min, 2)
        self.p_max = p_max
        
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        p = generate_prime(max(n, self.p_min), self.p_max)
        k = random.randint(0, p-1)
        
        # 处理n=p的特殊情况
        if n == p:
            a = list(range(p))
            random.shuffle(a)
        else:
            a = random.sample(range(p), n)
            
        return {'n':n, 'p':p, 'k':k, 'a':a}

    @staticmethod
    def prompt_func(case) -> str:
        return f"""Solve the following modular arithmetic puzzle:

Given:
- Prime number p = {case['p']}
- An array of {case['n']} distinct integers: {case['a']}
- Target value k = {case['k']}

Count the number of index pairs (i, j) with i < j that satisfy:
(a_i + a_j) * (a_i² + a_j²) ≡ k mod p

Format your answer as: [answer]N[/answer] where N is the integer solution."""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, case):
        p, k = case['p'], case['k']
        freq = {}
        for x in case['a']:
            val = (pow(x,4,p) - (k*x) % p) % p
            freq[val] = freq.get(val, 0) + 1
        correct = sum(c * (c-1) // 2 for c in freq.values())
        return solution == correct
