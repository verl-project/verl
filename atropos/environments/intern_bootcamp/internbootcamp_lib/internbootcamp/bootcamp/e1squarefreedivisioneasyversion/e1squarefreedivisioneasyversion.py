"""# 

### 谜题描述
This is the easy version of the problem. The only difference is that in this version k = 0.

There is an array a_1, a_2, …, a_n of n positive integers. You should divide it into a minimal number of continuous segments, such that in each segment there are no two numbers (on different positions), whose product is a perfect square.

Moreover, it is allowed to do at most k such operations before the division: choose a number in the array and change its value to any positive integer. But in this version k = 0, so it is not important.

What is the minimum number of continuous segments you should use if you will make changes optimally?

Input

The first line contains a single integer t (1 ≤ t ≤ 1000) — the number of test cases.

The first line of each test case contains two integers n, k (1 ≤ n ≤ 2 ⋅ 10^5, k = 0).

The second line of each test case contains n integers a_1, a_2, …, a_n (1 ≤ a_i ≤ 10^7).

It's guaranteed that the sum of n over all test cases does not exceed 2 ⋅ 10^5.

Output

For each test case print a single integer — the answer to the problem.

Example

Input


3
5 0
18 6 2 4 1
5 0
6 8 1 24 8
1 0
1


Output


3
2
1

Note

In the first test case the division may be as follows:

  * [18, 6] 
  * [2, 4] 
  * [1] 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
testing = len(sys.argv) == 4 and sys.argv[3] == \"myTest\"
if testing:
    cmd = sys.stdout
    from time import time
    start_time = int(round(time() * 1000)) 
    readAll = open(sys.argv[1], 'r').read
    sys.stdout = open(sys.argv[2], 'w')
else:
    readAll = sys.stdin.read

# ############ ---- I/O Functions ---- ############

flush = sys.stdout.flush
class InputData:
    def __init__(self):
        self.lines = readAll().split('\n')
        self.n = len(self.lines)
        self.ii = -1
    def input(self):
        self.ii += 1
        assert self.ii < self.n
        return self.lines[self.ii]
inputData = InputData()
input = inputData.input

def intin():
    return(int(input()))
def intlin():
    return(list(map(int,input().split())))
def chrin():
    return(list(input()))
def strin():
    return input()
def lout(l, sep=\"\n\", toStr=True):
    print(sep.join(map(str, l) if toStr else l))
    
# ############ ---- I/O Functions ---- ############

# from math import ceil
from collections import defaultdict as ddict, Counter
# from heapq import *
# from Queue import Queue

def isPrime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n%2==0 or n%3==0:
        return False
    i = 5
    while i*i <= n:
        if n%i==0 or n%(i+2)==0:
            return False
        i += 6
    return True

def getPrimeDivs(n, primes, bigPrimes):
    divs = 0
    i = 0
    p = primes[0]
    copy = n
    while p <= n:
        if n%p == 0:
            k = 0
            while n%p == 0:
                k += 1
                n /= p
            if k%2 == 1:
                divs += (1<<i)
        i += 1
        if i == len(primes):
            break
        p = primes[i]
    if n > p:
        if divs > 0:
            n = copy
        bigPrimes[copy] = -1*n
        divs = -1*n
    return divs

def isPerfectSquare(n):
    k = int(n**0.5)
    if n < 10**5:
        return k*k==n
    else:
        return k*k==n

primes = [2]
for i in xrange(3,int((10**7)**0.5)+1,2):
    if isPrime(i):
        primes.append(i)
bigPrimes = dict()

def main():
    n,k = intlin()
    a = intlin()
    b = []
    for x in a:
        if x in bigPrimes:
            b.append(bigPrimes[x])
        else:
            b.append(getPrimeDivs(x, primes, bigPrimes))
    # print(b)
    ans = 1
    i = 1
    flag = isPerfectSquare(a[0])
    cur = set([b[0]])
    while i < n:
        if b[i] in cur or (isPerfectSquare(a[i]) and flag):
            ans += 1
            flag = False
            cur = set([b[i]])
        else:
            cur.add(b[i])
        flag = isPerfectSquare(a[i])
        i += 1
    return ans


anss = []
for _ in xrange(intin()):
    anss.append(main())
    # anss.append(\"YES\" if main() else \"NO\")
lout(anss)

if testing:
    sys.stdout = cmd
    print(int(round(time() * 1000))  - start_time)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re
from math import isqrt

class E1squarefreedivisioneasyversionbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.max_prime = 3162  # sqrt(10^7) ≈ 3162.27
        self.primes = self._generate_primes(self.max_prime)
        self.big_primes_cache = {}

    def _generate_primes(self, max_limit):
        sieve = [True] * (max_limit + 1)
        sieve[0:2] = [False, False]
        for i in range(2, isqrt(max_limit) + 1):
            if sieve[i]:
                sieve[i*i::i] = [False] * len(sieve[i*i::i])
        return [i for i, prime in enumerate(sieve) if prime]

    def _get_square_signature(self, x):
        residual = 1
        for p in self.primes:
            if p*p > x:
                break
            exponent = 0
            while x % p == 0:
                exponent += 1
                x //= p
            if exponent % 2 != 0:
                residual *= p
        if x > 1:
            sqrt_x = isqrt(x)
            if sqrt_x * sqrt_x == x:
                return residual
            self.big_primes_cache[x] = residual * x  # 合并剩余大质数
            return self.big_primes_cache[x]
        return residual

    def _calculate_min_segments(self, a):
        seen = set()
        segments = 1
        for num in a:
            sig = self._get_square_signature(num)
            if sig in seen:
                seen = {sig}
                segments += 1
            else:
                seen.add(sig)
        return segments

    def case_generator(self):
        # 生成多样化的测试用例
        case_type = random.choice([1,2,3,4])
        n = random.randint(1, 20)
        
        if case_type == 1:  # 全平方数
            a = [random.randint(1, 30)**2 for _ in range(n)]
        elif case_type == 2:  # 含大质数
            a = [random.choice([9617497, 9999991, 32452843]) for _ in range(n)]
        elif case_type == 3:  # 混合类型
            a = [random.choice([p**2, p**3, p]) for p in random.choices(self.primes[-10:], k=n)]
        else:  # 随机正常案例
            a = [random.randint(1, 10**4) for _ in range(n)]
        
        # 确保至少一个非空案例
        a = a if n > 0 else [1]
        expected = self._calculate_min_segments(a)
        return {'n': len(a), 'k': 0, 'a': a, 'expected': expected}

    @staticmethod
    def prompt_func(question_case):
        return (
            "将数组划分为最少的连续段，使得每段内任意两数的乘积不是完全平方数。\n"
            f"输入：\n{question_case['n']} 0\n{' '.join(map(str, question_case['a']))}\n"
            "输出最小段数，答案置于[answer][/answer]中。"
        )

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        try:
            return int(matches[-1].strip()) if matches else None
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected']
