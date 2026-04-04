"""# 

### 谜题描述
Now you get Baby Ehab's first words: \"Given an integer n, find the longest subsequence of [1,2, …, n-1] whose product is 1 modulo n.\" Please solve the problem.

A sequence b is a subsequence of an array a if b can be obtained from a by deleting some (possibly all) elements. The product of an empty subsequence is equal to 1.

Input

The only line contains the integer n (2 ≤ n ≤ 10^5).

Output

The first line should contain a single integer, the length of the longest subsequence.

The second line should contain the elements of the subsequence, in increasing order.

If there are multiple solutions, you can print any.

Examples

Input


5


Output


3
1 2 3 

Input


8


Output


4
1 3 5 7 

Note

In the first example, the product of the elements is 6 which is congruent to 1 modulo 5. The only longer subsequence is [1,2,3,4]. Its product is 24 which is congruent to 4 modulo 5. Hence, the answer is [1,2,3].

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/usr/bin/env python
# https://github.com/cheran-senthil/PyRival
from __future__ import division, print_function

import os
import sys
from io import BytesIO, IOBase
from collections import defaultdict

if sys.version_info[0] < 3:
    from __builtin__ import xrange as range
    from future_builtins import ascii, filter, hex, map, oct, zip


def main():
    import math

    def prime_factors(n):
        factors = []
        while n % 2 == 0:
            factors.append(2)
            n //= 2
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            while n % i == 0:
                factors.append(i)
                n //= i
        if n > 2:
            factors.append(n)
        return factors

    N = int(input())
    s = [True] * N
    for p in set(prime_factors(N)):
        for k in range(p, N, p):
            s[k] = False
    p = 1
    ans = []
    for j in range(1, N):
        if s[j]:
            p = (p * j) % N
            ans.append(j)
    if p != 1:
        ans.remove(p)
    print(len(ans))
    print(' '.join(map(str, ans)))


# region fastio

BUFSIZE = 8192


class FastIO(IOBase):
    newlines = 0

    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = \"x\" in file.mode or \"r\" not in file.mode
        self.write = self.buffer.write if self.writable else None

    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()

    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b\"\n\") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()

    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)


class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode(\"ascii\"))
        self.read = lambda: self.buffer.read().decode(\"ascii\")
        self.readline = lambda: self.buffer.readline().decode(\"ascii\")


def print(*args, **kwargs):
    \"\"\"Prints the values to a stream, or to sys.stdout by default.\"\"\"
    sep, file = kwargs.pop(\"sep\", \" \"), kwargs.pop(\"file\", sys.stdout)
    at_start = True
    for x in args:
        if not at_start:
            file.write(sep)
        file.write(str(x))
        at_start = False
    file.write(kwargs.pop(\"end\", \"\n\"))
    if kwargs.pop(\"flush\", False):
        file.flush()


if sys.version_info[0] < 3:
    sys.stdin, sys.stdout = FastIO(sys.stdin), FastIO(sys.stdout)
else:
    sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)

input = lambda: sys.stdin.readline().rstrip(\"\r\n\")

# endregion

if __name__ == \"__main__\":
    main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import random
from bootcamp import Basebootcamp

class Cproduct1modulonbootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=10**5):
        self.min_n = max(2, min_n)
        self.max_n = min(10**5, max_n)
    
    def case_generator(self):
        # 基础随机生成（80%概率）
        if random.random() < 0.8:
            n = random.randint(self.min_n, self.max_n)
        else:  # 特殊案例生成（20%概率）
            candidates = self._generate_special_candidates()
            n = random.choice(candidates) if candidates else random.randint(self.min_n, self.max_n)
        return {'n': n}
    
    def _generate_special_candidates(self):
        candidates = []
        
        # 质数列表（动态生成范围内）
        primes = [p for p in [2,3,5,7,11,13,17,19,23,29] if self.min_n <= p <= self.max_n]
        if primes:
            candidates.append(random.choice(primes))
        
        # 平方数生成
        max_square_root = int(math.sqrt(self.max_n))
        squares = [x*x for x in range(2, max_square_root+1) if x*x >= self.min_n]
        if squares:
            candidates.append(random.choice(squares))
        
        # 合数生成（至少两个不同质因数）
        composites = []
        for x in range(max(4, self.min_n), min(100, self.max_n)+1):
            factors = set()
            num = x
            for i in [2,3,5,7]:
                if num % i == 0:
                    factors.add(i)
                    while num % i == 0:
                        num //= i
            if num > 1:
                factors.add(num)
            if len(factors) >= 2:
                composites.append(x)
        if composites:
            candidates.append(random.choice(composites))
            
        return candidates if candidates else [random.randint(self.min_n, self.max_n)]
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        return f"""Given integer n={n}, find the longest increasing subsequence of [1,2,...,{n-1}] where the product ≡1 mod n.

Output format:
[answer]
<length>
<sorted elements>
[/answer]"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
            
        last_match = matches[-1].strip()
        lines = [line.strip() for line in last_match.split('\n') if line.strip()]
        if len(lines) < 2:
            return None
            
        try:
            length = int(lines[0])
            elements = list(map(int, lines[1].split()))
            if len(elements) != length or elements != sorted(elements):
                return None
            return elements
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        
        # 基础校验
        if not solution or len(solution) == 0:
            return False
        if any(e < 1 or e >= n for e in solution):
            return False
        if solution != sorted(solution) or len(solution) != len(set(solution)):
            return False
        
        # 互质验证
        prime_factors = set()
        num = n
        for p in [2,3,5,7,11,13,17,19]:
            if num % p == 0:
                prime_factors.add(p)
                while num % p == 0:
                    num //= p
        if num > 1:
            prime_factors.add(num)
        
        for e in solution:
            if any(e % p == 0 for p in prime_factors):
                return False
        
        # 乘积验证
        product = 1
        for x in solution:
            product = (product * x) % n
        
        return product == 1 and len(solution) == cls._calculate_expected_length(n)
    
    @classmethod
    def _calculate_expected_length(cls, n):
        # 计算理论最大长度
        coprimes = [x for x in range(1, n) if math.gcd(x, n) == 1]
        product = 1
        for x in coprimes:
            product = (product * x) % n
        return len(coprimes) - (0 if product == 1 else 1)
