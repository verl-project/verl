"""# 

### 谜题描述
You are given an array a consisting of n non-negative integers. You have to choose a non-negative integer x and form a new array b of size n according to the following rule: for all i from 1 to n, b_i = a_i ⊕ x (⊕ denotes the operation [bitwise XOR](https://en.wikipedia.org/wiki/Bitwise_operation#XOR)).

An inversion in the b array is a pair of integers i and j such that 1 ≤ i < j ≤ n and b_i > b_j.

You should choose x in such a way that the number of inversions in b is minimized. If there are several options for x — output the smallest one.

Input

First line contains a single integer n (1 ≤ n ≤ 3 ⋅ 10^5) — the number of elements in a.

Second line contains n space-separated integers a_1, a_2, ..., a_n (0 ≤ a_i ≤ 10^9), where a_i is the i-th element of a.

Output

Output two integers: the minimum possible number of inversions in b, and the minimum possible value of x, which achieves those number of inversions.

Examples

Input


4
0 1 3 2


Output


1 0


Input


9
10 7 9 10 7 5 5 3 5


Output


4 14


Input


3
8 10 3


Output


0 8

Note

In the first sample it is optimal to leave the array as it is by choosing x = 0.

In the second sample the selection of x = 14 results in b: [4, 9, 7, 4, 9, 11, 11, 13, 11]. It has 4 inversions:

  * i = 2, j = 3; 
  * i = 2, j = 4; 
  * i = 3, j = 4; 
  * i = 8, j = 9. 



In the third sample the selection of x = 8 results in b: [0, 2, 11]. It has no inversions.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division, print_function
 
import os
import sys
from io import BytesIO, IOBase
 
if sys.version_info[0] < 3:
    from __builtin__ import xrange as range
    from future_builtins import ascii, filter, hex, map, oct, zip
 
 
def main():
    n=int(input())
    a=list(map(int,input().split()))
    v=30
    t=x=0
    while v:
        u=d=0
        r={}
        w=1<<(v-1)
        for i in a:
            p=i>>v
            b=i&w
            if b:r[2*p+1]=1+r.get(2*p+1,0)
            else:d+=r.get(2*p+1,0)
            r[2*p]=1+r.get(2*p,0)
        for p in r:
            if p%2:
                rp,cp=r.get(p,0),r.get(p-1,0)
                u+=(cp*(cp-1))//2-(rp*(rp-1))//2-((cp-rp)*(cp-rp-1))//2
        if d>u-d:
            x+=w
            d=u-d
        t+=d
        v-=1
    print(t,x)
 
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
import re
import random
from bootcamp import Basebootcamp

def solve_min_inversions(n, a):
    v = 30
    t = x = 0
    while v >= 0:  # 修正循环条件
        u = d = 0
        r = {}
        w = 1 << v
        for i in a:
            p = i >> (v + 1)
            b = i & w
            if b:
                key = 2*p + 1
                r[key] = r.get(key, 0) + 1
                d += r.get(2*p, 0)  # 修正d计算逻辑
            else:
                key = 2*p
                r[key] = r.get(key, 0) + 1
                d += r.get(2*p + 1, 0)  # 修正d计算逻辑
        for p in r:
            if p % 2:
                rp = r[p]
                cp = r.get(p-1, 0)
                u += (cp*(cp-1))//2 - (rp*(rp-1))//2 - ((cp-rp)*(cp-rp-1))//2
        if d > (u - d):
            x += w
            d = u - d
        t += d
        v -= 1
    return t, x

class Cxorinversebootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=1000, a_min=0, a_max=10**9):
        self.n_min = n_min
        self.n_max = n_max
        self.a_min = a_min
        self.a_max = a_max

    def case_generator(self):
        generation_strategy = random.choice([
            'zeros', 'uniform', 'random', 'high_bit_variation'
        ])
        
        n = random.randint(self.n_min, self.n_max)
        
        if generation_strategy == 'zeros':
            a = [0] * n
        elif generation_strategy == 'uniform':
            val = random.randint(self.a_min, self.a_max)
            a = [val] * n
        elif generation_strategy == 'high_bit_variation':
            base = random.randint(0, 1 << 20)
            a = [base ^ (random.randint(0, 1) << 30) for _ in range(n)]
        else:
            a = [random.randint(self.a_min, self.a_max) for _ in range(n)]
        
        t, x = solve_min_inversions(n, a)
        return {
            'n': n,
            'a': a,
            'expected_inversions': t,
            'optimal_x': x
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a_str = ' '.join(map(str, question_case['a']))
        return f"""You are given an array of {n} non-negative integers. Choose a non-negative integer x to form a new array b where each element b_i = a_i XOR x. Your goal is to minimize the number of inversions in b. If multiple x yield the same minimum, choose the smallest x.

Input:
{n}
{a_str}

Output format:
<inversion_count> <x>

Put your final answer within [answer] and [/answer] tags. Example: [answer]3 5[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            last = matches[-1].strip().split()
            return (int(last[0]), int(last[1]))
        except (ValueError, IndexError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or len(solution) != 2:
            return False
        return (solution[0] == identity['expected_inversions'] and 
                solution[1] == identity['optimal_x'])
