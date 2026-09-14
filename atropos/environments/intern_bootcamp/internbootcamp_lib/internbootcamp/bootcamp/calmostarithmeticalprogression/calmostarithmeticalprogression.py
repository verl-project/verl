"""# 

### 谜题描述
Gena loves sequences of numbers. Recently, he has discovered a new type of sequences which he called an almost arithmetical progression. A sequence is an almost arithmetical progression, if its elements can be represented as:

  * a1 = p, where p is some integer; 
  * ai = ai - 1 + ( - 1)i + 1·q (i > 1), where q is some integer. 



Right now Gena has a piece of paper with sequence b, consisting of n integers. Help Gena, find there the longest subsequence of integers that is an almost arithmetical progression.

Sequence s1, s2, ..., sk is a subsequence of sequence b1, b2, ..., bn, if there is such increasing sequence of indexes i1, i2, ..., ik (1 ≤ i1 < i2 < ... < ik ≤ n), that bij = sj. In other words, sequence s can be obtained from b by crossing out some elements.

Input

The first line contains integer n (1 ≤ n ≤ 4000). The next line contains n integers b1, b2, ..., bn (1 ≤ bi ≤ 106).

Output

Print a single integer — the length of the required longest subsequence.

Examples

Input

2
3 5


Output

2


Input

4
10 20 10 30


Output

3

Note

In the first test the sequence actually is the suitable subsequence. 

In the second test the following subsequence fits: 10, 20, 10.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
\"\"\"
// Author : snape_here - Susanta Mukherjee
     
 \"\"\"

from __future__ import division, print_function
 
import os,sys
from io import BytesIO, IOBase
 
if sys.version_info[0] < 3:
    from __builtin__ import xrange as range
    from future_builtins import ascii, filter, hex, map, oct, zip
 
def ii(): return int(input())
def fi(): return float(input())
def si(): return input()
def msi(): return map(str,input().split())
def mi(): return map(int,input().split())
def li(): return list(mi())
def lsi(): return list(msi())
 
def read():
    sys.stdin = open('input.txt', 'r')  
    sys.stdout = open('output.txt', 'w') 
 
def gcd(x, y):
    while y:
        x, y = y, x % y
    return x

def lcm(x, y):
    return (x*y)//(gcd(x,y))

mod=1000000007

def modInverse(b,m): 
    g = gcd(b, m)  
    if (g != 1):         
        return -1
    else:          
        return pow(b, m - 2, m) 

def ceil2(x,y):
    if x%y==0:
        return x//y
    else:
        return x//y+1

def modu(a,b,m): 

    a = a % m 
    inv = modInverse(b,m) 
    if(inv == -1): 
        return -999999999
    else: 
        return (inv*a)%m

from math import log,factorial,cos,tan,sin,radians,floor,sqrt,ceil

import bisect
import random
import string 

from decimal import *

getcontext().prec = 50

abc=\"abcdefghijklmnopqrstuvwxyz\"

pi=3.141592653589793238

def gcd1(a):

    if len(a) == 1:
        return a[0]

    ans = a[0]
    for i in range(1,len(a)):
        ans = gcd(ans,a[i])

    return ans

def mykey(x):
    return len(x)

def main():

    for _ in range(1):
        n=ii()
        a=li()
        d=dict()
        ind = -1
        for i in a:
            if i in d:
                pass
            else:
                ind += 1
                d[i] = ind 
        for i in range(n):
            a[i] = d[a[i]]
        #print(a)
        dp = []
        for i in range(n):
            c = [1]*n 
            dp.append(c)
        for i in range(n):
            for j in range(i):
                dp[i][a[j]] = max(1+dp[j][a[i]],dp[i][a[j]])
        ans = 0 
        for i in range(n):
            for j in range(n):
                ans = max(ans, dp[i][j])
        print(ans)


        # print(\"Case #\",end=\"\")
        # print(_+1,end=\"\")
        # print(\": \",end=\"\")
        # print(ans)



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
    #read()
    main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def solve_almost_arithmetic_progression(n, a):
    # 优化后的解题算法支持更高效的验证
    if n <= 1:
        return n
    
    value_map = {}
    idx = 0
    for num in a:
        if num not in value_map:
            value_map[num] = idx
            idx += 1
    compressed = [value_map[num] for num in a]
    
    dp = [[1] * idx for _ in range(n)]
    max_len = 1
    
    for i in range(n):
        for j in range(i):
            prev_val = compressed[j]
            current_val = compressed[i]
            dp[i][prev_val] = max(dp[i][prev_val], dp[j][current_val] + 1)
            max_len = max(max_len, dp[i][prev_val])
    
    return max_len

class Calmostarithmeticalprogressionbootcamp(Basebootcamp):
    CASE_TYPES = ['random', 'all_same', 'full_aap', 'alternating', 'minimal']
    
    def __init__(self, min_n=1, max_n=4000, min_val=1, max_val=10**6):
        self.min_n = max(1, min_n)  # 确保符合题目约束n≥1
        self.max_n = min(4000, max_n)  # 遵守题目最大限制
        self.min_val = min_val
        self.max_val = max_val
    
    def case_generator(self):
        case_type = random.choice(self.CASE_TYPES)
        
        # 特殊处理极小案例
        if case_type == 'minimal':
            n = random.choice([1, 2])
            array = [random.randint(self.min_val, self.max_val) for _ in range(n)]
            if n == 2 and random.random() > 0.5:
                array[1] = array[0]  # 50%概率生成全同序列
        else:
            n = random.randint(self.min_n, self.max_n)
            if case_type == 'random':
                array = [random.randint(self.min_val, self.max_val) for _ in range(n)]
            
            elif case_type == 'all_same':
                val = random.randint(self.min_val, self.max_val)
                array = [val] * n
            
            elif case_type == 'full_aap':
                p = random.randint(self.min_val, self.max_val)
                q = random.randint(1, self.max_val//2)  # 确保q≠0
                array = [p]
                for i in range(2, n+1):
                    sign = (-1) ** (i + 1)
                    array.append(array[-1] + sign * q)
            
            elif case_type == 'alternating':
                base = random.sample(range(self.min_val, self.max_val+1), 2)
                array = [base[i%2] for i in range(n)]
        
        expected_length = solve_almost_arithmetic_progression(n, array)
        return {
            'n': n,
            'array': array.copy(),
            'expected_length': expected_length
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        array = question_case['array']
        return f"""Gena的几乎等差数列定义如下：
1. 首项a₁是任意整数p
2. 后续项满足aᵢ = aᵢ₋₁ + (-1)^(i+1)*q（q为整数）

给定长度为{n}的整数序列：[{', '.join(map(str, array))}]
请找出其中最长的满足条件的子序列长度

答案请用[answer]答案[/answer]包裹，例如：[answer]5[/answer]"""

    @staticmethod
    def extract_output(output):
        try:
            matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
            if matches:
                value = matches[-1].strip()
                if '.' in value:  # 处理可能的浮点格式
                    return int(float(value))
                return int(value)
        except (ValueError, TypeError):
            pass
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return int(solution) == identity['expected_length']
        except:
            return False
