"""# 

### 谜题描述
Let's call an array a_1, a_2, ..., a_m of nonnegative integer numbers good if a_1 + a_2 + ... + a_m = 2⋅(a_1 ⊕ a_2 ⊕ ... ⊕ a_m), where ⊕ denotes the [bitwise XOR operation](https://en.wikipedia.org/wiki/Bitwise_operation#XOR).

For example, array [1, 2, 3, 6] is good, as 1 + 2 + 3 + 6 = 12 = 2⋅ 6 = 2⋅ (1⊕ 2 ⊕ 3 ⊕ 6). At the same time, array [1, 2, 1, 3] isn't good, as 1 + 2 + 1 + 3 = 7 ≠ 2⋅ 1 = 2⋅(1⊕ 2 ⊕ 1 ⊕ 3).

You are given an array of length n: a_1, a_2, ..., a_n. Append at most 3 elements to it to make it good. Appended elements don't have to be different. It can be shown that the solution always exists under the given constraints. If there are different solutions, you are allowed to output any of them. Note that you don't have to minimize the number of added elements!. So, if an array is good already you are allowed to not append elements.

Input

Each test contains multiple test cases. The first line contains the number of test cases t (1 ≤ t ≤ 10 000). The description of the test cases follows.

The first line of each test case contains a single integer n (1≤ n ≤ 10^5) — the size of the array.

The second line of each test case contains n integers a_1, a_2, ..., a_n (0≤ a_i ≤ 10^9) — the elements of the array.

It is guaranteed that the sum of n over all test cases does not exceed 10^5.

Output

For each test case, output two lines.

In the first line, output a single integer s (0≤ s≤ 3) — the number of elements you want to append.

In the second line, output s integers b_1, ..., b_s (0≤ b_i ≤ 10^{18}) — the elements you want to append to the array.

If there are different solutions, you are allowed to output any of them.

Example

Input


3
4
1 2 3 6
1
8
2
1 1


Output


0

2
4 4
3
2 6 2

Note

In the first test case of the example, the sum of all numbers is 12, and their ⊕ is 6, so the condition is already satisfied.

In the second test case of the example, after adding 4, 4, the array becomes [8, 4, 4]. The sum of numbers in it is 16, ⊕ of numbers in it is 8.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
# Author : raj1307 - Raj Singh
# Date   : 30.12.19

from __future__ import division, print_function

import os,sys
from io import BytesIO, IOBase

if sys.version_info[0] < 3:
    from __builtin__ import xrange as range
    from future_builtins import ascii, filter, hex, map, oct, zip


def ii(): return int(input())
def si(): return input()
def mi(): return map(int,input().strip().split(\" \"))
def msi(): return map(str,input().strip().split(\" \"))
def li(): return list(mi())

def dmain():
    sys.setrecursionlimit(100000000)
    threading.stack_size(40960000)
    thread = threading.Thread(target=main)
    thread.start()
    
#from collections import deque, Counter, OrderedDict,defaultdict
#from heapq import nsmallest, nlargest, heapify,heappop ,heappush, heapreplace
#from math import ceil,floor,log,sqrt,factorial,pi
#from bisect import bisect,bisect_left,bisect_right,insort,insort_left,insort_right
#from decimal import *,threading
#from itertools import permutations

abc='abcdefghijklmnopqrstuvwxyz'
abd={'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25}
mod=1000000007
#mod=998244353
inf = float(\"inf\")
vow=['a','e','i','o','u']
dx,dy=[-1,1,0,0],[0,0,1,-1]
def getKey(item): return item[0] 
def sort2(l):return sorted(l, key=getKey)
def d2(n,m,num):return [[num for x in range(m)] for y in range(n)]
def isPowerOfTwo (x): return (x and (not(x & (x - 1))) )
def decimalToBinary(n): return bin(n).replace(\"0b\",\"\")
def ntl(n):return [int(i) for i in str(n)]

def powerMod(x,y,p):
    res = 1
    x %= p
    while y > 0:
        if y&1:
            res = (res*x)%p
        y = y>>1
        x = (x*x)%p
    return res

def gcd(x, y):
    while y:
        x, y = y, x % y
    return x
    
def isPrime(n) : # Check Prime Number or not 
    if (n <= 1) : return False
    if (n <= 3) : return True
    if (n % 2 == 0 or n % 3 == 0) : return False
    i = 5
    while(i * i <= n) : 
        if (n % i == 0 or n % (i + 2) == 0) : 
            return False
        i = i + 6
    return True



def read():
    sys.stdin = open('input.txt', 'r')  
    sys.stdout = open('output.txt', 'w') 


 
def main():
 
    
    for _ in range(ii()):
        
    
        n=ii()
        a=li()

        s=sum(a)
        x=a[0]
        for i in range(1,n):
            x^=a[i]


        if 2*x==s:
            print(0)
            print()
            continue

        print(2)
        print(x,s+x)














    



























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
    #dmain()

# Comment Read()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cmakegoodbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10, max_element=10**9):
        """
        参数调整：移除ensure_non_trivial参数，允许生成所有合法情况
        """
        self.min_n = min_n
        self.max_n = max_n
        self.max_element = max_element
    
    def case_generator(self):
        """生成包含原始解或需要补元素的通用案例"""
        n = random.randint(self.min_n, self.max_n)
        arr = [random.randint(0, self.max_element) for _ in range(n)]
        
        # 预计算初始状态
        s = sum(arr)
        x = 0
        for num in arr:
            x ^= num
        
        return {
            'array': arr,
            '_sum': s,
            '_xor': x
        }
    
    @staticmethod
    def prompt_func(question_case):
        """完善格式说明：明确数值范围"""
        arr = question_case['array']
        return f"""给定一个长度为 {len(arr)} 的整数数组：{' '.join(map(str, arr))}

请追加最多3个元素（取值范围0-1e18），使其满足：
所有元素之和 = 2 × 所有元素的异或值

输出格式：
第一行：追加的元素数量s (0 ≤ s ≤ 3)
第二行（当s>0时）：s个整数，空格分隔

将最终答案放置在[answer]和[/answer]之间，示例：
[answer]
2
4 4
[/answer]"""

    @staticmethod
    def extract_output(output):
        """严格格式验证"""
        try:
            # 提取最后一个答案块
            answer_block = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)[-1]
            lines = [line.strip() for line in answer_block.strip().split('\n') if line.strip()]
            
            if not lines:
                return None
                
            # 解析元素数量
            s = int(lines[0])
            if not (0 <= s <= 3):
                return None
                
            # 验证行数匹配
            if s == 0:
                if len(lines) != 1:
                    return None
                return []
            else:
                if len(lines) != 2:
                    return None
                elements = list(map(int, lines[1].split()))
                if len(elements) != s:
                    return None
                return elements
        except (IndexError, ValueError, AttributeError):
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        """优化后的验证逻辑"""
        if not solution:  # 允许空列表
            return identity['_sum'] == 2 * identity['_xor']
        
        # 计算最终状态
        added_sum = sum(solution)
        final_sum = identity['_sum'] + added_sum
        final_xor = identity['_xor']
        for num in solution:
            final_xor ^= num
        
        return final_sum == 2 * final_xor
