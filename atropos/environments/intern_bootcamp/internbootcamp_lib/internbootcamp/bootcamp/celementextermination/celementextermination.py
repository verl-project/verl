"""# 

### 谜题描述
You are given an array a of length n, which initially is a permutation of numbers from 1 to n. In one operation, you can choose an index i (1 ≤ i < n) such that a_i < a_{i + 1}, and remove either a_i or a_{i + 1} from the array (after the removal, the remaining parts are concatenated). 

For example, if you have the array [1, 3, 2], you can choose i = 1 (since a_1 = 1 < a_2 = 3), then either remove a_1 which gives the new array [3, 2], or remove a_2 which gives the new array [1, 2].

Is it possible to make the length of this array equal to 1 with these operations?

Input

The first line contains a single integer t (1 ≤ t ≤ 2 ⋅ 10^4) — the number of test cases. The description of the test cases follows.

The first line of each test case contains a single integer n (2 ≤ n ≤ 3 ⋅ 10^5) — the length of the array.

The second line of each test case contains n integers a_1, a_2, ..., a_n (1 ≤ a_i ≤ n, a_i are pairwise distinct) — elements of the array.

It is guaranteed that the sum of n over all test cases doesn't exceed 3 ⋅ 10^5.

Output

For each test case, output on a single line the word \"YES\" if it is possible to reduce the array to a single element using the aforementioned operation, or \"NO\" if it is impossible to do so.

Example

Input


4
3
1 2 3
4
3 1 2 4
3
2 3 1
6
2 4 6 1 3 5


Output


YES
YES
NO
YES

Note

For the first two test cases and the fourth test case, we can operate as follow (the bolded elements are the pair chosen for that operation):

[1, 2, 3] → [1, 2] → [1]

[3, 1, 2, 4] → [3, 1, 4] → [3, 4] → [4]

[2, 4, 6, 1, 3, 5] → [4, 6, 1, 3, 5] → [4, 1, 3, 5] → [4, 1, 5] → [4, 5] → [4]

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
\"\"\"
    Satwik_Tiwari ;) .
    4th july , 2020  - Saturday
\"\"\"

#===============================================================================================
#importing some useful libraries.
from __future__ import division, print_function

from fractions import Fraction
import sys
import os
from io import BytesIO, IOBase


import bisect
from heapq import *
from math import *
from collections import deque
from collections import Counter as counter  # Counter(list)  return a dict with {key: count}
from itertools import combinations as comb # if a = [1,2,3] then print(list(comb(a,2))) -----> [(1, 2), (1, 3), (2, 3)]
from itertools import permutations as permutate
from bisect import bisect_left as bl
#If the element is already present in the list,
# the left most position where element has to be inserted is returned.
from bisect import bisect_right as br
from bisect import bisect
#If the element is already present in the list,
# the right most position where element has to be inserted is returned

#==============================================================================================

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

# inp = lambda: sys.stdin.readline().rstrip(\"\r\n\")

#===============================================================================================
#some shortcuts

mod = 1000000007
def inp(): return sys.stdin.readline().rstrip(\"\r\n\") #for fast input
def out(var): sys.stdout.write(str(var))  #for fast output, always take string
def lis(): return list(map(int, inp().split()))
def stringlis(): return list(map(str, inp().split()))
def sep(): return map(int, inp().split())
def strsep(): return map(str, inp().split())
# def graph(vertex): return [[] for i in range(0,vertex+1)]
def zerolist(n): return [0]*n
def nextline(): out(\"\n\")  #as stdout.write always print sring.
def testcase(t):
    for p in range(t):
        solve()
def printlist(a) :
    for p in range(0,len(a)):
        out(str(a[p]) + ' ')
def lcm(a,b): return (a*b)//gcd(a,b)
def power(a,b):
    ans = 1
    while(b>0):
        if(b%2==1):
            ans*=a
        a*=a
        b//=2
    return ans
def ncr(n,r): return factorial(n)//(factorial(r)*factorial(max(n-r,1)))
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

#===============================================================================================
# code here ;))
def bs(a,l,h,x):
    while(l<h):
        # print(l,h)
        mid = (l+h)//2
        if(a[mid] == x):
            return mid
        if(a[mid] < x):
            l = mid+1
        else:
            h = mid
    return l

def sieve(a): #O(n loglogn) nearly linear
    #all odd mark 1
    for i in range(3,((10**6)+1),2):
        a[i] = 1
    #marking multiples of i form i*i 0. they are nt prime
    for i in range(3,((10**6)+1),2):
        for j in range(i*i,((10**6)+1),i):
            a[j] = 0
    a[2] = 1 #special left case
    return (a)


def bfs(g,st):
    visited = [-1]*(len(g))
    visited[st] = 0
    queue = []
    queue.append(st)
    new = []
    while(len(queue) != 0):
        s = queue.pop()
        new.append(s)
        for i in g[s]:
            if(visited[i] == -1):
                visited[i] = visited[s]+1
                queue.append(i)
    return visited


def dfsusingstack(v,st):
    d = deque([])
    visited = [0]*(len(v))
    d.append(st)
    new = []
    visited[st] = 1
    while(len(d) != 0):
        curr = d.pop()
        new.append(curr)
        for i in v[curr]:
            if(visited[i] == 0):
                visited[i] = 1
                d.append(i)
    return new


def solve():
    n = int(inp())
    a= lis()
    f = True
    dp = [0]*(n)
    # dp[0] = 0
    curr = 0
    for i in range(1,n):
        if(a[i] < a[curr]):
            dp[i] = i
            curr = i
        else:
            dp[i] = curr

    end = n-2
    while(f):
        ind = dp[end]
        # print(end,ind)
        if(a[ind] > a[n-1]):
            print('NO')
            return
        end = ind-1
        if(end<0):
            f = False
            
    print('YES')





# testcase(1)
testcase(int(inp()))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Celementexterminationbootcamp(Basebootcamp):
    def __init__(self, min_length=2, max_length=10):
        self.min_length = min_length
        self.max_length = max_length
    
    def case_generator(self):
        n = random.randint(self.min_length, self.max_length)
        a = list(range(1, n + 1))
        random.shuffle(a)
        expected_answer = 'YES' if self.check_solution(a.copy()) else 'NO'  # 修复1：使用副本保证原始数据不变
        return {
            'n': n,
            'array': a,
            'expected_answer': expected_answer
        }
    
    @staticmethod 
    def check_solution(a):
        n = len(a)
        if n == 1:
            return True
        
        # 修复2：修正循环终止条件（参考原题正确解法）
        max_val = a[-1]
        for i in reversed(range(n-1)):
            if a[i] < max_val:
                max_val = a[i]
            else:
                return False
        return True
    
    @staticmethod
    def prompt_func(question_case) -> str:
        problem_desc = (
            "You are given an array a of length n, which is a permutation of numbers from 1 to n. "
            "In each operation, you can choose an index i (1 ≤ i < n) where a_i < a_{{i+1}}, "
            "and remove either a_i or a_{{i+1}}. The goal is to determine if it's possible to reduce "
            "the array to a single element using these operations.\n\n"
            "Input format:\n"
            "- The first line contains n (array length)\n"
            "- The second line contains the array elements\n\n"
            "Output 'YES' or 'NO'.\n\n"
            "Your task:\n"
            "Test case:\n"
            "n = {n}\n"
            "array: {a}\n\n"
            "Put your final answer within [answer] and [/answer] tags."
        ).format(
            n=question_case['n'],
            a=' '.join(map(str, question_case['array']))
        )
        return problem_desc
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(YES|NO)\s*\[/answer\]', output, re.IGNORECASE)
        return matches[-1].upper() if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected_answer']
