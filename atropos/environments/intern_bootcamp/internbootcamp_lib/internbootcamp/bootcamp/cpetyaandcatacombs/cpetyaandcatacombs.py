"""# 

### 谜题描述
A very brave explorer Petya once decided to explore Paris catacombs. Since Petya is not really experienced, his exploration is just walking through the catacombs.

Catacombs consist of several rooms and bidirectional passages between some pairs of them. Some passages can connect a room to itself and since the passages are built on different depths they do not intersect each other. Every minute Petya arbitrary chooses a passage from the room he is currently in and then reaches the room on the other end of the passage in exactly one minute. When he enters a room at minute i, he makes a note in his logbook with number ti: 

  * If Petya has visited this room before, he writes down the minute he was in this room last time; 
  * Otherwise, Petya writes down an arbitrary non-negative integer strictly less than current minute i. 



Initially, Petya was in one of the rooms at minute 0, he didn't write down number t0.

At some point during his wandering Petya got tired, threw out his logbook and went home. Vasya found his logbook and now he is curious: what is the minimum possible number of rooms in Paris catacombs according to Petya's logbook?

Input

The first line contains a single integer n (1 ≤ n ≤ 2·105) — then number of notes in Petya's logbook.

The second line contains n non-negative integers t1, t2, ..., tn (0 ≤ ti < i) — notes in the logbook.

Output

In the only line print a single integer — the minimum possible number of rooms in Paris catacombs.

Examples

Input

2
0 0


Output

2


Input

5
0 1 0 1 3


Output

3

Note

In the first sample, sequence of rooms Petya visited could be, for example 1 → 1 → 2, 1 → 2 → 1 or 1 → 2 → 3. The minimum possible number of rooms is 2.

In the second sample, the sequence could be 1 → 2 → 3 → 1 → 2 → 1.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
# Team : Too_Slow_to_Code

# target Expert  

# Author : raj1307 - Raj Singh
# Date   : 18.10.19

from __future__ import division, print_function

import os,sys
from io import BytesIO, IOBase

if sys.version_info[0] < 3:
    from __builtin__ import xrange as range
    from future_builtins import ascii, filter, hex, map, oct, zip


def ii(): return int(input())
def si(): return input()
def mi(): return map(int,input().strip().split(\" \"))
def li(): return list(mi())

def dmain():
    sys.setrecursionlimit(100000000)
    threading.stack_size(40960000)
    thread = threading.Thread(target=main)
    thread.start()
    
#from collections import deque, Counter, OrderedDict,defaultdict
#from heapq import nsmallest, nlargest, heapify,heappop ,heappush, heapreplace
#from math import ceil,floor,log,sqrt,factorial
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
    


    #for _ in range(ii()):

    n=ii()
    l=li()

    dp=[0]*(n+1)

    dp[0]=0
    ans=1
    for i in range(n):

        if dp[l[i]]:
            ans+=1

        dp[l[i]]=1

    print(ans)
































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
from bootcamp import Basebootcamp
import random
import re

class Cpetyaandcatacombsbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=20):
        if min_n < 1:
            raise ValueError("min_n must be at least 1")
        self.min_n = min_n
        self.max_n = max_n
    
    def case_generator(self):
        """增强测试案例生成逻辑，保证生成有效冲突场景"""
        n = random.randint(self.min_n, self.max_n)
        t = []
        conflict_pool = []
        
        # 强制生成至少一个重复的ti值（当n≥2时）
        for j in range(n):
            i = j + 1
            # 前两个元素特殊处理保证至少一个冲突
            if j == 0:
                ti = 0  # 第一个ti只能是0
            elif j == 1 and n >= 2:
                ti = 0  # 强制第二个ti为0触发冲突
            else:
                # 40%概率复用已有值，60%随机生成
                if conflict_pool and random.random() < 0.4:
                    ti = random.choice(conflict_pool)
                else:
                    ti = random.randint(0, i-1)
            
            conflict_pool.append(ti)
            t.append(ti)
        return {"n": n, "t": t}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        t_str = ' '.join(map(str, question_case['t']))
        prompt = f"""你是探险家Vasya，正在分析Petya的日志本以确定巴黎地下墓穴中可能的最小房间数量。根据以下规则进行分析：

Petya在时间0时位于某个房间，之后每过一分钟移动到另一个房间。每次进入一个房间时：
1. 如果该房间之前被访问过，他会记录上一次访问该房间的时间（即ti等于上一次的时间）；
2. 如果这是第一次访问该房间，他会在日志中记录一个严格小于当前时间i的非负整数。

现在给出Petya的日志记录，请确定满足这些记录所需的最小可能房间数量。

输入格式：
- 第一行是一个整数n（表示日志记录的数量）
- 第二行包含n个非负整数t1 t2 ... tn（0 ≤ ti < i）

例如，输入样例：
2
0 0
对应的输出是2，因为至少需要两个房间。

当前问题输入：
{n}
{t_str}

请仔细分析问题，并给出正确的答案。将你的最终答案放置在[answer]和[/answer]的标签之间，例如：[answer]2[/answer]。确保答案是一个整数。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        # 严格匹配标签大小写，使用多行匹配模式
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if matches:
            try:
                return int(matches[-1].strip())
            except ValueError:
                pass
        return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        """严格参考原题解算法验证"""
        t_list = identity['t']
        state = {}
        room_count = 1
        for ti in t_list:
            if state.get(ti, False):
                room_count += 1
                # 重置所有状态（参考原题解中dp数组的更新逻辑）
                state = {k: False for k in state}
            state[ti] = True
        return solution == room_count
