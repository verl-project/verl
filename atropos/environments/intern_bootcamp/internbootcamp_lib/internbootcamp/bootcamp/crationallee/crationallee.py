"""# 

### 谜题描述
Lee just became Master in Codeforces, and so, he went out to buy some gifts for his friends. He bought n integers, now it's time to distribute them between his friends rationally...

Lee has n integers a_1, a_2, …, a_n in his backpack and he has k friends. Lee would like to distribute all integers in his backpack between his friends, such that the i-th friend will get exactly w_i integers and each integer will be handed over to exactly one friend.

Let's define the happiness of a friend as the sum of the maximum and the minimum integer he'll get.

Lee would like to make his friends as happy as possible, in other words, he'd like to maximize the sum of friends' happiness. Now he asks you to calculate the maximum sum of friends' happiness.

Input

The first line contains one integer t (1 ≤ t ≤ 10^4) — the number of test cases.

Next 3t lines contain test cases — one per three lines.

The first line of each test case contains two integers n and k (1 ≤ n ≤ 2 ⋅ 10^5; 1 ≤ k ≤ n) — the number of integers Lee has and the number of Lee's friends.

The second line of each test case contains n integers a_1, a_2, …, a_n (-10^9 ≤ a_i ≤ 10^9) — the integers Lee has.

The third line contains k integers w_1, w_2, …, w_k (1 ≤ w_i ≤ n; w_1 + w_2 + … + w_k = n) — the number of integers Lee wants to give to each friend. 

It's guaranteed that the sum of n over test cases is less than or equal to 2 ⋅ 10^5.

Output

For each test case, print a single integer — the maximum sum of happiness Lee can achieve.

Example

Input


3
4 2
1 13 7 17
1 3
6 2
10 10 10 10 11 11
3 3
4 4
1000000000 1000000000 1000000000 1000000000
1 1 1 1


Output


48
42
8000000000

Note

In the first test case, Lee should give the greatest integer to the first friend (his happiness will be 17 + 17) and remaining integers to the second friend (his happiness will be 13 + 1).

In the second test case, Lee should give \{10, 10, 11\} to the first friend and to the second friend, so the total happiness will be equal to (11 + 10) + (11 + 10)

In the third test case, Lee has four friends and four integers, it doesn't matter how he distributes the integers between his friends.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division, print_function
from itertools import permutations 
import threading,bisect,math,heapq,sys
from collections import deque
# threading.stack_size(2**27)
# sys.setrecursionlimit(10**4)
from sys import stdin, stdout
i_m=9223372036854775807    
def cin():
    return map(int,sin().split())
def ain():                           #takes array as input
    return list(map(int,sin().split()))
def sin():
    return input()
def inin():
    return int(input()) 
prime=[]
def dfs(n,d,v):
    v[n]=1
    x=d[n]
    for i in x:
        if i not in v:
            dfs(i,d,v)
    return p 
def block(x): 
      
    v = []  
    while (x > 0): 
        v.append(int(x % 2)) 
        x = int(x / 2) 
    ans=[]
    for i in range(0, len(v)): 
        if (v[i] == 1): 
            ans.append(2**i)  
    return ans 
\"\"\"**************************MAIN*****************************\"\"\"
def main():
    t=inin()
    for _ in range(t):
        n,k=cin()
        a=ain()
        w=ain()
        a.sort(reverse=True)
        w.sort()
        p=a[:k]
        x=k-1
        ans=0
        for i in range(k):
            if w[i]==1:
                ans+=2*p[i]
            else:
                ans+=p[i]
                x+=w[i]-1
                ans+=a[x]
        print(ans)

\"\"\"***********************************************\"\"\"
def intersection(l,r,ll,rr):
    # print(l,r,ll,rr)
    if (ll > r or rr < l): 
            return 0
    else: 
        l = max(l, ll) 
        r = min(r, rr)
    return max(0,r-l+1) 
######## Python 2 and 3 footer by Pajenegod and c1729
fac=[]
def fact(n,mod):
    global fac
    fac.append(1)
    for i in range(1,n+1):
        fac.append((fac[i-1]*i)%mod)
    f=fac[:]
    return f
def nCr(n,r,mod):
    global fac
    x=fac[n]
    y=fac[n-r]
    z=fac[r]
    x=moddiv(x,y,mod)
    return moddiv(x,z,mod)
def moddiv(m,n,p):
    x=pow(n,p-2,p)
    return (m*x)%p
def GCD(x, y): 
    x=abs(x)
    y=abs(y)
    if(min(x,y)==0):
        return max(x,y)
    while(y): 
        x, y = y, x % y 
    return x 
def Divisors(n) : 
    l = []  
    ll=[]
    for i in range(1, int(math.sqrt(n) + 1)) :
        if (n % i == 0) : 
            if (n // i == i) : 
                l.append(i) 
            else : 
                l.append(i)
                ll.append(n//i)
    l.extend(ll[::-1])
    return l
def SieveOfEratosthenes(n): 
    global prime
    prime = [True for i in range(n+1)] 
    p = 2
    while (p * p <= n): 
        if (prime[p] == True): 
            for i in range(p * p, n+1, p): 
                prime[i] = False
        p += 1
    f=[]
    for p in range(2, n): 
        if prime[p]: 
            f.append(p)
    return f
def primeFactors(n): 
    a=[]
    while n % 2 == 0: 
        a.append(2) 
        n = n // 2 
    for i in range(3,int(math.sqrt(n))+1,2):  
        while n % i== 0: 
            a.append(i) 
            n = n // i  
    if n > 2: 
        a.append(n)
    return a
\"\"\"*******************************************************\"\"\"
py2 = round(0.5)
if py2:
    from future_builtins import ascii, filter, hex, map, oct, zip
    range = xrange
import os
from io import IOBase, BytesIO
BUFSIZE = 8192
class FastIO(BytesIO):
    newlines = 0
 
    def __init__(self, file):
        self._file = file
        self._fd = file.fileno()
        self.writable = \"x\" in file.mode or \"w\" in file.mode
        self.write = super(FastIO, self).write if self.writable else None
 
    def _fill(self):
        s = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
        self.seek((self.tell(), self.seek(0,2), super(FastIO, self).write(s))[0])
        return s
 
    def read(self):
        while self._fill(): pass
        return super(FastIO,self).read()
 
    def readline(self):
        while self.newlines == 0:
            s = self._fill(); self.newlines = s.count(b\"\n\") + (not s)
        self.newlines -= 1
        return super(FastIO, self).readline()
 
    def flush(self):
        if self.writable:
            os.write(self._fd, self.getvalue())
            self.truncate(0), self.seek(0)
 
class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        if py2:
            self.write = self.buffer.write
            self.read = self.buffer.read
            self.readline = self.buffer.readline
        else:
            self.write = lambda s:self.buffer.write(s.encode('ascii'))
            self.read = lambda:self.buffer.read().decode('ascii')
            self.readline = lambda:self.buffer.readline().decode('ascii')
 
sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
input = lambda: sys.stdin.readline().rstrip('\r\n') 
# Cout implemented in Python
class ostream:
    def __lshift__(self,a):
        sys.stdout.write(str(a))
        return self
cout = ostream()
endl = '\n'
 
# Read all remaining integers in stdin, type is given by optional argument, this is fast
def readnumbers(zero = 0):
    conv = ord if py2 else lambda x:x
    A = []; numb = zero; sign = 1; i = 0; s = sys.stdin.buffer.read()
    try:
        while True:
            if s[i] >= b'R' [0]:
                numb = 10 * numb + conv(s[i]) - 48
            elif s[i] == b'-' [0]: sign = -1
            elif s[i] != b'\r' [0]:
                A.append(sign*numb)
                numb = zero; sign = 1
            i += 1
    except:pass
    if s and s[-1] >= b'R' [0]:
        A.append(sign*numb)
    return A
 
# threading.Thread(target=main).start()
if __name__== \"__main__\":
  main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
# 确认原有代码正确，无修正必要。以下为原封不动的实现代码。

import random
import re
from bootcamp import Basebootcamp

class Crationalleebootcamp(Basebootcamp):
    def __init__(self, n_min=2, n_max=20, a_min=-10**9, a_max=10**9, **params):
        super().__init__(**params)
        self.n_min = n_min
        self.n_max = n_max
        self.a_min = a_min
        self.a_max = a_max
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        k = random.randint(1, n)
        # Generate w array with sum n, each >=1
        weights = [1] * k
        remaining = n - k
        for _ in range(remaining):
            idx = random.randint(0, k-1)
            weights[idx] += 1
        # Generate a array with n integers
        a = [random.randint(self.a_min, self.a_max) for _ in range(n)]
        return {
            'n': n,
            'k': k,
            'a': a,
            'w': weights
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k = question_case['k']
        a = ' '.join(map(str, question_case['a']))
        w = ' '.join(map(str, question_case['w']))
        problem = f"""Lee has {n} integers to distribute among his {k} friends. Each friend must receive exactly the specified number of integers. The happiness of a friend is the sum of the maximum and minimum integers they receive. Your task is to find the maximum possible total happiness.

Input for this case:
- First line: {n} {k}
- Second line: {a}
- Third line: {w}

Please compute the maximum sum of happiness and provide the numerical answer within [answer] tags. For example: [answer]123[/answer]."""
        return problem
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_answer = matches[-1].strip()
        try:
            return int(last_answer)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        a = identity['a']
        w = identity['w']
        n = identity['n']
        k = identity['k']
        # Sort a in descending order
        a_sorted = sorted(a, reverse=True)
        w_sorted = sorted(w)
        p = a_sorted[:k]
        x = k - 1
        ans = 0
        for i in range(k):
            wi = w_sorted[i]
            if wi == 1:
                ans += 2 * p[i]
            else:
                ans += p[i]
                x += wi - 1
                ans += a_sorted[x]
        return solution == ans
