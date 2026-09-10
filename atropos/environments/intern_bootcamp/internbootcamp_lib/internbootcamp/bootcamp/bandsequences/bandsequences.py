"""# 

### 谜题描述
A sequence of n non-negative integers (n ≥ 2) a_1, a_2, ..., a_n is called good if for all i from 1 to n-1 the following condition holds true: $$$a_1 \: \& \: a_2 \: \& \: ... \: \& \: a_i = a_{i+1} \: \& \: a_{i+2} \: \& \: ... \: \& \: a_n, where \&$$$ denotes the [bitwise AND operation](https://en.wikipedia.org/wiki/Bitwise_operation#AND).

You are given an array a of size n (n ≥ 2). Find the number of permutations p of numbers ranging from 1 to n, for which the sequence a_{p_1}, a_{p_2}, ... ,a_{p_n} is good. Since this number can be large, output it modulo 10^9+7.

Input

The first line contains a single integer t (1 ≤ t ≤ 10^4), denoting the number of test cases.

The first line of each test case contains a single integer n (2 ≤ n ≤ 2 ⋅ 10^5) — the size of the array.

The second line of each test case contains n integers a_1, a_2, …, a_n (0 ≤ a_i ≤ 10^9) — the elements of the array.

It is guaranteed that the sum of n over all test cases doesn't exceed 2 ⋅ 10^5.

Output

Output t lines, where the i-th line contains the number of good permutations in the i-th test case modulo 10^9 + 7.

Example

Input


4
3
1 1 1
5
1 2 3 4 5
5
0 2 0 3 0
4
1 3 5 1


Output


6
0
36
4

Note

In the first test case, since all the numbers are equal, whatever permutation we take, the sequence is good. There are a total of 6 permutations possible with numbers from 1 to 3: [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1].

In the second test case, it can be proved that no permutation exists for which the sequence is good.

In the third test case, there are a total of 36 permutations for which the sequence is good. One of them is the permutation [1,5,4,2,3] which results in the sequence s=[0,0,3,2,0]. This is a good sequence because 

  *  s_1 = s_2 \: \& \: s_3 \: \& \: s_4 \: \& \: s_5 = 0, 
  *  s_1 \: \& \: s_2 = s_3 \: \& \: s_4 \: \& \: s_5 = 0, 
  *  s_1 \: \& \: s_2 \: \& \: s_3 = s_4 \: \& \: s_5 = 0, 
  *  s_1 \: \& \: s_2 \: \& \: s_3 \: \& \: s_4 = s_5 = 0. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division, print_function
from itertools import permutations 
import threading,bisect,math,heapq,sys
from collections import deque
# threading.stack_size(2**27)
# sys.setrecursionlimit(10**6)
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
# 
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
    mod=10**9+7
    f=[1]
    for i in range(1,300000):
        f.append((f[-1]*i)%mod)
    for _ in range(t):
        n=inin()
        a=ain()
        x=a[0]
        for i in a:
            x&=i
        c=0
        for i in range(n):
            a[i]-=x
            if a[i]==0:
                c+=1
        if c<2:
            print(0)
        else:
            ans=c*(c-1)
            ans%=mod
            ans*=f[n-2]
            ans%=mod
            print(ans)
\"\"\"**********************************************\"\"\"
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
import random
import re

MOD = 10**9 + 7

class Bandsequencesbootcamp(Basebootcamp):
    max_n = 2 * 10**5
    fact = [1] * (max_n + 1)
    for i in range(1, max_n + 1):
        fact[i] = (fact[i-1] * i) % MOD

    def __init__(self, **params):
        super().__init__(**params)
        self.n_min = params.get('n_min', 2)
        self.n_max = params.get('n_max', 10)
        self.x_min = params.get('x_min', 0)
        self.x_max = params.get('x_max', 100)

    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        x = random.randint(self.x_min, self.x_max)
        c = random.randint(2, n)
        a = [x] * c
        for _ in range(n - c):
            a.append(random.randint(self.x_min, self.x_max))
        random.shuffle(a)
        return {
            'n': n,
            'a': a
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = question_case['a']
        a_str = ', '.join(map(str, a))
        prompt = (
            f"给定一个数组a，其中n={n}，数组元素为：{a}。找出所有符合条件的排列数目。输出结果模{MOD}。\n"
            f"一个排列是好的，当且仅当对于所有i从1到n-1，前i项的按位与等于后n-i项的按位与。\n"
            f"请将答案放在[answer]标签内，例如：[answer]42[/answer]。"
        )
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output)
        if matches:
            try:
                return int(matches[-1])
            except ValueError:
                return None
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        a = identity['a']
        n = identity['n']
        if n < 2:
            return False
        x = a[0]
        for num in a[1:]:
            x &= num
        c = a.count(x)
        if c < 2:
            correct = 0
        else:
            if n - 2 >= 0:
                fact = cls.fact[n - 2]
            else:
                fact = 1
            correct = (c * (c - 1) * fact) % MOD
        return solution == correct
