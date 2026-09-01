"""# 

### 谜题描述
You've been in love with Coronavirus-chan for a long time, but you didn't know where she lived until now. And just now you found out that she lives in a faraway place called Naha. 

You immediately decided to take a vacation and visit Coronavirus-chan. Your vacation lasts exactly x days and that's the exact number of days you will spend visiting your friend. You will spend exactly x consecutive (successive) days visiting Coronavirus-chan.

They use a very unusual calendar in Naha: there are n months in a year, i-th month lasts exactly d_i days. Days in the i-th month are numbered from 1 to d_i. There are no leap years in Naha.

The mood of Coronavirus-chan (and, accordingly, her desire to hug you) depends on the number of the day in a month. In particular, you get j hugs if you visit Coronavirus-chan on the j-th day of the month.

You know about this feature of your friend and want to plan your trip to get as many hugs as possible (and then maybe you can win the heart of Coronavirus-chan). 

Please note that your trip should not necessarily begin and end in the same year.

Input

The first line of input contains two integers n and x (1 ≤ n ≤ 2 ⋅ 10^5) — the number of months in the year and the number of days you can spend with your friend.

The second line contains n integers d_1, d_2, …, d_n, d_i is the number of days in the i-th month (1 ≤ d_i ≤ 10^6).

It is guaranteed that 1 ≤ x ≤ d_1 + d_2 + … + d_n.

Output

Print one integer — the maximum number of hugs that you can get from Coronavirus-chan during the best vacation in your life.

Examples

Input


3 2
1 3 1


Output


5

Input


3 6
3 3 3


Output


12

Input


5 6
4 2 3 1 3


Output


15

Note

In the first test case, the numbers of the days in a year are (indices of days in a corresponding month) \{1,1,2,3,1\}. Coronavirus-chan will hug you the most if you come on the third day of the year: 2+3=5 hugs.

In the second test case, the numbers of the days are \{1,2,3,1,2,3,1,2,3\}. You will get the most hugs if you arrive on the third day of the year: 3+1+2+3+1+2=12 hugs.

In the third test case, the numbers of the days are \{1,2,3,4,1,2, 1,2,3, 1, 1,2,3\}. You will get the most hugs if you come on the twelfth day of the year: your friend will hug you 2+3+1+2+3+4=15 times. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division, print_function
# import threading
# threading.stack_size(2**27)
# import sys
# sys.setrecursionlimit(10**7)
# sys.stdin = open('inpy.txt', 'r')
# sys.stdout = open('outpy.txt', 'w')
from sys import stdin, stdout
import bisect            #c++ upperbound
import math
import heapq
i_m=9223372036854775807
def modinv(n,p):
    return pow(n,p-2,p)
def cin():
    return map(int,sin().split())
def ain():                           #takes array as input
    return list(map(int,sin().split()))
def sin():
    return input()
def inin():
    return int(input())
import math  
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
    for i in range(1, int(math.sqrt(n) + 1)) :
        if (n % i == 0) : 
            if (n // i == i) : 
                l.append(i) 
            else : 
                l.append(i)
                l.append(n//i)
    return l
prime=[]
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
q=[]       
def dfs(n,d,v,c):
    global q
    v[n]=1
    x=d[n]
    q.append(n)
    j=c
    for i in x:
        if i not in v:
            f=dfs(i,d,v,c+1)
            j=max(j,f)
            # print(f)
    return j
store = {}
def findFrequency(arr, n, left, right, element): 
      
    # Find the position of  
    # first occurrence of element 
    a = lower_bound(store[element], left) 
  
    # Find the position of 
    # last occurrence of element 
    b = upper_bound(store[element], right)  
import random
   
\"\"\"*******************************************************\"\"\"
def main():
    n,k=cin()
    a=ain()
    te=a[:]
    a.extend(te)
    n=2*n
    b=[]
    for i in range(n):
        b.append((a[i]*(a[i]+1))//2)
    pa=[0]
    pb=[0]
    for i in range(0,n):
        pa.append(a[i]+pa[i])
    for i in range(0,n):
        pb.append(b[i]+pb[i])
    ans=0
    for i in range(n):
        x=bisect.bisect_left(pa,pa[i]+k)
        if x>n:
            continue
        # print(pb[i],pb[x])
        # print(i,x,pa)
        h=pb[x]-pb[i]
        hh=pa[x]-pa[i]-k
        hh=(hh*(hh+1))//2
        an=h-hh
        ans=max(an,ans)
        # print(an)
    print(ans)
######## Python 2 and 3 footer by Pajenegod and c1729
 
# Note because cf runs old PyPy3 version which doesn't have the sped up
# unicode strings, PyPy3 strings will many times be slower than pypy2.
# There is a way to get around this by using binary strings in PyPy3
# but its syntax is different which makes it kind of a mess to use.
 
# So on cf, use PyPy2 for best string performance.
 
py2 = round(0.5)
if py2:
    from future_builtins import ascii, filter, hex, map, oct, zip
    range = xrange
 
import os, sys
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
import sys
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
import bisect
import random
import re
from bootcamp import Basebootcamp

class Dthebestvacationbootcamp(Basebootcamp):
    def __init__(self, n_min=2, n_max=5, d_min=1, d_max=5):
        """
        初始化训练场参数，配置生成谜题实例的参数范围。
        :param n_min: 月份数的最小值，默认为2
        :param n_max: 月份数的最大值，默认为5
        :param d_min: 每月天数的最小值，默认为1
        :param d_max: 每月天数的最大值，默认为5
        """
        self.n_min = n_min
        self.n_max = n_max
        self.d_min = d_min
        self.d_max = d_max

    def case_generator(self):
        """
        生成符合要求的谜题实例，确保输入合法且有解。
        返回包含n, x, d列表及正确答案的字典。
        """
        n = random.randint(self.n_min, self.n_max)
        d = [random.randint(self.d_min, self.d_max) for _ in range(n)]
        sum_d = sum(d)
        x = random.randint(1, sum_d)
        correct_answer = self.calculate_max_hugs(n, x, d)
        return {
            'n': n,
            'x': x,
            'd': d,
            'correct_answer': correct_answer
        }

    @staticmethod
    def calculate_max_hugs(n_input, x_input, d_list):
        """
        根据参考代码逻辑计算最大拥抱数。
        """
        a = d_list.copy()
        a.extend(d_list)
        pa = [0]
        pb = [0]
        for day in a:
            pa.append(pa[-1] + day)
            pb.append(pb[-1] + (day * (day + 1)) // 2)
        k = x_input
        ans = 0
        for i in range(len(a)):
            target = pa[i] + k
            x = bisect.bisect_left(pa, target)
            if x > len(pa) - 1:
                continue
            total_hugs = pb[x] - pb[i]
            extra_days = pa[x] - pa[i] - k
            hh = (extra_days * (extra_days + 1)) // 2
            an = total_hugs - hh
            if an > ans:
                ans = an
        return ans

    @staticmethod
    def prompt_func(question_case) -> str:
        """
        将谜题实例转换为详细的自然语言问题，明确规则和输入格式。
        """
        n = question_case['n']
        x = question_case['x']
        d = question_case['d']
        d_str = ' '.join(map(str, d))
        prompt = f"""你是一位计划去Naha探望Coronavirus-chan的旅行者，想要规划你的假期以获得最多的拥抱次数。以下是详细规则：

1. Naha的日历有{n}个月，每个月的天数分别为{d_str}。每年结束后，月份循环重复（即第n月之后是第1月）。
2. 你需要选择连续的{x}天作为假期。假期可以跨年，例如，从某一年的最后一个月延续到下一年的第一个月。
3. 在第i个月的第j天，你会得到j次拥抱。
4. 你的目标是找到这连续的{x}天，使得所有天的拥抱次数总和最大。

输入数据的第一行是两个整数n和x，第二行是n个整数表示每个月的天数。请根据这些输入，计算出最大可能的拥抱次数。

将答案放在[answer]和[/answer]标签之间，例如：[answer]42[/answer]。

示例输入：
3 2
1 3 1

示例输出：
5（对应的正确格式是[answer]5[/answer]）

现在，请解决以下问题：

输入：
{n} {x}
{d_str}

答案："""
        return prompt

    @staticmethod
    def extract_output(output):
        """
        从模型输出中提取最后一个[answer]标签内的答案。
        """
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """
        验证答案是否正确。
        """
        return solution == identity['correct_answer']
