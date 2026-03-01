"""# 

### 谜题描述
Petya has come to the math exam and wants to solve as many problems as possible. He prepared and carefully studied the rules by which the exam passes.

The exam consists of n problems that can be solved in T minutes. Thus, the exam begins at time 0 and ends at time T. Petya can leave the exam at any integer time from 0 to T, inclusive.

All problems are divided into two types: 

  * easy problems — Petya takes exactly a minutes to solve any easy problem; 
  * hard problems — Petya takes exactly b minutes (b > a) to solve any hard problem. 



Thus, if Petya starts solving an easy problem at time x, then it will be solved at time x+a. Similarly, if at a time x Petya starts to solve a hard problem, then it will be solved at time x+b.

For every problem, Petya knows if it is easy or hard. Also, for each problem is determined time t_i (0 ≤ t_i ≤ T) at which it will become mandatory (required). If Petya leaves the exam at time s and there is such a problem i that t_i ≤ s and he didn't solve it, then he will receive 0 points for the whole exam. Otherwise (i.e if he has solved all such problems for which t_i ≤ s) he will receive a number of points equal to the number of solved problems. Note that leaving at time s Petya can have both \"mandatory\" and \"non-mandatory\" problems solved.

For example, if n=2, T=5, a=2, b=3, the first problem is hard and t_1=3 and the second problem is easy and t_2=2. Then:

  * if he leaves at time s=0, then he will receive 0 points since he will not have time to solve any problems; 
  * if he leaves at time s=1, he will receive 0 points since he will not have time to solve any problems; 
  * if he leaves at time s=2, then he can get a 1 point by solving the problem with the number 2 (it must be solved in the range from 0 to 2); 
  * if he leaves at time s=3, then he will receive 0 points since at this moment both problems will be mandatory, but he will not be able to solve both of them; 
  * if he leaves at time s=4, then he will receive 0 points since at this moment both problems will be mandatory, but he will not be able to solve both of them; 
  * if he leaves at time s=5, then he can get 2 points by solving all problems. 



Thus, the answer to this test is 2.

Help Petya to determine the maximal number of points that he can receive, before leaving the exam.

Input

The first line contains the integer m (1 ≤ m ≤ 10^4) — the number of test cases in the test.

The next lines contain a description of m test cases. 

The first line of each test case contains four integers n, T, a, b (2 ≤ n ≤ 2⋅10^5, 1 ≤ T ≤ 10^9, 1 ≤ a < b ≤ 10^9) — the number of problems, minutes given for the exam and the time to solve an easy and hard problem, respectively.

The second line of each test case contains n numbers 0 or 1, separated by single space: the i-th number means the type of the i-th problem. A value of 0 means that the problem is easy, and a value of 1 that the problem is hard.

The third line of each test case contains n integers t_i (0 ≤ t_i ≤ T), where the i-th number means the time at which the i-th problem will become mandatory.

It is guaranteed that the sum of n for all test cases does not exceed 2⋅10^5.

Output

Print the answers to m test cases. For each set, print a single integer — maximal number of points that he can receive, before leaving the exam.

Example

Input


10
3 5 1 3
0 0 1
2 1 4
2 5 2 3
1 0
3 2
1 20 2 4
0
16
6 20 2 5
1 1 0 1 0 0
0 8 2 9 11 6
4 16 3 6
1 0 1 1
8 3 5 6
6 20 3 6
0 1 0 0 1 0
20 11 3 20 16 17
7 17 1 6
1 1 0 1 0 0 0
1 7 0 11 10 15 10
6 17 2 6
0 0 1 0 0 1
7 6 3 7 10 12
5 17 2 5
1 1 1 1 0
17 11 10 6 4
1 1 1 2
0
1


Output


3
2
1
0
1
4
0
1
2
1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
mod=10**9+7
#import resource
#resource.setrlimit(resource.RLIMIT_STACK, [0x100000000, resource.RLIM_INFINITY])
#import threading
#threading.stack_size(2**27)
#import sys
#sys.setrecursionlimit(10**6)
#fact=[1]
#for i in range(1,10001):
#    fact.append((fact[-1]*i)%mod)
#ifact=[0]*10001
#ifact[10000]=pow(fact[10000],mod-2,mod)
#for i in range(10000,0,-1):
#    ifact[i-1]=(i*ifact[i])%mod
from sys import stdin, stdout
import bisect
from bisect import bisect_left as bl              #c++ lowerbound bl(array,element)
from bisect import bisect_right as br             #c++ upperbound
import itertools
import collections
import math
import heapq
from random import randint as rn
#from Queue import Queue as Q
class node:
    def __init__(self,s,l,p):
        self.child={}
        self.se=s
        self.le=l
        self.parent=p
        self.val=0
def modinv(n,p):
    return pow(n,p-2,p)
def ncr(n,r,p):                        #for using this uncomment the lines calculating fact and ifact
    t=((fact[n])*((ifact[r]*ifact[n-r])%p))%p
    return t
def ain():                           #takes array as input
    return list(map(int,sin().split()))
def sin():
    return input().strip()
def GCD(x,y):
    while(y):
        x, y = y, x % y
    return x
\"\"\"**************************************************************************\"\"\"
def main():
    for _ in range(int(input())):
        n,T,a1,b1=ain()
        d=ain()
        x=d.count(0)
        b=ain()
        for i in range(n):
            b[i]=[b[i],d[i]]
        b.sort()
        if(b[0][1]==0):
            p=[a1]
        else:
            p=[b1]
        r=[0]*n
        for i in range(n-1,0,-1):
            r[i-1]=r[i]
            if(b[i][1]==0):
                r[i-1]+=1
        for i in range(1,n):
            if(b[i][1]==0):
                p.append(p[-1]+a1)
            else:
                p.append(p[-1]+b1)
        if(p[-1]<=T):
            print n
            continue
        ans=max(0,min((b[0][0]-1)/a1,x))
        for i in range(n-1):
            if(p[i]<b[i+1][0]):
                z=min((b[i+1][0]-p[i]-1)/a1,r[i])
                ans=max(ans,i+1+z)
        print ans
######## Python 2 and 3 footer by Pajenegod and c1729
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

if __name__ == '__main__':
   main()
#threading.Thread(target=main).start()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cpetyaandexambootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=10, min_T=1, max_T=50, a_min=1, a_max=5, b_min=3, b_max=8):
        self.min_n = min_n
        self.max_n = max_n
        self.min_T = min_T
        self.max_T = max_T
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        T = random.randint(self.min_T, self.max_T)
        
        a = random.randint(self.a_min, self.a_max)
        b = random.randint(self.b_min, self.b_max)
        while b <= a:
            b = random.randint(self.b_min, self.b_max)
        
        types = [random.choice([0, 1]) for _ in range(n)]
        t_list = [random.randint(0, T) for _ in range(n)]
        
        # 添加更多边界案例：全easy/全hard问题
        if random.random() < 0.2:
            types = [0] * n
        elif random.random() < 0.2:
            types = [1] * n
        
        # 确保至少有一个有效案例
        while all(t > T for t in t_list):
            t_list = [random.randint(0, T) for _ in range(n)]
        
        correct_answer = self.solve_case(n, T, a, b, types, t_list)
        
        return {
            'n': n,
            'T': T,
            'a': a,
            'b': b,
            'types': types,
            't_list': t_list,
            'correct_answer': correct_answer
        }
    
    @staticmethod
    def solve_case(n, T, a, b, types, times):
        combined = sorted(zip(times, types), key=lambda x: (x[0], x[1]))
        sorted_times = [x[0] for x in combined]
        sorted_types = [x[1] for x in combined]
        
        # 计算前缀时间和剩余easy数量
        prefix = []
        total_time = 0
        for typ in sorted_types:
            total_time += a if typ == 0 else b
            prefix.append(total_time)
        
        max_points = 0
        
        # 情况1：解决所有问题
        if prefix[-1] <= T:
            return n
        
        # 情况2：在第一个问题强制前解决easy
        first_mandatory = sorted_times[0]
        if first_mandatory > 0:
            available = first_mandatory - 1
            max_easy = min(available // a, sum(1 for t in sorted_types if t == 0))
            max_points = max(max_points, max_easy)
        
        # 预处理剩余easy数量
        remaining_easy = [0] * (n + 1)
        count = 0
        for i in range(n-1, -1, -1):
            if sorted_types[i] == 0:
                count += 1
            remaining_easy[i] = count
        
        # 检查每个可能的分割点
        current_total_time = 0
        for i in range(n):
            current_total_time += a if sorted_types[i] == 0 else b
            if current_total_time > T:
                break
            
            # 计算后续可用时间
            next_mandatory = sorted_times[i+1] if i < n-1 else T + 1
            available_time = next_mandatory - current_total_time - 1
            if available_time < 0:
                continue
            
            # 计算可添加的easy数量
            possible = min(available_time // a, remaining_easy[i+1])
            max_points = max(max_points, i + 1 + possible)
        
        return max_points
    
    @staticmethod
    def prompt_func(question_case):
        params = question_case
        problem_desc = (
            "Petya has to solve math problems in an exam. The exam lasts {T} minutes with {n} problems. "
            "Easy problems take {a} minutes, hard ones take {b} minutes. Each problem becomes mandatory at a specific time. "
            "If Petya leaves at time s, he must have solved all problems with mandatory time ≤ s. Your task is to determine the maximum points he can earn.\n\n"
            "Problem Details:\n"
            "- Problem types (0=easy, 1=hard): {types}\n"
            "- Mandatory times: {t_list}\n\n"
            "Format your answer as: [answer]X[/answer], where X is the maximum points."
        ).format(
            n=params['n'],
            T=params['T'],
            a=params['a'],
            b=params['b'],
            types=params['types'],
            t_list=params['t_list']
        )
        return problem_desc
    
    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        last_answer = answer_blocks[-1].strip()
        try:
            return int(last_answer)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
