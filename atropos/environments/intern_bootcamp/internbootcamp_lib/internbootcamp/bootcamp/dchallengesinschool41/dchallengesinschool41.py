"""# 

### 谜题描述
There are n children, who study at the school №41. It is well-known that they are good mathematicians. Once at a break, they arranged a challenge for themselves. All children arranged in a row and turned heads either to the left or to the right.

Children can do the following: in one second several pairs of neighboring children who are looking at each other can simultaneously turn the head in the opposite direction. For instance, the one who was looking at the right neighbor turns left and vice versa for the second child. Moreover, every second at least one pair of neighboring children performs such action. They are going to finish when there is no pair of neighboring children who are looking at each other. 

You are given the number n, the initial arrangement of children and the number k. You have to find a way for the children to act if they want to finish the process in exactly k seconds. More formally, for each of the k moves, you need to output the numbers of the children who turn left during this move.

For instance, for the configuration shown below and k = 2 children can do the following steps: 

<image> At the beginning, two pairs make move: (1, 2) and (3, 4). After that, we receive the following configuration:  <image> At the second move pair (2, 3) makes the move. The final configuration is reached. Good job.  <image>

It is guaranteed that if the solution exists, it takes not more than n^2 \"headturns\".

Input

The first line of input contains two integers n and k (2 ≤ n ≤ 3000, 1 ≤ k ≤ 3000000) — the number of children and required number of moves.

The next line contains a string of length n and consists only of characters L and R, where L means that the child looks to the left and R means that the child looks to the right. 

Output

If there is no solution, print a single line with number -1.

Otherwise, output k lines. Each line has to start with a number n_i (1≤ n_i ≤ n/2) — the number of pairs of children, who turn at this move. After that print n_i distinct integers — the numbers of the children who will turn left during this move. 

After performing all \"headturns\", there can't be a pair of two neighboring children looking at each other.

If there are many solutions, print any of them.

Examples

Input


2 1
RL


Output


1 1 


Input


2 1
LR


Output


-1

Input


4 2
RLRL


Output


2 1 3 
1 2

Note

The first sample contains a pair of children who look at each other. After one move, they can finish the process.

In the second sample, children can't make any move. As a result, they can't end in k>0 moves.

The third configuration is described in the statement.

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

# 6
# 1 -1 2 3 -5 0
def solve():
    n,k = sep()
    s = list(inp())
    mini = 0
    maxi = 0
    ans = []
    i = 0
    while(i<n):
        j = 0
        temp = 0
        lol = []
        while(j<n-1):
            if(s[j] == 'R' and s[j+1] == 'L'):
                lol.append(j+1)
                s[j] = 'L'
                s[j+1] = 'R'
                j+=2
                temp +=1
                continue
            else:
                j+=1
        if(temp == 0):
            break
        maxi+=temp
        ans.append(lol)
        i+=1
    mini = i
    # print(maxi,mini)
    # print(ans)

    if(k<mini or k>maxi):
        print(-1)
        return

    # rem = k-mini
    cnt = 0
    while(cnt+(len(ans)-i)!=k):
        i = 0
        temp = -1
        while(i<len(ans)):
            for j in range(0,len(ans[i])):
                if(cnt+len(ans)-i != k):
                    print(1,ans[i][j])
                    # k -=1
                    cnt+=1
                else:
                    temp = j
                    break

            if(temp !=-1):
                print(len(ans[i])-temp,*ans[i][j:])
                # k-=len(ans[i])-temp
                cnt+=1

            i+=1










testcase(1)
# testcase(int(inp()))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Dchallengesinschool41bootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = params
        self.params.setdefault('max_n', 10)
        self.params.setdefault('min_n', 2)
        self.params.setdefault('max_k', 1000)

    def case_generator(self):
        def generate_valid_initial(n):
            # 确保生成至少一个RL对的初始配置
            for _ in range(10):
                s = [random.choice(['L', 'R']) for _ in range(n)]
                if any(s[i] == 'R' and s[i+1] == 'L' for i in range(n-1)):
                    return ''.join(s)
                # 强制插入一个RL对
                pos = random.randint(0, n-2) if n >=2 else 0
                s[pos] = 'R'
                s[pos+1] = 'L'
                return ''.join(s)
            return 'RL' + 'L'*(n-2) if n >=2 else 'RL'

        n = random.randint(self.params['min_n'], self.params['max_n'])
        # 50%概率生成有效案例
        if random.random() < 0.5:
            # 生成有效案例
            s = generate_valid_initial(n)
            mini, maxi, _ = self.compute_min_max(s)
            if maxi == 0:
                return {'n':4, 'k':2, 'initial':'RLRL'}  # 保底案例
            k = random.randint(mini, maxi)
            return {'n':n, 'k':k, 'initial':s}
        else:
            # 生成无效案例
            s = ''.join(random.choice(['L', 'R']) for _ in range(n))
            mini, maxi, _ = self.compute_min_max(s)
            # 生成无效的k值
            if maxi == 0:
                k = random.randint(1, self.params['max_k'])  # 无解情况
            else:
                k = random.choice([
                    random.randint(0, mini-1),
                    random.randint(maxi+1, self.params['max_k'])
                ])
            return {'n':n, 'k':k, 'initial':s}

    @staticmethod
    def compute_min_max(s):
        s_list = list(s)
        steps = []
        while True:
            pairs = []
            i = 0
            while i < len(s_list)-1:
                if s_list[i] == 'R' and s_list[i+1] == 'L':
                    pairs.append(i+1)  # 1-based左位置
                    s_list[i] = 'L'
                    s_list[i+1] = 'R'
                    i += 2
                else:
                    i += 1
            if not pairs:
                break
            steps.append(pairs)
        return len(steps), sum(len(step) for step in steps), steps

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        k = question_case['k']
        initial = question_case['initial']
        return f"""## 谜题挑战：学生转头协调

{n}个学生排成一列，初始朝向：{initial}  
（L=向左看，R=向右看）

**游戏规则**：
1. 每秒钟可以同时翻转多个相邻的RL对（左边学生向右看，右边向左看）
2. 每次翻转后，这对学生会变成LL和RR
3. 必须恰好经过{k}秒完成所有操作
4. 最终不能有任何相邻的RL对存在

**输出格式要求**：
- 共{k}行，每行表示每秒的操作
- 每行格式：首个数字表示翻转对数n_i，后面跟着n_i个**左学生的位置编号**（1-based）
- 位置编号必须按升序排列且不重复
- 同一秒的操作位置不能相邻

将最终答案放在[answer]和[/answer]之间，例如：
[answer]
2 1 3
1 2
[/answer]"""

    @staticmethod
    def extract_output(output):
        # 提取最后一个answer块并验证格式
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
            
        content = matches[-1].strip()
        steps = []
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 1:
                continue
            try:
                parts = list(map(int, parts))
                n_i = parts[0]
                if n_i != len(parts[1:]) or any(p <=0 for p in parts[1:]):
                    continue
                # 检查是否升序排列且不重复
                sorted_p = sorted(parts[1:])
                if sorted_p != parts[1:] or len(set(sorted_p)) != len(sorted_p):
                    continue
                steps.append(sorted_p)
            except:
                continue
        return steps if steps else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        k = identity['k']
        initial = list(identity['initial'])
        
        if len(solution) != k:
            return False
        
        current_state = initial.copy()
        for step in solution:
            # 检查步骤格式
            if not step or any(p <1 or p >=n for p in step):
                return False
            # 检查升序且不重复不连续
            prev = -1
            for p in step:
                if p <= prev or p - prev == 1:
                    return False
                prev = p
            # 验证RL对存在
            temp_state = current_state.copy()
            for p in step:
                idx = p-1
                if idx >= len(temp_state)-1 or temp_state[idx] != 'R' or temp_state[idx+1] != 'L':
                    return False
                temp_state[idx] = 'L'
                temp_state[idx+1] = 'R'
            current_state = temp_state
        
        # 检查最终状态
        return not any(current_state[i] == 'R' and current_state[i+1] == 'L' for i in range(n-1))
