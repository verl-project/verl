"""# 

### 谜题描述
The legendary Farmer John is throwing a huge party, and animals from all over the world are hanging out at his house. His guests are hungry, so he instructs his cow Bessie to bring out the snacks! Moo!

There are n snacks flavors, numbered with integers 1, 2, …, n. Bessie has n snacks, one snack of each flavor. Every guest has exactly two favorite flavors. The procedure for eating snacks will go as follows:

  * First, Bessie will line up the guests in some way. 
  * Then in this order, guests will approach the snacks one by one. 
  * Each guest in their turn will eat all remaining snacks of their favorite flavor. In case no favorite flavors are present when a guest goes up, they become very sad. 



Help Bessie to minimize the number of sad guests by lining the guests in an optimal way.

Input

The first line contains integers n and k (2 ≤ n ≤ 10^5, 1 ≤ k ≤ 10^5), the number of snacks and the number of guests. 

The i-th of the following k lines contains two integers x_i and y_i (1 ≤ x_i, y_i ≤ n, x_i ≠ y_i), favorite snack flavors of the i-th guest.

Output

Output one integer, the smallest possible number of sad guests.

Examples

Input


5 4
1 2
4 3
1 4
3 4


Output


1


Input


6 5
2 3
2 1
3 4
6 5
4 5


Output


0

Note

In the first example, Bessie can order the guests like this: 3, 1, 2, 4. Guest 3 goes first and eats snacks 1 and 4. Then the guest 1 goes and eats the snack 2 only, because the snack 1 has already been eaten. Similarly, the guest 2 goes up and eats the snack 3 only. All the snacks are gone, so the guest 4 will be sad. 

In the second example, one optimal ordering is 2, 1, 3, 5, 4. All the guests will be satisfied.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division, print_function
import bisect
def main():
    n,k=map(int,input().split())
    temp= [ [] for _ in range(n)]
    store={}
    for i in range(k):
        a,b=map(int,input().split())
        temp[a-1].append(i)
        temp[b-1].append(i)
        store[i]=(a-1,b-1)
    used=[0]*n
    ans=k
    visited=[0]*k
    stack=[]
    for i in range(k):
        if visited[i]==1:
            continue
        else:
            stack.append(i)
            visited[i]=1
        while stack:
            s=stack.pop()
            a,b=store[s][0],store[s][1]
            if used[a]==0 or used[b]==0:
                ans-=1
                if used[a]==0:
                    for item in temp[a]:
                        if visited[item]==0:
                            stack.append(item)
                            visited[item]=1
                if used[b]==0:
                    for item in temp[b]:
                        if visited[item]==0:
                            stack.append(item)
                            visited[item]=1
                used[a]=1
                used[b]=1
            
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
            if s[i] >= b'0' [0]:
                numb = 10 * numb + conv(s[i]) - 48
            elif s[i] == b'-' [0]: sign = -1
            elif s[i] != b'\r' [0]:
                A.append(sign*numb)
                numb = zero; sign = 1
            i += 1
    except:pass
    if s and s[-1] >= b'0' [0]:
        A.append(sign*numb)
    return A

if __name__== \"__main__\":
  main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Dcowandsnacksbootcamp(Basebootcamp):
    def __init__(self, max_n=20, max_k=20):
        self.max_n = max_n
        self.max_k = max_k
    
    def case_generator(self):
        # 修正k生成逻辑，确保符合题目输入要求
        n = random.randint(2, self.max_n)
        k = random.randint(1, self.max_k)  # 允许k超出n*(n-1)
        
        guests = []
        for _ in range(k):
            # 允许重复的口味组合
            x = random.randint(1, n)
            y = random.randint(1, n)
            while y == x:
                y = random.randint(1, n)
            guests.append([x, y])
        return {'n': n, 'k': k, 'guests': guests}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        k = question_case['k']
        guests = question_case['guests']
        problem = (
            "作为农场主John的助手Bessie，你需要安排客人顺序以最小化伤心人数。规则如下：\n\n"
            "1. 共有n种零食（1到n编号），每种恰好一个\n"
            "2. 每位客人有两个不同的偏爱口味\n"
            "3. 客人按顺序依次吃掉自己偏爱口味的剩余零食\n"
            "4. 如果没有剩余偏爱零食，客人会伤心\n\n"
            "输入格式：\n"
            f"第一行：{n} {k}\n"
            f"随后{k}行每行两个整数表示客人喜好\n\n"
            "当前问题：\n"
            f"{n} {k}\n"
        )
        for guest in guests:
            problem += f"{guest[0]} {guest[1]}\n"
        problem += "\n请输出最小的可能伤心人数，并将答案放在[answer]和[/answer]之间。"
        return problem
    
    @staticmethod
    def extract_output(output):
        # 增强异常处理
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL | re.IGNORECASE)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except (ValueError, TypeError):
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            # 直接调用计算函数验证
            return int(solution) == cls._compute_min_sad(
                identity['n'],
                identity['k'],
                identity['guests']
            )
        except:
            return False
    
    @staticmethod
    def _compute_min_sad(n, k, guests):
        # 重构验证算法
        flavor_map = [[] for _ in range(n)]
        guest_pairs = []
        
        for idx, (a, b) in enumerate(guests):
            a_idx = a - 1
            b_idx = b - 1
            flavor_map[a_idx].append(idx)
            flavor_map[b_idx].append(idx)
            guest_pairs.append((a_idx, b_idx))
        
        activated = [False] * n
        visited = [False] * k
        happy_count = 0
        
        for i in range(k):
            if visited[i]:
                continue
            stack = [i]
            visited[i] = True
            
            while stack:
                current = stack.pop()
                f1, f2 = guest_pairs[current]
                
                if not activated[f1] or not activated[f2]:
                    happy_count += 1
                    if not activated[f1]:
                        activated[f1] = True
                        for g in flavor_map[f1]:
                            if not visited[g]:
                                visited[g] = True
                                stack.append(g)
                    if not activated[f2]:
                        activated[f2] = True
                        for g in flavor_map[f2]:
                            if not visited[g]:
                                visited[g] = True
                                stack.append(g)
        
        return k - happy_count
