"""# 

### 谜题描述
There are n seats in the train's car and there is exactly one passenger occupying every seat. The seats are numbered from 1 to n from left to right. The trip is long, so each passenger will become hungry at some moment of time and will go to take boiled water for his noodles. The person at seat i (1 ≤ i ≤ n) will decide to go for boiled water at minute t_i.

Tank with a boiled water is located to the left of the 1-st seat. In case too many passengers will go for boiled water simultaneously, they will form a queue, since there can be only one passenger using the tank at each particular moment of time. Each passenger uses the tank for exactly p minutes. We assume that the time it takes passengers to go from their seat to the tank is negligibly small. 

Nobody likes to stand in a queue. So when the passenger occupying the i-th seat wants to go for a boiled water, he will first take a look on all seats from 1 to i - 1. In case at least one of those seats is empty, he assumes that those people are standing in a queue right now, so he would be better seating for the time being. However, at the very first moment he observes that all seats with numbers smaller than i are busy, he will go to the tank.

There is an unspoken rule, that in case at some moment several people can go to the tank, than only the leftmost of them (that is, seating on the seat with smallest number) will go to the tank, while all others will wait for the next moment.

Your goal is to find for each passenger, when he will receive the boiled water for his noodles.

Input

The first line contains integers n and p (1 ≤ n ≤ 100 000, 1 ≤ p ≤ 10^9) — the number of people and the amount of time one person uses the tank.

The second line contains n integers t_1, t_2, ..., t_n (0 ≤ t_i ≤ 10^9) — the moments when the corresponding passenger will go for the boiled water.

Output

Print n integers, where i-th of them is the time moment the passenger on i-th seat will receive his boiled water.

Example

Input


5 314
0 310 942 628 0


Output


314 628 1256 942 1570 

Note

Consider the example.

At the 0-th minute there were two passengers willing to go for a water, passenger 1 and 5, so the first passenger has gone first, and returned at the 314-th minute. At this moment the passenger 2 was already willing to go for the water, so the passenger 2 has gone next, and so on. In the end, 5-th passenger was last to receive the boiled water.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division, print_function
def main():
    n,p=map(int,input().split())
    l1=list(map(int,input().split()))
    temp=[]
    for i in range(n):
        temp.append((l1[i],i))
    temp.sort()
    import heapq
    from collections import deque
    people=deque()
    ready=[]
    heapq.heapify(ready)
    time=0
    i=0
    result=[0]*n
    while i<n:
        if len(people)==0:
            if ready:
                people.append(heapq.heappop(ready))
            else :
                people.append(temp[i][1])
                time=temp[i][0]
                i+=1
        
        while i<n and temp[i][0]<=time+p:
            if temp[i][1]<people[-1]:
                people.append(temp[i][1])
            else :
                heapq.heappush(ready,temp[i][1])
            i+=1
            
        time+=p
        result[people.popleft()]=time
    
    while people:
        result[people.popleft()]=time+p
        time+=p
    while ready:
        result[heapq.heappop(ready)]=time+p
        time+=p
        
    print(*result,sep=\" \")
        
            
        

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
from collections import deque
import heapq
import re
from bootcamp import Basebootcamp

class Cqueueinthetrainbootcamp(Basebootcamp):
    def __init__(self, max_n=100000, max_p=10**9, max_t=10**9):
        self.max_n = max_n
        self.max_p = max_p
        self.max_t = max_t

    def case_generator(self):
        n = random.randint(1, min(self.max_n, 1000))  # 控制测试规模
        p = random.randint(1, self.max_p)
        
        # 生成更有挑战性的测试数据
        if random.random() < 0.3:
            # 生成全相同时间
            t = random.randint(0, self.max_t)
            t_list = [t] * n
        elif random.random() < 0.5:
            # 生成严格递增序列
            t_list = sorted(random.sample(range(self.max_t), n))
        else:
            # 随机生成包含重复值的数据
            t_list = [random.choice([0, self.max_t]) for _ in range(n)]
        
        correct_output = self.solve(n, p, t_list)
        return {
            'n': n,
            'p': p,
            't_list': t_list,
            'correct_output': correct_output
        }

    @staticmethod
    def solve(n, p, t_list):
        temp = [(t, i) for i, t in enumerate(t_list)]
        temp.sort()
        people = deque()
        ready = []
        heapq.heapify(ready)
        time = 0
        i = 0
        result = [0] * n

        while i < n:
            if not people:
                if ready:
                    people.append(heapq.heappop(ready))
                else:
                    people.append(temp[i][1])
                    time = temp[i][0]
                    i += 1

            while i < n and temp[i][0] <= time + p:
                if temp[i][1] < people[-1]:
                    people.append(temp[i][1])
                else:
                    heapq.heappush(ready, temp[i][1])
                i += 1

            time += p
            passenger = people.popleft()
            result[passenger] = time

        while people:
            time += p
            passenger = people.popleft()
            result[passenger] = time

        while ready:
            time += p
            passenger = heapq.heappop(ready)
            result[passenger] = time

        return result

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        p = question_case['p']
        t_list = question_case['t_list']
        
        rule_desc = (
            "1. 每个乘客i在t_i分钟出发前往取水，使用时间为p分钟\n"
            "2. 出发时会检查所有左侧座位（1～i-1），如果任一左侧座位为空，则继续等待\n"
            "3. 如果所有左侧座位都有人，则立即加入队列\n"
            "4. 相同时间出发时，座位号小的乘客优先\n"
            "5. 队列遵循先到先服务原则，但需注意上述条件优先级"
        )
        
        example = (
            "输入示例：\n5 314\n0 310 942 628 0\n"
            "正确输出：\n314 628 1256 942 1570\n"
            "格式要求：用空格分隔的整数，座位1到n的完成时间"
        )

        return f"""解决火车车厢取水时间问题：

# 问题背景
{random.choice(["长途列车", "高铁动车"])}上有{n}个座位，每位乘客需要按特定规则使用饮水机。请计算各乘客完成取水的时间。

# 核心规则
{rule_desc}

# 输入参数
座位数n = {n}
单次使用时间p = {p}
出发时间列表t = [{' '.join(map(str, t_list))}]

# 输出要求
按座位顺序输出完成时间，格式为空格分隔的整数

{example}

请将最终答案放在[answer]和[/answer]标记之间：
[answer]
你的计算结果
[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\][\s\n]*(.*?)[\s\n]*\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            last_answer = matches[-1].strip().replace('\n', ' ')
            return list(map(int, last_answer.split()))
        except (ValueError, AttributeError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return solution == identity['correct_output']
        except (KeyError, TypeError):
            return False
