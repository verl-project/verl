"""# 

### 谜题描述
The city Valera lives in is going to hold elections to the city Parliament.

The city has n districts and n - 1 bidirectional roads. We know that from any district there is a path along the roads to any other district. Let's enumerate all districts in some way by integers from 1 to n, inclusive. Furthermore, for each road the residents decided if it is the problem road or not. A problem road is a road that needs to be repaired.

There are n candidates running the elections. Let's enumerate all candidates in some way by integers from 1 to n, inclusive. If the candidate number i will be elected in the city Parliament, he will perform exactly one promise — to repair all problem roads on the way from the i-th district to the district 1, where the city Parliament is located.

Help Valera and determine the subset of candidates such that if all candidates from the subset will be elected to the city Parliament, all problem roads in the city will be repaired. If there are several such subsets, you should choose the subset consisting of the minimum number of candidates.

Input

The first line contains a single integer n (2 ≤ n ≤ 105) — the number of districts in the city.

Then n - 1 lines follow. Each line contains the description of a city road as three positive integers xi, yi, ti (1 ≤ xi, yi ≤ n, 1 ≤ ti ≤ 2) — the districts connected by the i-th bidirectional road and the road type. If ti equals to one, then the i-th road isn't the problem road; if ti equals to two, then the i-th road is the problem road.

It's guaranteed that the graph structure of the city is a tree.

Output

In the first line print a single non-negative number k — the minimum size of the required subset of candidates. Then on the second line print k space-separated integers a1, a2, ... ak — the numbers of the candidates that form the required subset. If there are multiple solutions, you are allowed to print any of them.

Examples

Input

5
1 2 2
2 3 2
3 4 2
4 5 2


Output

1
5 


Input

5
1 2 1
2 3 2
2 4 1
4 5 1


Output

1
3 


Input

5
1 2 2
1 3 2
1 4 2
1 5 2


Output

4
5 4 3 2 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
\"\"\"Template for Python Competitive Programmers prepared by Mayank Chaudhary \"\"\"

# to use the print and division function of Python3
from __future__ import division, print_function

\"\"\"value of mod\"\"\"
MOD = 998244353
mod = 10**9 + 7

\"\"\"use resource\"\"\"
# import resource
# resource.setrlimit(resource.RLIMIT_STACK, [0x100000000, resource.RLIM_INFINITY])

\"\"\"for factorial\"\"\"

# def prepare_factorial():
#     fact = [1]
#     for i in range(1, 1000005):
#         fact.append((fact[-1] * i) % mod)
#     ifact = [0] * 1000005
#     ifact[1000004] = pow(fact[1000004], mod - 2, mod)
#     for i in range(1000004, 0, -1):
#         ifact[i - 1] = (i * ifact[i]) % mod
#
#     return fact, ifact

\"\"\"uncomment next 4 lines while doing recursion based question\"\"\"
import threading
threading.stack_size(1<<27)
import sys
sys.setrecursionlimit(10000)


\"\"\"uncomment modules according to your need\"\"\"
# from bisect import bisect_left, bisect_right, insort
# import itertools
# from math import floor, ceil, sqrt, degrees, atan, pi
# from heapq import heappop, heapify, heappush
# from random import randint as rn
# from Queue import Queue as Q
from collections import Counter, defaultdict, deque
# from copy import deepcopy
'''
def modinv(n, p):
    return pow(n, p - 2, p)
'''


# def ncr(n, r,  fact, ifact):  # for using this uncomment the lines calculating fact and ifact
#     t = (fact[n] * (ifact[r]*ifact[n-r]) % mod)% mod
#     return t



def get_ints(): return map(int, sys.stdin.readline().strip().split())
def get_array(): return list(map(int, sys.stdin.readline().strip().split()))
def input(): return sys.stdin.readline().strip()


# def GCD(x, y):
#     while (y):
#         x, y = y, x % y
#     return x
#
# def lcm(x, y):
#     return (x*y)//(GCD(x, y))

# def get_xor(n):
#     return [n,1,n+1,0][n%4]

# def binary_expo(a, b):
#
#     result = 1
#     while b:
#         if b&1:
#             result *= a
#             b-=1
#         else:
#             a *= a
#             b >>= 1
#     return result



\"\"\"*******************************************************\"\"\"

class Tree:

    def __init__(self, n):

        self.n = n
        self.mydict = {i: [] for i in range(1, n+1)}              # it represents tree
        self.white = set()
        self.ans = [0]*(10**5 + 6)
        self.visited = [False]*(n+1)


    def addEdge(self, x, y, quality):

        self.mydict[x].append(y)
        self.mydict[y].append(x)
        if quality==2:
            self.white.add((x, y))
            self.white.add((y, x))

    def BFS(self, curr):

        self.Q = deque()
        self.Q.append((curr, curr))

        while self.Q:

            curr, par = self.Q.popleft()
            self.visited[curr] = True
            for child in self.mydict[curr]:
                if not self.visited[child]:
                    if (curr, child) in self.white:
                        self.ans[par] = 0
                        self.ans[child] = 1
                        self.Q.append((child, child))
                    else:
                        self.Q.append((child, par))



    def solve(self):

        self.BFS(1)
        answer = 0
        store = []
        # print(self.ans)
        for index, value in enumerate(self.ans):
            if value==1:
                answer += 1
                store.append(index)
        print(answer)
        print(*store)


def main():

    n = int(input())
    tree = Tree(n)
    for i in range(n-1):
        x, y, q = get_ints()
        tree.addEdge(x, y, q)

    tree.solve()





\"\"\" -------- Python 2 and 3 footer by Pajenegod and c1729 ---------\"\"\"

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
        self.seek((self.tell(), self.seek(0, 2), super(FastIO, self).write(s))[0])
        return s

    def read(self):
        while self._fill(): pass
        return super(FastIO, self).read()

    def readline(self):
        while self.newlines == 0:
            s = self._fill();
            self.newlines = s.count(b\"\n\") + (not s)
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
            self.write = lambda s: self.buffer.write(s.encode('ascii'))
            self.read = lambda: self.buffer.read().decode('ascii')
            self.readline = lambda: self.buffer.readline().decode('ascii')


sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
input = lambda: sys.stdin.readline().rstrip('\r\n')

# sys.stdin = open('input.txt', 'r')
# sys.stdout = open('output.txt', 'w')

\"\"\" main function\"\"\"

if __name__ == '__main__':
    # main()
    threading.Thread(target=main).start()
    # thread.start()
    # thread.join()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from collections import defaultdict
from bootcamp import Basebootcamp

class Cvaleraandelectionsbootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=10, problem_prob=0.5):
        self.min_n = min_n
        self.max_n = max_n
        self.problem_prob = problem_prob

    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        nodes = list(range(1, n+1))
        random.shuffle(nodes)  # 确保各种结构的树都能生成
        
        parent_map = {}
        edges = []
        # 使用更均匀的树生成算法
        for i in range(1, n):
            parent = random.choice(nodes[:i])
            child = nodes[i]
            ti = 2 if random.random() < self.problem_prob else 1
            edges.append((parent, child, ti))
            parent_map[child] = parent
        return {
            'n': n,
            'edges': edges,
            'parent_map': parent_map  # 缓存父节点关系加速验证
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        edges = question_case['edges']
        roads = '\n'.join(f"{parent} {child} {ti}" for parent, child, ti in edges)
        return f"""The city Valera lives in has {n} districts connected by {n-1} roads forming a tree. Each road is either a problem (needs repair) or not. When candidate from district i is elected, they repair all problem roads on the path from i to district 1 (root).

Your task is to find the smallest subset of candidates such that all problem roads are repaired. If multiple solutions exist, output any.

Input:
{n}
{roads}

Output your answer as:

[answer]
k
a1 a2 ... ak
[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        content = matches[-1].strip()
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if len(lines) < 2:
            return None
        try:
            k = int(lines[0])
            candidates = list(map(int, lines[1].split()))
            if len(candidates) != k or len(set(candidates)) != k:
                return None
            return sorted(candidates)
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 处理特殊情况：无问题道路
        problem_edges = [e for e in identity['edges'] if e[2] == 2]
        if not problem_edges:
            return solution == []
        
        # 构建完整的父节点关系（包含根节点的子节点）
        parent_map = defaultdict(lambda: None)
        parent_map.update(identity.get('parent_map', {}))
        
        # 验证问题边覆盖
        required_nodes = set()
        for u, v, _ in problem_edges:
            current = v
            while current is not None:
                required_nodes.add(current)
                current = parent_map[current]
        
        # 检查候选节点是否覆盖所有必要节点
        covered = set()
        for candidate in solution:
            current = candidate
            while current is not None:
                covered.add(current)
                current = parent_map[current]
        
        if not required_nodes.issubset(covered):
            return False

        # 验证极小性：检查每个候选是否必要
        for candidate in solution:
            temp_covered = set()
            for c in solution:
                if c == candidate:
                    continue
                current = c
                while current is not None:
                    temp_covered.add(current)
                    current = parent_map[current]
            if required_nodes.issubset(temp_covered):
                return False
        
        return True
