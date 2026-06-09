"""# 

### 谜题描述
You're given an undirected graph with n nodes and m edges. Nodes are numbered from 1 to n.

The graph is considered harmonious if and only if the following property holds:

  * For every triple of integers (l, m, r) such that 1 ≤ l < m < r ≤ n, if there exists a path going from node l to node r, then there exists a path going from node l to node m. 



In other words, in a harmonious graph, if from a node l we can reach a node r through edges (l < r), then we should able to reach nodes (l+1), (l+2), …, (r-1) too.

What is the minimum number of edges we need to add to make the graph harmonious? 

Input

The first line contains two integers n and m (3 ≤ n ≤ 200\ 000 and 1 ≤ m ≤ 200\ 000).

The i-th of the next m lines contains two integers u_i and v_i (1 ≤ u_i, v_i ≤ n, u_i ≠ v_i), that mean that there's an edge between nodes u and v.

It is guaranteed that the given graph is simple (there is no self-loop, and there is at most one edge between every pair of nodes).

Output

Print the minimum number of edges we have to add to the graph to make it harmonious.

Examples

Input


14 8
1 2
2 7
3 4
6 3
5 7
3 8
6 8
11 12


Output


1


Input


200000 3
7 9
9 8
4 5


Output


0

Note

In the first example, the given graph is not harmonious (for instance, 1 < 6 < 7, node 1 can reach node 7 through the path 1 → 2 → 7, but node 1 can't reach node 6). However adding the edge (2, 4) is sufficient to make it harmonious.

In the second example, the given graph is already harmonious.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/usr/bin/env python
from __future__ import division, print_function

import os
import sys
from io import BytesIO, IOBase

if sys.version_info[0] < 3:
    from __builtin__ import xrange as range
    from future_builtins import ascii, filter, hex, map, oct, zip


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, a):
        acopy = a
        while a != self.parent[a]:
            a = self.parent[a]
        while acopy != a:
            self.parent[acopy], acopy = a, self.parent[acopy]
        return a

    def union(self, a, b):
        self.parent[self.find(b)] = self.find(a)


def main():
    n, m = map(int, input().split())

    uf = UnionFind(n)

    for _ in range(m):
        u, v = map(int, input().split())
        uf.union(u - 1, v - 1)

    p = uf.find(0)
    ea = []
    es = set()
    o = 0

    for i in range(n):
        f = uf.find(i)

        if f != p:
            ea.append(p)
            es.add(p)

        if f in es:
            k = ea.pop()
            es.remove(k)
            while k != f:
                uf.union(k, f)
                o += 1
                k = ea.pop()
                es.remove(k)

        p = uf.find(i)

    print(o)


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
    main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Dharmoniousgraphbootcamp(Basebootcamp):
    def __init__(self, n_min=3, n_max=2000, m_min=1, max_edges_factor=0.3, harmonious_ratio=0.5):
        """
        参数优化：增大默认n_max范围，增加和谐图生成比例
        
        参数:
            harmonious_ratio (float): 生成和谐图案例的概率比例（0-1）
        """
        self.n_min = n_min
        self.n_max = n_max
        self.m_min = m_min
        self.max_edges_factor = max_edges_factor
        self.harmonious_ratio = harmonious_ratio  # 控制和谐图生成比例

    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))
        
        def find(self, a):
            acopy = a
            while a != self.parent[a]:
                a = self.parent[a]
            while acopy != a:
                self.parent[acopy], acopy = a, self.parent[acopy]
            return a
        
        def union(self, a, b):
            self.parent[self.find(b)] = self.find(a)

    @classmethod
    def compute_answer(cls, n, m, edges):
        uf = cls.UnionFind(n)
        for u, v in edges:
            uf.union(u-1, v-1)

        p = uf.find(0)
        ea = []
        es = set()
        o = 0
        
        for i in range(n):
            f = uf.find(i)
            
            if f != p:
                ea.append(p)
                es.add(p)
            
            if f in es:
                k = ea.pop()
                es.remove(k)
                while k != f:
                    uf.union(k, f)
                    o += 1
                    k = ea.pop()
                    es.remove(k)
            
            p = uf.find(i)
        
        return o

    def case_generator(self):
        if random.random() < self.harmonious_ratio:
            return self._generate_harmonious_case()
        else:
            return self._generate_random_case()

    def _generate_harmonious_case(self):
        """生成满足和谐条件的图案例（答案为零）"""
        n = random.randint(self.n_min, self.n_max)
        
        # 基础链式结构（确保完全连通）
        edges = []
        for u in range(1, n):
            edges.append((u, u+1))
        
        # 添加允许的随机附加边（不会破坏和谐性）
        existing_edges = set(edges)
        possible_extra = [
            (i, j) for i in range(1, n+1) for j in range(i+1, n+1)
            if (i, j) not in existing_edges and abs(i-j) > 1  # 避免重复基础边
        ]
        
        max_extra = min(
            len(possible_extra),
            int(n * 0.2)  # 最多添加节点数20%的附加边
        )
        if max_extra > 0:
            extra = random.sample(possible_extra, k=random.randint(0, max_extra))
            edges.extend(extra)
        
        random.shuffle(edges)
        m_actual = len(edges)
        
        return {
            'n': n,
            'm': m_actual,
            'edges': edges,
            'correct_answer': 0  # 确保答案为零
        }

    def _generate_random_case(self):
        """原始随机生成逻辑"""
        n = random.randint(self.n_min, self.n_max)
        max_possible = n * (n - 1) // 2
        m_max = int(max_possible * self.max_edges_factor)
        m_max = max(m_max, self.m_min)
        m = random.randint(self.m_min, m_max)
        
        edges = set()
        while len(edges) < m:
            u = random.randint(1, n)
            v = random.randint(1, n)
            if u == v:
                continue
            u, v = sorted((u, v))
            edges.add((u, v))
        
        edges = list(edges)
        correct_answer = self.compute_answer(n, m, edges)
        return {
            'n': n,
            'm': m,
            'edges': edges,
            'correct_answer': correct_answer
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        # 保持原有实现不变
        n = question_case['n']
        m = question_case['m']
        edges = question_case['edges']
        input_lines = [f"{u} {v}" for u, v in edges]
        problem_text = (
            "You are given an undirected graph with {n} nodes and {m} edges. Nodes are numbered from 1 to {n}. "
            "The graph is considered harmonious if for every triple of integers (l, m, r) where 1 ≤ l < m < r ≤ n, "
            "if there exists a path from node l to node r, then there must also exist a path from node l to node m. "
            "Your task is to determine the minimum number of edges required to make the graph harmonious.\n\n"
            "Input format:\n"
            "The first line contains two integers n and m.\n"
            "The next m lines each contain two integers u and v, denoting an edge between nodes u and v.\n\n"
            "Problem Input:\n"
            "{n} {m}\n"
            "{edges}\n\n"
            "Please provide your answer as an integer enclosed within [answer] tags, like [answer]5[/answer]. "
            "Ensure your answer is correct and properly formatted."
        ).format(
            n=n,
            m=m,
            edges='\n'.join(input_lines)
        )
        return problem_text

    @staticmethod
    def extract_output(output):
        # 保持原有实现不变
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
        # 保持原有实现不变
        return solution == identity['correct_answer']
