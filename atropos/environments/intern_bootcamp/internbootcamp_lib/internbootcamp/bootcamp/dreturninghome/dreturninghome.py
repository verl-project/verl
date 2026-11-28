"""# 

### 谜题描述
Yura has been walking for some time already and is planning to return home. He needs to get home as fast as possible. To do this, Yura can use the instant-movement locations around the city.

Let's represent the city as an area of n × n square blocks. Yura needs to move from the block with coordinates (s_x,s_y) to the block with coordinates (f_x,f_y). In one minute Yura can move to any neighboring by side block; in other words, he can move in four directions. Also, there are m instant-movement locations in the city. Their coordinates are known to you and Yura. Yura can move to an instant-movement location in no time if he is located in a block with the same coordinate x or with the same coordinate y as the location.

Help Yura to find the smallest time needed to get home.

Input

The first line contains two integers n and m — the size of the city and the number of instant-movement locations (1 ≤ n ≤ 10^9, 0 ≤ m ≤ 10^5).

The next line contains four integers s_x s_y f_x f_y — the coordinates of Yura's initial position and the coordinates of his home ( 1 ≤ s_x, s_y, f_x, f_y ≤ n).

Each of the next m lines contains two integers x_i y_i — coordinates of the i-th instant-movement location (1 ≤ x_i, y_i ≤ n).

Output

In the only line print the minimum time required to get home.

Examples

Input


5 3
1 1 5 5
1 2
4 1
3 3


Output


5


Input


84 5
67 59 41 2
39 56
7 2
15 3
74 18
22 7


Output


42

Note

In the first example Yura needs to reach (5, 5) from (1, 1). He can do that in 5 minutes by first using the second instant-movement location (because its y coordinate is equal to Yura's y coordinate), and then walking (4, 1) → (4, 2) → (4, 3) → (5, 3) → (5, 4) → (5, 5).

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
range = xrange
input = raw_input

class segtree:
    def __init__(self, n):
        self.m = 1
        while self.m < n: self.m *= 2
        self.data = [-1] * (self.m * 2)

    def __setitem__(self, i, val):
        i += self.m
        while i and (self.data[i] > val or self.data[i] == -1):
            self.data[i] = val
            i >>= 1

    def __call__(self):
        i = 1
        goal = self.data[1]
        
        while i < self.m:
            i *= 2
            if self.data[i] != goal:
                i += 1

        j = i >> 1
        self.data[i] = -1
        while j:
            x = self.data[2 * j]
            y = self.data[2 * j + 1]

            a1 = min(x,y)
            a2 = max(x,y)

            self.data[j] = a1 if a1 != -1 else a2
            j >>= 1
        return i - self.m, goal

inp = [int(x) for x in sys.stdin.read().split()]; ii = 0

n = inp[ii]; ii += 1
m = inp[ii]; ii += 1

sx,sy,fx,fy = inp[ii: ii + 4]; ii += 4

X = inp[ii + 0: ii + 2 * m: 2]
Y = inp[ii + 1: ii + 2 * m: 2]

N = m + 2
graph = [[] for _ in range(N)]
W = []
V = []

orderX = sorted(range(m), key = X.__getitem__)
orderY = sorted(range(m), key = Y.__getitem__)

for ind in range(m - 1):
    i1 = orderX[ind]
    i2 = orderX[ind + 1]

    w = X[i2] - X[i1]

    eind = len(W)
    W.append(w)
    W.append(w)
    V.append(i2)
    V.append(i1)

    graph[i1].append(eind)
    graph[i2].append(eind + 1)

for ind in range(m - 1):
    i1 = orderY[ind]
    i2 = orderY[ind + 1]

    w = Y[i2] - Y[i1]

    eind = len(W)
    W.append(w)
    W.append(w)
    V.append(i2)
    V.append(i1)

    graph[i1].append(eind)
    graph[i2].append(eind + 1)

for i in range(m):
    w = abs(X[i] - fx) + abs(Y[i] - fy)
    eind = len(W)
    W.append(w)
    V.append(i)
    graph[m].append(eind)

for i in range(m):
    w = min(abs(X[i] - sx), abs(Y[i] - sy))
    eind = len(W)
    W.append(w)
    V.append(m + 1)
    graph[i].append(eind)

# Directly
i1 = m
i2 = m + 1
w = abs(sx - fx) + abs(sy - fy)
eind = len(W)
W.append(w)
V.append(i2)
graph[i1].append(eind)

dist = [-1] * N
seg = segtree(N)
seg[m] = 0
dist[m] = 0

while True:
    i, d = seg()
    if d == -1:
        break 
    for eind in graph[i]:
        v = V[eind]
        w = W[eind]
        if dist[v] == -1 or dist[v] > d + w:
            dist[v] = seg[v] = d + w
print dist[m + 1]
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
import heapq
from bootcamp import Basebootcamp

class Dreturninghomebootcamp(Basebootcamp):
    def __init__(self, max_m=100, max_n=10**9):
        self.max_m = max_m
        self.max_n = max_n

    def case_generator(self):
        n = random.randint(1, self.max_n)
        m = random.randint(0, min(self.max_m, 10**5))
        
        sx, sy = random.randint(1, n), random.randint(1, n)
        while True:
            fx, fy = random.randint(1, n), random.randint(1, n)
            if (fx, fy) != (sx, sy):
                break
        
        teleports = [(random.randint(1, n), random.randint(1, n)) for _ in range(m)]
        return {
            'n': n,
            'm': m,
            'sx': sx,
            'sy': sy,
            'fx': fx,
            'fy': fy,
            'teleports': teleports
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        teleports = "\n".join(f"{x} {y}" for x, y in question_case['teleports'])
        return f"""Yura needs to get home in minimum time in a {question_case['n']}x{question_case['n']} grid.
Start: ({question_case['sx']}, {question_case['sy']})
Home: ({question_case['fx']}, {question_case['fy']})
Instant-movement locations ({question_case['m']}):
{teleports}

Rules:
1. Normal movement: 1 minute per adjacent block
2. Instant movement: Free if current x/y matches any location's x/y

Calculate minimal time. Put answer in [answer][/answer]. Example: [answer]5[/answer]"""

    @staticmethod
    def extract_output(output):
        answers = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answers:
            return None
        try:
            return int(answers[-1].strip())
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            expected = cls.calculate_min_time(
                identity['sx'], identity['sy'],
                identity['fx'], identity['fy'],
                identity['teleports']
            )
            return int(solution) == expected
        except:
            return False

    @staticmethod
    def calculate_min_time(sx, sy, fx, fy, teleports):
        """
        优化后的最短路径算法实现
        """
        # 直接行走时间
        direct_time = abs(sx - fx) + abs(sy - fy)
        if not teleports:
            return direct_time

        # 建立瞬移点图
        nodes = [(sx, sy, True)] + [(x, y, False) for x, y in teleports] + [(fx, fy, True)]
        node_count = len(nodes)
        
        # 邻接表存储
        graph = [[] for _ in range(node_count)]
        
        # 添加瞬移点之间的边
        for i in range(1, node_count-1):
            # 起点到瞬移点的接入成本
            start_cost = min(abs(nodes[0][0] - nodes[i][0]), abs(nodes[0][1] - nodes[i][1]))
            graph[0].append((i, start_cost))
            
            # 瞬移点到终点的离场成本
            end_cost = abs(nodes[i][0] - fx) + abs(nodes[i][1] - fy)
            graph[i].append((node_count-1, end_cost))
            
            # 瞬移点之间的连接
            for j in range(i+1, node_count-1):
                if nodes[i][0] == nodes[j][0] or nodes[i][1] == nodes[j][1]:
                    graph[i].append((j, 0))
                    graph[j].append((i, 0))

        # Dijkstra算法
        heap = [(0, 0)]
        dist = [float('inf')] * node_count
        dist[0] = 0

        while heap:
            current_dist, u = heapq.heappop(heap)
            if u == node_count-1:
                return min(direct_time, int(current_dist))
            if current_dist > dist[u]:
                continue
                
            for v, w in graph[u]:
                if dist[v] > dist[u] + w:
                    dist[v] = dist[u] + w
                    heapq.heappush(heap, (dist[v], v))
        
        return direct_time
