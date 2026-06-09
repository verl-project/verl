"""# 

### 谜题描述
Students of Winter Informatics School are going to live in a set of houses connected by underground passages. Teachers are also going to live in some of these houses, but they can not be accommodated randomly. For safety reasons, the following must hold:

  * All passages between two houses will be closed, if there are no teachers in both of them. All other passages will stay open. 
  * It should be possible to travel between any two houses using the underground passages that are open. 
  * Teachers should not live in houses, directly connected by a passage. 



Please help the organizers to choose the houses where teachers will live to satisfy the safety requirements or determine that it is impossible.

Input

The first input line contains a single integer t — the number of test cases (1 ≤ t ≤ 10^5). 

Each test case starts with two integers n and m (2 ≤ n ≤ 3 ⋅ 10^5, 0 ≤ m ≤ 3 ⋅ 10^5) — the number of houses and the number of passages.

Then m lines follow, each of them contains two integers u and v (1 ≤ u, v ≤ n, u ≠ v), describing a passage between the houses u and v. It is guaranteed that there are no two passages connecting the same pair of houses.

The sum of values n over all test cases does not exceed 3 ⋅ 10^5, and the sum of values m over all test cases does not exceed 3 ⋅ 10^5.

Output

For each test case, if there is no way to choose the desired set of houses, output \"NO\". Otherwise, output \"YES\", then the total number of houses chosen, and then the indices of the chosen houses in arbitrary order.

Examples

Input


2
3 2
3 2
2 1
4 2
1 4
2 3


Output


YES
2
1 3 
NO


Input


1
17 27
1 8
2 9
3 10
4 11
5 12
6 13
7 14
8 9
8 14
8 15
9 10
9 15
10 11
10 15
10 17
11 12
11 17
12 13
12 16
12 17
13 14
13 16
14 16
14 15
15 16
15 17
16 17


Output


YES
8
1 3 4 5 6 9 14 17 

Note

The picture below shows the second example test. 

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
\"\"\" Python 3 compatibility tools. \"\"\"
from __future__ import division, print_function
import itertools
import sys
import os
from io import BytesIO
from atexit import register

if sys.version_info[0] < 3:
    input = raw_input
    range = xrange

    filter = itertools.ifilter
    map = itertools.imap
    zip = itertools.izip

def gcd(x, y):
    \"\"\" greatest common divisor of x and y \"\"\"
    while y:
        x, y = y, x % y
    return x

sys.stdin = BytesIO(os.read(0, os.fstat(0).st_size))
sys.stdout = BytesIO()
register(lambda: os.write(1, sys.stdout.getvalue()))
input = lambda: sys.stdin.readline()

from collections import deque

cases = int(input())
for _ in range(cases):
    n, m = map(int, input().split())
    adj = [set() for i in range(n + 1)]
    visited = [False] * (n + 1)
    for i in range(m):
        start, end = map(int, input().split())
        adj[start].add(end)
        adj[end].add(start)
    queue = deque()
    uncolored = deque()
    uncolored.append(1)
    solution = []
    allE = 0
    while uncolored:
        index = uncolored.popleft()
        if visited[index]:
            continue
        visited[index] = True
        allE += 1
        solution.append(str(index))
        for i in adj[index]:
            if not visited[i]:
                queue.append(i)
                
        while queue:
            indexQ = queue.popleft()
            if visited[indexQ]:
                continue
            visited[indexQ] = True
            allE += 1
            for i in adj[indexQ]:
                if not visited[i]:
                    uncolored.append(i)
    
    # print(visited)                        
    if allE == n:
        print(\"YES\")
        print(len(solution))
        print(\" \".join(solution))
    else:
        print(\"NO\")
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import deque
from bootcamp import Basebootcamp

class Dstrangehousingbootcamp(Basebootcamp):
    def __init__(self, max_houses=10, max_paths=15):
        self.max_houses = max_houses
        self.max_paths = max_paths

    def case_generator(self):
        # 生成连通二分图或不可解图（概率各50%）
        if random.random() < 0.5:
            return self._generate_valid_case()
        else:
            return self._generate_invalid_case()

    def _generate_valid_case(self):
        """生成保证有解的连通二分图"""
        n = random.randint(3, self.max_houses)
        partition = {i: i%2 for i in range(1, n+1)}  # 简单二分
        
        # 确保图连通
        edges = []
        visited = set([1])
        queue = deque([1])
        
        while len(visited) < n:
            u = queue.popleft()
            candidates = [v for v in range(1, n+1) 
                        if v not in visited and partition[v] != partition[u]]
            if candidates:
                v = random.choice(candidates)
                edges.append((u, v))
                visited.add(v)
                queue.append(v)
            else:  # 添加跨分区边保持连通
                for v in range(1, n+1):
                    if v not in visited and partition[v] == partition[u]:
                        edges.append((u, v))
                        visited.add(v)
                        queue.append(v)
                        break
        
        # 添加额外边（保持二分性）
        possible_edges = []
        for u in range(1, n):
            for v in range(u+1, n+1):
                if partition[u] != partition[v] and (u, v) not in edges:
                    possible_edges.append((u, v))
        
        add_num = min(len(possible_edges), self.max_paths - len(edges))
        edges.extend(random.sample(possible_edges, add_num))
        
        return {'n': n, 'm': len(edges), 'edges': edges}

    def _generate_invalid_case(self):
        """生成包含奇数环的不可解案例"""
        cycle_size = random.choice([3, 5, 7])
        n = cycle_size
        edges = [(i, i%cycle_size +1) for i in range(1, cycle_size+1)]
        
        # 添加额外边保持连通
        for _ in range(random.randint(0, 3)):
            u = random.randint(1, n)
            v = random.randint(1, n)
            if u != v and (u, v) not in edges and (v, u) not in edges:
                edges.append((u, v))
        
        return {'n': n, 'm': len(edges), 'edges': edges}

    @staticmethod
    def prompt_func(case) -> str:
        edges = "\n".join(f"{u} {v}" for u, v in case['edges'])
        return f"""# 冬季信息学校的教师住房问题

## 问题描述
需要选择若干房屋安排教师，满足：
1. 任何两个有教师的房屋不能直接相连
2. 无教师的房屋之间的通道关闭后，剩余通道必须保持全图连通
3. 必须同时满足上述两个条件

## 输入格式
房屋数：{case['n']}
通道数：{case['m']}
通道列表：
{edges}

## 输出要求
若存在可行方案：
[answer]
YES
K
a₁ a₂ ... a_K
[/answer]

若无解：
[answer]
NO
[/answer]

请严格遵循输出格式，答案标记必须使用[answer]标签。"""

    @staticmethod
    def extract_output(output):
        # 提取最后一个合法答案块
        answer_blocks = []
        start = -1
        for i in range(len(output)):
            if output.startswith('[answer]', i):
                start = i + 8
            elif output.startswith('[/answer]', i) and start != -1:
                answer_blocks.append(output[start:i].strip())
                start = -1
        
        if not answer_blocks:
            return None
            
        last_answer = answer_blocks[-1].split('\n')
        cleaned = [line.strip() for line in last_answer if line.strip()]
        
        if not cleaned:
            return None
            
        if cleaned[0].upper() == 'NO':
            return 'NO'
            
        if len(cleaned) >=2 and cleaned[0].upper() == 'YES':
            try:
                k = int(cleaned[1])
                if len(cleaned) == 2 + k:
                    nums = list(map(int, cleaned[2:]))
                    return f"YES {k} {' '.join(map(str, nums))}"
            except:
                pass
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 处理无效解
        if not solution:
            return False
            
        if solution == 'NO':
            return not cls._is_bipartite(identity['edges'], identity['n'])
        
        # 解析答案
        parts = solution.split()
        if len(parts) < 3 or parts[0] != 'YES':
            return False
            
        try:
            k = int(parts[1])
            S = set(map(int, parts[2:2+k]))
        except:
            return False
        
        # 条件1：独立集验证
        adj = {u: set() for u in range(1, identity['n']+1)}
        for u, v in identity['edges']:
            adj[u].add(v)
            adj[v].add(u)
            if u in S and v in S:
                return False
        
        # 条件2：开放通道后的连通性验证
        open_edges = set()
        T = set(range(1, identity['n']+1)) - S
        for u, v in identity['edges']:
            if u in S or v in S:
                open_edges.add((u, v))
        
        # 构建邻接表
        graph = {u: [] for u in range(1, identity['n']+1)}
        for u, v in open_edges:
            graph[u].append(v)
            graph[v].append(u)
        
        # BFS检查连通性
        visited = set()
        start_node = next((u for u in T if u in graph and graph[u]), None) or next(iter(S), None)
        if not start_node:
            return False
            
        queue = deque([start_node])
        visited.add(start_node)
        while queue:
            u = queue.popleft()
            for v in graph[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        
        return len(visited) == identity['n']

    @classmethod
    def _is_bipartite(cls, edges, n):
        """判断是否为二分图（可解条件）"""
        color = {}
        adj = {u: [] for u in range(1, n+1)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for u in range(1, n+1):
            if u not in color:
                queue = deque([u])
                color[u] = 0
                while queue:
                    current = queue.popleft()
                    for neighbor in adj[current]:
                        if neighbor not in color:
                            color[neighbor] = color[current] ^ 1
                            queue.append(neighbor)
                        elif color[neighbor] == color[current]:
                            return False
        return True
