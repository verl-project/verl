"""# 

### 谜题描述
Lunar New Year is approaching, and Bob decides to take a wander in a nearby park.

The park can be represented as a connected graph with n nodes and m bidirectional edges. Initially Bob is at the node 1 and he records 1 on his notebook. He can wander from one node to another through those bidirectional edges. Whenever he visits a node not recorded on his notebook, he records it. After he visits all nodes at least once, he stops wandering, thus finally a permutation of nodes a_1, a_2, …, a_n is recorded.

Wandering is a boring thing, but solving problems is fascinating. Bob wants to know the lexicographically smallest sequence of nodes he can record while wandering. Bob thinks this problem is trivial, and he wants you to solve it.

A sequence x is lexicographically smaller than a sequence y if and only if one of the following holds: 

  * x is a prefix of y, but x ≠ y (this is impossible in this problem as all considered sequences have the same length); 
  * in the first position where x and y differ, the sequence x has a smaller element than the corresponding element in y. 

Input

The first line contains two positive integers n and m (1 ≤ n, m ≤ 10^5), denoting the number of nodes and edges, respectively.

The following m lines describe the bidirectional edges in the graph. The i-th of these lines contains two integers u_i and v_i (1 ≤ u_i, v_i ≤ n), representing the nodes the i-th edge connects.

Note that the graph can have multiple edges connecting the same two nodes and self-loops. It is guaranteed that the graph is connected.

Output

Output a line containing the lexicographically smallest sequence a_1, a_2, …, a_n Bob can record.

Examples

Input


3 2
1 2
1 3


Output


1 2 3 


Input


5 5
1 4
3 4
5 4
3 2
1 5


Output


1 4 3 2 5 


Input


10 10
1 4
6 8
2 5
3 7
9 4
5 6
3 4
8 10
8 9
1 10


Output


1 4 3 7 9 8 6 5 2 10 

Note

In the first sample, Bob's optimal wandering path could be 1 → 2 → 1 → 3. Therefore, Bob will obtain the sequence \{1, 2, 3\}, which is the lexicographically smallest one.

In the second sample, Bob's optimal wandering path could be 1 → 4 → 3 → 2 → 3 → 4 → 1 → 5. Therefore, Bob will obtain the sequence \{1, 4, 3, 2, 5\}, which is the lexicographically smallest one.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import heapq
n,m = map(int,raw_input().split())
g = []
for i in range(n):
	g.append([])
for i in range(m):
	u,v = map(int,raw_input().split())
	g[u-1].append(v-1)
	g[v-1].append(u-1)
stck = []
heapq.heappush(stck,0)
vis = [0]*n
ans = []
while len(stck)!=0:
	v = heapq.heappop(stck)
	if vis[v]:
		continue
	vis[v] = 1
	ans.append(v+1)
	for i in g[v]:
		if not vis[i]:
			heapq.heappush(stck,i)
for i in ans:
	print i,
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import heapq
import random
import re
from bootcamp import Basebootcamp

def compute_min_lex_sequence(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        u_zero = u - 1
        v_zero = v - 1
        adj[u_zero].append(v_zero)
        adj[v_zero].append(u_zero)
    
    heap = []
    heapq.heappush(heap, 0)
    visited = [False] * n
    result = []
    
    while heap:
        current = heapq.heappop(heap)
        if visited[current]:
            continue
        visited[current] = True
        result.append(current + 1)
        for neighbor in sorted(adj[current]):  # 需要排序保证稳定性
            if not visited[neighbor]:
                heapq.heappush(heap, neighbor)
    return result

class Dlunarnewyearandawanderbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', 5)
        self.m = params.get('m', self.n)
        if self.m < self.n-1:
            raise ValueError(f"m must be ≥ n-1 ({self.n-1}), got {self.m}")
        if self.n < 1:
            raise ValueError(f"n must be ≥ 1, got {self.n}")

    def case_generator(self):
        n = self.n
        m = self.m
        edges = []
        visited = {1}
        unvisited = set(range(2, n+1))
        
        # 生成生成树确保连通性
        while unvisited:
            u = random.choice(list(unvisited))
            v = random.choice(list(visited))
            edges.append((v, u))
            visited.add(u)
            unvisited.remove(u)
        
        # 生成剩余边（允许自环和重复）
        for _ in range(m - (n-1)):
            u = random.randint(1, n)
            v = random.randint(1, n)
            edges.append((u, v))
        
        # 洗牌避免边顺序影响生成结果
        random.shuffle(edges)
        return {
            'n': n,
            'm': m,
            'edges': edges,
            'correct_answer': compute_min_lex_sequence(n, edges)
        }

    @staticmethod
    def prompt_func(question_case):
        edges = [" ".join(map(str, e)) for e in question_case['edges']]
        input_str = f"{question_case['n']} {question_case['m']}\n" + "\n".join(edges)
        return f"""作为公园路径规划AI，你需要找到字典序最小的节点访问序列。规则：

1. 从节点1出发并记录
2. 每次访问新节点立即记录
3. 使用双向通道移动
4. 序列字典序最小标准：首个不同位置数值更小则更优

输入格式：
n m
u1 v1
...
um vm

当前输入：
{input_str}

答案格式：[answer]1 3 2 ...[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return list(map(int, re.findall(r'\d+', matches[-1]))) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
