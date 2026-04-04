"""# 

### 谜题描述
Sergey Semyonovich is a mayor of a county city N and he used to spend his days and nights in thoughts of further improvements of Nkers' lives. Unfortunately for him, anything and everything has been done already, and there are no more possible improvements he can think of during the day (he now prefers to sleep at night). However, his assistants have found a solution and they now draw an imaginary city on a paper sheet and suggest the mayor can propose its improvements.

Right now he has a map of some imaginary city with n subway stations. Some stations are directly connected with tunnels in such a way that the whole map is a tree (assistants were short on time and enthusiasm). It means that there exists exactly one simple path between each pair of station. We call a path simple if it uses each tunnel no more than once.

One of Sergey Semyonovich's favorite quality objectives is the sum of all pairwise distances between every pair of stations. The distance between two stations is the minimum possible number of tunnels on a path between them.

Sergey Semyonovich decided to add new tunnels to the subway map. In particular, he connected any two stations u and v that were not connected with a direct tunnel but share a common neighbor, i.e. there exists such a station w that the original map has a tunnel between u and w and a tunnel between w and v. You are given a task to compute the sum of pairwise distances between all pairs of stations in the new map.

Input

The first line of the input contains a single integer n (2 ≤ n ≤ 200 000) — the number of subway stations in the imaginary city drawn by mayor's assistants. Each of the following n - 1 lines contains two integers u_i and v_i (1 ≤ u_i, v_i ≤ n, u_i ≠ v_i), meaning the station with these indices are connected with a direct tunnel.

It is guaranteed that these n stations and n - 1 tunnels form a tree.

Output

Print one integer that is equal to the sum of distances between all pairs of stations after Sergey Semyonovich draws new tunnels between all pairs of stations that share a common neighbor in the original map.

Examples

Input

4
1 2
1 3
1 4


Output

6


Input

4
1 2
2 3
3 4


Output

7

Note

In the first sample, in the new map all pairs of stations share a direct connection, so the sum of distances is 6.

In the second sample, the new map has a direct tunnel between all pairs of stations except for the pair (1, 4). For these two stations the distance is 2.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import os
import sys
from atexit import register
from io import BytesIO
sys.stdin = BytesIO(os.read(0, os.fstat(0).st_size))
sys.stdout = BytesIO()
register(lambda: os.write(1, sys.stdout.getvalue()))
input = lambda: sys.stdin.readline().rstrip('\r\n')
raw_input = lambda: sys.stdin.readline().rstrip('\r\n')

n = int(input())
edges = [[]for i in range(n+1)]
for i in range(n-1):
    a,b = map(int,raw_input().split(\" \"))
    edges[a].append(b)
    edges[b].append(a)
    


root = 1

q = [root]

odd = [0]*(1+n)
even = [0]*(1+n)
odd_size = [0]*(1+n)
even_size = [1]*(1+n)

rank = [0]*(1+n)
rank[root] = 1
i = 0
grandson = [0]*(1+n)
while i<len(q):
    node = q[i]
    for v in edges[node]:
        if rank[v] == 0:
            rank[v] = rank[node]+1
            q.append(v)
    i += 1 

for node in q[::-1]:
    for v in edges[node]:
        if rank[v]>rank[node]:
            odd[node] += even[v]+even_size[v]
            even[node] += odd[v]+odd_size[v]
            even_size[node] += odd_size[v]
            odd_size[node] += even_size[v]
ans = 0
for node in q:
    for v in edges[node]:
        if rank[v]>rank[node]:
            deven = odd[node]-(even[v]+even_size[v])+(odd_size[node]-even_size[v])
            dodd = even[node]-(odd[v]+odd_size[v])+ (even_size[node]-odd_size[v])
            even[v] += deven
            odd[v] +=  dodd
            even_size[v] = odd_size[node]
            odd_size[v] = even_size[node]

for i in range(1,n+1):
    ans += even[i]/2
    ans += (odd[i]+odd_size[i])/2
print ans/2
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Esergeyandsubwaybootcamp(Basebootcamp):
    def __init__(self, max_n=100):
        """
        初始化训练场参数，设置生成树的最大节点数。
        """
        self.max_n = max_n
        self.min_n = 2
    
    def case_generator(self):
        """
        生成一个树结构实例，并计算正确的答案。
        """
        n = random.randint(self.min_n, self.max_n)
        edges = self.generate_random_tree(n)
        correct_answer = self.solve(n, edges)
        return {
            'n': n,
            'edges': edges,
            'correct_answer': correct_answer
        }
    
    @staticmethod
    def generate_random_tree(n):
        """
        使用Prüfer序列生成随机树。
        """
        if n == 1:
            return []
        if n == 2:
            return [(1, 2)]
        
        prufer = [random.randint(1, n) for _ in range(n-2)]
        degree = [0] * (n + 1)
        for node in prufer:
            degree[node] += 1
        
        leaves = []
        for i in range(1, n + 1):
            if degree[i] == 0:
                leaves.append(i)
        
        edges = []
        for node in prufer:
            leaf = leaves.pop(0)
            edges.append((leaf, node))
            degree[node] -= 1
            if degree[node] == 0:
                leaves.append(node)
            leaves.sort()
        
        edges.append((leaves[0], leaves[1]))
        return edges
    
    @staticmethod
    def solve(n, edges_list):
        """
        根据给定的树结构计算正确的结果。
        """
        adj = [[] for _ in range(n+1)]
        for a, b in edges_list:
            adj[a].append(b)
            adj[b].append(a)
        
        root = 1
        q = [root]
        odd = [0] * (n+1)
        even = [0] * (n+1)
        odd_size = [0] * (n+1)
        even_size = [1] * (n+1)
        rank = [0] * (n+1)
        rank[root] = 1
        
        i = 0
        while i < len(q):
            node = q[i]
            for v in adj[node]:
                if rank[v] == 0:
                    rank[v] = rank[node] + 1
                    q.append(v)
            i += 1
        
        for node in reversed(q):
            for v in adj[node]:
                if rank[v] > rank[node]:
                    odd[node] += even[v] + even_size[v]
                    even[node] += odd[v] + odd_size[v]
                    even_size[node] += odd_size[v]
                    odd_size[node] += even_size[v]
        
        for node in q:
            for v in adj[node]:
                if rank[v] > rank[node]:
                    deven = odd[node] - (even[v] + even_size[v]) + (odd_size[node] - even_size[v])
                    dodd = even[node] - (odd[v] + odd_size[v]) + (even_size[node] - odd_size[v])
                    even[v] += deven
                    odd[v] += dodd
                    even_size[v] = odd_size[node]
                    odd_size[v] = even_size[node]
        
        ans = 0
        for i in range(1, n+1):
            ans += even[i] // 2
            ans += (odd[i] + odd_size[i]) // 2
        ans = ans // 2
        return ans
    
    @staticmethod
    def prompt_func(question_case) -> str:
        edges = question_case['edges']
        n = question_case['n']
        edges_str = '\n'.join(f"{u} {v}" for u, v in edges)
        prompt = f"""Sergey Semyonovich, the mayor of city N, wants to improve the subway system. The subway network is a tree with {n} stations. Sergey adds new tunnels between any two stations u and v that were not directly connected but share a common neighbor. Your task is to calculate the sum of all pairwise distances between stations after these new tunnels are added.

Input:
The first line contains an integer n (2 ≤ n ≤ 200000) — the number of stations. The next n-1 lines describe the tunnels, each with two integers u and v.

Problem instance:
n = {n}
The tunnels are:
{edges_str}

Your answer must be a single integer. Please place your final answer within [answer] and [/answer] tags. For example, [answer]42[/answer]."""
        return prompt
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
