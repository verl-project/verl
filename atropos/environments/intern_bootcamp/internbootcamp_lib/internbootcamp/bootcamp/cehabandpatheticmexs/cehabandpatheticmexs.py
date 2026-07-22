"""# 

### 谜题描述
You are given a tree consisting of n nodes. You want to write some labels on the tree's edges such that the following conditions hold:

  * Every label is an integer between 0 and n-2 inclusive. 
  * All the written labels are distinct. 
  * The largest value among MEX(u,v) over all pairs of nodes (u,v) is as small as possible. 



Here, MEX(u,v) denotes the smallest non-negative integer that isn't written on any edge on the unique simple path from node u to node v.

Input

The first line contains the integer n (2 ≤ n ≤ 10^5) — the number of nodes in the tree.

Each of the next n-1 lines contains two space-separated integers u and v (1 ≤ u,v ≤ n) that mean there's an edge between nodes u and v. It's guaranteed that the given graph is a tree.

Output

Output n-1 integers. The i^{th} of them will be the number written on the i^{th} edge (in the input order).

Examples

Input


3
1 2
1 3


Output


0
1


Input


6
1 2
1 3
2 4
2 5
5 6


Output


0
3
2
4
1

Note

The tree from the second sample:

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin, stdout
n=int(stdin.readline())
edges=[]
sz=[0]*(n+1)
ans=[-1]*(n-1)
for i in xrange(n-1):
    u,v=map(int, stdin.readline().split())
    edges.append([u,v,i])
    edges.append([v,u,i])
    sz[u]+=1
    sz[v]+=1

m=0

for i in xrange(len(sz)):
    if sz[i]>sz[m]:
        m=i

res=0
for i in xrange(len(edges)):
    if edges[i][0]==m:
        ans[edges[i][2]]=res
        res+=1

for i in xrange(len(edges)):
    if ans[edges[i][2]]==-1:
        ans[edges[i][2]]=res
        res+=1
print '\n'.join(map(str, ans))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Cehabandpatheticmexsbootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=10):
        self.min_n = min_n
        self.max_n = max_n
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        edges = self._generate_random_tree(n)
        sz = [0] * (n + 1)
        for u, v in edges:
            sz[u] += 1
            sz[v] += 1
        m = max(range(1, n+1), key=lambda x: sz[x])  # 修正：限定节点范围
        k = sz[m]
        m_edges = []
        other_edges = []
        for edge in edges:
            u, v = edge
            if u == m or v == m:
                m_edges.append(edge)
            else:
                other_edges.append(edge)
        edges = m_edges + other_edges
        return {
            'n': n,
            'edges': edges,
            'm': m,
            'k': k,
        }
    
    @staticmethod
    def _generate_random_tree(n):
        if n == 1:
            return []
        if n == 2:
            return [(1, 2)]
        prufer = [random.randint(1, n) for _ in range(n-2)]
        degree = [1] * (n + 1)
        for node in prufer:
            degree[node] += 1
        edges = []
        for node in prufer:
            for v in range(1, n+1):
                if degree[v] == 1 and degree[node] > 0:
                    edges.append((node, v))
                    degree[node] -= 1
                    degree[v] -= 1
                    break
        left = [v for v in range(1, n+1) if degree[v] == 1]
        edges.append((left[0], left[1]))
        return edges
    
    @staticmethod
    def prompt_func(question_case):
        edges = question_case['edges']
        n = question_case['n']
        edge_lines = '\n'.join([f"{u} {v}" for u, v in edges])
        prompt = f"You are given a tree consisting of {n} nodes. Your task is to label each edge with a distinct integer between 0 and {n-2} such that the maximum MEX value across all pairs of nodes is minimized.\n\n"
        prompt += "Input Format:\n"
        prompt += f"- The first line contains the integer {n}.\n"
        prompt += f"- The next {n-1} lines describe the edges. Each line contains two integers u and v.\n\n"
        prompt += "Output Format:\n"
        prompt += f"Output {n-1} integers, each corresponding to the label of the edges in the order they were input.\n\n"
        prompt += "Rules for labeling:\n"
        prompt += "1. Labels must be unique integers from 0 to n-2.\n"
        prompt += "2. The goal is to minimize the largest MEX value over all node pairs.\n"
        prompt += "   MEX(u, v) is the smallest non-negative integer not present on the path between u and v.\n\n"
        prompt += "Example Input and Output:\n"
        prompt += "Example 1:\n"
        prompt += "Input:\n3\n1 2\n1 3\nOutput:\n0 1\n"
        prompt += "Example 2:\n"
        prompt += "Input:\n6\n1 2\n1 3\n2 4\n2 5\n5 6\nOutput:\n0 3 2 4 1\n\n"
        prompt += "Current Input:\n"
        prompt += f"{n}\n{edge_lines}\n\n"
        prompt += "Please provide your answer as a space-separated list of integers enclosed within [answer] and [/answer] tags.\n"
        return prompt
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            solution = list(map(int, last_match.split()))
            return solution
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        n = identity['n']
        edges = identity['edges']
        m = identity['m']
        k = identity['k']
        
        if len(solution) != n - 1:
            return False
        
        if sorted(solution) != list(range(n - 1)):
            return False
        
        m_edge_indices = []
        for idx, (u, v) in enumerate(edges):
            if u == m or v == m:
                m_edge_indices.append(idx)
        
        if len(m_edge_indices) != k:
            return False
        
        m_labels = [solution[idx] for idx in m_edge_indices]
        if sorted(m_labels) != list(range(k)):
            return False
        
        non_m_labels = [solution[idx] for idx in range(n-1) if idx not in m_edge_indices]
        if any(label < k for label in non_m_labels):
            return False
        
        return True
