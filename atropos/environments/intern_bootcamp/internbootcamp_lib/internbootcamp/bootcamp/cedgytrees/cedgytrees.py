"""# 

### 谜题描述
You are given a tree (a connected undirected graph without cycles) of n vertices. Each of the n - 1 edges of the tree is colored in either black or red.

You are also given an integer k. Consider sequences of k vertices. Let's call a sequence [a_1, a_2, …, a_k] good if it satisfies the following criterion:

  * We will walk a path (possibly visiting same edge/vertex multiple times) on the tree, starting from a_1 and ending at a_k. 
  * Start at a_1, then go to a_2 using the shortest path between a_1 and a_2, then go to a_3 in a similar way, and so on, until you travel the shortest path between a_{k-1} and a_k.
  * If you walked over at least one black edge during this process, then the sequence is good. 

<image>

Consider the tree on the picture. If k=3 then the following sequences are good: [1, 4, 7], [5, 5, 3] and [2, 3, 7]. The following sequences are not good: [1, 4, 6], [5, 5, 5], [3, 7, 3].

There are n^k sequences of vertices, count how many of them are good. Since this number can be quite large, print it modulo 10^9+7.

Input

The first line contains two integers n and k (2 ≤ n ≤ 10^5, 2 ≤ k ≤ 100), the size of the tree and the length of the vertex sequence.

Each of the next n - 1 lines contains three integers u_i, v_i and x_i (1 ≤ u_i, v_i ≤ n, x_i ∈ \{0, 1\}), where u_i and v_i denote the endpoints of the corresponding edge and x_i is the color of this edge (0 denotes red edge and 1 denotes black edge).

Output

Print the number of good sequences modulo 10^9 + 7.

Examples

Input


4 4
1 2 1
2 3 1
3 4 1


Output


252

Input


4 6
1 2 0
1 3 0
1 4 0


Output


0

Input


3 5
1 2 1
2 3 0


Output


210

Note

In the first example, all sequences (4^4) of length 4 except the following are good: 

  * [1, 1, 1, 1]
  * [2, 2, 2, 2]
  * [3, 3, 3, 3]
  * [4, 4, 4, 4] 



In the second example, all edges are red, hence there aren't any good sequences.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
mod = 10**9 + 7
nk = map(int, raw_input().strip().split())
n = nk[0]
k = nk[1]
parent = [0] * (n+1)
rank = [0] * (n +1)
for i in range(1, n+1):
    parent[i] = i
    rank[i] = 1
for i in range(n-1):
  abc = map(int, raw_input().strip().split())
  if abc[2] == 1:
      continue
  def find(node):
      while(parent[node] != node):
          parent[node] = parent[parent[node]]
          node = parent[node]
      return node
  ra = find(abc[0])
  rb = find(abc[1])
  parent[ra] = parent[rb]
  rank[rb] += rank[ra]

sum_excluded = 0
for i in range(1,n+1):
  if parent[i] == i:
    sum_excluded += (rank[i] ** k) % mod
    sum_excluded %= mod

start = (n**k) % mod
print (start - sum_excluded + mod)%mod
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import defaultdict

mod = 10**9 + 7

class Cedgytreesbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n_min = params.get('n_min', 2)
        self.n_max = params.get('n_max', 20)
        self.k_min = params.get('k_min', 2)
        self.k_max = params.get('k_max', 10)
        self.red_prob = params.get('red_prob', 0.5)
        self.seed = params.get('seed', None)
        if self.seed is not None:
            random.seed(self.seed)
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        k = random.randint(self.k_min, self.k_max)
        edges = []
        parent = [i for i in range(n + 1)]
        
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        
        for i in range(1, n):
            u = random.randint(1, n)
            v = random.randint(1, n)
            while u == v:
                v = random.randint(1, n)
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                edges.append((u, v))
                parent[root_u] = root_v
        
        colored_edges = []
        for u, v in edges:
            x = 0 if random.random() < self.red_prob else 1
            colored_edges.append({'u': u, 'v': v, 'x': x})
        
        return {
            'n': n,
            'k': k,
            'edges': colored_edges
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k = question_case['k']
        edges = question_case['edges']
        edge_descriptions = []
        for edge in edges:
            u = edge['u']
            v = edge['v']
            x = edge['x']
            color = '黑色' if x == 1 else '红色'
            edge_descriptions.append(f"边{u}-{v}，颜色{color}")
        prompt = f"""
        你被给予了一棵由{n}个顶点组成的树。每条边被染成红色或黑色。你需要计算长度为{k}的顶点序列中有多少是好的。好的序列满足在移动过程中至少经过一条黑色边。移动规则是从a1开始，依次走到a2，a3，直到ak，每次走最短路径。请计算这样的序列总数，结果模1e9+7。将答案放在[answer]标签中。

        边信息如下：
        {"; ".join(edge_descriptions)}

        你的任务是计算符合条件的序列数，并将答案放在[answer]标签中。
        """
        return prompt.strip()
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](\d+)\[/answer\]', output)
        if matches:
            return int(matches[-1])
        else:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        k = identity['k']
        edges = identity['edges']
        
        parent = list(range(n + 1))
        rank = [1] * (n + 1)
        
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        
        for edge in edges:
            u = edge['u']
            v = edge['v']
            x = edge['x']
            if x == 0:
                root_u = find(u)
                root_v = find(v)
                if root_u != root_v:
                    if rank[root_u] > rank[root_v]:
                        parent[root_v] = root_u
                    else:
                        parent[root_u] = root_v
                        if rank[root_u] == rank[root_v]:
                            rank[root_v] += 1
        
        root_counts = defaultdict(int)
        for i in range(1, n + 1):
            root = find(i)
            root_counts[root] += 1
        
        sum_excluded = 0
        for m in root_counts.values():
            sum_excluded = (sum_excluded + pow(m, k, mod)) % mod
        
        total = pow(n, k, mod)
        correct = (total - sum_excluded + mod) % mod
        
        return solution == correct
