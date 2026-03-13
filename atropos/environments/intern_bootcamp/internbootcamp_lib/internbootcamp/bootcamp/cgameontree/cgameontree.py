"""# 

### 谜题描述
Momiji has got a rooted tree, consisting of n nodes. The tree nodes are numbered by integers from 1 to n. The root has number 1. Momiji decided to play a game on this tree.

The game consists of several steps. On each step, Momiji chooses one of the remaining tree nodes (let's denote it by v) and removes all the subtree nodes with the root in node v from the tree. Node v gets deleted as well. The game finishes when the tree has no nodes left. In other words, the game finishes after the step that chooses the node number 1.

Each time Momiji chooses a new node uniformly among all the remaining nodes. Your task is to find the expectation of the number of steps in the described game.

Input

The first line contains integer n (1 ≤ n ≤ 105) — the number of nodes in the tree. The next n - 1 lines contain the tree edges. The i-th line contains integers ai, bi (1 ≤ ai, bi ≤ n; ai ≠ bi) — the numbers of the nodes that are connected by the i-th edge.

It is guaranteed that the given graph is a tree.

Output

Print a single real number — the expectation of the number of steps in the described game.

The answer will be considered correct if the absolute or relative error doesn't exceed 10 - 6.

Examples

Input

2
1 2


Output

1.50000000000000000000


Input

3
1 2
1 3


Output

2.00000000000000000000

Note

In the first sample, there are two cases. One is directly remove the root and another is remove the root after one step. Thus the expected steps are: 

1 × (1 / 2) + 2 × (1 / 2) = 1.5

In the second sample, things get more complex. There are two cases that reduce to the first sample, and one case cleaned at once. Thus the expected steps are: 

1 × (1 / 3) + (1 + 1.5) × (2 / 3) = (1 / 3) + (5 / 3) = 2

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = (int)(raw_input())
ed = [[] for x in range(0, n)]
vis = [0 for x in range(0, n)]
for i in range(1, n):
    x, y = map(int, raw_input().split())
    x -= 1
    y -= 1
    ed[x].append(y)
    ed[y].append(x)
ret = 0.0;
d = [0 for x in range(0, n)]
Q = [0]
d[0] = 1
vis[0] =1
pos = 0
while (pos < n):
    u = Q[pos]
    pos += 1
    ret += 1.0 / d[u]
    for v in ed[u]:
        if(vis[v] == 0):
            vis[v] = 1
            d[v] = d[u] + 1
            Q.append(v)
print ret
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import deque
from bootcamp import Basebootcamp

class Cgameontreebootcamp(Basebootcamp):
    def __init__(self, min_nodes=2, max_nodes=20):
        if min_nodes < 2 or max_nodes < min_nodes:
            raise ValueError("Node range must satisfy 2 ≤ min_nodes ≤ max_nodes")
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
    
    def case_generator(self):
        n = random.randint(self.min_nodes, self.max_nodes)
        edges = []
        parents = {}
        
        # Generate valid tree structure
        for node in range(2, n+1):
            parent = random.randint(1, node-1)
            parents[node] = parent
            edges.append((parent, node))

        # Build adjacency list
        adj = [[] for _ in range(n)]
        for a, b in edges:
            a_idx = a-1
            b_idx = b-1
            adj[a_idx].append(b_idx)
            adj[b_idx].append(a_idx)

        # BFS for depth calculation
        depths = [0]*n
        visited = [False]*n
        q = deque([0])  # Root node (1) has index 0
        visited[0] = True
        expectation = 0.0

        while q:
            u = q.popleft()
            expectation += 1.0 / (depths[u] + 1)  # Depth starts from 0
            
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    depths[v] = depths[u] + 1
                    q.append(v)

        return {
            'n': n,
            'edges': edges,
            'expected': expectation,
            '_depth_info': depths  # For debug purposes
        }

    @staticmethod
    def prompt_func(question_case):
        edges = '\n'.join(f"{a} {b}" for a, b in question_case['edges'])
        return f"""Given a rooted tree with {question_case['n']} nodes (root=1), calculate the expected number of steps to delete all nodes through random subtree removal.

Input:
{question_case['n']}
{edges}

Output requirements:
- Compute expectation with 12+ decimal places
- Enclose final answer in [answer][/answer]
- Example: [answer]2.000000000000[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return float(matches[-1].strip())
        except (ValueError, TypeError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False
            
        expected = identity['expected']
        
        # Handle near-zero expected values
        if abs(expected) < 1e-12:
            return abs(solution) < 1e-6
        
        abs_error = abs(solution - expected)
        rel_error = abs_error / abs(expected)
        
        return abs_error <= 1e-6 or rel_error <= 1e-6
