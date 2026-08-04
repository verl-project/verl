"""# 

### 谜题描述
Let's call an undirected graph of n vertices p-interesting, if the following conditions fulfill: 

  * the graph contains exactly 2n + p edges; 
  * the graph doesn't contain self-loops and multiple edges; 
  * for any integer k (1 ≤ k ≤ n), any subgraph consisting of k vertices contains at most 2k + p edges. 



A subgraph of a graph is some set of the graph vertices and some set of the graph edges. At that, the set of edges must meet the condition: both ends of each edge from the set must belong to the chosen set of vertices. 

Your task is to find a p-interesting graph consisting of n vertices.

Input

The first line contains a single integer t (1 ≤ t ≤ 5) — the number of tests in the input. Next t lines each contains two space-separated integers: n, p (5 ≤ n ≤ 24; p ≥ 0; <image>) — the number of vertices in the graph and the interest value for the appropriate test. 

It is guaranteed that the required graph exists.

Output

For each of the t tests print 2n + p lines containing the description of the edges of a p-interesting graph: the i-th line must contain two space-separated integers ai, bi (1 ≤ ai, bi ≤ n; ai ≠ bi) — two vertices, connected by an edge in the resulting graph. Consider the graph vertices numbered with integers from 1 to n. 

Print the answers to the tests in the order the tests occur in the input. If there are multiple solutions, you can print any of them.

Examples

Input

1
6 0


Output

1 2
1 3
1 4
1 5
1 6
2 3
2 4
2 5
2 6
3 4
3 5
3 6

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
t=int(input())
while t>0:
    t-=1
    n,p=map(int,raw_input().split())
    k=2*n+p
    for i in range(1,n+1):
        for j in range(i+1,n+1):
            print i,j
            k-=1
            if k==0:
                break
        if k==0:
            break
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from itertools import combinations
from bootcamp import Basebootcamp

class Asearchingforgraphbootcamp(Basebootcamp):
    def __init__(self, n_min=5, n_max=24, p_min=0):
        self.n_min = n_min
        self.n_max = n_max
        self.p_min = p_min

    def case_generator(self):
        while True:
            n = random.randint(self.n_min, self.n_max)
            max_possible_edges = n * (n - 1) // 2
            max_p = max_possible_edges - 2 * n
            if max_p >= self.p_min:
                break
        p = random.randint(self.p_min, max_p)
        edges = self.generate_reference_edges(n, p)
        return {'n': n, 'p': p, 'edges': edges}

    @staticmethod
    def generate_reference_edges(n, p):
        edges = []
        required = 2 * n + p
        for i in range(1, n+1):
            for j in range(i+1, n+1):
                edges.append((i, j))
                if len(edges) == required:
                    return edges
        return edges

    @staticmethod
    def prompt_func(case):
        n, p = case['n'], case['p']
        example = "\n".join(f"{u} {v}" for u, v in case['edges'][:3]) + "\n..."
        return f"""Construct a {p}-interesting graph with {n} vertices. Conditions:
1. Exactly {2*n+p} edges
2. No self-loops/multi-edges
3. Any k-vertex subgraph has ≤ {2*'k'}+{p} edges

Output {2*n+p} edges in [answer]...[/answer] format.

Example (n=6, p=0):
{example}"""

    @staticmethod
    def extract_output(output):
        answers = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answers:
            return None
        edges = set()
        for line in answers[-1].strip().split('\n'):
            u, v = map(int, line.strip().split())
            if u != v:
                edges.add((min(u, v), max(u, v)))
        return sorted(edges)

    @classmethod
    def _verify_correction(cls, solution, case):
        # Condition 1: Edge count
        if len(solution) != 2*case['n'] + case['p']:
            return False
        
        # Condition 2: Validate edge structure
        all_nodes = set(range(1, case['n']+1))
        for u, v in solution:
            if u not in all_nodes or v not in all_nodes or u == v:
                return False
        if len(set(solution)) != len(solution):
            return False

        # Condition 3: Generate reference edges and check subset
        ref_edges = set(cls.generate_reference_edges(case['n'], case['p']))
        user_edges = set(solution)
        if not user_edges.issubset(ref_edges):
            return False  # Ensure user edges follow reference structure
        
        return True
