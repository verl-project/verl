"""# 

### 谜题描述
You are given a tree (an undirected connected graph without cycles) and an integer s.

Vanya wants to put weights on all edges of the tree so that all weights are non-negative real numbers and their sum is s. At the same time, he wants to make the diameter of the tree as small as possible.

Let's define the diameter of a weighed tree as the maximum sum of the weights of the edges lying on the path between two some vertices of the tree. In other words, the diameter of a weighed tree is the length of the longest simple path in the tree, where length of a path is equal to the sum of weights over all edges in the path.

Find the minimum possible diameter that Vanya can get.

Input

The first line contains two integer numbers n and s (2 ≤ n ≤ 10^5, 1 ≤ s ≤ 10^9) — the number of vertices in the tree and the sum of edge weights.

Each of the following n−1 lines contains two space-separated integer numbers a_i and b_i (1 ≤ a_i, b_i ≤ n, a_i ≠ b_i) — the indexes of vertices connected by an edge. The edges are undirected.

It is guaranteed that the given edges form a tree.

Output

Print the minimum diameter of the tree that Vanya can get by placing some non-negative real weights on its edges with the sum equal to s.

Your answer will be considered correct if its absolute or relative error does not exceed 10^{-6}.

Formally, let your answer be a, and the jury's answer be b. Your answer is considered correct if \frac {|a-b|} {max(1, b)} ≤ 10^{-6}.

Examples

Input


4 3
1 2
1 3
1 4


Output


2.000000000000000000

Input


6 1
2 1
2 3
2 5
5 4
5 6


Output


0.500000000000000000

Input


5 5
1 2
2 3
3 4
3 5


Output


3.333333333333333333

Note

In the first example it is necessary to put weights like this:

<image>

It is easy to see that the diameter of this tree is 2. It can be proved that it is the minimum possible diameter.

In the second example it is necessary to put weights like this:

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division
from sys import stdin

rints = lambda: [int(x) for x in stdin.readline().split()]
n, s = rints()
deg = [0] * (n + 1)
for i in range(n - 1):
    u, v = rints()
    deg[u] += 1
    deg[v] += 1

print((2 * s) / len(list(filter(lambda x: x == 1, deg))))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import defaultdict
from bootcamp import Basebootcamp

class Dminimumdiametertreebootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=10, min_s=1, max_s=10**9):
        self.min_n = min_n
        self.max_n = max_n
        self.min_s = min_s
        self.max_s = max_s
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        s = random.randint(self.min_s, self.max_s)
        
        edges = self._generate_tree(n)
        L = self._count_leaves(edges, n)
        correct_answer = (2 * s) / L
        
        return {
            'n': n,
            's': s,
            'edges': edges,
            'correct_answer': correct_answer
        }
    
    def _generate_tree(self, n):
        """Generates a random tree using BFS-based approach for better diversity"""
        if n == 1:
            return []
            
        nodes = list(range(1, n+1))
        random.shuffle(nodes)
        root = nodes[0]
        parent = {}
        q = [root]
        
        for node in nodes[1:]:
            u = random.choice(q)
            parent[node] = u
            q.append(node)
            if random.random() < 0.3:  # Control branching factor
                q.remove(u)
        
        edges = []
        for v, u in parent.items():
            edges.append((u, v))
        return edges
    
    def _count_leaves(self, edges, n):
        degree = defaultdict(int)
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        return sum(1 for d in degree.values() if d == 1)
    
    @staticmethod
    def prompt_func(question_case):
        input_lines = [f"{question_case['n']} {question_case['s']}"]
        for u, v in sorted(question_case['edges']):
            input_lines.append(f"{u} {v}")
        input_str = '\n'.join(input_lines)
        
        prompt = f"""Given a tree with {question_case['n']} nodes and total edge weight sum {question_case['s']}, find the minimal possible diameter. 
Use [answer]result[/answer] with 12 decimal places. 

Input:
{input_str}"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            user_val = float(solution)
            correct = identity['correct_answer']
            abs_err = abs(user_val - correct)
            return abs_err <= 1e-6 or abs_err / max(1.0, correct) <= 1e-6
        except (ValueError, TypeError, KeyError):
            return False
