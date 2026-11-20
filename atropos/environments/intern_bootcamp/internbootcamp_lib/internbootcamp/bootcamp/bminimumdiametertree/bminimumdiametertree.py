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
\"\"\"
NTC here
\"\"\"
from sys import stdin

 
# print(\"Case #{}: {} {}\".format(i, n + m, n * m))
 
 
def iin(): return int(stdin.readline())
def lin(): return list(map(int, stdin.readline().split()))

range = xrange
input = raw_input


n,s=lin()

adj=[[] for i in range(n)]
for _ in range(n-1):
    i,j=lin()
    adj[i-1].append(j-1)
    adj[j-1].append(i-1)
l=0
for i in adj:
    if len(i)==1:
        l+=1

print 2*(float(s)/float(l))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import defaultdict, deque
import re
from bootcamp import Basebootcamp

class Bminimumdiametertreebootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=10, s_min=1, s_max=10**9):
        self.min_n = min_n
        self.max_n = max_n
        self.s_min = s_min
        self.s_max = s_max
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        s = random.randint(self.s_min, self.s_max)
        
        # 使用BFS生成更平衡的树结构
        edges = []
        nodes = list(range(1, n+1))
        random.shuffle(nodes)
        
        root = nodes[0]
        available = deque([root])
        used = {root}
        
        for node in nodes[1:]:
            parent = random.choice(available)
            edges.append((parent, node))
            used.add(node)
            available.append(node)
            
            # 保持连接数多样性
            if len(available) > 3 and random.random() < 0.5:
                available.popleft()

        # 正确计算叶节点数
        adj = defaultdict(set)
        for a, b in edges:
            adj[a].add(b)
            adj[b].add(a)
        
        leaf_count = sum(1 for node in adj if len(adj[node]) == 1)
        
        return {
            'n': n,
            's': s,
            'edges': edges,
            'leaf_count': leaf_count
        }
    
    @staticmethod
    def prompt_func(question_case):
        edges_str = "\n".join(f"{a} {b}" for a, b in question_case['edges'])
        return f"""Given a tree with {question_case['n']} nodes and total edge weight sum {question_case['s']}. 
Assign non-negative weights to edges such that:
1. Sum of weights equals {question_case['s']}
2. The diameter (max path weight between any two nodes) is minimized

Edges:
{edges_str}

Calculate the minimal possible diameter. Format your answer with 12+ decimal places within [answer] tags like:
[answer]3.141592653589[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]([\d.]+)\[\/answer\]', output)
        return float(matches[-1]) if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution: return False
        expected = 2 * identity['s'] / identity['leaf_count']
        return abs(solution - expected) < 1e-6 * max(1, expected)
