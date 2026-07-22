"""# 

### 谜题描述
[INSPION FullBand Master - INSPION](https://www.youtube.com/watch?v=kwsciXm_7sA)

[INSPION - IOLITE-SUNSTONE](https://www.youtube.com/watch?v=kwsciXm_7sA)

On another floor of the A.R.C. Markland-N, the young man Simon \"Xenon\" Jackson, takes a break after finishing his project early (as always). Having a lot of free time, he decides to put on his legendary hacker \"X\" instinct and fight against the gangs of the cyber world.

His target is a network of n small gangs. This network contains exactly n - 1 direct links, each of them connecting two gangs together. The links are placed in such a way that every pair of gangs is connected through a sequence of direct links.

By mining data, Xenon figured out that the gangs used a form of cross-encryption to avoid being busted: every link was assigned an integer from 0 to n - 2 such that all assigned integers are distinct and every integer was assigned to some link. If an intruder tries to access the encrypted data, they will have to surpass S password layers, with S being defined by the following formula:

$$$S = ∑_{1 ≤ u < v ≤ n} mex(u, v)$$$

Here, mex(u, v) denotes the smallest non-negative integer that does not appear on any link on the unique simple path from gang u to gang v.

Xenon doesn't know the way the integers are assigned, but it's not a problem. He decides to let his AI's instances try all the passwords on his behalf, but before that, he needs to know the maximum possible value of S, so that the AIs can be deployed efficiently.

Now, Xenon is out to write the AI scripts, and he is expected to finish them in two hours. Can you find the maximum possible S before he returns?

Input

The first line contains an integer n (2 ≤ n ≤ 3000), the number of gangs in the network.

Each of the next n - 1 lines contains integers u_i and v_i (1 ≤ u_i, v_i ≤ n; u_i ≠ v_i), indicating there's a direct link between gangs u_i and v_i.

It's guaranteed that links are placed in such a way that each pair of gangs will be connected by exactly one simple path.

Output

Print the maximum possible value of S — the number of password layers in the gangs' network.

Examples

Input


3
1 2
2 3


Output


3


Input


5
1 2
1 3
1 4
3 5


Output


10

Note

In the first example, one can achieve the maximum S with the following assignment:

<image>

With this assignment, mex(1, 2) = 0, mex(1, 3) = 2 and mex(2, 3) = 1. Therefore, S = 0 + 2 + 1 = 3.

In the second example, one can achieve the maximum S with the following assignment:

<image>

With this assignment, all non-zero mex value are listed below: 

  * mex(1, 3) = 1 
  * mex(1, 5) = 2 
  * mex(2, 3) = 1 
  * mex(2, 5) = 2 
  * mex(3, 4) = 1 
  * mex(4, 5) = 3 



Therefore, S = 1 + 2 + 1 + 2 + 1 + 3 = 10.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
range = xrange
input = raw_input
 
# Read input and build the graph
inp = [int(x) for x in sys.stdin.read().split()]; ii = 0
n = inp[ii]; ii += 1
coupl = [[] for _ in range(n)]
for _ in range(n - 1):
    u = inp[ii] - 1; ii += 1
    v = inp[ii] - 1; ii += 1
    coupl[u].append(v)
    coupl[v].append(u)
 
# Relabel to speed up n^2 operations later on
bfs = [0]
found = [0]*n
found[0] = 1
for node in bfs:
    for nei in coupl[node]:
        if not found[nei]:
            found[nei] = 1
            bfs.append(nei)
 
new_label = [0]*n
for i in range(n):
    new_label[bfs[i]] = i
 
coupl = [coupl[i] for i in bfs]
for c in coupl:
    c[:] = [new_label[x] for x in c]
 
##### DP using multisource bfs
 
DP = [0] * (n * n)
size = [1] * (n * n)
P = [-1] * (n * n)
 
# Create the bfs ordering
bfs = [root * n + root for root in range(n)]
for ind in bfs:
    P[ind] = ind

for ind in bfs:
    node, root = divmod(ind, n)
    for nei in coupl[node]:
        ind2 = nei * n + root
        if P[ind2] == -1:
            bfs.append(ind2)
            P[ind2] = ind
 
del bfs[:n]
 
# Do the DP
for ind in reversed(bfs):
    node, root = divmod(ind, n)
    ind2 = root * n + node
    pind = P[ind]
    parent = pind//n
    
    # Update size of (root, parent)
    size[pind] += size[ind]
 
    # Update DP value of (root, parent)
    DP[pind] = max(DP[pind], max(DP[ind], DP[ind2]) + size[ind] * size[ind2])
print max(DP[root * n + root] for root in range(n))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cxenonsattackonthegangsbootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=10):
        self.min_n = min_n
        self.max_n = max_n
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        edges = self.generate_balanced_tree(n)
        correct_S = self.calculate_max_S(n, edges)
        return {
            'n': n,
            'edges': edges,
            'correct_S': correct_S
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        edges_str = '\n'.join(f"{u} {v}" for u, v in question_case['edges'])
        prompt = f"""You are Xenon, a legendary hacker tasked with calculating the maximum possible value of S for a network of gangs. The network forms a tree structure with {question_case['n']} nodes. Your goal is to determine the maximum S, defined as the sum of the mex values for all pairs of nodes (u, v) where u < v.

**Problem Details:**
- The network has {question_case['n']} gangs connected by {question_case['n']-1} direct links.
- Each link is assigned a distinct integer from 0 to {question_case['n']-2}.
- The mex(u, v) is the smallest non-negative integer not present on the path between u and v.

**Input:**
{question_case['n']}
{edges_str}

**Task:**
Compute the maximum possible value of S and provide your answer within [answer] tags. For example, if the answer is 5, write [answer]5[/answer]."""

        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_S']
    
    # Improved helper methods
    @staticmethod
    def generate_balanced_tree(n):
        """Generate more diverse tree structures using BFS-based approach"""
        if n == 1:
            return []
        nodes = list(range(1, n+1))
        random.shuffle(nodes)
        root = nodes[0]
        edges = []
        available = [root]
        used = {root}
        
        for node in nodes[1:]:
            parent = random.choice(available)
            edges.append((parent, node))
            used.add(node)
            available.append(node)
            # Randomly remove parent from available to create branching
            if random.random() < 0.3:
                available.remove(parent)
        
        random.shuffle(edges)
        return sorted(edges, key=lambda x: (x[0], x[1]))
    
    @staticmethod
    def calculate_max_S(n, edges):
        # Convert edges to adjacency list with 0-based indexing
        coupl = [[] for _ in range(n)]
        for u, v in edges:
            u0 = u - 1
            v0 = v - 1
            coupl[u0].append(v0)
            coupl[v0].append(u0)
        
        # BFS relabeling for optimal traversal
        bfs_order = [0]
        visited = [False] * n
        visited[0] = True
        for node in bfs_order:
            for nei in coupl[node]:
                if not visited[nei]:
                    visited[nei] = True
                    bfs_order.append(nei)
        
        # Rebuild adjacency with new labels
        new_coupl = [[] for _ in range(n)]
        relabel = {old: new for new, old in enumerate(bfs_order)}
        for old_node in range(n):
            new_node = relabel[old_node]
            new_coupl[new_node] = [relabel[nei] for nei in coupl[old_node]]
        
        # Dynamic programming initialization
        DP = [0] * (n * n)
        size = [1] * (n * n)
        parent_map = [-1] * (n * n)
        
        # Initialize BFS queue with all root nodes
        queue = [root * n + root for root in range(n)]
        for ind in queue:
            parent_map[ind] = ind
        
        # Build parent pointers using BFS
        for ind in queue:
            node, root = divmod(ind, n)
            for nei in new_coupl[node]:
                child_ind = nei * n + root
                if parent_map[child_ind] == -1:
                    parent_map[child_ind] = ind
                    queue.append(child_ind)
        
        # Process nodes in reverse BFS order
        for ind in reversed(queue):
            if parent_map[ind] == ind:  # Skip root nodes
                continue
            
            node, root = divmod(ind, n)
            parent_ind = parent_map[ind]
            symmetric_ind = root * n + node
            
            # Update subtree size
            size[parent_ind] += size[ind]
            
            # Update DP value using correct formula
            DP[parent_ind] = max(
                DP[parent_ind],
                max(DP[ind], DP[symmetric_ind]) + size[ind] * size[symmetric_ind]
            )
        
        return max(DP[root * n + root] for root in range(n))
