"""# 

### 谜题描述
A tree of size n is an undirected connected graph consisting of n vertices without cycles.

Consider some tree with n vertices. We call a tree invariant relative to permutation p = p1p2... pn, if for any two vertices of the tree u and v the condition holds: \"vertices u and v are connected by an edge if and only if vertices pu and pv are connected by an edge\".

You are given permutation p of size n. Find some tree size n, invariant relative to the given permutation.

Input

The first line contains number n (1 ≤ n ≤ 105) — the size of the permutation (also equal to the size of the sought tree).

The second line contains permutation pi (1 ≤ pi ≤ n).

Output

If the sought tree does not exist, print \"NO\" (without the quotes).

Otherwise, print \"YES\", and then print n - 1 lines, each of which contains two integers — the numbers of vertices connected by an edge of the tree you found. The vertices are numbered from 1, the order of the edges and the order of the vertices within the edges does not matter.

If there are multiple solutions, output any of them.

Examples

Input

4
4 3 2 1


Output

YES
4 1
4 2
1 3


Input

3
3 1 2


Output

NO

Note

In the first sample test a permutation transforms edge (4, 1) into edge (1, 4), edge (4, 2) into edge (1, 3) and edge (1, 3) into edge (4, 2). These edges all appear in the resulting tree.

It can be shown that in the second sample test no tree satisfies the given condition.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from collections import defaultdict

n = int(raw_input())
p = [int(v) - 1 for v in raw_input().split()]

was = [False] * n
cyc = defaultdict(list)

for i in xrange(n):
	if was[i]:
		continue

	was[i] = True
	c = [i]
	cur = p[i]
	while cur != i:
		was[cur] = True
		c.append(cur)
		cur = p[cur]

	cyc[len(c)].append(c)

lengths = cyc.keys()
lengths.sort()

roots = []
parents = {}

for i in xrange(len(lengths) - 1, -1, -1):
	for j in xrange(i - 1, -1, -1):
		if lengths[i] % lengths[j] == 0:
			cyc[lengths[i]][0].append(lengths[j])
			break
	else:
		cyc[lengths[i]][0].append(None)
		roots.append(lengths[i])

if len(roots) > 1 or roots[0] > 2:
	print(\"NO\")
	exit()

print(\"YES\")

ans = []
if roots[0] == 2:
	ans.append((cyc[2][0][0], cyc[2][0][1]))
	for k in xrange(1, len(cyc[2])):
		ans.append((cyc[2][0][0], cyc[2][k][0]))
		ans.append((cyc[2][0][1], cyc[2][k][1]))
else:
	for k in xrange(1, len(cyc[1])):
		ans.append((cyc[1][0][0], cyc[1][k][0]))

for l in lengths:
	if l == roots[0]:
		continue
	parent = cyc[l][0].pop()
	for cc in cyc[l]:
		for i in xrange(len(cc)):
			ans.append((cyc[parent][0][i % parent], cc[i]))

print('\n'.join('%d %d' % (p[0] + 1, p[1] + 1) for p in ans))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import defaultdict
from bootcamp import Basebootcamp
import re

def check_permutation_solution(n, p_list_1based):
    if n == 0:
        return (False, [])
    p_list = [x - 1 for x in p_list_1based]  # Convert to 0-based
    was = [False] * n
    cyc = defaultdict(list)

    # Find all cycles
    for i in range(n):
        if was[i]:
            continue
        cycle = []
        j = i
        while not was[j]:
            was[j] = True
            cycle.append(j)
            j = p_list[j]
        cyc[len(cycle)].append(cycle)
    
    lengths = sorted(cyc.keys(), reverse=True)
    parent = {}
    roots = []
    
    # Determine parents for each cycle length
    for l in lengths:
        found = False
        for m in lengths:
            if m < l and l % m == 0:
                parent[l] = m
                found = True
                break
        if not found:
            parent[l] = None
            roots.append(l)
    
    # Check validity of roots
    if len(roots) > 1 or (len(roots) == 1 and roots[0] > 2):
        return (False, None)
    
    # Construct the tree edges
    edges = []
    if roots:
        root_len = roots[0]
    else:
        return (False, None)
    
    # Handle root cycle(s)
    if root_len == 2:
        root_cycle = cyc[2][0]
        edges.append((root_cycle[0], root_cycle[1]))
        for cycle in cyc[2][1:]:
            edges.append((root_cycle[0], cycle[0]))
            edges.append((root_cycle[1], cycle[1]))
    elif root_len == 1 and 1 in cyc:
        main_node = cyc[1][0][0]
        for cycle in cyc[1][1:]:
            edges.append((main_node, cycle[0]))
    
    # Attach other cycles to their parents
    for l in lengths:
        if l == root_len:
            continue
        if l not in parent:
            continue
        parent_len = parent[l]
        if parent_len is None:
            continue
        parent_cycles = cyc[parent_len]
        for cycle in cyc[l]:
            for i in range(len(cycle)):
                parent_node = parent_cycles[0][i % parent_len]
                edges.append((parent_node, cycle[i]))
    
    # Convert edges back to 1-based
    edges_1based = [(u + 1, v + 1) for u, v in edges]
    return (True, edges_1based)

class Binvarianceoftreebootcamp(Basebootcamp):
    def __init__(self, max_n=10):
        self.max_n = max_n  # Control the size for case generation
    
    def case_generator(self):
        n = random.randint(1, self.max_n)
        p = list(range(1, n+1))
        random.shuffle(p)
        exists, edges = check_permutation_solution(n, p)
        case = {
            "n": n,
            "p": p,
            "exists": exists
        }
        if exists:
            case["edges"] = edges
        return case
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case["n"]
        p = question_case["p"]
        p_str = ' '.join(map(str, p))
        problem = (
            "You are given a permutation of size n. Your task is to determine if there exists a tree of size n that is invariant under this permutation. If it exists, output YES followed by the edges of the tree; otherwise, output NO.\n\n"
            f"Input:\n{n}\n{p_str}\n\n"
            "Output your answer as follows:\n"
            "- If no such tree exists, output: NO\n"
            "- If it exists, output: YES followed by n-1 edges, each on a new line.\n"
            "Enclose your final answer within [answer] and [/answer] tags."
        )
        return problem
    
    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        last_answer = answer_blocks[-1].strip()
        return last_answer
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        exists = identity['exists']
        lines = solution.strip().split('\n')
        if not lines:
            return False
        
        first_line = lines[0].strip().upper()
        if exists and first_line != 'YES':
            return False
        if not exists and first_line != 'NO':
            return False
        if not exists:
            return True
        
        n = identity['n']
        p = identity['p']
        edges = []
        edge_lines = lines[1:] if len(lines) > 1 else []
        if len(edge_lines) != n - 1:
            return False
        
        for line in edge_lines:
            parts = line.strip().split()
            if len(parts) != 2:
                return False
            try:
                u = int(parts[0])
                v = int(parts[1])
            except ValueError:
                return False
            edges.append((u, v))
        
        # Check tree validity
        parent = list(range(n + 1))
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        
        for u, v in edges:
            if u < 1 or u > n or v < 1 or v > n:
                return False
            pu, pv = find(u), find(v)
            if pu == pv:
                return False
            parent[pv] = pu
        
        root = find(1)
        for node in range(2, n + 1):
            if find(node) != root:
                return False
        
        # Check permutation invariance
        original_edges = set(frozenset((u, v)) for u, v in edges)
        permuted_edges = set()
        for u, v in edges:
            pu = p[u - 1]
            pv = p[v - 1]
            permuted_edges.add(frozenset((pu, pv)))
        
        return original_edges == permuted_edges
