"""# 

### 谜题描述
PolandBall lives in a forest with his family. There are some trees in the forest. Trees are undirected acyclic graphs with k vertices and k - 1 edges, where k is some integer. Note that one vertex is a valid tree.

There is exactly one relative living in each vertex of each tree, they have unique ids from 1 to n. For each Ball i we know the id of its most distant relative living on the same tree. If there are several such vertices, we only know the value of the one with smallest id among those.

How many trees are there in the forest?

Input

The first line contains single integer n (1 ≤ n ≤ 104) — the number of Balls living in the forest.

The second line contains a sequence p1, p2, ..., pn of length n, where (1 ≤ pi ≤ n) holds and pi denotes the most distant from Ball i relative living on the same tree. If there are several most distant relatives living on the same tree, pi is the id of one with the smallest id.

It's guaranteed that the sequence p corresponds to some valid forest.

Hacking: To hack someone, you should provide a correct forest as a test. The sequence p will be calculated according to the forest and given to the solution you try to hack as input. Use the following format:

In the first line, output the integer n (1 ≤ n ≤ 104) — the number of Balls and the integer m (0 ≤ m < n) — the total number of edges in the forest. Then m lines should follow. The i-th of them should contain two integers ai and bi and represent an edge between vertices in which relatives ai and bi live. For example, the first sample is written as follows:
    
    
      
    5 3  
    1 2  
    3 4  
    4 5  
    

Output

You should output the number of trees in the forest where PolandBall lives.

Interaction

From the technical side, this problem is interactive. However, it should not affect you (except hacking) since there is no interaction.

Examples

Input

5
2 1 5 3 3

Output

2

Input

1
1


Output

1

Note

In the first sample testcase, possible forest is: 1-2 3-4-5. 

There are 2 trees overall.

In the second sample testcase, the only possible graph is one vertex and no edges. Therefore, there is only one tree.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import re
import sys
import time

def read(t=None):
	string = raw_input()
	return string if t is None else [t(x) for x in string.split()]

def solve():
	n = int(read())
	a = read(int)
	a = [x-1 for x in a]

	nsingles = 0
	ndoubles = 0
	for i, x in enumerate(a):
		if a[x] == i and a[i] == x:
			if i == x:
				nsingles += 1
			else:
				ndoubles += 1
	print nsingles+ndoubles/2

def solve2():
	n = int(read())
	a = read(int)
	a = [x-1 for x in a]

	nsingles = 0
	for i, x in enumerate(a):
		if x == i:
			nsingles += 1

	#print \"nsingles: %d\"%nsingles
	ndoubles = len(set(a)) - nsingles
	print nsingles+ndoubles/2

if __name__ == \"__main__\":
	solve2()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from collections import deque
from bootcamp import Basebootcamp

class Cpolandballandforestbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=20):
        self.min_n = min_n
        self.max_n = max_n
    
    def generate_valid_forest(self, n):
        """生成满足条件的森林结构"""
        nodes = list(range(1, n+1))
        random.shuffle(nodes)
        forest = []
        
        while nodes:
            tree_size = random.randint(1, min(5, len(nodes)))
            current_tree = nodes[:tree_size]
            del nodes[:tree_size]
            
            # 生成树结构
            if len(current_tree) == 1:
                forest.append((current_tree[0], []))
            else:
                edges = []
                available = [current_tree[0]]
                for node in current_tree[1:]:
                    parent = random.choice(available)
                    edges.append((parent, node))
                    available.append(node)
                forest.append((current_tree, edges))
        return forest
    
    def compute_diameter(self, tree):
        """精确计算树直径端点"""
        root, edges = tree
        adj = {}
        for u, v in edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
        
        def bfs(start):
            visited = {start: 0}
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj.get(u, []):
                    if v not in visited:
                        visited[v] = visited[u] + 1
                        q.append(v)
            return visited
        
        # 第一次BFS找到最远点a
        init_visit = bfs(root[0] if isinstance(root, list) else root)
        a = min([k for k, v in init_visit.items() if v == max(init_visit.values())])
        
        # 第二次BFS找到直径端点b
        final_visit = bfs(a)
        b = min([k for k, v in final_visit.items() if v == max(final_visit.values())])
        
        return (a, b) if a < b else (b, a)
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        forest = self.generate_valid_forest(n)
        
        p = [0] * n
        for tree in forest:
            node, edges = tree
            if isinstance(node, list):  # 多节点树
                a, b = self.compute_diameter(tree)
                for u in node:
                    dist_a = self.calculate_distance(u, a, edges)
                    dist_b = self.calculate_distance(u, b, edges)
                    p[u-1] = a if dist_a > dist_b else b if dist_b > dist_a else min(a, b)
            else:  # 单节点树
                p[node-1] = node
        return {'n': n, 'p': p}
    
    def calculate_distance(self, start, end, edges):
        """精确计算两节点间距"""
        if start == end:
            return 0
        adj = {}
        for u, v in edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
        
        visited = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj.get(u, []):
                if v == end:
                    return visited[u] + 1
                if v not in visited:
                    visited[v] = visited[u] + 1
                    q.append(v)
        return float('inf')

    @staticmethod
    def prompt_func(question_case) -> str:
        p_str = ' '.join(map(str, question_case['p']))
        return f"""给定波兰球家族森林：
- 总人数 n = {question_case['n']}
- 距离参数 p = [{p_str}]

请计算森林中的树木数量，答案用[answer]标签包裹。"""

    @staticmethod
    def extract_output(output):
        answers = re.findall(r'\[answer\](\d+)\[\/answer\]', output, flags=re.DOTALL)
        return int(answers[-1]) if answers else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        a = [x-1 for x in identity['p']]
        unique = len(set(a))
        singles = sum(1 for i, v in enumerate(a) if i == v)
        return solution == singles + (unique - singles) // 2
