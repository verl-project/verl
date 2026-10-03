"""# 

### 谜题描述
Twilight Sparkle learnt that the evil Nightmare Moon would return during the upcoming Summer Sun Celebration after one thousand years of imprisonment on the moon. She tried to warn her mentor Princess Celestia, but the princess ignored her and sent her to Ponyville to check on the preparations for the celebration.

<image>

Twilight Sparkle wanted to track the path of Nightmare Moon. Unfortunately, she didn't know the exact path. What she knew is the parity of the number of times that each place Nightmare Moon visited. Can you help Twilight Sparkle to restore any path that is consistent with this information?

Ponyville can be represented as an undirected graph (vertices are places, edges are roads between places) without self-loops and multi-edges. The path can start and end at any place (also it can be empty). Each place can be visited multiple times. The path must not visit more than 4n places.

Input

The first line contains two integers n and m (2 ≤ n ≤ 105; 0 ≤ m ≤ 105) — the number of places and the number of roads in Ponyville. Each of the following m lines contains two integers ui, vi (1 ≤ ui, vi ≤ n; ui ≠ vi), these integers describe a road between places ui and vi.

The next line contains n integers: x1, x2, ..., xn (0 ≤ xi ≤ 1) — the parity of the number of times that each place must be visited. If xi = 0, then the i-th place must be visited even number of times, else it must be visited odd number of times.

Output

Output the number of visited places k in the first line (0 ≤ k ≤ 4n). Then output k integers — the numbers of places in the order of path. If xi = 0, then the i-th place must appear in the path even number of times, else i-th place must appear in the path odd number of times. Note, that given road system has no self-loops, therefore any two neighbouring places in the path must be distinct.

If there is no required path, output -1. If there multiple possible paths, you can output any of them.

Examples

Input

3 2
1 2
2 3
1 1 1


Output

3
1 2 3


Input

5 7
1 2
1 3
1 4
1 5
3 4
3 5
4 5
0 1 0 1 0


Output

10
2 1 3 4 5 4 5 4 3 1 

Input

2 0
0 0


Output

0

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def satisfiable():
	colored = [-1]*(n)
	color = 1
	odd_components = 0
	
	for node in range(n):
		if colored[node] == -1:
			mark_component(node, color, colored)			
			for node_aux in range(n):
				if colored[node_aux]==color and parity[node_aux]==1:
					odd_components += 1
					break
			color+=1
	if odd_components > 1:
		return False
	else:
		return True
	

def mark_component(node, color, colored):
	colored[node] = color
	for neighbor in graph[node]:
		if colored[neighbor] != color:
			mark_component(neighbor, color, colored)

n, m = map(int, raw_input().split())

graph = [[] for i in range(n)]

for i in range(m):
	path = map(int, raw_input().split())
	graph[path[0] - 1].append(path[1] - 1)
	graph[path[1] - 1].append(path[0] - 1)

parity = map(int, raw_input().split())

def root():
	for i in range(n):
		if parity[i] == 1:
			return i
	return 0

def wrong_parity(node):
	return ((parity[node]%2==0 and new_parity[node]%2==1) or (parity[node]%2==1 and new_parity[node]%2==0))

def path(node, parent):
	visited[node] = True
	new_parity[node] += 1
	ans_path.append(node + 1)
	
	for neighbor in graph[node]:
		if not visited[neighbor]:
			path(neighbor, node)
			ans_path.append(node + 1)
			new_parity[node] += 1
	
	if wrong_parity(node) and parent != -1:
		ans_path.append(parent + 1)
		new_parity[parent] += 1
		ans_path.append(node + 1)
		new_parity[node] += 1 

if satisfiable():
	visited = [False] * n
	new_parity = [0] * n
	
	ans_path = []
	
	root = root()
	
	path(root, -1)
	
	if wrong_parity(root):
		ans_path.pop()
	
	print len(ans_path)
	if len(ans_path)>0:
		print ' '.join(map(str,ans_path))
else:
	print -1
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import deque
from bootcamp import Basebootcamp

class Elittleponyandsummersuncelebrationbootcamp(Basebootcamp):
    def __init__(self, n_min=2, n_max=5, solvable_prob=0.5, max_edges=10):
        self.n_min = n_min
        self.n_max = n_max
        self.solvable_prob = solvable_prob
        self.max_edges = max_edges

    def build_connected_graph(self, n):
        parent = list(range(n))
        edges = []
        
        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]
        
        for _ in range(n*2):  # 确保连通
            u, v = random.sample(range(n), 2)
            if find(u) != find(v):
                parent[find(v)] = find(u)
                edges.append((u, v))
        
        # 添加冗余边
        candidates = [(i, j) for i in range(n) for j in range(i+1, n) if find(i) != find(j)]
        for _ in range(min(len(candidates), self.max_edges - len(edges))):
            u, v = candidates.pop(random.randint(0, len(candidates)-1))
            edges.append((u, v))
            parent[find(v)] = find(u)
        
        return edges

    def case_generator(self):
        solvable = random.random() < self.solvable_prob
        
        if solvable:
            n = random.randint(self.n_min, self.n_max)
            edges = self.build_connected_graph(n)
            graph = [[] for _ in range(n)]
            for u, v in edges:
                graph[u].append(v)
                graph[v].append(u)
            
            parity = [0]*n
            components = self.get_components(graph)
            
            # 确保最多一个组件有奇数结点
            odd_components = 0
            for comp in components:
                has_odd = False
                for node in comp:
                    if random.random() < 0.5:
                        parity[node] = 1
                        has_odd = True
                if not has_odd and random.random() < 0.5:
                    parity[comp[0]] = 1
                    has_odd = True
                if has_odd:
                    odd_components += 1
            
            # 修正至多一个奇数组件
            if odd_components > 1:
                return self.case_generator()
            
            return {
                'n': n,
                'm': len(edges),
                'edges': [(u+1, v+1) for u, v in edges],
                'parity': parity,
                'graph': graph,
                'solvable': True
            }
        else:
            n = random.randint(max(self.n_min, 3), self.n_max)
            # 生成两个不连通组件
            split = random.randint(1, n-1)
            group1 = list(range(split))
            group2 = list(range(split, n))
            
            # 构建各自子图
            edges1 = self.build_connected_graph(len(group1)) if len(group1) > 1 else []
            edges2 = self.build_connected_graph(len(group2)) if len(group2) > 1 else []
            edges = [(group1[u], group1[v]) for u, v in edges1] + [(group2[u], group2[v]) for u, v in edges2]
            
            # 设置奇偶性
            parity = [0]*n
            if group1:
                parity[random.choice(group1)] = 1
            if group2:
                parity[random.choice(group2)] = 1
            
            # 验证不可解条件
            graph = [[] for _ in range(n)]
            for u, v in edges:
                graph[u].append(v)
                graph[v].append(u)
            components = self.get_components(graph)
            odd_comps = sum(1 for comp in components if any(parity[n] for n in comp))
            if odd_comps < 2:
                return self.case_generator()
            
            return {
                'n': n,
                'm': len(edges),
                'edges': [(u+1, v+1) for u, v in edges],
                'parity': parity,
                'graph': graph,
                'solvable': False
            }

    def get_components(self, graph):
        visited = [False] * len(graph)
        components = []
        for i in range(len(graph)):
            if not visited[i]:
                q = deque([i])
                visited[i] = True
                comp = []
                while q:
                    node = q.popleft()
                    comp.append(node)
                    for neighbor in graph[node]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            q.append(neighbor)
                components.append(comp)
        return components

    @staticmethod
    def prompt_func(question_case) -> str:
        input_data = f"{question_case['n']} {question_case['m']}\n"
        input_data += "\n".join(f"{u} {v}" for u, v in question_case['edges'])
        input_data += f"\n{' '.join(map(str, question_case['parity']))}"
        
        return f"""
根据Ponyville的地图，找到满足奇偶访问次数的路径。地图结构：
{input_data}

要求：
1. 路径长度 ≤ 4*{question_case['n']}
2. 相邻地点必须直接相连
3. 输出格式必须严格符合规范

将最终答案放在[answer]标签内，例如：
[answer]
3
1 2 3
[/answer]
或
[answer]
-1
[/answer]
"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        answer = matches[-1].strip()
        if answer == '-1':
            return -1
        if '\n' not in answer:
            return None
        k_line, *path_lines = answer.split('\n')
        try:
            k = int(k_line.strip())
            path = []
            for line in path_lines:
                path.extend(map(int, line.strip().split()))
            if len(path) != k:
                return None
            return {'k': k, 'path': path}
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution == -1:
            return not identity['solvable']
        
        if not identity['solvable']:
            return False
        
        n = identity['n']
        path = solution.get('path', [])
        k = solution.get('k', 0)
        
        # 基本长度校验
        if k != len(path) or k > 4*n:
            return False
        
        # 奇偶校验
        counts = [0] * n
        for node in path:
            if not 1 <= node <= n:
                return False
            counts[node-1] += 1
        for i in range(n):
            if counts[i] % 2 != identity['parity'][i]:
                return False
        
        # 路径连续性校验
        if k > 0:
            graph = identity['graph']
            current = path[0] - 1
            for next_node in path[1:]:
                next_idx = next_node - 1
                if next_idx not in graph[current]:
                    return False
                current = next_idx
        return True
