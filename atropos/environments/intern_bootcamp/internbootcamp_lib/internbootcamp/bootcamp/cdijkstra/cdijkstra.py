"""# 

### 谜题描述
You are given a weighted undirected graph. The vertices are enumerated from 1 to n. Your task is to find the shortest path between the vertex 1 and the vertex n.

Input

The first line contains two integers n and m (2 ≤ n ≤ 105, 0 ≤ m ≤ 105), where n is the number of vertices and m is the number of edges. Following m lines contain one edge each in form ai, bi and wi (1 ≤ ai, bi ≤ n, 1 ≤ wi ≤ 106), where ai, bi are edge endpoints and wi is the length of the edge.

It is possible that the graph has loops and multiple edges between pair of vertices.

Output

Write the only integer -1 in case of no path. Write the shortest path in opposite case. If there are many solutions, print any of them.

Examples

Input

5 6
1 2 2
2 5 5
2 3 4
1 4 1
4 3 3
3 5 1


Output

1 4 3 5 

Input

5 6
1 2 2
2 5 5
2 3 4
1 4 1
4 3 3
3 5 1


Output

1 4 3 5 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import print_function
import heapq as hq

Inf = float(\"inf\")

def dijkstra(n, adj, src):
    d = [Inf] * (n+1)
    d[src] = 0
    q = []
    parent = [None] *(n+1)
    hq.heappush(q, (d[src], src))
    while q:
        _, u = hq.heappop(q)
        for v, w in adj[u]:
            if d[v] > d[u] + w:
                d[v] = d[u] + w
                hq.heappush(q, (d[v], v))
                parent[v]= u

    return parent

def path_print(v,parent):
	parent_path = []
	while v is not None:
		parent_path.append(v)
		v = parent[v]
	l = len(parent_path)
	for i in range(l):
		print(parent_path[l-i-1], end = \" \")


n, m = map(int, raw_input().split())
adj = {}
for i in range(1, n+1):
	adj[i] = []
for _ in range(m):
	u, v, w = map(int, raw_input().split())
	adj[u].append([v, w])
	adj[v].append([u, w])
parent = dijkstra(n, adj, 1)
if parent[n] == None:
	print(\"-1\")
else:
    path_print(n,parent)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp  # 确保导入基类
from collections import defaultdict
import heapq
import random
import re

class Cdijkstrabootcamp(Basebootcamp):  # 添加基类继承
    def __init__(self, n_min=5, n_max=15, edge_density=0.3, max_weight=100, ensure_path=True):
        super().__init__()  # 初始化基类
        self.n_min = n_min
        self.n_max = n_max
        self.edge_density = edge_density
        self.max_weight = max_weight
        self.ensure_path = ensure_path
    
    def case_generator(self):
        # 生成节点数时至少包含2个节点
        n = max(2, random.randint(self.n_min, self.n_max))
        max_possible_edges = n * (n - 1) // 2
        m = int(max_possible_edges * self.edge_density)
        
        # 生成基础边集
        edges = []
        node_list = list(range(1, n+1))
        random.shuffle(node_list)
        
        # 生成保证连通性的最小边集
        for i in range(1, len(node_list)):
            a, b = node_list[i-1], node_list[i]
            w = random.randint(1, self.max_weight)
            edges.append((a, b, w))
        
        # 添加随机边
        existing_edges = set()
        for _ in range(m - (n-1)):
            while True:
                a = random.choice(node_list)
                b = random.choice(node_list)
                if a != b and (a, b) not in existing_edges:
                    existing_edges.add((a, b))
                    existing_edges.add((b, a))
                    break
            w = random.randint(1, self.max_weight)
            edges.append((a, b, w))
        
        # 构建邻接表
        adj = defaultdict(list)
        for a, b, w in edges:
            adj[a].append((b, w))
            adj[b].append((a, w))
        
        # Dijkstra算法实现
        dist = {i: float('inf') for i in node_list}
        prev = {i: None for i in node_list}
        dist[1] = 0
        heap = [(0, 1)]
        
        while heap:
            current_dist, u = heapq.heappop(heap)
            if current_dist > dist[u]:
                continue
            for v, w in adj[u]:
                if dist[v] > current_dist + w:
                    dist[v] = current_dist + w
                    prev[v] = u
                    heapq.heappush(heap, (dist[v], v))
        
        # 处理无路径情况
        has_path = dist[n] != float('inf')
        if self.ensure_path and not has_path:
            # 添加直达路径确保连通
            w = random.randint(1, self.max_weight)
            edges.append((1, n, w))
            adj[1].append((n, w))
            adj[n].append((1, w))
            dist[n] = w
            has_path = True
        
        # 转换为可序列化格式
        return {
            'n': n,
            'edges': edges,
            'expected_distance': float(dist[n]) if has_path else -1,
            'has_path': has_path
        }
    
    @staticmethod
    def prompt_func(case):
        input_lines = [f"{case['n']} {len(case['edges'])}"]
        input_lines += [f"{a} {b} {w}" for a, b, w in case['edges']]
        
        return f"""Solve the shortest path problem in an undirected weighted graph. Find the minimal path from node 1 to node {case['n']}.

Input Format:
First line: n m (number of nodes, edges)
Next m lines: a b w (edges with weights)

Output Format:
- If path exists: space-separated node sequence
- If no path: -1

Example:
Input:
5 6
1 2 2
2 5 5
2 3 4
1 4 1
4 3 3
3 5 1

Output:
[answer]1 4 3 5[/answer]

Your Task:
{" ".join(input_lines)}
Put your final answer within [answer] tags like: [answer]your path here[/answer]"""

    @staticmethod
    def extract_output(text):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', text, re.IGNORECASE | re.DOTALL)
        if not matches:
            return None
        # 取最后一个有效答案并标准化输出
        last_answer = matches[-1].strip()
        # 清理多余空格和换行符
        return ' '.join(last_answer.split()) if last_answer != '-1' else '-1'

    @classmethod
    def _verify_correction(cls, solution, case):
        # 处理无解情况
        if solution == "-1":
            return not case['has_path']
        
        # 解析路径
        try:
            path = list(map(int, solution.split()))
        except ValueError:
            return False
        
        # 验证端点
        if path[0] != 1 or path[-1] != case['n']:
            return False
        
        # 构建最小权重邻接字典
        min_weights = defaultdict(dict)
        for a, b, w in case['edges']:
            if b not in min_weights[a] or w < min_weights[a][b]:
                min_weights[a][b] = w
            if a not in min_weights[b] or w < min_weights[b][a]:
                min_weights[b][a] = w
        
        # 计算路径总权重
        total = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if v not in min_weights.get(u, {}):
                return False
            total += min_weights[u][v]
        
        # 浮点数精度处理
        return abs(total - case['expected_distance']) < 1e-9
