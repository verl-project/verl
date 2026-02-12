"""# 

### 谜题描述
Emuskald considers himself a master of flow algorithms. Now he has completed his most ingenious program yet — it calculates the maximum flow in an undirected graph. The graph consists of n vertices and m edges. Vertices are numbered from 1 to n. Vertices 1 and n being the source and the sink respectively.

However, his max-flow algorithm seems to have a little flaw — it only finds the flow volume for each edge, but not its direction. Help him find for each edge the direction of the flow through this edges. Note, that the resulting flow should be correct maximum flow.

More formally. You are given an undirected graph. For each it's undirected edge (ai, bi) you are given the flow volume ci. You should direct all edges in such way that the following conditions hold:

  1. for each vertex v (1 < v < n), sum of ci of incoming edges is equal to the sum of ci of outcoming edges; 
  2. vertex with number 1 has no incoming edges; 
  3. the obtained directed graph does not have cycles. 

Input

The first line of input contains two space-separated integers n and m (2 ≤ n ≤ 2·105, n - 1 ≤ m ≤ 2·105), the number of vertices and edges in the graph. The following m lines contain three space-separated integers ai, bi and ci (1 ≤ ai, bi ≤ n, ai ≠ bi, 1 ≤ ci ≤ 104), which means that there is an undirected edge from ai to bi with flow volume ci.

It is guaranteed that there are no two edges connecting the same vertices; the given graph is connected; a solution always exists.

Output

Output m lines, each containing one integer di, which should be 0 if the direction of the i-th edge is ai → bi (the flow goes from vertex ai to vertex bi) and should be 1 otherwise. The edges are numbered from 1 to m in the order they are given in the input.

If there are several solutions you can print any of them.

Examples

Input

3 3
3 2 10
1 2 10
3 1 5


Output

1
0
1


Input

4 5
1 2 10
1 3 10
2 3 5
4 2 15
3 4 5


Output

0
0
1
1
0

Note

In the first test case, 10 flow units pass through path <image>, and 5 flow units pass directly from source to sink: <image>.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int maxm = 200000 + 5;
const int maxn = 200000 + 5;
struct Edge {
  int from, to, w, id;
};
vector<Edge> edges;
vector<int> G[maxn];
int ans[maxm];
int n, m;
int win[maxn], wall[maxn];
inline void solve() {
  memset(ans, -1, sizeof ans);
  queue<int> Q;
  Q.push(1);
  while (!Q.empty()) {
    int now = Q.front();
    Q.pop();
    vector<int> tocheck;
    for (const int& i : G[now]) {
      Edge& e = edges[i];
      if (ans[e.id] != -1) continue;
      int to;
      if (now == e.from)
        to = e.to, ans[e.id] = 0;
      else
        to = e.from, ans[e.id] = 1;
      win[to] += e.w;
      wall[to] -= e.w;
      if (to != n) tocheck.push_back(to);
    }
    for (const int& nx : tocheck)
      if (win[nx] == wall[nx]) Q.push(nx);
  }
}
int main() {
  scanf(\"%d%d\", &n, &m);
  for (int i = 0, a, b, c; i < m; i++) {
    scanf(\"%d%d%d\", &a, &b, &c);
    edges.push_back((Edge){a, b, c, i});
    G[a].push_back((int)edges.size() - 1);
    G[b].push_back((int)edges.size() - 1);
    wall[a] += c;
    wall[b] += c;
  }
  solve();
  for (int i = 0; i < m; i++) printf(\"%d\n\", ans[i]);
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import defaultdict, deque
from bootcamp import Basebootcamp

class Edge:
    def __init__(self, from_, to_, w_, id_):
        self.from_ = from_
        self.to_ = to_
        self.w_ = w_
        self.id_ = id_

class Cflawedflowbootcamp(Basebootcamp):
    def __init__(self, max_n=8, max_m=15):
        self.max_n = max_n
        self.max_m = max_m
    
    def case_generator(self):
        n = random.randint(2, self.max_n)
        max_edges = n * (n - 1) // 2
        m = random.randint(n-1, min(max_edges, self.max_m))
        edges = self._generate_connected_edges(n, m)
        solution = self._generate_solution(n, edges)
        return {
            "n": n,
            "m": m,
            "edges": edges,
            "solution": solution
        }
    
    @staticmethod
    def _generate_connected_edges(n, m):
        parent = list(range(n+1))
        
        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]
        
        edges = []
        existing = set()
        
        # Generate spanning tree to ensure connectivity
        nodes = list(range(1, n+1))
        random.shuffle(nodes)
        root = nodes[0]
        for node in nodes[1:]:
            a, b = root, node
            if a > b:
                a, b = b, a
            c = random.randint(1, 10000)
            edges.append((a, b, c))
            existing.add((a, b))
            parent[b] = a
        
        # Add remaining edges
        remaining = m - (n-1)
        candidates = [(i, j) for i in range(1, n+1) for j in range(i+1, n+1) if (i, j) not in existing]
        while remaining > 0 and candidates:
            add_num = min(remaining, len(candidates))
            selected = random.sample(candidates, add_num)
            for a, b in selected:
                c = random.randint(1, 10000)
                edges.append((a, b, c))
                existing.add((a, b))
                candidates.remove((a, b))  # Prevent duplicate selection
            remaining -= add_num
        
        random.shuffle(edges)
        return edges[:m]
    
    @staticmethod
    def _generate_solution(n, edges):
        m = len(edges)
        graph = [[] for _ in range(n+1)]
        wall = [0]*(n+1)
        for idx, (a, b, c) in enumerate(edges):
            edge = Edge(a, b, c, idx)
            graph[a].append(edge)
            graph[b].append(edge)
            wall[a] += c
            wall[b] += c
        
        ans = [-1]*m
        win = [0]*(n+1)
        q = deque([1])
        
        while q:
            u = q.popleft()
            to_check = []
            for edge in graph[u]:
                if ans[edge.id_] != -1:
                    continue
                if edge.from_ == u:
                    v = edge.to_
                    ans[edge.id_] = 0
                else:
                    v = edge.from_
                    ans[edge.id_] = 1
                win[v] += edge.w_
                wall[v] -= edge.w_
                if v != n:
                    to_check.append(v)
            
            for v in to_check:
                if win[v] == wall[v]:
                    q.append(v)
        
        return ans
    
    @staticmethod
    def prompt_func(question_case):
        edges = question_case["edges"]
        n = question_case["n"]
        m = question_case["m"]
        edge_lines = "\n".join([f"{a} {b} {c}" for a, b, c in edges])
        return f"""作为流量算法专家，请确定无向图中各边方向，并遵守以下规则：
1. 源点（1号顶点）没有入边
2. 中间节点（非源点和汇点）流入等于流出
3. 最终图必须无环

输入格式：
{n} {m}
{edge_lines}

请输出{m}个0或1（对应每条边的方向），答案用[answer]标签包裹，如：
[answer]
0 1 1 0
[/answer]"""
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        digits = re.findall(r'\b[01]\b', matches[-1])
        return [int(d) for d in digits] if digits and len(digits) == len(matches[-1].split()) else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if len(solution) != identity["m"]:
            return False
        
        edges = identity["edges"]
        n = identity["n"]
        
        # Check source has no incoming edges
        for (a, b, _), d in zip(edges, solution):
            if (d == 0 and b == 1) or (d == 1 and a == 1):
                return False
        
        # Flow conservation check
        inflow = defaultdict(int)
        outflow = defaultdict(int)
        for (a, b, c), d in zip(edges, solution):
            if d == 0:
                outflow[a] += c
                inflow[b] += c
            else:
                outflow[b] += c
                inflow[a] += c
        
        for v in range(2, n):
            if inflow.get(v, 0) != outflow.get(v, 0):
                return False
        
        # Acyclicity check with topological sort
        adj = [[] for _ in range(n+1)]
        in_degree = [0]*(n+1)
        for (a, b, _), d in zip(edges, solution):
            u, v = (a, b) if d == 0 else (b, a)
            adj[u].append(v)
            in_degree[v] += 1
        
        q = deque([u for u in range(1, n+1) if in_degree[u] == 0])
        visited = 0
        while q:
            u = q.popleft()
            visited += 1
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    q.append(v)
        
        return visited == n  # All nodes must be visited for acyclic graph
