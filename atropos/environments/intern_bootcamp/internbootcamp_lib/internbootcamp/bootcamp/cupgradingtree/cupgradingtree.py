"""# 

### 谜题描述
You are given a tree with n vertices and you are allowed to perform no more than 2n transformations on it. Transformation is defined by three vertices x, y, y' and consists of deleting edge (x, y) and adding edge (x, y'). Transformation x, y, y' could be performed if all the following conditions are satisfied:

  1. There is an edge (x, y) in the current tree. 
  2. After the transformation the graph remains a tree. 
  3. After the deletion of edge (x, y) the tree would consist of two connected components. Let's denote the set of nodes in the component containing vertex x by Vx, and the set of nodes in the component containing vertex y by Vy. Then condition |Vx| > |Vy| should be satisfied, i.e. the size of the component with x should be strictly larger than the size of the component with y. 



You should minimize the sum of squared distances between all pairs of vertices in a tree, which you could get after no more than 2n transformations and output any sequence of transformations leading initial tree to such state.

Note that you don't need to minimize the number of operations. It is necessary to minimize only the sum of the squared distances.

Input

The first line of input contains integer n (1 ≤ n ≤ 2·105) — number of vertices in tree.

The next n - 1 lines of input contains integers a and b (1 ≤ a, b ≤ n, a ≠ b) — the descriptions of edges. It is guaranteed that the given edges form a tree.

Output

In the first line output integer k (0 ≤ k ≤ 2n) — the number of transformations from your example, minimizing sum of squared distances between all pairs of vertices.

In each of the next k lines output three integers x, y, y' — indices of vertices from the corresponding transformation.

Transformations with y = y' are allowed (even though they don't change tree) if transformation conditions are satisfied.

If there are several possible answers, print any of them.

Examples

Input

3
3 2
1 3


Output

0


Input

7
1 2
2 3
3 4
4 5
5 6
6 7


Output

2
4 3 2
4 5 6

Note

This is a picture for the second sample. Added edges are dark, deleted edges are dotted.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int c = 200002;
int n, f[c], rf[c], h, l, ki, ut;
vector<int> sz[c];
vector<pair<int, pair<int, int> > > sol;
bool v1[c], v2[c], cen[c];
void add(int x, int y1, int y2) { sol.push_back({x, {y1, y2}}); }
void dfs1(int a) {
  v1[a] = true, rf[a] = 1;
  int maxi = 0;
  for (int x : sz[a]) {
    if (!v1[x]) {
      dfs1(x);
      maxi = max(maxi, rf[x]);
      rf[a] += rf[x];
    }
  }
  if (rf[a] >= h && n - maxi >= h) {
    cen[a] = 1;
    v2[a] = 1;
  }
}
void dfs3(int a) {
  v2[a] = true;
  for (int x : sz[a]) {
    if (!v2[x]) {
      f[x] = a;
      dfs3(x);
      add(l, ut, x);
      add(x, a, ki);
      ut = x;
    }
  }
}
void dfs2(int a) {
  v2[a] = true;
  for (int x : sz[a]) {
    if (!v2[x]) {
      f[x] = a, l = a, ki = x, ut = x;
      dfs3(x);
      add(a, ut, x);
    }
  }
}
int main() {
  ios_base::sync_with_stdio(false);
  cin >> n;
  for (int i = 1; i < n; i++) {
    int a, b;
    cin >> a >> b;
    sz[a].push_back(b), sz[b].push_back(a);
  }
  h = (n + 1) / 2;
  dfs1(1);
  for (int i = 1; i <= n; i++) {
    if (cen[i]) {
      dfs2(i);
    }
  }
  int si = sol.size();
  cout << si << \"\n\";
  for (int i = 0; i < si; i++) {
    cout << sol[i].first << \" \" << sol[i].second.first << \" \"
         << sol[i].second.second << \"\n\";
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from collections import defaultdict, deque
import random
import re
from bootcamp import Basebootcamp

class Cupgradingtreebootcamp(Basebootcamp):
    def __init__(self, max_n=20, default_n=None):
        self.max_n = max_n
        self.default_n = default_n if default_n is not None else random.randint(1, max_n)
    
    def case_generator(self):
        n = self.default_n
        if n == 1:
            return {'n':1, 'edges':[]}
        
        parent = list(range(n+1))
        edges = []
        
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        
        nodes = list(range(1, n+1))
        random.shuffle(nodes)
        connected = [nodes[0]]
        
        for u in nodes[1:]:
            while True:
                v = random.choice(connected)
                if find(u) != find(v):
                    edges.append((u, v))
                    parent[find(u)] = find(v)
                    connected.append(u)
                    break
        
        return {'n':n, 'edges':edges}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        edges = question_case['edges']
        edges_str = '\n'.join(f"{a} {b}" for a, b in edges)
        return f"""Given a tree with {n} vertices. Perform transformations (x y y') meeting all conditions to minimize squared distances. Format your answer as:

[answer]
k
x1 y1 y1'
...
[/answer]

Input tree:
{n}
{edges_str}"""

    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        try:
            lines = [l.strip() for l in answer_blocks[-1].split('\n') if l.strip()]
            k = int(lines[0])
            steps = [tuple(map(int, line.split())) for line in lines[1:k+1]]
            if len(steps) != k:
                return None
            return steps
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 空操作验证
        if not solution:
            return cls._is_optimal_structure(identity)
        
        # 重建邻接表
        adj = defaultdict(set)
        for u, v in identity['edges']:
            adj[u].add(v)
            adj[v].add(u)
        
        # 逐步验证操作
        for x, y, yp in solution:
            # 边存在性检查
            if y not in adj[x]:
                return False
            
            # 前置分割检查
            size_x = cls._component_size(x, adj, exclude=y)
            size_y = cls._component_size(y, adj, exclude=x)
            if size_x <= size_y:
                return False
            
            # 有效性检查
            vy_nodes = cls._find_component(y, adj, exclude=x)
            if yp not in vy_nodes:
                return False
            
            # 执行变换
            adj[x].remove(y)
            adj[y].remove(x)
            adj[x].add(yp)
            adj[yp].add(x)
            
            # 环路检查
            if cls._has_cycle(adj):
                return False
        
        return cls._is_optimal_structure({'n':identity['n'], 'edges':[
            (u, v) for u in adj for v in adj[u] if u < v
        ]})
    
    @staticmethod
    def _component_size(root, adj, exclude=None):
        visited = set()
        q = deque([root])
        while q:
            u = q.popleft()
            if u == exclude:
                continue
            visited.add(u)
            for v in adj[u]:
                if v not in visited and v != exclude:
                    q.append(v)
        return len(visited)
    
    @staticmethod
    def _find_component(root, adj, exclude=None):
        visited = set()
        q = deque([root])
        while q:
            u = q.popleft()
            if u in visited or u == exclude:
                continue
            visited.add(u)
            for v in adj[u]:
                if v != exclude:
                    q.append(v)
        return visited
    
    @staticmethod
    def _has_cycle(adj):
        visited = {}
        for node in adj:
            if node not in visited:
                stack = [(node, None)]
                while stack:
                    u, parent = stack.pop()
                    if u in visited:
                        return True
                    visited[u] = True
                    for v in adj[u]:
                        if v != parent:
                            stack.append((v, u))
        return False
    
    @classmethod
    def _is_optimal_structure(cls, identity):
        n = identity['n']
        adj = defaultdict(set)
        for u, v in identity['edges']:
            adj[u].add(v)
            adj[v].add(u)
        
        centroid = cls._find_centroid(adj, n)
        return len(centroid) > 0
    
    @staticmethod
    def _find_centroid(adj, n):
        def dfs(u, parent):
            size = 1
            max_sub = 0
            for v in adj[u]:
                if v != parent:
                    s = dfs(v, u)
                    size += s
                    max_sub = max(max_sub, s)
            max_sub = max(max_sub, n - size)
            if max_sub <= n // 2:
                centroids.append(u)
            return size
        
        centroids = []
        if n >= 1:
            dfs(next(iter(adj.keys())), None)
        return centroids
