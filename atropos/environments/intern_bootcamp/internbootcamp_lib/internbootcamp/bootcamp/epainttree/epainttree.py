"""# 

### 谜题描述
You are given a tree with n vertexes and n points on a plane, no three points lie on one straight line.

Your task is to paint the given tree on a plane, using the given points as vertexes. 

That is, you should correspond each vertex of the tree to exactly one point and each point should correspond to a vertex. If two vertexes of the tree are connected by an edge, then the corresponding points should have a segment painted between them. The segments that correspond to non-adjacent edges, should not have common points. The segments that correspond to adjacent edges should have exactly one common point.

Input

The first line contains an integer n (1 ≤ n ≤ 1500) — the number of vertexes on a tree (as well as the number of chosen points on the plane).

Each of the next n - 1 lines contains two space-separated integers ui and vi (1 ≤ ui, vi ≤ n, ui ≠ vi) — the numbers of tree vertexes connected by the i-th edge.

Each of the next n lines contain two space-separated integers xi and yi ( - 109 ≤ xi, yi ≤ 109) — the coordinates of the i-th point on the plane. No three points lie on one straight line.

It is guaranteed that under given constraints problem has a solution.

Output

Print n distinct space-separated integers from 1 to n: the i-th number must equal the number of the vertex to place at the i-th point (the points are numbered in the order, in which they are listed in the input).

If there are several solutions, print any of them.

Examples

Input

3
1 3
2 3
0 0
1 1
2 0


Output

1 3 2


Input

4
1 2
2 3
1 4
-1 -2
3 5
-3 3
2 0


Output

4 2 1 3

Note

The possible solutions for the sample are given below.

<image> <image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
struct pt {
  int x, y, id;
};
vector<pt> p;
vector<vector<int> > g;
vector<int> size, ans;
int n;
int dfs(int v, int parent = -1) {
  for (int i = 0; i < g[v].size(); i++) {
    int to = g[v][i];
    if (to == parent) continue;
    size[v] += dfs(to, v);
  }
  return size[v];
}
int gx, gy;
bool cmp(pt a, pt b) {
  return 1LL * (a.x - gx) * (b.y - gy) - 1LL * (b.x - gx) * (a.y - gy) > 0;
}
bool cmp2(pt a, pt b) { return a.y > b.y || a.y == b.y && a.x < b.x; }
void rec(int v, vector<pt> p, int parent = -1) {
  gx = p.front().x, gy = p.front().y;
  ans[p.front().id] = v;
  p.erase(p.begin());
  sort(p.begin(), p.end(), cmp);
  vector<pt> buf;
  int cur = 0;
  for (int i = 0; i < g[v].size(); i++) {
    int to = g[v][i];
    if (to != parent) {
      buf.clear();
      for (int x = cur; cur < x + size[to]; cur++) {
        buf.push_back(p[cur]);
      }
      rec(to, buf, v);
    }
  }
}
int main() {
  cin >> n;
  g.resize(n);
  p.resize(n);
  ans.resize(n);
  size.assign(n, 1);
  for (int i = 0; i < n - 1; i++) {
    int a, b;
    cin >> a >> b;
    g[--a].push_back(--b);
    g[b].push_back(a);
  }
  for (int i = 0; i < n; i++) {
    cin >> p[i].x >> p[i].y;
    p[i].id = i;
  }
  dfs(0);
  sort(p.begin(), p.end(), cmp2);
  rec(0, p);
  for (int i = 0; i < n; i++) cout << (ans[i] + 1) << \" \";
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import functools
from collections import defaultdict
from bootcamp import Basebootcamp

def generate_random_tree_edges(n):
    if n == 1:
        return []
    if n == 2:
        return [(1, 2)]
    prufer = [random.randint(1, n) for _ in range(n-2)]
    degree = defaultdict(int)
    for node in prufer:
        degree[node] += 1
    leaves = []
    for v in range(1, n+1):
        if degree[v] == 0:
            leaves.append(v)
    edges = []
    for node in prufer:
        leaf = leaves.pop(0)
        edges.append((leaf, node))
        degree[leaf] -= 1
        degree[node] -= 1
        if degree[node] == 0:
            leaves.append(node)
        leaves.sort()
    edges.append((leaves[0], leaves[1]))
    edges = [tuple(sorted(e)) for e in edges]
    return edges[:n-1]

def generate_points(n, min_coord=-10**9, max_coord=10**9):
    xs = random.sample(range(min_coord, max_coord + 1), n)
    ys = [x**2 + random.randint(-1000, 1000) for x in xs]
    return list(zip(xs, ys))

def generate_solution(n, edges, points):
    g = [[] for _ in range(n)]
    for u, v in edges:
        u0 = u - 1
        v0 = v - 1
        g[u0].append(v0)
        g[v0].append(u0)
    p_list = [{'x': x, 'y': y, 'id': i} for i, (x, y) in enumerate(points)]
    size = [1] * n

    def dfs(v, parent):
        total = 1
        for to in g[v]:
            if to != parent:
                total += dfs(to, v)
        size[v] = total
        return total
    dfs(0, -1)
    sorted_p = sorted(p_list, key=lambda pt: (-pt['y'], pt['x']))
    ans = [0] * n

    def rec(v, pts, parent):
        if not pts:
            return
        current = pts[0]
        ans[current['id']] = v
        remaining = pts[1:]
        if not remaining:
            return
        gx, gy = current['x'], current['y']
        def compare(a, b):
            val = (a['x'] - gx) * (b['y'] - gy) - (b['x'] - gx) * (a['y'] - gy)
            return -1 if val > 0 else 1 if val < 0 else 0
        remaining_sorted = sorted(remaining, key=functools.cmp_to_key(compare))
        cur = 0
        for to in g[v]:
            if to != parent:
                subset = remaining_sorted[cur:cur + size[to]]
                cur += size[to]
                rec(to, subset, v)
    rec(0, sorted_p, -1)
    return [ans[i] + 1 for i in range(n)]

def segments_intersect(a, b, c, d):
    def ccw(A, B, C):
        return (B[0]-A[0])*(C[1]-A[1]) - (B[1]-A[1])*(C[0]-A[0])
    ccw1 = ccw(a, b, c)
    ccw2 = ccw(a, b, d)
    ccw3 = ccw(c, d, a)
    ccw4 = ccw(c, d, b)
    
    if (ccw1 * ccw2 < 0) and (ccw3 * ccw4 < 0):
        return True
    
    def on_segment(p, a, b):
        return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0])) and \
               (min(a[1], b[1]) <= p[1] <= max(a[1], b[1])) and \
               (ccw(a, b, p) == 0)
    
    return on_segment(c, a, b) or on_segment(d, a, b) or \
           on_segment(a, c, d) or on_segment(b, c, d)

class Epainttreebootcamp(Basebootcamp):
    def __init__(self, n=3, min_coord=-10**9, max_coord=10**9):
        self.n = n
        self.min_coord = min_coord
        self.max_coord = max_coord
    
    def case_generator(self):
        while True:
            edges = generate_random_tree_edges(self.n)
            points = generate_points(self.n, self.min_coord, self.max_coord)
            try:
                solution = generate_solution(self.n, edges, points)
                if sorted(solution) == list(range(1, self.n+1)):
                    return {
                        'n': self.n,
                        'edges': edges,
                        'points': points
                    }
            except:
                continue
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        edges = question_case['edges']
        points = question_case['points']
        problem = f"""You are given a tree with {n} vertices and {n} points on a plane. No three points lie on the same straight line. Your task is to assign each vertex of the tree to exactly one of the given points such that when the tree is drawn on the plane with edges as line segments between corresponding points, the following conditions are met:
1. Each vertex is assigned to a distinct point.
2. For any two adjacent vertices in the tree, their corresponding points are connected by a segment.
3. No two segments corresponding to non-adjacent edges in the tree intersect each other, except at common endpoints for adjacent edges.

Input format:
- The first line contains an integer n ({n} in this case).
- The next {n-1} lines describe the edges of the tree.
- The next {n} lines give the coordinates of the points.

Output format:
Print {n} distinct integers where the i-th integer represents the vertex assigned to the i-th point (in the order the points are given). The vertices are numbered from 1 to {n}.

The input for this case is:
{n}
"""
        for u, v in edges:
            problem += f"{u} {v}\n"
        for x, y in points:
            problem += f"{x} {y}\n"
        problem += "\nProvide your answer within [answer] and [/answer] tags."
        return problem
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last = matches[-1].strip()
        try:
            return list(map(int, last.split()))
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        edges = identity['edges']
        points = identity['points']
        if solution is None or len(solution) != n or sorted(solution) != list(range(1, n+1)):
            return False
        
        vertex_to_point = {solution[i]: points[i] for i in range(n)}
        segments = []
        adj_edges = defaultdict(set)
        for u, v in edges:
            adj_edges[u].add(v)
            adj_edges[v].add(u)
            p1 = vertex_to_point[u]
            p2 = vertex_to_point[v]
            segments.append((p1, p2))
        
        for i in range(len(segments)):
            a, b = segments[i]
            u1, v1 = edges[i]
            for j in range(i+1, len(segments)):
                c, d = segments[j]
                u2, v2 = edges[j]
                if u2 in adj_edges[u1] or u2 in adj_edges[v1] or v2 in adj_edges[u1] or v2 in adj_edges[v1]:
                    continue
                if segments_intersect(a, b, c, d):
                    return False
        return True
