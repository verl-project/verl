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
template <class T>
inline bool chkmin(T& x, T y) {
  return y < x ? x = y, 1 : 0;
}
template <class T>
inline bool chkmax(T& x, T y) {
  return x < y ? x = y, 1 : 0;
}
inline long long Max(long long x, long long y) { return x > y ? x : y; }
inline long long Min(long long x, long long y) { return x < y ? x : y; }
int n;
vector<int> E[1502];
int sz[1502];
struct point {
  int x, y, id;
  double jj;
} p[1502];
int ans[1502];
bool cmp(point a, point b) { return a.jj < b.jj; }
void dfs(int x, int f) {
  sz[x] = 1;
  for (int i = (0), i_end_ = (E[x].size()); i < i_end_; i++) {
    int y = E[x][i];
    if (y == f) continue;
    dfs(y, x);
    sz[x] += sz[y];
  }
}
void solve(int x, int f, int l, int r) {
  if (l > r) return;
  int pos = l;
  for (int i = l + 1; i <= r; i++)
    if (p[i].y < p[pos].y || (p[i].y == p[pos].y && p[i].x < p[pos].x)) pos = i;
  swap(p[pos], p[l]);
  ans[p[l].id] = x;
  if (l == r) return;
  for (int i = l + 1; i <= r; i++) {
    p[i].x -= p[l].x, p[i].y -= p[l].y;
    p[i].jj = atan2(p[i].y, p[i].x);
  }
  sort(p + l + 1, p + r + 1, cmp);
  int now = l + 1;
  for (int i = (0), i_end_ = (E[x].size()); i < i_end_; i++) {
    int y = E[x][i];
    if (y == f) continue;
    solve(y, x, now, now + sz[y] - 1);
    now += sz[y];
  }
}
int main() {
  scanf(\"%d\", &n);
  for (int i = (1), i_end_ = (n); i < i_end_; i++) {
    int a, b;
    scanf(\"%d%d\", &a, &b);
    E[a].push_back(b);
    E[b].push_back(a);
  }
  for (int i = (1), i_end_ = (n); i <= i_end_; i++) {
    scanf(\"%d%d\", &p[i].x, &p[i].y);
    p[i].id = i;
  }
  dfs(1, 0);
  solve(1, 0, 1, n);
  for (int i = (1), i_end_ = (n); i <= i_end_; i++)
    printf(\"%d%c\", ans[i], i < n ? ' ' : '\n');
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import json
import re
import random
from math import atan2

class Cpainttreebootcamp(Basebootcamp):
    def __init__(self, max_n=10, default_n=5):
        self.max_n = max_n
        self.default_n = default_n
    
    def case_generator(self):
        n = self.default_n
        edges = self._generate_tree(n)
        points = self._generate_points(n)
        return {'n': n, 'edges': edges, 'points': points}
    
    @staticmethod
    def prompt_func(question_case):
        edges_str = '\n'.join(f"{u} {v}" for u, v in question_case['edges'])
        points_str = '\n'.join(f"{x} {y}" for x, y in question_case['points'])
        prompt = (
            "You are given a tree with {n} vertices and {n} distinct points on a plane. "
            "No three points are collinear.\n\n"
            "Your task is to assign each tree vertex to a point such that edges are drawn as straight lines "
            "without unnecessary intersections. Specifically, edges that are not adjacent in the tree must not "
            "intersect at any point, and adjacent edges should only share their common endpoint.\n\n"
            "Input Format:\n"
            "First line: {n}\n"
            "Next {n_minus_1} lines: Pairs of vertices connected by edges\n"
            "Next {n} lines: Coordinates of each point\n\n"
            "Output Format:\n"
            "Space-separated integers where the i-th number indicates the vertex assigned to the i-th input point.\n\n"
            "Input Example:\n"
            "3\n"
            "1 3\n"
            "2 3\n"
            "0 0\n"
            "1 1\n"
            "2 0\n\n"
            "Expected Output:\n"
            "1 3 2\n\n"
            "Your Task Input:\n"
            "{n}\n{edges}\n{points}\n\n"
            "Output your answer within [answer] and [/answer], for example: [answer]1 3 2[/answer]"
        ).format(
            n=question_case['n'],
            n_minus_1=question_case['n'] - 1,
            edges=edges_str,
            points=points_str
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        last_answer = answer_blocks[-1].strip()
        try:
            return list(map(int, last_answer.split()))
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            n = identity['n']
            edges = identity['edges']
            points = identity['points']
            
            # Validate permutation
            if len(solution) != n or set(solution) != set(range(1, n+1)):
                return False
            
            # Build vertex to point mapping
            vertex_point = {v: points[i] for i, v in enumerate(solution)}
            
            # Build adjacency list
            adj = {i: set() for i in range(1, n+1)}
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            
            # Check all edge segments
            segments = []
            for u, v in edges:
                p1 = vertex_point[u]
                p2 = vertex_point[v]
                segments.append((p1, p2, u, v))  # Store with vertices for adjacency check
            
            # Check pairwise intersections
            for i in range(len(segments)):
                seg1 = segments[i]
                for j in range(i+1, len(segments)):
                    seg2 = segments[j]
                    if cls._segments_intersect(seg1[:2], seg2[:2]):
                        # Check if edges are adjacent
                        u1, v1, _, _ = seg1
                        u2, v2, _, _ = seg2
                        if not ({u1, v1} & {u2, v2}):
                            return False
            return True
        except Exception as e:
            print(f"Verification error: {e}")
            return False
    
    @staticmethod
    def _generate_tree(n):
        parents = [0]*(n+1)
        edges = []
        for v in range(2, n+1):
            u = random.randint(1, v-1)
            parents[v] = u
            edges.append((u, v))
        return edges
    
    @staticmethod
    def _generate_points(n):
        points = []
        max_attempts = 1000
        
        def is_collinear(p1, p2, p3):
            return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0])
        
        for _ in range(n):
            attempts = 0
            while True:
                x = random.randint(-10, 10)
                y = random.randint(-10, 10)
                new_point = (x, y)
                
                # Check uniqueness and collinearity
                if new_point in points:
                    continue
                
                collinear = False
                for i in range(len(points)):
                    for j in range(i+1, len(points)):
                        if is_collinear(points[i], points[j], new_point):
                            collinear = True
                            break
                    if collinear:
                        break
                
                if not collinear:
                    points.append(new_point)
                    break
                
                attempts += 1
                if attempts > max_attempts:
                    raise RuntimeError("Failed to generate non-collinear points")
        return points
    
    @classmethod
    def _segments_intersect(cls, seg1, seg2):
        (a, b), (c, d) = seg1, seg2
        
        def ccw(p, q, r):
            return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
        
        ccw1 = ccw(a, b, c)
        ccw2 = ccw(a, b, d)
        if (ccw1 > 0 and ccw2 > 0) or (ccw1 < 0 and ccw2 < 0):
            return False
        
        ccw3 = ccw(c, d, a)
        ccw4 = ccw(c, d, b)
        if (ccw3 > 0 and ccw4 > 0) or (ccw3 < 0 and ccw4 < 0):
            return False
        
        # Check overlapping colinear segments
        if ccw1 == 0 and ccw2 == 0 and ccw3 == 0 and ccw4 == 0:
            def on_segment(p, q, r):
                return (min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and 
                        min(p[1], q[1]) <= r[1] <= max(p[1], q[1]))
            
            return any(on_segment(a, b, p) for p in [c, d]) or \
                   any(on_segment(c, d, p) for p in [a, b])
        
        return True
