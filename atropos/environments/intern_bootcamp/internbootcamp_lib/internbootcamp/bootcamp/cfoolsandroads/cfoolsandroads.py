"""# 

### 谜题描述
They say that Berland has exactly two problems, fools and roads. Besides, Berland has n cities, populated by the fools and connected by the roads. All Berland roads are bidirectional. As there are many fools in Berland, between each pair of cities there is a path (or else the fools would get upset). Also, between each pair of cities there is no more than one simple path (or else the fools would get lost). 

But that is not the end of Berland's special features. In this country fools sometimes visit each other and thus spoil the roads. The fools aren't very smart, so they always use only the simple paths.

A simple path is the path which goes through every Berland city not more than once.

The Berland government knows the paths which the fools use. Help the government count for each road, how many distinct fools can go on it.

Note how the fools' paths are given in the input.

Input

The first line contains a single integer n (2 ≤ n ≤ 105) — the number of cities. 

Each of the next n - 1 lines contains two space-separated integers ui, vi (1 ≤ ui, vi ≤ n, ui ≠ vi), that means that there is a road connecting cities ui and vi. 

The next line contains integer k (0 ≤ k ≤ 105) — the number of pairs of fools who visit each other. 

Next k lines contain two space-separated numbers. The i-th line (i > 0) contains numbers ai, bi (1 ≤ ai, bi ≤ n). That means that the fool number 2i - 1 lives in city ai and visits the fool number 2i, who lives in city bi. The given pairs describe simple paths, because between every pair of cities there is only one simple path.

Output

Print n - 1 integer. The integers should be separated by spaces. The i-th number should equal the number of fools who can go on the i-th road. The roads are numbered starting from one in the order, in which they occur in the input.

Examples

Input

5
1 2
1 3
2 4
2 5
2
1 4
3 5


Output

2 1 1 1 


Input

5
3 4
4 5
1 4
2 4
3
2 3
1 3
3 5


Output

3 1 1 1 

Note

In the first sample the fool number one goes on the first and third road and the fool number 3 goes on the second, first and fourth ones.

In the second sample, the fools number 1, 3 and 5 go on the first road, the fool number 5 will go on the second road, on the third road goes the fool number 3, and on the fourth one goes fool number 1.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int N = 1e5 + 1;
struct my {
  int first, second, col, hld;
} b[N];
int l = 1, C = 1, tin[N], tout[N], timer, n, m, first, second, z, up[N][25],
    d[N], sz[N], s, v, ans[N];
pair<int, int> pred[N];
vector<pair<int, int> > a[N];
vector<int> tree[N], hld[N];
void dfs(int v, int pr = 0) {
  tin[v] = ++timer;
  up[v][0] = pr;
  for (int i = 1; i <= l; ++i) up[v][i] = up[up[v][i - 1]][i - 1];
  for (int i = 0; i < a[v].size(); ++i) {
    int to = a[v][i].first;
    if (to == pr) continue;
    pred[to] = make_pair(v, a[v][i].second);
    dfs(to, v);
    d[v] += d[to] + 1;
  }
  tout[v] = ++timer;
}
bool upper(int a, int b) { return (tin[a] <= tin[b] && tout[a] >= tout[b]); }
int lca(int a, int b) {
  if (upper(a, b)) return a;
  if (upper(b, a)) return b;
  for (int i = l; i >= 0; --i)
    if (up[a][i] != 0 && !upper(up[a][i], b)) a = up[a][i];
  return up[a][0];
}
void update(int level, int v, int l, int r, int L, int R) {
  if (r < L || l > R) return;
  if (l >= L && r <= R) {
    ++tree[level][v];
    return;
  }
  int mid = (l + r) >> 1;
  update(level, v + v, l, mid, L, R);
  update(level, v + v + 1, mid + 1, r, L, R);
}
void dfs2(int v, int pr = 0, int c = 1) {
  pair<int, int> z = make_pair(0, 0);
  for (int i = 0; i < a[v].size(); ++i) {
    pair<int, int> to = a[v][i];
    if (to.first == pr) continue;
    if (b[to.second].first != v) swap(b[to.second].first, b[to.second].second);
    if (d[to.first] >= d[z.first]) z = to;
  }
  if (!z.first) return;
  hld[C].push_back(z.second);
  b[z.second].col = C;
  b[z.second].hld = hld[C].size() - 1;
  dfs2(z.first, v, c);
  for (int i = 0; i < a[v].size(); ++i) {
    pair<int, int> to = a[v][i];
    if (to.first == pr || to.first == z.first) continue;
    C++;
    hld[C].push_back(to.second);
    b[to.second].col = C;
    b[to.second].hld = 0;
    dfs2(to.first, v, C);
  }
}
void solve(int first, int second) {
  int v = first;
  for (int i = l; i >= 0; --i)
    if (up[v][i] != 0 && !upper(up[v][i], second)) v = up[v][i];
  int f = pred[v].second, s = pred[first].second;
  while (b[s].col != b[f].col) {
    update(b[s].col, 1, 1, sz[b[s].col], 1, b[s].hld + 1);
    s = pred[b[hld[b[s].col][0]].first].second;
  }
  update(b[s].col, 1, 1, sz[b[s].col], b[f].hld + 1, b[s].hld + 1);
}
int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cin >> n;
  for (int i = 1; i < n; ++i) {
    cin >> b[i].first >> b[i].second;
    first = b[i].first, second = b[i].second;
    a[first].push_back(make_pair(second, i));
    a[second].push_back(make_pair(first, i));
  }
  while ((1 << l) < n) ++l;
  dfs(1);
  dfs2(1);
  for (int i = 1; i <= C; ++i) {
    tree[i].resize(hld[i].size() << 2);
    sz[i] = 1;
    while (sz[i] < hld[i].size()) sz[i] <<= 1;
  }
  cin >> m;
  while (m--) {
    cin >> first >> second;
    z = lca(first, second);
    if (z != first) solve(first, z);
    if (z != second) solve(second, z);
  }
  for (int i = 1; i <= C; ++i)
    for (int j = 0; j < hld[i].size(); ++j) {
      s = 0, v = sz[i] + j;
      while (v) {
        s += tree[i][v];
        v >>= 1;
      }
      ans[hld[i][j]] = s;
    }
  for (int i = 1; i < n; ++i) printf(\"%d \", ans[i]);
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import deque

from bootcamp import Basebootcamp

class Cfoolsandroadsbootcamp(Basebootcamp):
    def __init__(self, n_min=2, n_max=10, k_min=0, k_max=10):
        self.n_min = n_min
        self.n_max = n_max
        self.k_min = k_min
        self.k_max = k_max
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        nodes = list(range(1, n+1))
        random.shuffle(nodes)
        root = nodes.pop()
        edges = []
        available = [root]
        while nodes:
            u = random.choice(available)
            v = nodes.pop()
            edges.append((u, v) if random.random() < 0.5 else (v, u))
            available.append(v)
        
        random.shuffle(edges)
        
        k = random.randint(self.k_min, self.k_max)
        pairs = []
        valid_nodes = list(range(1, n+1))
        for _ in range(k):
            a, b = random.sample(valid_nodes, 2)
            pairs.append((a, b))
        
        return {
            "n": n,
            "edges": edges,
            "k": k,
            "pairs": pairs
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        input_lines = [
            str(question_case['n']),
            *[f"{u} {v}" for u, v in question_case['edges']],
            str(question_case['k']),
            *[f"{a} {b}" for a, b in question_case['pairs']]
        ]
        input_block = '\n'.join(input_lines)
        
        return f"""You are a Berland road analyst. Given a tree of cities and visiting pairs, count path usage for each road in input order.

Input format:
n
u1 v1
...
uk vk
k
a1 b1
...
ak bk

Output: space-separated integers corresponding to the roads in INPUT ORDER

Input:
{input_block}

Put your final answer within [answer] and [/answer], like:
[answer]1 2 3 4[/answer]"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        
        try:
            numbers = list(map(int, matches[-1].strip().split()))
            return numbers
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or len(solution) != len(identity["edges"]):
            return False
        
        edge_index = {}
        for idx, (u, v) in enumerate(identity["edges"]):
            edge_key = tuple(sorted((u, v)))
            edge_index[edge_key] = idx
        
        counters = [0] * len(identity["edges"])
        for a, b in identity["pairs"]:
            path = cls._find_path(identity["edges"], a, b)
            for u, v in path:
                edge_key = tuple(sorted((u, v)))
                counters[edge_index[edge_key]] += 1
        
        return solution == counters
    
    @classmethod
    def _find_path(cls, edges, start, end):
        adjacency = {}
        for u, v in edges:
            adjacency.setdefault(u, []).append(v)
            adjacency.setdefault(v, []).append(u)
        
        visited = {}
        queue = deque([start])
        visited[start] = None
        
        while queue:
            current = queue.popleft()
            if current == end:
                break
            for neighbor in adjacency.get(current, []):
                if neighbor not in visited:
                    visited[neighbor] = current
                    queue.append(neighbor)
        
        path = []
        current = end
        while current != start and current in visited:
            parent = visited[current]
            path.append((parent, current))
            current = parent
        
        return path
