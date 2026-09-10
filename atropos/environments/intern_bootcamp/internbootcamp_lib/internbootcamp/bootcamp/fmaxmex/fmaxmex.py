"""# 

### 谜题描述
Once Grisha found a tree (connected graph without cycles) with a root in node 1.

But this tree was not just a tree. A permutation p of integers from 0 to n - 1 is written in nodes, a number p_i is written in node i.

As Grisha likes to invent some strange and interesting problems for himself, but not always can solve them, you need to help him deal with two types of queries on this tree.

Let's define a function MEX(S), where S is a set of non-negative integers, as a smallest non-negative integer that is not included in this set.

Let l be a simple path in this tree. So let's define indices of nodes which lie on l as u_1, u_2, …, u_k. 

Define V(l) as a set {p_{u_1}, p_{u_2}, … , p_{u_k}}. 

Then queries are: 

  1. For two nodes i and j, swap p_i and p_j. 
  2. Find the maximum value of MEX(V(l)) in all possible l. 

Input

The first line contains a single integer n (2 ≤ n ≤ 2 ⋅ 10^5) — the number of nodes of a tree.

The second line contains n integers — p_1, p_2, …, p_n (0≤ p_i < n) — the permutation p, it's guaranteed that all numbers are different .

The third line contains n - 1 integers — d_2, d_3, …, d_n (1 ≤ d_i < i), where d_i is a direct ancestor of node i in a tree.

The fourth line contains a single integer q (1 ≤ q ≤ 2 ⋅ 10^5) — the number of queries.

The following q lines contain the description of queries:

At the beginning of each of next q lines, there is a single integer t (1 or 2) — the type of a query: 

  1. If t = 1, the line also contains two integers i and j (1 ≤ i, j ≤ n) — the indices of nodes, where values of the permutation should be swapped. 
  2. If t = 2, you need to find the maximum value of MEX(V(l)) in all possible l. 

Output

For each type 2 query print a single integer — the answer for this query.

Examples

Input


6
2 5 0 3 1 4
1 1 3 3 3
3
2
1 6 3
2


Output


3
2


Input


6
5 2 1 4 3 0
1 1 1 3 3
9
2
1 5 3
2
1 6 1
2
1 4 2
2
1 1 6
2


Output


3
2
4
4
2

Note

Number written in brackets is a permutation value of a node. 

<image> In the first example, for the first query, optimal path is a path from node 1 to node 5. For it, set of values is \{0, 1, 2\} and MEX is 3.  <image> For the third query, optimal path is a path from node 5 to node 6. For it, set of values is \{0, 1, 4\} and MEX is 2.  <image> In the second example, for the first query, optimal path is a path from node 2 to node 6. For it, set of values is \{0, 1, 2, 5\} and MEX is 3.  <image> For the third query, optimal path is a path from node 5 to node 6. For it, set of values is \{0, 1, 3\} and MEX is 2.  <image> For the fifth query, optimal path is a path from node 5 to node 2. For it, set of values is \{0, 1, 2, 3\} and MEX is 4.  <image> For the seventh query, optimal path is a path from node 5 to node 4. For it, set of values is \{0, 1, 2, 3\} and MEX is 4.  <image> For the ninth query, optimal path is a path from node 6 to node 5. For it, set of values is \{0, 1, 3\} and MEX is 2. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int N = 2e5 + 5;
int n, tme, m = 1, q;
int a[N], pl[N], p[N], dep[N], B[N], E[N];
int lg[N * 2];
pair<int, int> MN[22][N * 2];
vector<int> son[N];
int lca(int x, int y) {
  if (B[x] > B[y]) swap(x, y);
  x = B[x], y = E[y];
  int l = lg[y - x + 1];
  return min(MN[l][x], MN[l][y - (1 << l) + 1]).second;
}
void dfs(int x) {
  B[x] = E[x] = ++tme;
  MN[0][tme] = make_pair(dep[x], x);
  for (int i = 0; i < int(son[x].size()); i++) {
    int to = son[x][i];
    dep[to] = dep[x] + 1;
    dfs(to);
    E[x] = ++tme;
    MN[0][tme] = make_pair(dep[x], x);
  }
}
struct Line {
  int status;
  int px, py;
  Line() : px(0), py(0), status(0){};
  Line(int px, int py) : px(px), py(py), status(1){};
  int length() const { return dep[px] + dep[py] - 2 * dep[lca(px, py)]; }
  bool operator<(const Line &oth) const { return length() < oth.length(); }
  bool operator>(const Line &oth) const { return oth < (*this); }
  bool operator==(const Line &oth) const {
    return !((*this) < oth) && !(oth < (*this));
  }
  Line operator+(const Line &oth) const {
    if (!status) return oth;
    if (!oth.status) return (*this);
    if (status == 2) return (*this);
    if (oth.status == 2) return oth;
    Line r[7], mx, res;
    mx = r[1] = (*this);
    r[2] = oth;
    r[3] = Line(px, oth.px);
    r[4] = Line(px, oth.py);
    r[5] = Line(py, oth.px);
    r[6] = Line(py, oth.py);
    for (int i = 1; i <= 6; i++) mx = max(mx, r[i]);
    for (int i = 1; i <= 6; i++)
      if (mx == r[i]) {
        if (Line(mx.px, px).length() + Line(px, mx.py).length() ==
                mx.length() &&
            Line(mx.px, py).length() + Line(py, mx.py).length() ==
                mx.length() &&
            Line(mx.px, oth.px).length() + Line(oth.px, mx.py).length() ==
                mx.length() &&
            Line(mx.px, oth.py).length() + Line(oth.py, mx.py).length() ==
                mx.length())
          res = mx;
        else
          res.status = 2;
        break;
      }
    return res;
  }
};
int lson[N * 2], rson[N * 2];
Line val[N * 2];
void build(int pos, int x, int y) {
  if (x == y) {
    val[pos] = Line(pl[x], pl[x]);
    return;
  }
  int mid = (x + y) >> 1;
  lson[pos] = ++m;
  rson[pos] = ++m;
  build(lson[pos], x, mid);
  build(rson[pos], mid + 1, y);
  val[pos] = val[lson[pos]] + val[rson[pos]];
}
void modify(int pos, int x, int y, int l) {
  if (x >= l && y <= l) {
    val[pos] = Line(pl[l], pl[l]);
    return;
  }
  if (x > l || y < l) return;
  int mid = (x + y) >> 1;
  modify(lson[pos], x, mid, l);
  modify(rson[pos], mid + 1, y, l);
  val[pos] = val[lson[pos]] + val[rson[pos]];
}
int query(int pos, int x, int y, Line v) {
  Line nv;
  if (x == y) {
    nv = v + val[pos];
    if (nv.status == 2) x--;
    return x;
  }
  nv = v + val[lson[pos]];
  int mid = (x + y) >> 1;
  if (nv.status != 2)
    return query(rson[pos], mid + 1, y, nv);
  else
    return query(lson[pos], x, mid, v);
}
int main() {
  lg[1] = 0;
  for (int i = 2; i < N * 2; i++) lg[i] = lg[i / 2] + 1;
  scanf(\"%d\", &n);
  for (int i = 1; i <= n; i++) {
    int x;
    scanf(\"%d\", &x);
    x++;
    a[i] = x;
    pl[x] = i;
  }
  for (int i = 2; i <= n; i++) {
    scanf(\"%d\", p + i);
    son[p[i]].push_back(i);
  }
  dfs(1);
  for (int i = 1; i <= 21; i++)
    for (int j = 1; j + (1 << i) - 1 <= tme; j++)
      MN[i][j] = min(MN[i - 1][j], MN[i - 1][j + (1 << (i - 1))]);
  build(1, 1, n);
  scanf(\"%d\", &q);
  while (q--) {
    int o;
    scanf(\"%d\", &o);
    if (o == 1) {
      int x, y;
      scanf(\"%d%d\", &x, &y);
      swap(a[x], a[y]);
      pl[a[x]] = x;
      pl[a[y]] = y;
      modify(1, 1, n, a[x]);
      modify(1, 1, n, a[y]);
    } else
      printf(\"%d\n\", query(1, 1, n, Line()));
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Fmaxmexbootcamp(Basebootcamp):
    def __init__(self, max_nodes=6, q_max=4):
        self.max_nodes = max_nodes  # 限制节点数提高计算可靠性
        self.q_max = q_max

    def _build_tree(self, n):
        parent_map = {1: None}
        for i in range(2, n+1):
            parent_map[i] = random.randint(1, i-1)
        return [parent_map[i] for i in range(2, n+1)], parent_map

    @staticmethod
    def _find_lca(u, v, parent_map):
        path = set()
        while u:
            path.add(u)
            u = parent_map.get(u)
        while v not in path:
            v = parent_map.get(v)
        return v

    def _calc_mex(self, n, p, parent_map):
        max_mex = 0
        for start in range(1, n+1):
            for end in range(start, n+1):
                path = set()
                current = start
                lca = self._find_lca(start, end, parent_map)
                
                while current != lca:
                    path.add(current)
                    current = parent_map[current]
                path.add(lca)
                
                current = end
                while current != lca:
                    path.add(current)
                    current = parent_map[current]
                
                values = {p[node-1] for node in path}
                mex = 0
                while mex in values:
                    mex += 1
                max_mex = max(max_mex, mex)
        return max_mex

    def case_generator(self):
        n = random.randint(2, self.max_nodes)
        p = list(range(n))
        random.shuffle(p)
        d, parent_map = self._build_tree(n)
        
        queries = []
        expected = []
        current_p = p.copy()
        
        for _ in range(random.randint(1, self.q_max)):
            if random.random() < 0.4 and len(queries) > 0:
                i, j = random.sample(range(1, n+1), 2)
                queries.append({'type': 1, 'i': i, 'j': j})
                current_p[i-1], current_p[j-1] = current_p[j-1], current_p[i-1]
            else:
                queries.append({'type': 2})
                expected.append(self._calc_mex(n, current_p, parent_map))
        
        return {
            'n': n,
            'p': current_p,
            'd': d,
            'queries': queries,
            'expected_answers': expected,
            'parent_map': parent_map
        }

    @staticmethod
    def prompt_func(case):
        prompt = f"""Tree with {case['n']} nodes (root=1)
Permutation: {case['p']}
Parent list (nodes 2-{case['n']}): {case['d']}
Queries:
"""
        for i, q in enumerate(case['queries'], 1):
            if q['type'] == 1:
                prompt += f"{i}. Swap nodes {q['i']} and {q['j']}\n"
            else:
                prompt += f"{i}. Find max MEX\n"
        return prompt + "\nAnswer each type 2 query with [answer]number[/answer]"

    @staticmethod
    def extract_output(output):
        import re
        return list(map(int, re.findall(r'\[answer\](\d+)\[\/answer\]', output)))

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected_answers']
