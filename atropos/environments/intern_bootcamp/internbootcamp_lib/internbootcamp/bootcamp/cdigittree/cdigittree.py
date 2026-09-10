"""# 

### 谜题描述
ZS the Coder has a large tree. It can be represented as an undirected connected graph of n vertices numbered from 0 to n - 1 and n - 1 edges between them. There is a single nonzero digit written on each edge.

One day, ZS the Coder was bored and decided to investigate some properties of the tree. He chose a positive integer M, which is coprime to 10, i.e. <image>.

ZS consider an ordered pair of distinct vertices (u, v) interesting when if he would follow the shortest path from vertex u to vertex v and write down all the digits he encounters on his path in the same order, he will get a decimal representaion of an integer divisible by M.

Formally, ZS consider an ordered pair of distinct vertices (u, v) interesting if the following states true:

  * Let a1 = u, a2, ..., ak = v be the sequence of vertices on the shortest path from u to v in the order of encountering them; 
  * Let di (1 ≤ i < k) be the digit written on the edge between vertices ai and ai + 1; 
  * The integer <image> is divisible by M. 



Help ZS the Coder find the number of interesting pairs!

Input

The first line of the input contains two integers, n and M (2 ≤ n ≤ 100 000, 1 ≤ M ≤ 109, <image>) — the number of vertices and the number ZS has chosen respectively.

The next n - 1 lines contain three integers each. i-th of them contains ui, vi and wi, denoting an edge between vertices ui and vi with digit wi written on it (0 ≤ ui, vi < n, 1 ≤ wi ≤ 9).

Output

Print a single integer — the number of interesting (by ZS the Coder's consideration) pairs.

Examples

Input

6 7
0 1 2
4 2 4
2 0 1
3 0 9
2 5 7


Output

7


Input

5 11
1 2 3
2 0 3
3 0 3
4 3 3


Output

8

Note

In the first sample case, the interesting pairs are (0, 4), (1, 2), (1, 5), (3, 2), (2, 5), (5, 2), (3, 5). The numbers that are formed by these pairs are 14, 21, 217, 91, 7, 7, 917 respectively, which are all multiples of 7. Note that (2, 5) and (5, 2) are considered different. 

<image>

In the second sample case, the interesting pairs are (4, 0), (0, 4), (3, 2), (2, 3), (0, 1), (1, 0), (4, 1), (1, 4), and 6 of these pairs give the number 33 while 2 of them give the number 3333, which are all multiples of 11.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const long long N = 100005;
long long n, m, k, sz[N], mx[N], vis[N], rt, pw[N] = {1}, iv[N] = {0}, ans;
vector<pair<long long, long long>> e[N], b;
vector<long long> a, c;
void exgcd(long long a, long long b, long long &x, long long &y) {
  if (!b) {
    x = 1, y = 0;
    return;
  }
  exgcd(b, a % b, x, y);
  long long t = x;
  x = y, y = t - a / b * y;
}
long long inv(long long a, long long k) {
  long long x, y;
  exgcd(a, k, x, y);
  x = (x + k) % k;
  return x;
}
void root(long long u, long long f) {
  sz[u] = 1, mx[u] = 0;
  for (auto i : e[u]) {
    long long v = i.first;
    if (v == f || vis[v]) continue;
    root(v, u);
    if (sz[v] > mx[u]) mx[u] = sz[v];
    sz[u] += sz[v];
  }
  mx[u] = max(mx[u], m - sz[u]);
  if (mx[u] < mx[rt]) rt = u;
}
void dfs1(long long u, long long f, long long p, long long d) {
  a.push_back(p);
  for (auto i : e[u]) {
    long long v = i.first, w = i.second;
    if (v == f || vis[v]) continue;
    dfs1(v, u, (p + w * d % k) % k, d * 10 % k);
  }
}
void dfs2(long long u, long long f, long long p, long long d) {
  b.emplace_back(p, d);
  for (auto i : e[u]) {
    long long v = i.first, w = i.second;
    if (v == f || vis[v]) continue;
    dfs2(v, u, (p * 10 + w) % k, d + 1);
  }
}
long long cal(long long u, long long d) {
  long long s = 0;
  a.clear();
  b.clear();
  c.clear();
  if (!d) {
    dfs1(u, 0, 0, 1);
    dfs2(u, 0, 0, 0);
  } else {
    dfs1(u, 0, d, 10);
    dfs2(u, 0, d, 1);
  }
  for (auto i : b) c.push_back((k - i.first) * iv[i.second] % k);
  for (long long i = 0; i < a.size(); i++)
    if (a[i] == c[i]) s--;
  sort(c.begin(), c.end());
  for (auto i : a)
    s +=
        upper_bound(c.begin(), c.end(), i) - lower_bound(c.begin(), c.end(), i);
  return s;
}
void sol(long long u) {
  vis[u] = 1;
  ans += cal(u, 0);
  for (auto i : e[u]) {
    long long v = i.first, w = i.second;
    if (vis[v]) continue;
    ans -= cal(v, w);
    rt = 0, m = sz[v];
    root(v, 0);
    sol(rt);
  }
}
signed main() {
  ios::sync_with_stdio(false);
  cin >> n >> k;
  if (k == 1) {
    cout << n * (n - 1) << endl;
    return 0;
  }
  for (long long i = 1; i <= n; i++) {
    pw[i] = pw[i - 1] * 10 % k;
    iv[i] = inv(pw[i], k);
  }
  for (long long i = 1; i < n; i++) {
    long long u, v, w;
    cin >> u >> v >> w;
    w %= k;
    u++, v++;
    e[u].emplace_back(v, w);
    e[v].emplace_back(u, w);
  }
  mx[0] = 1e9, m = n;
  root(1, 0);
  sol(rt);
  cout << ans << endl;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import math
import re
from bootcamp import Basebootcamp

class Cdigittreebootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__()
        self.n = params.get('n', 5)
        self.M = params.get('M', 7)
        
        # Ensure M is coprime with 10
        if math.gcd(self.M, 10) != 1:
            while True:
                new_M = random.randint(1, 100)
                if math.gcd(new_M, 10) == 1:
                    self.M = new_M
                    break
    
    def case_generator(self):
        n = self.n
        M = self.M
        
        edges = self._generate_random_tree(n)
        adj = self._build_adjacency_list(n, edges)
        correct_answer = self._calculate_correct_answer(n, M, adj)
        
        return {
            'n': n,
            'M': M,
            'edges': edges,
            'correct_answer': correct_answer
        }
    
    def _generate_random_tree(self, n):
        if n == 1:
            return []
        edges = []
        for i in range(1, n):
            parent = random.randint(0, i-1)
            w = random.randint(1, 9)
            edges.append((parent, i, w))
        return edges
    
    def _build_adjacency_list(self, n, edges):
        adj = [[] for _ in range(n)]
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        return adj
    
    def _calculate_correct_answer(self, n, M, adj):
        correct = 0
        for u in range(n):
            for v in range(n):
                if u == v:
                    continue
                path = self._get_path_weights(u, v, adj)
                mod = 0
                for d in path:
                    mod = (mod * 10 + d) % M
                if mod == 0:
                    correct += 1
        return correct
    
    def _get_path_weights(self, u, v, adj):
        parent = {}
        visited = set([u])
        queue = [u]
        parent[u] = (None, None)
        found = False
        
        while queue:
            current = queue.pop(0)
            if current == v:
                found = True
                break
            for neighbor, w in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = (current, w)
                    queue.append(neighbor)
        
        if not found:
            return []
        
        path = []
        current = v
        while current != u:
            prev, w = parent[current]
            path.append(w)
            current = prev
        path.reverse()
        return path

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        M = question_case['M']
        edges = question_case['edges']
        
        input_lines = [f"{n} {M}"]
        for u, v, w in edges:
            input_lines.append(f"{u} {v} {w}")
        input_str = '\n'.join(input_lines)
        
        return f"""ZS the Coder has a tree with {n} vertices. Each edge contains a non-zero digit. Find the number of ordered pairs (u, v) where u ≠ v and the integer formed by the path's digits is divisible by {M}.

Input:
{input_str}

Output a single integer. Place your answer within [answer] and [/answer] tags, e.g., [answer]7[/answer]."""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity.get('correct_answer', -1)
