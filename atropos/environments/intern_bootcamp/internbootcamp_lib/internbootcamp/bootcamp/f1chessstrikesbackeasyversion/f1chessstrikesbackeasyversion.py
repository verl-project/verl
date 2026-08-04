"""# 

### 谜题描述
Note that the difference between easy and hard versions is that in hard version unavailable cells can become available again and in easy version can't. You can make hacks only if all versions are solved.

Ildar and Ivan are tired of chess, but they really like the chessboard, so they invented a new game. The field is a chessboard 2n × 2m: it has 2n rows, 2m columns, and the cell in row i and column j is colored white if i+j is even, and is colored black otherwise.

The game proceeds as follows: Ildar marks some of the white cells of the chessboard as unavailable, and asks Ivan to place n × m kings on the remaining white cells in such way, so that there are no kings attacking each other. A king can attack another king if they are located in the adjacent cells, sharing an edge or a corner.

Ildar would like to explore different combinations of cells. Initially all cells are marked as available, and then he has q queries. In each query he marks a cell as unavailable. After each query he would like to know whether it is possible to place the kings on the available cells in a desired way. Please help him!

Input

The first line of input contains three integers n, m, q (1 ≤ n, m, q ≤ 200 000) — the size of the board and the number of queries.

q lines follow, each of them contains a description of a query: two integers i and j, denoting a white cell (i, j) on the board (1 ≤ i ≤ 2n, 1 ≤ j ≤ 2m, i + j is even) that becomes unavailable. It's guaranteed, that each cell (i, j) appears in input at most once.

Output

Output q lines, i-th line should contain answer for a board after i queries of Ildar. This line should contain \"YES\" if it is possible to place the kings on the available cells in the desired way, or \"NO\" otherwise.

Examples

Input


1 3 3
1 1
1 5
2 4


Output


YES
YES
NO


Input


3 2 7
4 2
6 4
1 3
2 2
2 4
4 4
3 1


Output


YES
YES
NO
NO
NO
NO
NO

Note

In the first example case after the second query only cells (1, 1) and (1, 5) are unavailable. Then Ivan can place three kings on cells (2, 2), (2, 4) and (2, 6).

After the third query three cells (1, 1), (1, 5) and (2, 4) are unavailable, so there remain only 3 available cells: (2, 2), (1, 3) and (2, 6). Ivan can not put 3 kings on those cells, because kings on cells (2, 2) and (1, 3) attack each other, since these cells share a corner.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
using ll = long long;
unsigned euclidean_gcd(unsigned a, unsigned b) {
  if (a < b) return euclidean_gcd(b, a);
  unsigned r;
  while ((r = a % b)) {
    a = b;
    b = r;
  }
  return b;
}
ll ll_gcd(ll a, ll b) {
  if (a < b) return ll_gcd(b, a);
  ll r;
  while ((r = a % b)) {
    a = b;
    b = r;
  }
  return b;
}
struct UnionFind {
  vector<ll> par;
  vector<ll> siz;
  UnionFind(ll sz_) : par(sz_), siz(sz_, 1LL) {
    for (ll i = 0; i < sz_; ++i) par[i] = i;
  }
  void init(ll sz_) {
    par.resize(sz_);
    siz.assign(sz_, 1LL);
    for (ll i = 0; i < sz_; ++i) par[i] = i;
  }
  ll root(ll x) {
    while (par[x] != x) {
      x = par[x] = par[par[x]];
    }
    return x;
  }
  bool merge(ll x, ll y) {
    x = root(x);
    y = root(y);
    if (x == y) return false;
    if (siz[x] < siz[y]) swap(x, y);
    siz[x] += siz[y];
    par[y] = x;
    return true;
  }
  bool issame(ll x, ll y) { return root(x) == root(y); }
  ll size(ll x) { return siz[root(x)]; }
};
long long modpow(long long a, long long n, long long mod) {
  long long res = 1;
  while (n > 0) {
    if (n & 1) res = res * a % mod;
    a = a * a % mod;
    n >>= 1;
  }
  return res;
}
long long modinv(long long a, long long mod) { return modpow(a, mod - 2, mod); }
vector<int> tpsort(vector<vector<int>>& G) {
  int V = G.size();
  vector<int> sorted_vertices;
  queue<int> que;
  vector<int> indegree(V);
  for (int i = 0; i < V; i++) {
    for (int j = 0; j < G[i].size(); j++) {
      indegree[G[i][j]]++;
    }
  }
  for (int i = 0; i < V; i++) {
    if (indegree[i] == 0) {
      que.push(i);
    }
  }
  while (que.empty() == false) {
    int v = que.front();
    que.pop();
    for (int i = 0; i < G[v].size(); i++) {
      int u = G[v][i];
      indegree[u] -= 1;
      if (indegree[u] == 0) que.push(u);
    }
    sorted_vertices.push_back(v);
  }
  return sorted_vertices;
}
struct Point {
  double x;
  double y;
};
struct LineSegment {
  Point start;
  Point end;
};
double tenkyori(const LineSegment& line, const Point& point) {
  double x0 = point.x, y0 = point.y;
  double x1 = line.start.x, y1 = line.start.y;
  double x2 = line.end.x, y2 = line.end.y;
  double a = x2 - x1;
  double b = y2 - y1;
  double a2 = a * a;
  double b2 = b * b;
  double r2 = a2 + b2;
  double tt = -(a * (x1 - x0) + b * (y1 - y0));
  if (tt < 0)
    return sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
  else if (tt > r2)
    return sqrt((x2 - x0) * (x2 - x0) + (y2 - y0) * (y2 - y0));
  double f1 = a * (y1 - y0) - b * (x1 - x0);
  return sqrt((f1 * f1) / r2);
}
void dfs1(vector<vector<ll>>& z, ll k, ll oya, ll& ans, vector<ll>& b) {
  for (auto m : z[k]) {
    if (m != oya) dfs1(z, m, k, ans, b);
  }
  vector<ll> s;
  for (auto m : z[k]) {
    if (m != oya) s.push_back(b[m]);
  }
  ll m = b.size() - 1;
  for (auto d : s) {
    m -= d;
  }
  b[k] = b.size() - m;
  if (m != 0) s.push_back(m);
  ll a = modinv(2, 1000000007);
  for (auto d : s) {
    a += 1000000007 - modinv(modpow(2, b.size() - d, 1000000007), 1000000007);
  }
  a += modinv(modpow(2, b.size(), 1000000007), 1000000007) * (z[k].size() - 1);
  ans += a;
  ans %= 1000000007;
  return;
}
ll merge_cnt(vector<int>& a) {
  int n = a.size();
  if (n <= 1) {
    return 0;
  }
  ll cnt = 0;
  vector<int> b(a.begin(), a.begin() + n / 2);
  vector<int> c(a.begin() + n / 2, a.end());
  cnt += merge_cnt(b);
  cnt += merge_cnt(c);
  int ai = 0, bi = 0, ci = 0;
  while (ai < n) {
    if (bi < b.size() && (ci == c.size() || b[bi] <= c[ci])) {
      a[ai++] = b[bi++];
    } else {
      cnt += n / 2 - bi;
      a[ai++] = c[ci++];
    }
  }
  return cnt;
}
int main() {
  ll n, m, q;
  cin >> n >> m >> q;
  vector<pair<ll, ll>> z(q);
  for (int i = 0; i < q; i++) {
    cin >> z[i].first >> z[i].second;
    z[i].first--;
    z[i].second--;
  }
  ll ok = 0;
  ll ng = q + 1;
  while (ng - ok > 1) {
    ll mid = (ok + ng) / 2;
    vector<ll> f(n, -1);
    vector<ll> g(n, 20000000);
    for (int i = 0; i < mid; i++) {
      ll s = z[i].first / 2;
      ll t = z[i].second / 2;
      if (z[i].first % 2 == 1) {
        f[s] = max(t, f[s]);
      } else {
        g[s] = min(t, g[s]);
      }
    }
    for (int i = n - 1; i > 0; i--) {
      f[i - 1] = max(f[i], f[i - 1]);
    }
    for (int i = 0; i < n - 1; i++) {
      g[i + 1] = min(g[i], g[i + 1]);
    }
    for (int i = 0; i < n; i++) {
      if (g[i] <= f[i]) ng = mid;
    }
    if (ng != mid) ok = mid;
  }
  for (int i = 0; i < q; i++) {
    if (i < ok)
      cout << \"YES\" << endl;
    else
      cout << \"NO\" << endl;
  }
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class F1chessstrikesbackeasyversionbootcamp(Basebootcamp):
    def __init__(self, max_n=3, max_m=3, max_q=5):
        self.max_n = max_n
        self.max_m = max_m
        self.max_q = max_q

    def case_generator(self):
        n = random.randint(1, self.max_n)
        m = random.randint(1, self.max_m)
        
        # 生成所有合法的白色单元格
        white_cells = []
        for i in range(1, 2*n + 1):
            for j in range(1, 2*m + 1):
                if (i + j) % 2 == 0:
                    white_cells.append((i, j))
        
        # 确定最大可能的q值
        max_possible_q = min(len(white_cells), self.max_q)
        if max_possible_q < 1:
            return {'n': n, 'm': m, 'q': 0, 'queries': [], 'answers': []}
        
        q = random.randint(1, max_possible_q)
        random.shuffle(white_cells)
        queries = [[i, j] for i, j in white_cells[:q]]
        
        answers = self.compute_answers(n, m, queries)
        return {
            'n': n,
            'm': m,
            'q': q,
            'queries': queries,
            'answers': answers,
        }

    @staticmethod
    def compute_answers(n, m, queries):
        q = len(queries)
        answers = []
        for k in range(1, q + 1):
            f = [-1] * n
            g = [m] * n
            
            for i, j in queries[:k]:
                s = (i - 1) // 2    # 转换为0-based行号后除2
                t = (j - 1) // 2    # 转换为0-based列号后除2
                # 判断0-based行号的奇偶性
                if (i - 1) % 2 == 1:
                    if t > f[s]:
                        f[s] = t
                else:
                    if t < g[s]:
                        g[s] = t
            
            # 向右传播f的最大值
            for i in range(n-1, 0, -1):
                if f[i] > f[i-1]:
                    f[i-1] = f[i]
            
            # 向左传播g的最小值
            for i in range(n-1):
                if g[i] < g[i+1]:
                    g[i+1] = g[i]
            
            possible = all(g[i] > f[i] for i in range(n))
            answers.append("YES" if possible else "NO")
        return answers

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        q = question_case['q']
        queries = question_case['queries']
        prompt = f"""Ildar和Ivan正在玩一个基于棋盘的策咯游戏。棋盘的大小为{2*n}行×{2*m}列，其中白色单元格（满足i + j为偶数的位置）可以放置国王。游戏开始时，所有白色单元格都是可用的。

Ildar进行了{q}次操作，每次操作将一个特定的白色单元格标记为不可用。每次操作后，你需要判断是否能够在剩余的可用白色单元格中放置{n*m}个国王，使得这些国王两两之间无法相互攻击（即不能位于相邻的单元格，包括上下、左右以及对角线相邻）。

操作记录如下：
"""
        for idx, (i, j) in enumerate(queries, 1):
            prompt += f"操作{idx}：标记单元格 ({i}, {j}) 为不可用。\n"
        prompt += "\n请针对每个操作后的棋盘状态，依次输出“YES”或“NO”，每个结果占一行，按操作顺序排列，并确保所有结果被包裹在[answer]和[/answer]标签内。例如：\n\n[answer]\nYES\nNO\nYES\n[/answer]\n\n请确保严格按照上述格式要求作答。"
        return prompt.strip()

    @staticmethod
    def extract_output(output):
        import re
        pattern = r'\[answer\](.*?)\[/answer\]'
        matches = re.findall(pattern, output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        lines = [line.strip().upper() for line in last_match.split('\n') if line.strip()]
        return lines if lines else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity.get('answers', [])
