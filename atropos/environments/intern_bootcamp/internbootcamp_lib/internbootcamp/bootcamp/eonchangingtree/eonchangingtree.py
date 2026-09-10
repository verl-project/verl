"""# 

### 谜题描述
You are given a rooted tree consisting of n vertices numbered from 1 to n. The root of the tree is a vertex number 1.

Initially all vertices contain number 0. Then come q queries, each query has one of the two types:

  * The format of the query: 1 v x k. In response to the query, you need to add to the number at vertex v number x; to the numbers at the descendants of vertex v at distance 1, add x - k; and so on, to the numbers written in the descendants of vertex v at distance i, you need to add x - (i·k). The distance between two vertices is the number of edges in the shortest path between these vertices. 
  * The format of the query: 2 v. In reply to the query you should print the number written in vertex v modulo 1000000007 (109 + 7). 



Process the queries given in the input.

Input

The first line contains integer n (1 ≤ n ≤ 3·105) — the number of vertices in the tree. The second line contains n - 1 integers p2, p3, ... pn (1 ≤ pi < i), where pi is the number of the vertex that is the parent of vertex i in the tree.

The third line contains integer q (1 ≤ q ≤ 3·105) — the number of queries. Next q lines contain the queries, one per line. The first number in the line is type. It represents the type of the query. If type = 1, then next follow space-separated integers v, x, k (1 ≤ v ≤ n; 0 ≤ x < 109 + 7; 0 ≤ k < 109 + 7). If type = 2, then next follows integer v (1 ≤ v ≤ n) — the vertex where you need to find the value of the number.

Output

For each query of the second type print on a single line the number written in the vertex from the query. Print the number modulo 1000000007 (109 + 7).

Examples

Input

3
1 1
3
1 1 2 1
2 1
2 2


Output

2
1

Note

You can read about a rooted tree here: http://en.wikipedia.org/wiki/Tree_(graph_theory).

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
const int N = 300200;
using namespace std;
int n;
int tim;
int tin[N];
int tout[N];
int dep[N];
int mod = 1e9 + 7;
pair<int, int> t[4 * N];
vector<int> v[N];
void dfs(int x, int g) {
  tin[x] = tout[x] = ++tim;
  dep[tin[x]] = g;
  for (auto y : v[x]) {
    dfs(y, g - 1);
    tout[x] = tout[y];
  }
}
void push(int x) {
  t[x * 2].first += t[x].first;
  t[x * 2].second += t[x].second;
  t[x * 2 + 1].first += t[x].first;
  t[x * 2 + 1].second += t[x].second;
  t[x * 2].first %= mod;
  t[x * 2].second %= mod;
  t[x * 2 + 1].first %= mod;
  t[x * 2 + 1].second %= mod;
  t[x].first = t[x].second = 0;
}
void upd(int x, int l, int r, int tl, int tr, int f1, int f2) {
  if (tl > tr) return;
  if (l == tl && r == tr) {
    t[x].first = (t[x].first + f1) % mod;
    t[x].second = (t[x].second + f2) % mod;
    return;
  }
  push(x);
  int m = (l + r) / 2;
  upd(x * 2, l, m, tl, min(m, tr), f1, f2);
  upd(x * 2 + 1, m + 1, r, max(m + 1, tl), tr, f1, f2);
}
int get(int x, int l, int r, int g) {
  if (l == r) {
    int ans = (1ll * t[x].second * dep[l]) % mod;
    ans = (t[x].first + ans) % mod;
    return ans;
  }
  push(x);
  int m = (l + r) / 2;
  if (g <= m)
    return get(x * 2, l, m, g);
  else
    return get(x * 2 + 1, m + 1, r, g);
}
int main() {
  ios_base::sync_with_stdio(0);
  scanf(\"%d\", &n);
  for (int i = 2; i <= n; i++) {
    int x;
    scanf(\"%d\", &x);
    v[x].push_back(i);
  }
  dfs(1, n);
  int q;
  scanf(\"%d\", &q);
  for (int i = 1; i <= q; i++) {
    int t, v, x, k;
    scanf(\"%d\", &t);
    if (t == 1) {
      scanf(\"%d%d%d\", &v, &x, &k);
      long long f = 1ll * x - 1ll * dep[tin[v]] * k;
      f = f % mod + mod;
      upd(1, 1, n, tin[v], tout[v], f % mod, k);
    } else {
      scanf(\"%d\", &v);
      printf(\"%d\n\", get(1, 1, n, tin[v]));
    }
  }
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

mod = 10**9 + 7

def build_adj(parents, n):
    adj = {i: [] for i in range(1, n+1)}
    for i in range(2, n+1):
        parent = parents[i-2]
        adj[parent].append(i)
    return adj

def dfs(x, parent_adj, tin, tout, dep, current_time, current_g):
    current_time[0] += 1
    tin[x] = current_time[0]
    dep[tin[x]] = current_g
    for child in parent_adj.get(x, []):
        dfs(child, parent_adj, tin, tout, dep, current_time, current_g-1)
    tout[x] = current_time[0]

def perform_dfs(n, parent_adj):
    tin = [0] * (n + 1)
    tout = [0] * (n + 1)
    dep = [0] * (n + 2)  # tin values are 1-based
    current_time = [0]
    dfs(1, parent_adj, tin, tout, dep, current_time, n)
    return tin, tout, dep

def process_queries_for_identity(queries, n, tin_dict, tout_dict, dep_list):
    a = [0] * (n + 2)
    b = [0] * (n + 2)
    expected_outputs = []
    for query in queries:
        if query['type'] == 1:
            v = query['v']
            x = query['x']
            k = query['k']
            tin_v = tin_dict[v]
            tout_v = tout_dict[v]
            f1 = (x - dep_list[tin_v] * k) % mod
            f2 = k % mod
            for u in range(1, n+1):
                u_tin = tin_dict[u]
                if tin_v <= u_tin <= tout_v:
                    a[u_tin] = (a[u_tin] + f1) % mod
                    b[u_tin] = (b[u_tin] + f2) % mod
        else:
            v = query['v']
            u_tin = tin_dict[v]
            res = (a[u_tin] + b[u_tin] * dep_list[u_tin]) % mod
            expected_outputs.append(res)
    return expected_outputs

class Eonchangingtreebootcamp(Basebootcamp):
    def __init__(self, max_n=5, max_q=5):
        self.max_n = max_n
        self.max_q = max_q
    
    def case_generator(self):
        n = random.randint(1, self.max_n)
        parents = []
        if n > 1:
            parents = [random.randint(1, i-1) for i in range(2, n+1)]
        adj = build_adj(parents, n)
        tin, tout, dep = perform_dfs(n, adj)
        tin_dict = {x: tin[x] for x in range(1, n+1)}
        tout_dict = {x: tout[x] for x in range(1, n+1)}
        q = random.randint(1, self.max_q)
        queries = []
        for _ in range(q):
            if random.random() < 0.3 or not any(q.get('type') == 2 for q in queries):
                v = random.randint(1, n)
                queries.append({'type': 2, 'v': v})
            else:
                v = random.randint(1, n)
                x = random.randint(0, mod-1)
                k = random.randint(0, mod-1)
                queries.append({'type': 1, 'v': v, 'x': x, 'k': k})
        expected_outputs = process_queries_for_identity(queries, n, tin_dict, tout_dict, dep)
        new_queries = []
        output_idx = 0
        for q in queries:
            if q['type'] == 2:
                new_q = q.copy()
                new_q['expected'] = expected_outputs[output_idx]
                new_queries.append(new_q)
                output_idx += 1
            else:
                new_queries.append(q)
        return {
            'n': n,
            'parents': parents,
            'queries': new_queries,
            'tin_dict': tin_dict,
            'tout_dict': tout_dict,
            'dep_list': dep
        }
    
    @staticmethod
    def prompt_func(question_case):
        input_lines = [
            str(question_case['n']),
            ' '.join(map(str, question_case['parents'])) if question_case['n'] > 1 else '',
            str(len(question_case['queries']))
        ]
        for q in question_case['queries']:
            if q['type'] == 1:
                input_lines.append(f"1 {q['v']} {q['x']} {q['k']}")
            else:
                input_lines.append(f"2 {q['v']}")
        input_str = '\n'.join(input_lines)
        return f"""你正在解决一个树处理问题。处理所有查询并将每个类型2的答案放在[answer]和[/answer]之间。输入数据如下：

{input_str}

规则：
1. 类型1查询为节点v及其后代按距离加值。
2. 类型2查询输出节点值模10^9+7的结果。

答案格式示例：
[answer]结果1[/answer]
[answer]结果2[/answer]"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        try:
            return [int(m) % mod for m in matches] if matches else None
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = [q['expected'] for q in identity['queries'] if q['type'] == 2]
        return solution == expected if solution else False
