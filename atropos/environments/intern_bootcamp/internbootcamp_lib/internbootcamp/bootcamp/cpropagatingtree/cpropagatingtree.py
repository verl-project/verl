"""# 

### 谜题描述
Iahub likes trees very much. Recently he discovered an interesting tree named propagating tree. The tree consists of n nodes numbered from 1 to n, each node i having an initial value ai. The root of the tree is node 1.

This tree has a special property: when a value val is added to a value of node i, the value -val is added to values of all the children of node i. Note that when you add value -val to a child of node i, you also add -(-val) to all children of the child of node i and so on. Look an example explanation to understand better how it works.

This tree supports two types of queries:

  * \"1 x val\" — val is added to the value of node x; 
  * \"2 x\" — print the current value of node x. 



In order to help Iahub understand the tree better, you must answer m queries of the preceding type.

Input

The first line contains two integers n and m (1 ≤ n, m ≤ 200000). The second line contains n integers a1, a2, ..., an (1 ≤ ai ≤ 1000). Each of the next n–1 lines contains two integers vi and ui (1 ≤ vi, ui ≤ n), meaning that there is an edge between nodes vi and ui.

Each of the next m lines contains a query in the format described above. It is guaranteed that the following constraints hold for all queries: 1 ≤ x ≤ n, 1 ≤ val ≤ 1000.

Output

For each query of type two (print the value of node x) you must print the answer to the query on a separate line. The queries must be answered in the order given in the input.

Examples

Input

5 5
1 2 1 1 2
1 2
1 3
2 4
2 5
1 2 3
1 1 2
2 1
2 2
2 4


Output

3
3
0

Note

The values of the nodes are [1, 2, 1, 1, 2] at the beginning.

Then value 3 is added to node 2. It propagates and value -3 is added to it's sons, node 4 and node 5. Then it cannot propagate any more. So the values of the nodes are [1, 5, 1, - 2, - 1].

Then value 2 is added to node 1. It propagates and value -2 is added to it's sons, node 2 and node 3. From node 2 it propagates again, adding value 2 to it's sons, node 4 and node 5. Node 3 has no sons, so it cannot propagate from there. The values of the nodes are [3, 3, - 1, 0, 1].

You can see all the definitions about the tree at the following link: http://en.wikipedia.org/wiki/Tree_(graph_theory)

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int MAX = 2e5 + 7;
vector<int> tree[MAX];
int idx[MAX];
int child[MAX];
int parity[MAX];
bool vst[MAX];
vector<int> euler;
void dfs(int node, int h = 0) {
  vst[node] = true;
  parity[node] = h & 1;
  idx[node] = euler.size();
  euler.push_back(node);
  child[node] = 1;
  for (int to : tree[node]) {
    if (!vst[to]) {
      dfs(to, h + 1);
      child[node] += child[to];
    }
  }
}
long long bit[2][MAX];
inline void add(int p, int x, long long val) {
  for (int i = x; i < MAX; i += (i & -i)) bit[p][i] += val;
}
inline long long sum(int p, int x) {
  long long res = 0;
  for (int i = x; i > 0; i -= (i & -i)) res += bit[p][i];
  return res;
}
int ar[MAX];
int main() {
  ios ::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  int n, q;
  cin >> n >> q;
  for (int i = 1; i <= n; ++i) cin >> ar[i];
  for (int i = 1; i <= n - 1; ++i) {
    int u, v;
    cin >> u >> v;
    tree[u].push_back(v);
    tree[v].push_back(u);
  }
  euler.push_back(-1);
  dfs(1);
  while (q--) {
    int t;
    cin >> t;
    if (t == 1) {
      int x, val;
      cin >> x >> val;
      add(parity[x], idx[x], val);
      add(parity[x], idx[x] + child[x], -val);
      add(1 ^ parity[x], idx[x], -val);
      add(1 ^ parity[x], idx[x] + child[x], val);
    } else {
      int x;
      cin >> x;
      cout << ar[x] + sum(parity[x], idx[x]) << '\n';
    }
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # 1-based indexing

    def update_point(self, idx, delta):
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx

    def query_prefix(self, idx):
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

    def update_range(self, l, r, delta):
        self.update_point(l, delta)
        self.update_point(r + 1, -delta)

class Cpropagatingtreebootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', 5)
        self.m = params.get('m', 5)
        self.max_val = params.get('max_val', 1000)
        self.max_query_val = params.get('max_query_val', 1000)

    def generate_tree(self, n):
        if n == 1:
            return []
        edges = []
        nodes = [1]
        for i in range(2, n + 1):
            parent = random.choice(nodes)
            edges.append((parent, i))
            nodes.append(i)
        return edges

    def case_generator(self):
        n, m = self.n, self.m
        a = [random.randint(1, self.max_val) for _ in range(n)]
        edges = self.generate_tree(n)
        queries = []
        for _ in range(m):
            if random.random() < 0.3:
                x = random.randint(1, n)
                queries.append(('2', x))
            else:
                x = random.randint(1, n)
                val = random.randint(1, self.max_query_val)
                queries.append(('1', x, val))
        
        case = {
            'n': n,
            'm': m,
            'a': a,
            'edges': edges,
            'queries': queries
        }
        expected = self.simulate_case(case)
        case['expected_outputs'] = expected
        return case

    def simulate_case(self, case):
        n, a = case['n'], case['a']
        edges, queries = case['edges'], case['queries']
        tree = [[] for _ in range(n + 1)]
        for u, v in edges:
            tree[u].append(v)
            tree[v].append(u)

        # Euler Tour初始化
        euler = [-1]
        idx = [0] * (n + 1)
        child = [0] * (n + 1)
        parity = [0] * (n + 1)
        vst = [False] * (n + 1)

        def dfs(u, depth):
            vst[u] = True
            parity[u] = depth % 2
            idx[u] = len(euler)
            euler.append(u)
            child[u] = 1
            for v in tree[u]:
                if not vst[v] and v != u:
                    dfs(v, depth + 1)
                    child[u] += child[v]

        dfs(1, 0)
        max_size = len(euler) - 1

        # 初始化两个BIT
        bit0 = FenwickTree(max_size)
        bit1 = FenwickTree(max_size)
        expected = []

        # 处理查询
        for query in queries:
            if query[0] == '1':
                x = int(query[1])
                val = int(query[2])
                p = parity[x]
                L = idx[x]
                R = L + child[x] - 1  # 闭区间

                if p == 0:
                    bit0.update_range(L, R, val)
                    bit1.update_range(L, R, -val)
                else:
                    bit1.update_range(L, R, val)
                    bit0.update_range(L, R, -val)
            else:
                x = int(query[1])
                p = parity[x]
                sum_p = bit0.query_prefix(idx[x]) if p == 0 else bit1.query_prefix(idx[x])
                expected.append(a[x-1] + sum_p)
        return expected

    @staticmethod
    def prompt_func(question_case):
        input_lines = [
            f"{question_case['n']} {question_case['m']}",
            ' '.join(map(str, question_case['a']))
        ]
        input_lines.extend(f"{u} {v}" for u, v in question_case['edges'])
        input_lines.extend(' '.join(map(str, q)) for q in question_case['queries'])
        input_str = '\n'.join(input_lines)
        return f"""Solve the propagating tree problem. Process all queries and output the results for type 2 queries. Enclose your answers within [answer] and [/answer]. Here is the input:

{input_str}"""

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
        return solution == identity.get('expected_outputs', [])
