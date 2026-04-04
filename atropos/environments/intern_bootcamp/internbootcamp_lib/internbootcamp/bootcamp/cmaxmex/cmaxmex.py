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
template <typename T>
struct Segtree {
  int n;
  T e;
  vector<T> dat;
  typedef function<T(T a, T b)> Func;
  Func f;
  Segtree() {}
  Segtree(int n_input, Func f_input, T e_input) {
    initialize(n_input, f_input, e_input);
  }
  void initialize(int n_input, Func f_input, T e_input) {
    f = f_input;
    e = e_input;
    n = 1;
    while (n < n_input) n <<= 1;
    dat.resize(2 * n - 1, e);
  }
  void update(int k, T a) {
    k += n - 1;
    dat[k] = a;
    while (k > 0) {
      k = (k - 1) / 2;
      dat[k] = f(dat[2 * k + 1], dat[2 * k + 2]);
    }
  }
  T get(int k) { return dat[k + n - 1]; }
  T between(int a, int b) { return query(a, b + 1, 0, 0, n); }
  T query(int a, int b, int k, int l, int r) {
    if (r <= a || b <= l) return e;
    if (a <= l && r <= b) return dat[k];
    T vl = query(a, b, 2 * k + 1, l, (l + r) / 2);
    T vr = query(a, b, 2 * k + 2, (l + r) / 2, r);
    return f(vl, vr);
  }
  int bisect() {
    if (dat[0].first != -1) return n;
    int ans = 0, pt = 0, gain = n / 2;
    T now = e;
    while (true) {
      auto nxt = f(now, dat[2 * pt + 1]);
      if (nxt.first == -1) {
        pt = 2 * pt + 1;
      } else {
        now = nxt;
        ans += gain;
        pt = 2 * pt + 2;
      }
      gain /= 2;
      if (gain == 0) break;
    }
    return ans;
  }
};
template <typename T>
struct SparseTable {
  vector<vector<T>> dat;
  vector<int> lookup;
  void initialize(const vector<T>& v) {
    int N = v.size();
    int b = 0;
    while ((1 << b) <= N) b++;
    dat.assign(b, vector<T>(1 << b));
    for (int i = 0; i < N; i++) dat[0][i] = v[i];
    for (int i = 1; i < b; i++) {
      for (int j = 0; j + (1 << i) <= (1 << b); j++) {
        dat[i][j] = min(dat[i - 1][j], dat[i - 1][j + (1 << (i - 1))]);
      }
    }
    lookup.resize(N + 1, 0);
    for (int i = 2; i <= N; i++) {
      lookup[i] = lookup[i >> 1] + 1;
    }
  }
  T rmq(int l, int r) {
    r++;
    int b = lookup[r - l];
    return min(dat[b][l], dat[b][r - (1 << b)]);
  }
};
vector<int> children[200000];
int depth[200000];
int dfs_order[200000];
vector<int> euler_tour;
int euler_rev[200000];
void dfs(int i, int p, int& n, int d) {
  dfs_order[i] = n;
  depth[i] = d;
  n++;
  euler_tour.push_back(i);
  for (int j : children[i])
    if (j != p) {
      dfs(j, i, n, d + 1);
      euler_tour.push_back(i);
    }
}
SparseTable<pair<int, int>> spt;
int calc_lca(int i, int j) {
  i = euler_rev[i];
  j = euler_rev[j];
  if (i > j) swap(i, j);
  return spt.rmq(i, j).second;
}
bool isPC(int parent, int child) { return (calc_lca(parent, child) == parent); }
pair<int, int> merge_one(pair<int, int> a, int v) {
  if (a.first == -1 || v == -1) return {-1, -1};
  vector<int> nums = {a.first, a.second, v};
  sort(nums.begin(), nums.end());
  int root = calc_lca(calc_lca(a.first, a.second), v);
  if (nums[0] == root) {
    if (isPC(nums[1], nums[2])) {
      return {nums[0], nums[2]};
    } else if (calc_lca(nums[1], nums[2]) == nums[0]) {
      return {nums[1], nums[2]};
    }
  } else if (isPC(nums[1], nums[2])) {
    return {nums[0], nums[2]};
  } else if (isPC(nums[0], nums[1])) {
    return {nums[1], nums[2]};
  }
  return {-1, -1};
}
pair<int, int> merge_P(pair<int, int> a, pair<int, int> b) {
  if (a.first == -2) return b;
  if (b.first == -2) return a;
  auto ret = merge_one(merge_one(a, b.first), b.second);
  return ret;
}
int main() {
  int N, p[200000], p2[200000];
  cin >> N;
  for (int i = 0; i < N; i++) scanf(\"%d\", &p[i]);
  for (int i = 1; i < N; i++) {
    int par;
    scanf(\"%d\", &par);
    children[par - 1].push_back(i);
  }
  int tmp = 0;
  dfs(0, -1, tmp, 0);
  vector<pair<int, int>> v;
  for (int k = 0; k < euler_tour.size(); k++) {
    int i = euler_tour[k];
    euler_rev[dfs_order[i]] = k;
    v.emplace_back(depth[i], dfs_order[i]);
  }
  spt.initialize(v);
  Segtree<pair<int, int>> st(N, merge_P, {-2, -2});
  for (int i = 0; i < N; i++) {
    int pos = dfs_order[i];
    p2[pos] = p[i];
    st.update(p[i], {pos, pos});
  }
  int Q;
  cin >> Q;
  while (Q--) {
    int t;
    scanf(\"%d\", &t);
    if (t == 1) {
      int i, j;
      scanf(\"%d %d\", &i, &j);
      i = dfs_order[i - 1];
      j = dfs_order[j - 1];
      st.update(p2[i], {-1, -1});
      st.update(p2[j], {-1, -1});
      st.update(p2[i], {j, j});
      st.update(p2[j], {i, i});
      swap(p2[i], p2[j]);
    } else {
      int ans = st.bisect();
      printf(\"%d\n\", ans);
    }
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cmaxmexbootcamp(Basebootcamp):
    def __init__(self, max_n=6, max_q=5, **params):
        super().__init__(**params)
        self.max_n = max_n
        self.max_q = max_q
    
    def case_generator(self):
        n = random.randint(2, self.max_n)
        # Generate tree structure
        d = []
        parent = {}
        for i in range(2, n + 1):
            possible_parents = list(range(1, i))
            di = random.choice(possible_parents)
            d.append(di)
            parent[i] = di
        # Generate permutation p
        p = list(range(n))
        random.shuffle(p)
        # Generate queries
        q = random.randint(1, self.max_q)
        current_p = p.copy()
        queries = []
        for _ in range(q):
            query_type = random.choices([1, 2], weights=[0.3, 0.7], k=1)[0]
            if query_type == 1:
                i = random.randint(1, n)
                j = random.randint(1, n)
                while i == j:
                    j = random.randint(1, n)
                queries.append({'type': 1, 'i': i, 'j': j})
                # Swap p values
                current_p[i-1], current_p[j-1] = current_p[j-1], current_p[i-1]
            else:
                # Compute current max MEX
                max_mex = 0
                for u in range(1, n + 1):
                    for v in range(u, n + 1):
                        path = self.get_path(u, v, parent)
                        values = {current_p[node-1] for node in path}
                        mex = self.compute_mex(values)
                        if mex > max_mex:
                            max_mex = mex
                queries.append({'type': 2, 'answer': max_mex})
        return {
            'n': n,
            'p': p,
            'd': d,
            'queries': queries
        }
    
    @staticmethod
    def get_lca(u, v, parent):
        ancestors = set()
        current = u
        while current is not None:
            ancestors.add(current)
            current = parent.get(current, None)
        current = v
        while current is not None and current not in ancestors:
            current = parent.get(current, None)
        return current if current is not None else 1
    
    @classmethod
    def get_path(cls, u, v, parent):
        lca = cls.get_lca(u, v, parent)
        path_u = []
        current = u
        while current != lca:
            path_u.append(current)
            current = parent.get(current, None)
        path_u.append(lca)
        # Get v to lca path
        path_v = []
        current = v
        while current != lca:
            path_v.append(current)
            current = parent.get(current, None)
        # Combine paths
        return path_u + path_v[::-1]
    
    @staticmethod
    def compute_mex(s):
        mex = 0
        while mex in s:
            mex += 1
        return mex
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        p = ' '.join(map(str, question_case['p']))
        d = ' '.join(map(str, question_case['d']))
        queries = []
        for q in question_case['queries']:
            if q['type'] == 1:
                queries.append(f"1 {q['i']} {q['j']}")
            else:
                queries.append("2")
        input_str = (
            f"{n}\n{p}\n{d}\n{len(queries)}\n" + '\n'.join(queries)
        )
        prompt = f"""给定一个以节点1为根的树，每个节点包含唯一的0到{n-1}的数值。处理以下两种查询：
1. 交换两个节点的数值。
2. 查询所有路径中MEX的最大值。

输入格式：
- 首行：节点数n
- 第二行：各节点的初始数值p_1到p_n
- 第三行：节点2到n的父节点列表
- 第四行：查询数q
- 接下来q行：每个查询的描述（类型1为交换操作，类型2为查询）

请处理所有查询，并将类型2查询的结果按顺序放入[answer]标签中。例如：
[answer]
3
2
[/answer]

输入数据：
{input_str}

请根据上述输入，给出所有类型2查询的结果："""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        answers = []
        for line in last_match.split('\n'):
            stripped = line.strip()
            if stripped:
                answers.append(stripped)
        return answers if answers else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        correct_answers = [q['answer'] for q in identity['queries'] if q['type'] == 2]
        if not solution or len(solution) != len(correct_answers):
            return False
        try:
            user_answers = list(map(int, solution))
            return user_answers == correct_answers
        except ValueError:
            return False
