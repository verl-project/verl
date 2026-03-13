"""# 

### 谜题描述
Since Boboniu finished building his Jianghu, he has been doing Kungfu on these mountains every day. 

Boboniu designs a map for his n mountains. He uses n-1 roads to connect all n mountains. Every pair of mountains is connected via roads.

For the i-th mountain, Boboniu estimated the tiredness of doing Kungfu on the top of it as t_i. He also estimated the height of each mountain as h_i.

A path is a sequence of mountains M such that for each i (1 ≤ i < |M|), there exists a road between M_i and M_{i+1}. Boboniu would regard the path as a challenge if for each i (1≤ i<|M|), h_{M_i}≤ h_{M_{i+1}}.

Boboniu wants to divide all n-1 roads into several challenges. Note that each road must appear in exactly one challenge, but a mountain may appear in several challenges. 

Boboniu wants to minimize the total tiredness to do all the challenges. The tiredness of a challenge M is the sum of tiredness of all mountains in it, i.e. ∑_{i=1}^{|M|}t_{M_i}. 

He asked you to find the minimum total tiredness. As a reward for your work, you'll become a guardian in his Jianghu.

Input

The first line contains a single integer n (2 ≤ n ≤ 2 ⋅ 10^5), denoting the number of the mountains.

The second line contains n integers t_1, t_2, …, t_n (1 ≤ t_i ≤ 10^6), denoting the tiredness for Boboniu to do Kungfu on each mountain.

The third line contains n integers h_1, h_2, …, h_n (1 ≤ h_i ≤ 10^6), denoting the height of each mountain.

Each of the following n - 1 lines contains two integers u_i, v_i (1 ≤ u_i, v_i ≤ n, u_i ≠ v_i), denoting the ends of the road. It's guaranteed that all mountains are connected via roads.

Output

Print one integer: the smallest sum of tiredness of all challenges.

Examples

Input


5
40 10 30 50 20
2 3 2 3 1
1 2
1 3
2 4
2 5


Output


160


Input


5
1000000 1 1 1 1
1000000 1 1 1 1
1 2
1 3
1 4
1 5


Output


4000004


Input


10
510916 760492 684704 32545 484888 933975 116895 77095 127679 989957
402815 705067 705067 705067 623759 103335 749243 138306 138306 844737
1 2
3 2
4 3
1 5
6 4
6 7
8 7
8 9
9 10


Output


6390572

Note

For the first example:

<image>

In the picture, the lighter a point is, the higher the mountain it represents. One of the best divisions is:

  * Challenge 1: 3 → 1 → 2 
  * Challenge 2: 5 → 2 → 4 



The total tiredness of Boboniu is (30 + 40 + 10) + (20 + 10 + 50) = 160. It can be shown that this is the minimum total tiredness.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
using lint = long long;
using pi = pair<lint, lint>;
const lint inf = 1e12;
const int MAXN = 500005;
vector<int> gph[MAXN];
lint up[MAXN], dn[MAXN];
lint t[MAXN], h[MAXN];
void dfs(int x, int p) {
  vector<pi> v;
  lint tot = 0;
  lint sum = 0;
  for (auto &i : gph[x]) {
    if (i != p) {
      dfs(i, x);
      if (h[i] > h[x]) up[i] = -inf;
      if (h[i] < h[x]) dn[i] = -inf;
      v.emplace_back(up[i], dn[i]);
      sum += up[i];
    }
  }
  sort((v).begin(), (v).end(), [&](const pi &a, const pi &b) {
    return a.second - a.first > b.second - b.first;
  });
  up[x] = dn[x] = -inf;
  {
    lint foo = sum;
    int in = ((int)(gph[x]).size()) - 1, out = 1;
    up[x] = max(up[x], foo + min(in, out) * t[x]);
    for (auto &i : v) {
      foo += i.second - i.first;
      in--;
      out++;
      up[x] = max(up[x], foo + min(in, out) * t[x]);
    }
  }
  {
    lint foo = sum;
    int in = ((int)(gph[x]).size()), out = 0;
    dn[x] = max(dn[x], foo + min(in, out) * t[x]);
    for (auto &i : v) {
      foo += i.second - i.first;
      in--;
      out++;
      dn[x] = max(dn[x], foo + min(in, out) * t[x]);
    }
  }
}
lint solve() {
  vector<pi> v;
  lint sum = 0;
  int x = 1;
  for (auto &i : gph[1]) {
    dfs(i, 1);
    if (h[i] > h[x]) up[i] = -inf;
    if (h[i] < h[x]) dn[i] = -inf;
    v.emplace_back(up[i], dn[i]);
    sum += up[i];
  }
  sort((v).begin(), (v).end(), [&](const pi &a, const pi &b) {
    return a.second - a.first > b.second - b.first;
  });
  lint foo = sum;
  int in = ((int)(gph[x]).size()), out = 0;
  lint dap = -inf;
  dap = max(dap, foo + min(in, out) * t[x]);
  for (auto &i : v) {
    foo += i.second - i.first;
    in--;
    out++;
    dap = max(dap, foo + min(in, out) * t[x]);
  }
  return dap;
}
int main() {
  int n;
  scanf(\"%d\", &n);
  for (int i = 1; i <= n; i++) scanf(\"%lld\", &t[i]);
  for (int i = 1; i <= n; i++) scanf(\"%lld\", &h[i]);
  for (int i = 1; i < n; i++) {
    int u, v;
    scanf(\"%d %d\", &u, &v);
    gph[u].push_back(v);
    gph[v].push_back(u);
  }
  lint ret = 0;
  for (int i = 1; i <= n; i++) {
    ret += 1ll * ((int)(gph[i]).size()) * t[i];
  }
  cout << ret - solve() << endl;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from math import inf

class Dboboniuandjianghubootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = params
        self.min_n = params.get('min_n', 2)
        self.max_n = params.get('max_n', 5)
        self.min_t = params.get('min_t', 1)
        self.max_t = params.get('max_t', 1e6)
        self.min_h = params.get('min_h', 1)
        self.max_h = params.get('max_h', 1e6)

    def case_generator(self):
        while True:
            try:
                n = random.randint(self.min_n, self.max_n)
                edges = []
                nodes = list(range(1, n+1))
                random.shuffle(nodes)
                parent_map = {}
                for i in range(1, n):
                    parent = nodes[random.randint(0, i-1)]
                    child = nodes[i]
                    edges.append((parent, child))
                
                # Convert to undirected graph representation
                undirected_edges = []
                for u, v in edges:
                    undirected_edges.append((u, v))
                    undirected_edges.append((v, u))
                
                h_list = [random.randint(self.min_h, self.max_h) for _ in range(n)]
                t_list = [random.randint(self.min_t, self.max_t) for _ in range(n)]
                
                # Calculate correct answer
                correct_answer = self.calculate_min_total(
                    n, t_list, h_list, undirected_edges
                )
                
                return {
                    'n': n,
                    't': t_list,
                    'h': h_list,
                    'edges': [(u, v) for u, v in edges],  # return original directed edges
                    'correct_answer': correct_answer
                }
            except Exception as e:
                continue

    @staticmethod
    def prompt_func(question_case):
        edges_str = '\n'.join(f"{u} {v}" for u, v in question_case['edges'])
        return f"""Boboniu's Jianghu Challenge:
Given {question_case['n']} mountains with:
Tiredness values: {', '.join(map(str, question_case['t']))}
Height values: {', '.join(map(str, question_case['h']))}
Connected by roads:
{edges_str}

Divide all roads into non-decreasing height paths to minimize total tiredness. 
Provide your answer within [answer] tags like: [answer]160[/answer]"""

    @staticmethod
    def extract_output(output):
        answers = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return answers[-1].strip() if answers else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return int(solution.strip()) == identity['correct_answer']
        except:
            return False

    @staticmethod
    def dfs(x, p, up, dn, gph, h, t):
        v = []
        sum_val = 0
        for i in gph[x]:
            if i != p:
                Dboboniuandjianghubootcamp.dfs(i, x, up, dn, gph, h, t)
                if h[i] > h[x]:
                    up[i] = -inf
                if h[i] < h[x]:
                    dn[i] = -inf
                v.append((up[i], dn[i]))
                sum_val += up[i]
        
        v.sort(key=lambda a: (a[1]-a[0]), reverse=True)
        
        # Calculate up[x]
        up[x] = -inf
        foo = sum_val
        in_degree = len(gph[x]) - 1
        out_degree = 1
        up[x] = max(up[x], foo + min(in_degree, out_degree)*t[x])
        for a, b in v:
            foo += (b - a)
            in_degree -= 1
            out_degree += 1
            up[x] = max(up[x], foo + min(in_degree, out_degree)*t[x])
        
        # Calculate dn[x]
        dn[x] = -inf
        foo = sum_val
        in_degree = len(gph[x])
        out_degree = 0
        dn[x] = max(dn[x], foo + min(in_degree, out_degree)*t[x])
        for a, b in v:
            foo += (b - a)
            in_degree -= 1
            out_degree += 1
            dn[x] = max(dn[x], foo + min(in_degree, out_degree)*t[x])

    def calculate_min_total(self, n, t_list, h_list, edges):
        h = [0]*(n+2)
        t = [0]*(n+2)
        for i in range(n):
            h[i+1] = h_list[i]
            t[i+1] = t_list[i]
        
        # Build adjacency list
        gph = [[] for _ in range(n+2)]
        for u, v in edges:
            if v not in gph[u]:
                gph[u].append(v)
        
        total = sum(t[i] * len(gph[i]) for i in range(1, n+1))
        
        up = [-inf]*(n+2)
        dn = [-inf]*(n+2)
        
        # Special handling for root node
        v_list = []
        sum_val = 0
        root = 1
        for neighbor in gph[root]:
            if neighbor == root: 
                continue
            Dboboniuandjianghubootcamp.dfs(neighbor, root, up, dn, gph, h, t)
            if h[neighbor] > h[root]:
                up[neighbor] = -inf
            if h[neighbor] < h[root]:
                dn[neighbor] = -inf
            v_list.append((up[neighbor], dn[neighbor]))
            sum_val += up[neighbor]
        
        v_list.sort(key=lambda a: (a[1]-a[0]), reverse=True)
        
        max_val = -inf
        foo = sum_val
        in_degree = len(gph[root])
        out_degree = 0
        max_val = foo + min(in_degree, out_degree)*t[root]
        
        for a, b in v_list:
            foo += (b - a)
            in_degree -= 1
            out_degree += 1
            max_val = max(max_val, foo + min(in_degree, out_degree)*t[root])
        
        return total - max_val
