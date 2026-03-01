"""# 

### 谜题描述
You are given a graph with 3 ⋅ n vertices and m edges. You are to find a matching of n edges, or an independent set of n vertices.

A set of edges is called a matching if no two edges share an endpoint.

A set of vertices is called an independent set if no two vertices are connected with an edge.

Input

The first line contains a single integer T ≥ 1 — the number of graphs you need to process. The description of T graphs follows.

The first line of description of a single graph contains two integers n and m, where 3 ⋅ n is the number of vertices, and m is the number of edges in the graph (1 ≤ n ≤ 10^{5}, 0 ≤ m ≤ 5 ⋅ 10^{5}).

Each of the next m lines contains two integers v_i and u_i (1 ≤ v_i, u_i ≤ 3 ⋅ n), meaning that there is an edge between vertices v_i and u_i.

It is guaranteed that there are no self-loops and no multiple edges in the graph.

It is guaranteed that the sum of all n over all graphs in a single test does not exceed 10^{5}, and the sum of all m over all graphs in a single test does not exceed 5 ⋅ 10^{5}.

Output

Print your answer for each of the T graphs. Output your answer for a single graph in the following format.

If you found a matching of size n, on the first line print \"Matching\" (without quotes), and on the second line print n integers — the indices of the edges in the matching. The edges are numbered from 1 to m in the input order.

If you found an independent set of size n, on the first line print \"IndSet\" (without quotes), and on the second line print n integers — the indices of the vertices in the independent set.

If there is no matching and no independent set of the specified size, print \"Impossible\" (without quotes).

You can print edges and vertices in any order.

If there are several solutions, print any. In particular, if there are both a matching of size n, and an independent set of size n, then you should print exactly one of such matchings or exactly one of such independent sets.

Example

Input


4
1 2
1 3
1 2
1 2
1 3
1 2
2 5
1 2
3 1
1 4
5 1
1 6
2 15
1 2
1 3
1 4
1 5
1 6
2 3
2 4
2 5
2 6
3 4
3 5
3 6
4 5
4 6
5 6


Output


Matching
2
IndSet
1
IndSet
2 4
Matching
1 15

Note

The first two graphs are same, and there are both a matching of size 1 and an independent set of size 1. Any of these matchings and independent sets is a correct answer.

The third graph does not have a matching of size 2, however, there is an independent set of size 2. Moreover, there is an independent set of size 5: 2 3 4 5 6. However such answer is not correct, because you are asked to find an independent set (or matching) of size exactly n.

The fourth graph does not have an independent set of size 2, but there is a matching of size 2.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const long long N = 3e5 + 5, inf = 1e18 + 100;
vector<long long> g[N];
long long used[N];
void solve() {
  long long n, m;
  cin >> n >> m;
  for (long long i = 1; i <= 3 * n; ++i) g[i].clear(), used[i] = 0;
  vector<long long> seq;
  for (long long i = 1; i <= m; ++i) {
    long long u, v;
    cin >> u >> v;
    if (!used[u] && !used[v]) used[u] = 1, used[v] = 1, seq.push_back(i);
  }
  if (seq.size() >= n) {
    cout << \"Matching\" << '\n';
    for (long long i = 0; i < n; ++i) cout << seq[i] << ' ';
    cout << '\n';
    return;
  }
  seq.clear();
  for (long long i = 1; i <= 3 * n; ++i)
    if (!used[i]) seq.push_back(i);
  if (seq.size() >= n) {
    cout << \"IndSet\" << '\n';
    for (long long i = 0; i < n; ++i) cout << seq[i] << ' ';
    cout << '\n';
    return;
  }
  cout << \"Impossible\" << '\n';
}
signed main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  long long q;
  cin >> q;
  while (q--) solve();
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Ematchingvsindependentsetbootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_m=50):
        self.max_n = max_n
        self.max_m = max_m
    
    def case_generator(self):
        solution_type = random.choice(['Matching', 'IndSet'])
        n = random.randint(1, self.max_n)
        
        if solution_type == 'Matching':
            vertices = list(range(1, 3 * n + 1))
            random.shuffle(vertices)
            core_edges = [(vertices[i*2], vertices[i*2+1]) for i in range(n)]
            
            remaining = list(set(vertices) - {u for edge in core_edges for u in edge})
            other_edges = []
            for _ in range(min(self.max_m - n, (len(remaining)*(len(remaining)-1))//2)):
                if len(remaining) < 2: break
                u, v = random.sample(remaining, 2)
                other_edges.append((u, v))
            
            edges = core_edges + other_edges
        else:
            ind_set = list(range(2*n + 1, 3*n + 1))
            vertices = list(range(1, 2*n + 1))
            random.shuffle(vertices)
            
            core_edges = [(vertices[i*2], vertices[i*2+1]) for i in range(n-1)]
            used_vertices = {u for edge in core_edges for u in edge}
            
            remaining = [v for v in vertices if v not in used_vertices]
            other_edges = []
            for _ in range(self.max_m - (n-1)):
                if len(remaining) < 2: break
                u = random.choice(remaining)
                candidates = [v for v in remaining if v != u]
                if not candidates: break
                v = random.choice(candidates)
                other_edges.append((u, v))
            
            edges = core_edges + other_edges
            edges = [e for e in edges if not (e[0] in ind_set and e[1] in ind_set)]
        
        edges = list({tuple(sorted(e)) for e in edges if e[0] != e[1]})[:self.max_m]
        return {'n': n, 'm': len(edges), 'edges': edges}
    
    @staticmethod
    def prompt_func(question_case):
        edge_list = '\n'.join([f"{i+1}: {u} {v}" for i, (u, v) in enumerate(question_case['edges'])])
        return f"""Given a graph with {3*question_case['n']} vertices and {question_case['m']} edges. 
Find either:
1. {question_case['n']} edge matching (no shared vertices) OR 
2. {question_case['n']} vertex independent set (no connected vertices)

Formatted answer required between [answer] and [/answer]:
[answer]
Matching
1 3  # Example edge indices
[/answer] or
[answer]
IndSet
2 4  # Example vertex numbers
[/answer]

Current graph:
{edge_list}"""

    @staticmethod
    def extract_output(output):
        answer_block = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_block: return None
        lines = [l.strip() for l in answer_block[-1].strip().split('\n') if l.strip()]
        try:
            return {
                'type': lines[0], 
                'elements': list(map(int, lines[1].split()))
            } if len(lines)>=2 else None
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or not solution.get('elements'): return False
        
        n = identity['n']
        elements = solution['elements']
        if len(elements) != n: return False
        
        if solution['type'] == 'Matching':
            edges = identity['edges']
            seen = set()
            for idx in elements:
                if not (1 <= idx <= len(edges)): return False
                u, v = edges[idx-1]
                if u in seen or v in seen: return False
                seen.update({u, v})
            return True
        
        elif solution['type'] == 'IndSet':
            vertices = set(elements)
            if len(vertices) != n: return False
            for u, v in identity['edges']:
                if u in vertices and v in vertices: return False
            return True
        
        return False
