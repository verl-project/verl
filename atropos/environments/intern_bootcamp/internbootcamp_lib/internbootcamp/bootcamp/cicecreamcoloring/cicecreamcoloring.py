"""# 

### 谜题描述
Isart and Modsart were trying to solve an interesting problem when suddenly Kasra arrived. Breathless, he asked: \"Can you solve a problem I'm stuck at all day?\"

We have a tree T with n vertices and m types of ice cream numerated from 1 to m. Each vertex i has a set of si types of ice cream. Vertices which have the i-th (1 ≤ i ≤ m) type of ice cream form a connected subgraph. We build a new graph G with m vertices. We put an edge between the v-th and the u-th (1 ≤ u, v ≤ m, u ≠ v) vertices in G if and only if there exists a vertex in T that has both the v-th and the u-th types of ice cream in its set. The problem is to paint the vertices of G with minimum possible number of colors in a way that no adjacent vertices have the same color.

Please note that we consider that empty set of vertices form a connected subgraph in this problem.

As usual, Modsart don't like to abandon the previous problem, so Isart wants you to solve the new problem.

Input

The first line contains two integer n and m (1 ≤ n, m ≤ 3·105) — the number of vertices in T and the number of ice cream types.

n lines follow, the i-th of these lines contain single integer si (0 ≤ si ≤ 3·105) and then si distinct integers, each between 1 and m — the types of ice cream in the i-th vertex. The sum of si doesn't exceed 5·105.

n - 1 lines follow. Each of these lines describes an edge of the tree with two integers u and v (1 ≤ u, v ≤ n) — the indexes of connected by this edge vertices.

Output

Print single integer c in the first line — the minimum number of colors to paint the vertices in graph G.

In the second line print m integers, the i-th of which should be the color of the i-th vertex. The colors should be between 1 and c. If there are some answers, print any of them.

Examples

Input

3 3
1 1
2 2 3
1 2
1 2
2 3


Output

2
1 1 2 

Input

4 5
0
1 1
1 3
3 2 4 5
2 1
3 2
4 3


Output

3
1 1 1 2 3 

Note

In the first example the first type of ice cream is present in the first vertex only, so we can color it in any color. The second and the third ice cream are both presented in the second vertex, so we should paint them in different colors.

In the second example the colors of the second, the fourth and the fifth ice cream should obviously be distinct.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int MAXN = 500005;
int N, M, C, ban[MAXN], ans[MAXN], done[MAXN];
set<int> pset;
vector<int> color[MAXN], E[MAXN];
int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  ;
  cin >> N >> M;
  C = 0;
  int rt = 0;
  for (int i = (1); i <= (N); i++) {
    int K;
    cin >> K;
    color[i].resize(K);
    for (int j = 0; j < (K); j++) cin >> color[i][j];
    if (K > C) {
      C = K;
      rt = i;
    }
    sort(begin(color[i]), end(color[i]));
  }
  for (int _ = 0; _ < (N - 1); _++) {
    int u, v;
    cin >> u >> v;
    E[u].push_back(v);
    E[v].push_back(u);
  }
  queue<int> que;
  for (int i = (1); i <= (C); i++) {
    ans[color[rt][i - 1]] = i;
    pset.insert(i);
  }
  done[rt] = 1;
  que.push(rt);
  while (!que.empty()) {
    int u = que.front();
    que.pop();
    for (auto it : color[u]) ban[it] = 1;
    for (auto v : E[u]) {
      if (done[v]) continue;
      done[v] = 1;
      for (auto it : color[v]) {
        if (ban[it]) {
          pset.erase(ans[it]);
          assert(ans[it]);
        }
      }
      for (auto it : color[v]) {
        if (ans[it]) continue;
        assert(!pset.empty());
        ans[it] = *pset.begin();
        pset.erase(pset.begin());
      }
      que.push(v);
      for (auto it : color[v]) {
        pset.insert(ans[it]);
      }
    }
    for (auto it : color[u]) ban[it] = 0;
  }
  for (int i = (1); i <= (M); i++) {
    if (!ans[i]) ans[i] = 1;
  }
  C = max(C, 1);
  cout << C << endl;
  for (int i = (1); i <= (M); i++) cout << ans[i] << \" \n\"[i == M];
  for (int i = (1); i <= (N); i++) {
    int k = ((int)((color[i]).size()));
    for (int j = 0; j < (k - 1); j++)
      assert(ans[color[i][j]] != ans[color[i][j + 1]]);
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from collections import deque
from bootcamp import Basebootcamp

class Cicecreamcoloringbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.default_params = {
            'n_range': (3, 5),
            'm_range': (2, 5),
            'max_type_nodes': 3,
            'prob_assign': 0.8
        }
        self.params = self.default_params.copy()
        self.params.update(params)

    def generate_random_tree(self, n):
        if n == 1:
            return []
        parents = [0] * n
        for i in range(1, n):
            parents[i] = random.randint(0, i-1)
        return [(i+1, parents[i]+1) for i in range(1, n)]

    def bfs_connected_subset(self, adj, start, max_size):
        visited = set([start])
        q = deque([start])
        while q and len(visited) < max_size:
            u = q.popleft()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    q.append(v)
                    if len(visited) == max_size:
                        break
        return list(visited)
    
    def case_generator(self):
        params = self.params
        n = random.randint(*params['n_range'])
        m = random.randint(*params['m_range'])
        edges = self.generate_random_tree(n)
        
        adj = [[] for _ in range(n+1)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        nodes = [[] for _ in range(n+1)]
        
        for ice_type in range(1, m+1):
            if random.random() > params['prob_assign']:
                continue
                
            max_size = random.randint(1, params['max_type_nodes'])
            start_node = random.randint(1, n)
            connected_nodes = self.bfs_connected_subset(adj, start_node, max_size)
            
            for node in connected_nodes:
                nodes[node].append(ice_type)
        
        nodes_data = []
        for i in range(1, n+1):
            types = sorted(nodes[i])
            nodes_data.append({'types': types})
        
        total_si = sum(len(nd['types']) for nd in nodes_data)
        if total_si > 5*10**5:
            return self.case_generator()
        
        return {
            'n': n,
            'm': m,
            'nodes': nodes_data,
            'edges': edges
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        nodes = question_case['nodes']
        edges = question_case['edges']
        
        input_lines = [f"{n} {m}"]
        for node in nodes:
            si = len(node['types'])
            if si == 0:
                input_lines.append("0")
            else:
                input_lines.append(f"{si} {' '.join(map(str, node['types']))}")
        for u, v in edges:
            input_lines.append(f"{u} {v}")
        
        # 显式生成input_str避免转义问题
        input_str = '\n'.join(input_lines)
        
        return f"""请解决以下冰淇淋着色问题：

输入：
{input_str}

输出要求：
1. 第一行输出最小颜色数c
2. 第二行输出m个颜色值（空格分隔）
答案请按如下格式包裹在[answer]标签内：
[answer]
c值
颜色列表
[/answer]"""

    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        
        last_answer = answer_blocks[-1].strip()
        lines = [line.strip() for line in last_answer.split('\n') if line.strip()]
        
        try:
            c = int(lines[0])
            colors = list(map(int, lines[1].split()))
            return {'c': c, 'colors': colors}
        except (IndexError, ValueError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or 'colors' not in solution:
            return False
        colors = solution['colors']
        m = identity['m']
        nodes = identity['nodes']
        
        if len(colors) != m:
            return False
        
        max_si = max(len(node['types']) for node in nodes)
        correct_c = max(max_si, 1)
        if solution.get('c') != correct_c:
            return False
        
        if any(c < 1 or c > correct_c for c in colors):
            return False
        
        edge_set = set()
        for node in nodes:
            types = node['types']
            for i in range(len(types)):
                for j in range(i+1, len(types)):
                    u, v = types[i], types[j]
                    edge_set.add((u-1, v-1))
                    edge_set.add((v-1, u-1))
        
        for u, v in edge_set:
            if colors[u] == colors[v]:
                return False
        return True
