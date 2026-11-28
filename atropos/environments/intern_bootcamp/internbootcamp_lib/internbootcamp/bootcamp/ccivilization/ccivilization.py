"""# 

### 谜题描述
Andrew plays a game called \"Civilization\". Dima helps him.

The game has n cities and m bidirectional roads. The cities are numbered from 1 to n. Between any pair of cities there either is a single (unique) path, or there is no path at all. A path is such a sequence of distinct cities v1, v2, ..., vk, that there is a road between any contiguous cities vi and vi + 1 (1 ≤ i < k). The length of the described path equals to (k - 1). We assume that two cities lie in the same region if and only if, there is a path connecting these two cities.

During the game events of two types take place:

  1. Andrew asks Dima about the length of the longest path in the region where city x lies. 
  2. Andrew asks Dima to merge the region where city x lies with the region where city y lies. If the cities lie in the same region, then no merging is needed. Otherwise, you need to merge the regions as follows: choose a city from the first region, a city from the second region and connect them by a road so as to minimize the length of the longest path in the resulting region. If there are multiple ways to do so, you are allowed to choose any of them. 



Dima finds it hard to execute Andrew's queries, so he asks you to help him. Help Dima.

Input

The first line contains three integers n, m, q (1 ≤ n ≤ 3·105; 0 ≤ m < n; 1 ≤ q ≤ 3·105) — the number of cities, the number of the roads we already have and the number of queries, correspondingly.

Each of the following m lines contains two integers, ai and bi (ai ≠ bi; 1 ≤ ai, bi ≤ n). These numbers represent the road between cities ai and bi. There can be at most one road between two cities.

Each of the following q lines contains one of the two events in the following format:

  * 1 xi. It is the request Andrew gives to Dima to find the length of the maximum path in the region that contains city xi (1 ≤ xi ≤ n). 
  * 2 xi yi. It is the request Andrew gives to Dima to merge the region that contains city xi and the region that contains city yi (1 ≤ xi, yi ≤ n). Note, that xi can be equal to yi. 

Output

For each event of the first type print the answer on a separate line.

Examples

Input

6 0 6
2 1 2
2 3 4
2 5 6
2 3 2
2 5 3
1 1


Output

4

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int maxn = 3e5;
int n, m;
int f[maxn];
int diameter[maxn];
int find(int x) { return x == f[x] ? x : f[x] = find(f[x]); }
vector<int> G[maxn];
int d1[maxn], d2[maxn];
int vis[maxn], t;
queue<int> q;
int bfs1(int u, int d[]) {
  t++;
  q.push(u);
  d[u] = 0;
  vis[u] = t;
  while (!q.empty()) {
    u = q.front();
    q.pop();
    for (int i = 0; i < G[u].size(); i++) {
      int v = G[u][i];
      if (vis[v] == t) continue;
      q.push(v);
      d[v] = d[u] + 1;
      vis[v] = t;
    }
  }
  return u;
}
int bfs2(int u, int fa, int tag) {
  int ret = -1;
  t++;
  q.push(u);
  d2[u] = 0;
  vis[u] = t;
  while (!q.empty()) {
    u = q.front();
    q.pop();
    f[u] = fa;
    if (d1[u] + d2[u] == tag) {
      if (d1[u] == tag / 2 || d2[u] == tag / 2) ret = u;
    }
    for (int i = 0; i < G[u].size(); i++) {
      int v = G[u][i];
      if (vis[v] == t) continue;
      q.push(v);
      d2[v] = d2[u] + 1;
      vis[v] = t;
    }
  }
  return ret;
}
void bfs3(int u) {
  int fa = u;
  t++;
  q.push(u);
  vis[u] = t;
  while (!q.empty()) {
    u = q.front();
    q.pop();
    f[u] = fa;
    for (int i = 0; i < G[u].size(); i++) {
      int v = G[u][i];
      if (vis[v] == t) continue;
      q.push(v);
      vis[v] = t;
    }
  }
}
int main() {
  int Q;
  scanf(\"%d%d%d\", &n, &m, &Q);
  for (int i = 0; i < m; i++) {
    int a, b;
    scanf(\"%d%d\", &a, &b);
    a--, b--;
    G[a].push_back(b);
    G[b].push_back(a);
  }
  for (int i = 0; i < n; i++)
    if (!vis[i]) {
      int x = bfs1(i, d1);
      int y = bfs1(x, d1);
      int z = bfs2(y, y, d1[y]);
      diameter[z] = d1[z] + d2[z];
      bfs3(z);
    }
  for (int i = 0; i < Q; i++) {
    int op;
    scanf(\"%d\", &op);
    if (op == 1) {
      int x;
      scanf(\"%d\", &x);
      x--;
      x = find(x);
      printf(\"%d\n\", diameter[x]);
    } else {
      int x, y;
      scanf(\"%d%d\", &x, &y);
      x--, y--;
      x = find(x), y = find(y);
      if (x != y) {
        if (diameter[x] > diameter[y]) {
          int t1 = diameter[x] / 2, t2 = diameter[x] - t1;
          int t3 = diameter[y] / 2, t4 = diameter[y] - t3;
          f[y] = x;
          diameter[x] = max(max(diameter[x], diameter[y]), t2 + t4 + 1);
        } else {
          int t1 = diameter[x] / 2, t2 = diameter[x] - t1;
          int t3 = diameter[y] / 2, t4 = diameter[y] - t3;
          f[x] = y;
          diameter[y] = max(max(diameter[x], diameter[y]), t2 + t4 + 1);
        }
      }
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

class Ccivilizationbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__()
        self.params = {
            'n': params.get('n', 6),
            'm': params.get('m', 0),
            'q': params.get('q', 6),
            'max_retries': 10  # 用于生成道路时的重试次数
        }
    
    def case_generator(self):
        n = self.params['n']
        m = self.params['m']
        q = self.params['q']
        max_retries = self.params['max_retries']
        
        parent = list(range(n + 1))  # 1-based indexing
        diameter = [0] * (n + 1)
        
        # 生成 m 条道路，确保形成树结构
        roads = []
        for _ in range(m):
            added = False
            retries = 0
            while not added and retries < max_retries:
                a = random.randint(1, n)
                b = random.randint(1, n)
                if a == b:
                    continue
                root_a = self.find(parent, a)
                root_b = self.find(parent, b)
                if root_a != root_b:
                    roads.append((a, b))
                    # 计算新直径
                    t1_a = diameter[root_a] // 2
                    t2_a = diameter[root_a] - t1_a
                    t1_b = diameter[root_b] // 2
                    t2_b = diameter[root_b] - t1_b
                    new_diam = max(diameter[root_a], diameter[root_b], t2_a + t2_b + 1)
                    # 合并区域
                    if diameter[root_a] > diameter[root_b]:
                        root_a, root_b = root_b, root_a
                    parent[root_a] = root_b
                    diameter[root_b] = new_diam
                    added = True
                retries += 1
        
        # 生成 q 个查询
        queries = []
        results = []
        for _ in range(q):
            op = random.choice([1, 2])
            if op == 1:
                x = random.randint(1, n)
                root = self.find(parent, x)
                current_diam = diameter[root]
                queries.append(('1', x))
                results.append(current_diam)
            else:
                x = random.randint(1, n)
                y = random.randint(1, n)
                queries.append(('2', x, y))
                root_x = self.find(parent, x)
                root_y = self.find(parent, y)
                if root_x != root_y:
                    if diameter[root_x] > diameter[root_y]:
                        root_x, root_y = root_y, root_x
                    t1 = diameter[root_x] // 2
                    t2 = diameter[root_x] - t1
                    t3 = diameter[root_y] // 2
                    t4 = diameter[root_y] - t3
                    new_diam = max(diameter[root_x], diameter[root_y], t2 + t4 + 1)
                    parent[root_x] = root_y
                    diameter[root_y] = new_diam
        
        case = {
            'n': n,
            'm': m,
            'q': q,
            'roads': roads,
            'queries': queries,
            'results': results
        }
        return case
    
    @staticmethod
    def find(parent, x):
        if parent[x] != x:
            parent[x] = Ccivilizationbootcamp.find(parent, parent[x])
        return parent[x]
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        q = question_case['q']
        roads = question_case['roads']
        queries = question_case['queries']
        
        road_descriptions = []
        for a, b in roads:
            road_descriptions.append("城市 {} 和 {} 之间有一条道路。".format(a, b))
        
        query_descriptions = []
        for i, query in enumerate(queries, 1):
            if query[0] == '1':
                query_descriptions.append("{}. 查询城市 {} 所在区域的最长路径长度。".format(i, query[1]))
            else:
                query_descriptions.append("{}. 合并城市 {} 和 {} 所在的区域。".format(i, query[1], query[2]))
        
        prompt = (
            "游戏开始时有 {} 个城市，其中 {} 条道路。道路情况如下：\n"
            "{}\n"
            "\n接下来有 {} 个查询：\n"
            "{}\n"
            "\n对于每个类型1的查询，请输出该区域的最长路径长度。请将所有类型1查询的结果按顺序用逗号分隔，并放在 [answer] 标签中。例如：[answer]4,5,6[/answer]\n"
            "\n请详细解答每个查询，并将所有类型1查询的结果放在上述格式中。"
        ).format(
            n, m, '\n'.join(road_descriptions),
            q, '\n'.join(query_descriptions)
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output)
        if not matches:
            return None
        last_match = matches[-1]
        answers = last_match.split(',')
        try:
            answers = [int(a.strip()) for a in answers]
        except ValueError:
            return None
        return answers
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity['results']
        if not solution or not isinstance(solution, list):
            return False
        if len(solution) != len(expected):
            return False
        for s, e in zip(solution, expected):
            if s != e:
                return False
        return True
