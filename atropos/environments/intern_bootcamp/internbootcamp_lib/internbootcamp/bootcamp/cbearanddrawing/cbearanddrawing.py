"""# 

### 谜题描述
Limak is a little bear who learns to draw. People usually start with houses, fences and flowers but why would bears do it? Limak lives in the forest and he decides to draw a tree.

Recall that tree is a connected graph consisting of n vertices and n - 1 edges.

Limak chose a tree with n vertices. He has infinite strip of paper with two parallel rows of dots. Little bear wants to assign vertices of a tree to some n distinct dots on a paper so that edges would intersect only at their endpoints — drawn tree must be planar. Below you can see one of correct drawings for the first sample test.

<image>

Is it possible for Limak to draw chosen tree?

Input

The first line contains single integer n (1 ≤ n ≤ 105).

Next n - 1 lines contain description of a tree. i-th of them contains two space-separated integers ai and bi (1 ≤ ai, bi ≤ n, ai ≠ bi) denoting an edge between vertices ai and bi. It's guaranteed that given description forms a tree.

Output

Print \"Yes\" (without the quotes) if Limak can draw chosen tree. Otherwise, print \"No\" (without the quotes).

Examples

Input

8
1 2
1 3
1 6
6 4
6 7
6 5
7 8


Output

Yes


Input

13
1 2
1 3
1 4
2 5
2 6
2 7
3 8
3 9
3 10
4 11
4 12
4 13


Output

No

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int INF = 1000001000;
const long long INFL = 2000000000000001000;
int solve();
int main() {
  srand(2317);
  cout.precision(10);
  cout.setf(ios::fixed);
  int tn = 1;
  for (int i = 0; i < tn; ++i) solve();
}
const int maxn = 2e5;
vector<int> g[maxn];
bool del[maxn];
int bamboo[maxn];
bool type2[maxn];
queue<int> o;
int solve() {
  int n;
  cin >> n;
  for (int i = 0; i < int(n - 1); ++i) {
    int u, v;
    cin >> u >> v;
    --u, --v;
    g[u].push_back(v);
    g[v].push_back(u);
  }
  for (int i = 0; i < int(n); ++i)
    if (((int)(g[i]).size()) == 1) o.push(i);
  while (!o.empty()) {
    int u = o.front();
    del[u] = true;
    o.pop();
    for (int v : g[u]) {
      if (del[v]) continue;
      if (((int)(g[v]).size()) == 2)
        o.push(v);
      else
        ++bamboo[v];
    }
  }
  bool fail = false;
  for (int i = 0; i < int(n); ++i) {
    if (del[i]) continue;
    if (((int)(g[i]).size()) <= 3 && ((int)(g[i]).size()) <= bamboo[i] + 1)
      type2[i] = true;
  }
  for (int i = 0; i < int(n); ++i) {
    if (type2[i] || del[i]) continue;
    int deg = 0;
    for (int v : g[i])
      if (!type2[v] && !del[v]) ++deg;
    if (deg > 2) fail = true;
  }
  if (fail)
    cout << \"No\n\";
  else
    cout << \"Yes\n\";
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import deque
from bootcamp import Basebootcamp

class Cbearanddrawingbootcamp(Basebootcamp):
    def __init__(self, max_nodes=20, seed=None):
        super().__init__()
        self.max_nodes = max_nodes
        self.seed = seed or random.randint(0, 999999)
        random.seed(self.seed)
    
    def case_generator(self):
        n = random.randint(1, self.max_nodes)
        edges = self._generate_random_tree(n)
        answer = self._check_tree_planar(n, edges)
        return {
            'n': n,
            'edges': edges,
            'answer': answer
        }
    
    def _generate_random_tree(self, n):
        """使用Prüfer序列生成随机树"""
        if n == 1:
            return []
        # 生成Prüfer序列
        prufer = [random.randint(1, n) for _ in range(n-2)]
        # 生成树边
        degree = [1] * (n + 1)  # 节点编号1-based
        for node in prufer:
            degree[node] += 1
        
        edges = []
        for node in prufer:
            for leaf in range(1, n+1):
                if degree[leaf] == 1:
                    edges.append((node, leaf))
                    degree[node] -= 1
                    degree[leaf] -= 1
                    break
        
        # 处理最后两个节点
        leaves = [i for i in range(1, n+1) if degree[i] == 1]
        edges.append((leaves[0], leaves[1]))
        return edges
    
    def _check_tree_planar(self, n, edges):
        if n <= 2:
            return "Yes"
        
        # 构建邻接表（0-based）
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u-1].append(v-1)
            adj[v-1].append(u-1)
        
        del_ = [False]*n
        bamboo = [0]*n
        q = deque()
        
        # 初始化队列
        for i in range(n):
            if len(adj[i]) == 1:
                q.append(i)
        
        # BFS处理叶子节点
        while q:
            u = q.popleft()
            del_[u] = True
            for v in adj[u]:
                if del_[v]: continue
                original_degree = len(adj[v])
                if original_degree == 2:
                    q.append(v)
                else:
                    bamboo[v] += 1
        
        # 标记type2节点
        type2 = [False]*n
        for i in range(n):
            if del_[i]: continue
            original_degree = len(adj[i])
            if original_degree <= 3 and original_degree <= bamboo[i] + 1:
                type2[i] = True
        
        # 最终验证
        fail = False
        for i in range(n):
            if del_[i] or type2[i]: continue
            cnt = 0
            for v in adj[i]:
                if not del_[v] and not type2[v]:
                    cnt += 1
                    if cnt > 2:
                        fail = True
                        break
            if fail: break
        
        return "Yes" if not fail else "No"

    @staticmethod
    def prompt_func(question_case):
        edges = '\n'.join(f"{u} {v}" for u, v in question_case['edges'])
        return f"""Cbearanddrawing wants to draw a tree on a two-row grid without edge crossings. Determine if this tree can be drawn planarly.

Input:
{question_case['n']}
{edges}

Output format: [answer]Yes[/answer] or [answer]No[/answer]"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](Yes|No)\[/answer\]', output, re.IGNORECASE)
        return matches[-1].upper() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return (solution or "").lower() == identity['answer'].lower()
