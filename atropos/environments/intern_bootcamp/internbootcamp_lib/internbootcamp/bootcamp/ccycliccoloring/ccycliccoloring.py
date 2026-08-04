"""# 

### 谜题描述
You are given a directed graph G with n vertices and m arcs (multiple arcs and self-loops are allowed). You have to paint each vertex of the graph into one of the k (k ≤ n) colors in such way that for all arcs of the graph leading from a vertex u to vertex v, vertex v is painted with the next color of the color used to paint vertex u.

The colors are numbered cyclically 1 through k. This means that for each color i (i < k) its next color is color i + 1. In addition, the next color of color k is color 1. Note, that if k = 1, then the next color for color 1 is again color 1.

Your task is to find and print the largest possible value of k (k ≤ n) such that it's possible to color G as described above with k colors. Note that you don't necessarily use all the k colors (that is, for each color i there does not necessarily exist a vertex that is colored with color i).

Input

The first line contains two space-separated integers n and m (1 ≤ n, m ≤ 105), denoting the number of vertices and the number of arcs of the given digraph, respectively.

Then m lines follow, each line will contain two space-separated integers ai and bi (1 ≤ ai, bi ≤ n), which means that the i-th arc goes from vertex ai to vertex bi.

Multiple arcs and self-loops are allowed.

Output

Print a single integer — the maximum possible number of the colors that can be used to paint the digraph (i.e. k, as described in the problem statement). Note that the desired value of k must satisfy the inequality 1 ≤ k ≤ n.

Examples

Input

4 4
1 2
2 1
3 4
4 3


Output

2


Input

5 2
1 4
2 5


Output

5


Input

4 5
1 2
2 3
3 1
2 4
4 1


Output

3


Input

4 4
1 1
1 2
2 1
1 2


Output

1

Note

For the first example, with k = 2, this picture depicts the two colors (arrows denote the next color of that color).

<image>

With k = 2 a possible way to paint the graph is as follows.

<image>

It can be proven that no larger value for k exists for this test case.

For the second example, here's the picture of the k = 5 colors.

<image>

A possible coloring of the graph is:

<image>

For the third example, here's the picture of the k = 3 colors.

<image>

A possible coloring of the graph is:

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
template <typename T>
T abs(T a) {
  return a < 0 ? -a : a;
}
template <typename T>
T sqr(T a) {
  return a * a;
}
const int INF = (int)1e9;
const long double EPS = 1e-9;
const long double PI = 3.1415926535897932384626433832795;
const int N = 100500;
int n, m;
vector<int> g[N], rg[N];
bool used[N];
int c[N];
vector<int> q;
int minC;
void dfs(int v) {
  for (int i = 0; i < int(int((g[v]).size())); ++i) {
    int u = g[v][i];
    if (!used[u]) {
      c[u] = c[v] + 1;
      q.push_back(u);
      minC = min(minC, c[u]);
      used[u] = true;
      dfs(u);
    }
  }
  for (int i = 0; i < int(int((rg[v]).size())); ++i) {
    int u = rg[v][i];
    if (!used[u]) {
      c[u] = c[v] - 1;
      q.push_back(u);
      minC = min(minC, c[u]);
      used[u] = true;
      dfs(u);
    }
  }
}
bool check(int k) {
  for (int v = 0; v < int(n); ++v)
    for (int j = 0; j < int(int((g[v]).size())); ++j) {
      int u = g[v][j];
      if ((c[v] + 1) % k != c[u] % k) return false;
    }
  return true;
}
int ans = 1;
void update(int k) {
  if (k > ans && check(k)) ans = k;
}
int main() {
  cin >> n >> m;
  for (int i = 0; i < int(m); ++i) {
    int x, y;
    scanf(\"%d %d\", &x, &y);
    x--;
    y--;
    if (x == y) {
      puts(\"1\");
      return 0;
    }
    g[x].push_back(y);
    rg[y].push_back(x);
  }
  for (int i = 0; i < int(n); ++i) {
    sort((g[i]).begin(), (g[i]).end());
    g[i].erase(unique((g[i]).begin(), (g[i]).end()), g[i].end());
  }
  memset(c, -1, sizeof(c));
  for (int i = int(n) - 1; i >= 0; --i) {
    int v = i;
    if (!used[v]) {
      c[v] = 0;
      q.clear();
      used[v] = true;
      q.push_back(v);
      minC = 0;
      dfs(v);
      for (int i = 0; i < int(int((q).size())); ++i) c[q[i]] -= minC;
    }
  }
  for (int v = 0; v < int(n); ++v) {
    for (int i = 0; i < int(int((g[v]).size())); ++i) {
      int u = g[v][i];
      if (c[v] + 1 != c[u]) {
        cerr << v << \" \" << u << \" \" << c[v] << \" \" << c[u] << endl;
        int d = abs(c[v] + 1 - c[u]);
        for (int i = 1; i * i <= d; ++i) {
          if (d % i == 0) {
            update(i);
            update(d / i);
          }
        }
        cout << ans << endl;
        return 0;
      }
    }
  }
  cout << n << endl;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from math import gcd
from collections import deque
from bootcamp import Basebootcamp

class Ccycliccoloringbootcamp(Basebootcamp):
    def __init__(self, max_n=10, min_k=1, max_k=5):
        self.max_n = max_n
        self.min_k = min_k
        self.max_k = max_k
    
    def case_generator(self):
        # 核心改进：确保生成合法k的测试用例
        while True:
            try:
                n = random.randint(2, self.max_n)
                k = random.randint(self.min_k, min(self.max_k, n))
                
                # 生成合法颜色分配
                colors = {}
                nodes = list(range(n))
                random.shuffle(nodes)
                
                # 创建至少一个长度为k的环确保解至少为k
                cycle = nodes[:k]
                for i, node in enumerate(cycle):
                    colors[node] = i
                
                # 分配剩余节点颜色
                for node in nodes[k:]:
                    colors[node] = random.randint(0, k-1)
                
                # 生成合法边集合
                edges = []
                # 强制生成环
                for i in range(k):
                    u = cycle[i]
                    v = cycle[(i+1) % k]
                    edges.append((u+1, v+1))  # 1-based
                
                # 添加合法随机边
                additional_edges = []
                for _ in range(random.randint(0, 3)):  # 控制边的数量
                    u = random.choice(nodes)
                    valid_color = (colors[u] + 1) % k
                    valid_nodes = [node for node in nodes if colors[node] == valid_color]
                    if valid_nodes:
                        v = random.choice(valid_nodes)
                        additional_edges.append((u+1, v+1))
                
                # 合并边并检查自环
                edges += additional_edges
                if any(u == v for u, v in edges):
                    return {'n': n, 'm': len(edges), 'edges': edges}
                
                # 最终校验
                test_case = {'n': n, 'edges': edges}
                if self._verify_correction(k, test_case):
                    return {'n': n, 'm': len(edges), 'edges': edges}
            except:
                continue

    @staticmethod
    def prompt_func(question_case):
        edges_str = '\n'.join(f"{u} {v}" for u, v in question_case['edges'])
        return (
            f"给定一个有向图，{question_case['n']}个顶点，{question_case['m']}条边。"
            f"找到最大的k使得存在颜色1~k的着色方案，满足每条边u→v中v的颜色是u颜色的下一个（循环顺序）。\n"
            f"输入：\n{question_case['n']} {question_case['m']}\n{edges_str}\n"
            f"答案请写在[answer]和[/answer]之间。"
        )

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return int(matches[-1].strip()) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 精确验证算法
        n = identity['n']
        edges = identity['edges']
        
        # 自环特判
        if any(u == v for u, v in edges):
            return solution == 1
        
        # 构建邻接表
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u-1].append(v-1)
        
        # 计算最大k的算法实现
        visited = [False] * n
        color = [0] * n
        
        def dfs(u, c):
            color[u] = c
            visited[u] = True
            for v in adj[u]:
                if not visited[v]:
                    if not dfs(v, c + 1):
                        return False
                elif color[v] != (c + 1) % solution:
                    return False
            return True
        
        # 检查所有连通分量
        for i in range(n):
            if not visited[i]:
                if not dfs(i, 0):
                    return False
        return True
