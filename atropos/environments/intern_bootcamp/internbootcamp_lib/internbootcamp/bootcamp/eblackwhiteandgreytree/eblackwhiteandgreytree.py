"""# 

### 谜题描述
You are given a tree with each vertex coloured white, black or grey. You can remove elements from the tree by selecting a subset of vertices in a single connected component and removing them and their adjacent edges from the graph. The only restriction is that you are not allowed to select a subset containing a white and a black vertex at once.

What is the minimum number of removals necessary to remove all vertices from the tree?

Input

Each test contains multiple test cases. The first line contains an integer t (1 ≤ t ≤ 100 000), denoting the number of test cases, followed by a description of the test cases.

The first line of each test case contains an integer n (1 ≤ n ≤ 200 000): the number of vertices in the tree.

The second line of each test case contains n integers a_v (0 ≤ a_v ≤ 2): colours of vertices. Gray vertices have a_v=0, white have a_v=1, black have a_v=2.

Each of the next n-1 lines contains two integers u, v (1 ≤ u, v ≤ n): tree edges.

The sum of all n throughout the test is guaranteed to not exceed 200 000.

Output

For each test case, print one integer: the minimum number of operations to solve the problem.

Example

Input


4
2
1 1
1 2
4
1 2 1 2
1 2
2 3
3 4
5
1 1 0 1 2
1 2
2 3
3 4
3 5
8
1 2 1 2 2 2 1 2
1 3
2 3
3 4
4 5
5 6
5 7
5 8


Output


1
3
2
3

Note

<image>

In the first test case, both vertices are white, so you can remove them at the same time.

<image>

In the second test case, three operations are enough. First, we need to remove both black vertices (2 and 4), then separately remove vertices 1 and 3. We can't remove them together because they end up in different connectivity components after vertex 2 is removed.

<image>

In the third test case, we can remove vertices 1, 2, 3, 4 at the same time, because three of them are white and one is grey. After that, we can remove vertex 5.

<image>

In the fourth test case, three operations are enough. One of the ways to solve the problem is to remove all black vertices at once, then remove white vertex 7, and finally, remove connected white vertices 1 and 3.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
template <typename T1, typename T2>
bool mini(T1 &a, T2 b) {
  if (a > b) {
    a = b;
    return true;
  }
  return false;
}
template <typename T1, typename T2>
bool maxi(T1 &a, T2 b) {
  if (a < b) {
    a = b;
    return true;
  }
  return false;
}
const int N = 2e5 + 5;
const int oo = 1e9;
vector<int> adj[N];
int dp[N][2];
int a[N];
int n, ans;
void dfs(int u, int p = -1) {
  for (int v : adj[u])
    if (v != p) dfs(v, u);
  dp[u][0] = dp[u][1] = oo;
  if (a[u] == 0) {
    int d = oo;
    for (int val = 0; val < 2; val++) {
      dp[u][val] = 0;
      pair<int, int> tmp = make_pair(0, 0);
      for (int v : adj[u])
        if (v != p) {
          int res = min(dp[v][0] + val, dp[v][1] + (!val));
          maxi(dp[u][val], res);
          maxi(tmp.second, res);
          if (tmp.second > tmp.first) swap(tmp.second, tmp.first);
        }
      mini(d, tmp.first + tmp.second);
    }
    maxi(ans, d);
  } else {
    int val = a[u] - 1;
    dp[u][val] = 0;
    pair<int, int> tmp = make_pair(0, 0);
    for (int v : adj[u])
      if (v != p) {
        int res = min(dp[v][0] + val, dp[v][1] + (!val));
        maxi(dp[u][val], res);
        maxi(tmp.second, res);
        if (tmp.second > tmp.first) swap(tmp.second, tmp.first);
      }
    maxi(ans, tmp.first + tmp.second);
  }
}
void solve() {
  cin >> n;
  for (int i = 1; i <= n; i++) {
    adj[i].clear();
    cin >> a[i];
  }
  for (int i = 1; i < n; i++) {
    int u, v;
    cin >> u >> v;
    adj[u].push_back(v);
    adj[v].push_back(u);
  }
  ans = 0;
  dfs(1);
  cout << ((ans + 1) >> 1) + 1 << \"\n\";
}
int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  int t;
  cin >> t;
  while (t--) solve();
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp
import re

class Eblackwhiteandgreytreebootcamp(Basebootcamp):
    def __init__(self, min_nodes=2, max_nodes=200, grey_prob=0.2, white_prob=0.4, black_prob=0.4):
        self.min_nodes = min_nodes
        self.max_nodes = max(max_nodes, min_nodes)
        total = grey_prob + white_prob + black_prob
        if total <= 0:
            self.grey_prob = 0.0
            self.white_prob = 0.0
            self.black_prob = 0.0
        else:
            self.grey_prob = grey_prob / total
            self.white_prob = white_prob / total
            self.black_prob = black_prob / total

    def case_generator(self):
        n = random.randint(self.min_nodes, self.max_nodes)
        colors = []
        for _ in range(n):
            rand = random.random()
            if rand < self.grey_prob:
                colors.append(0)
            elif rand < self.grey_prob + self.white_prob:
                colors.append(1)
            else:
                colors.append(2)
        
        edges = []
        if n > 1:
            parent = {}
            for i in range(2, n+1):
                parent[i] = random.randint(1, i-1)
            edges = [(parent[i], i) for i in range(2, n+1)]
        
        correct_answer = self._solve_cpp_style(n, colors, edges)
        return {
            'n': n,
            'colors': colors,
            'edges': edges,
            'correct_answer': correct_answer
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        colors = ' '.join(map(str, question_case['colors']))
        edges = [' '.join(map(str, edge)) for edge in question_case['edges']]
        edges_str = '\n'.join(edges)
        return f"""你是一个算法专家，需要解决一个树结构的最小移除操作次数问题。问题规则如下：

给定一个树结构，每个节点被涂成白色（1）、黑色（2）或灰色（0）。每次操作，你可以选择一个连通的顶点子集，但所选子集不能同时包含白色和黑色节点。移除该子集及其相连的边。目标是找出移除所有节点的最小操作次数。

当前测试用例的输入数据为：

第一行：{n}
第二行：{colors}
接下来的{n-1}行，每行两个整数表示边：
{edges_str}

请计算正确的最小操作次数，并将答案用[answer]...[/answer]标签包裹。例如，若答案是3，写成[answer]3[/answer]。"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity.get('correct_answer')

    def _solve_cpp_style(self, n, colors, edges):
        adj = [[] for _ in range(n+1)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        a = [0] * (n+1)
        for i in range(1, n+1):
            a[i] = colors[i-1]
        ans = 0

        dp = [[float('inf')] * 2 for _ in range(n+1)]

        # 迭代后序遍历
        stack = [(1, -1, False)]
        post_order = []

        while stack:
            u, p, visited = stack.pop()
            if not visited:
                stack.append((u, p, True))
                # Reverse to maintain original order
                for v in reversed(adj[u]):
                    if v != p:
                        stack.append((v, u, False))
            else:
                post_order.append((u, p))

        for u, p in post_order:
            if a[u] == 0:
                d = float('inf')
                for val in [0, 1]:
                    current_max = 0
                    tmp = [0, 0]
                    res_list = []
                    for v in adj[u]:
                        if v == p:
                            continue
                        res = min(dp[v][0] + val, dp[v][1] + (1 - val))
                        res_list.append(res)
                    
                    for res in res_list:
                        if res >= tmp[0]:
                            tmp[1] = tmp[0]
                            tmp[0] = res
                        elif res > tmp[1]:
                            tmp[1] = res
                    current_max = tmp[0]
                    current_d = tmp[0] + tmp[1]
                    if current_d < d:
                        d = current_d
                    dp[u][val] = current_max
                if d != float('inf'):
                    ans = max(ans, d)
            else:
                val = a[u] - 1
                tmp = [0, 0]
                res_list = []
                for v in adj[u]:
                    if v == p:
                        continue
                    res = min(dp[v][0] + val, dp[v][1] + (1 - val))
                    res_list.append(res)
                
                for res in res_list:
                    if res >= tmp[0]:
                        tmp[1] = tmp[0]
                        tmp[0] = res
                    elif res > tmp[1]:
                        tmp[1] = res
                current_d = tmp[0] + tmp[1]
                ans = max(ans, current_d)
                dp[u][val] = tmp[0]

        return ((ans + 1) // 2) + 1
