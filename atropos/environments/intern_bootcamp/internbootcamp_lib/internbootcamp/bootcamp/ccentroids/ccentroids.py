"""# 

### 谜题描述
Tree is a connected acyclic graph. Suppose you are given a tree consisting of n vertices. The vertex of this tree is called centroid if the size of each connected component that appears if this vertex is removed from the tree doesn't exceed <image>.

You are given a tree of size n and can perform no more than one edge replacement. Edge replacement is the operation of removing one edge from the tree (without deleting incident vertices) and inserting one new edge (without adding new vertices) in such a way that the graph remains a tree. For each vertex you have to determine if it's possible to make it centroid by performing no more than one edge replacement.

Input

The first line of the input contains an integer n (2 ≤ n ≤ 400 000) — the number of vertices in the tree. Each of the next n - 1 lines contains a pair of vertex indices ui and vi (1 ≤ ui, vi ≤ n) — endpoints of the corresponding edge.

Output

Print n integers. The i-th of them should be equal to 1 if the i-th vertex can be made centroid by replacing no more than one edge, and should be equal to 0 otherwise.

Examples

Input

3
1 2
2 3


Output

1 1 1 


Input

5
1 2
1 3
1 4
1 5


Output

1 0 0 0 0 

Note

In the first sample each vertex can be made a centroid. For example, in order to turn vertex 1 to centroid one have to replace the edge (2, 3) with the edge (1, 3).

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
inline long long read() {
  long long x = 0;
  char ch = getchar(), w = 1;
  while (ch < '0' || ch > '9') {
    if (ch == '-') w = -1;
    ch = getchar();
  }
  while (ch >= '0' && ch <= '9') {
    x = x * 10 + ch - '0';
    ch = getchar();
  }
  return x * w;
}
void write(long long x) {
  if (x < 0) putchar('-'), x = -x;
  if (x > 9) write(x / 10);
  putchar(x % 10 + '0');
}
inline void writeln(long long x) {
  write(x);
  puts(\"\");
}
using namespace std;
int n;
const int N = 420000 * 2;
struct Edge {
  int u, v, nxt;
} e[N];
int head[N], en;
void addl(int x, int y) {
  e[++en].u = x, e[en].v = y, e[en].nxt = head[x], head[x] = en;
}
bool ans[N];
int siz[N], rt, res = 1e9;
void dfs(int x, int F) {
  siz[x] = 1;
  int mx = 0;
  for (int i = head[x]; i; i = e[i].nxt) {
    int y = e[i].v;
    if (y == F) continue;
    dfs(y, x);
    siz[x] += siz[y];
    mx = max(mx, siz[y]);
  }
  mx = max(mx, n - siz[x]);
  if (mx < res) {
    res = mx;
    rt = x;
  }
}
vector<pair<int, int> > sub;
void solve(int x, int F, int sum, int pre) {
  if (sum <= n / 2) ans[x] = 1;
  for (int i = 0; i < 2 && i < sub.size(); ++i) {
    if (sub[i].second == pre) continue;
    if (n - siz[x] - sub[i].first <= n / 2) ans[x] = 1;
  }
  for (int i = head[x]; i; i = e[i].nxt) {
    int y = e[i].v;
    if (y == F) continue;
    solve(y, x, sum, pre);
  }
}
int main() {
  n = read();
  for (int i = 1; i < n; ++i) {
    int x = read(), y = read();
    addl(x, y);
    addl(y, x);
  }
  dfs(1, 0);
  dfs(rt, 0);
  ans[rt] = 1;
  for (int i = head[rt]; i; i = e[i].nxt)
    sub.push_back(make_pair(siz[e[i].v], e[i].v));
  sort(sub.begin(), sub.end(), greater<pair<int, int> >());
  for (int i = head[rt]; i; i = e[i].nxt) {
    int to = e[i].v;
    solve(to, rt, n - siz[to], to);
  }
  for (int i = 1; i <= n; ++i) printf(\"%d \", ans[i]);
  puts(\"\");
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Ccentroidsbootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=20):
        self.min_n = min_n
        self.max_n = max_n
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        edges = self.generate_random_tree(n)
        correct_answer = self.solve_problem(n, edges)
        return {
            'n': n,
            'edges': edges,
            'correct_answer': correct_answer
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        edges_str = '\n'.join(f"{u} {v}" for u, v in question_case['edges'])
        prompt = f"""你是一名网络结构工程师，负责优化一个树形结构的网络节点。你的任务是通过最多一次边的替换操作，使得某个特定节点成为网络的重心。树的重心定义如下：移除该节点后，剩下的每个连通块的大小都不超过原树节点数的一半。边替换操作是指删除一条边并添加一条新边，替换后仍保持树的结构。你需要确定每个节点是否可以通过最多一次这样的操作成为重心。

输入格式：
第一行是一个整数n，表示节点的数量。接下来的n-1行每行两个整数，表示一条边的两个端点。

你的任务是为每个节点i（从1到n），输出1或0，表示是否可以通过最多一次边替换使其成为重心。输出为一个由空格分隔的n个数字组成的字符串。

请按照输入示例的格式进行解答，并将最终答案放置在[answer]和[/answer]标签之间。例如，若正确输出是1 1 1，则你的回答应为：
[answer]1 1 1[/answer]

当前的问题实例：
n = {question_case['n']}
边列表：
{edges_str}"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        parts = last_match.split()
        try:
            solution = list(map(int, parts))
        except ValueError:
            return None
        return solution
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        correct = identity['correct_answer']
        if not isinstance(solution, list):
            return False
        if len(solution) != len(correct):
            return False
        return solution == correct
    
    @staticmethod
    def generate_random_tree(n):
        if n == 1:
            return []
        edges = []
        for i in range(2, n+1):
            p = random.randint(1, i-1)
            edges.append((p, i))
        random.shuffle(edges)
        return edges
    
    @staticmethod
    def solve_problem(n, edges):
        adj = [[] for _ in range(n+1)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # 第一次DFS寻找重心
        siz = [0]*(n+1)
        res = float('inf')
        rt = 0
        def dfs1(x, F):
            nonlocal res, rt
            siz[x] = 1
            mx = 0
            for y in adj[x]:
                if y == F:
                    continue
                dfs1(y, x)
                siz[x] += siz[y]
                mx = max(mx, siz[y])
            mx = max(mx, n - siz[x])
            if mx < res or (mx == res and x < rt):
                res = mx
                rt = x
        dfs1(1, 0)
        
        # 第二次DFS建立父节点关系
        siz = [0]*(n+1)
        parent = {}
        def dfs2(x, F):
            parent[x] = F
            siz[x] = 1
            for y in adj[x]:
                if y == F:
                    continue
                dfs2(y, x)
                siz[x] += siz[y]
        dfs2(rt, 0)
        
        # 获取直接子节点
        sub = []
        for y in adj[rt]:
            if parent[y] == rt:  # 关键修正：确保只处理子节点
                sub.append((siz[y], y))
        sub.sort(reverse=True, key=lambda x: x[0])
        
        ans = [0]*(n+1)
        ans[rt] = 1
        
        # 递归求解答案
        def solve(x, F, sum_val, pre):
            if sum_val <= n//2:
                ans[x] = 1
            for i in range(min(2, len(sub))):
                s, node = sub[i]
                if node == pre:
                    continue
                if (n - siz[x] - s) <= n//2:
                    ans[x] = 1
            for y in adj[x]:
                if y == F:
                    continue
                solve(y, x, sum_val, pre)
        
        # 遍历所有子节点
        for y in adj[rt]:
            if parent[y] != rt:  # 关键修正：过滤非子节点
                continue
            solve(y, rt, n - siz[y], y)
        
        return ans[1:n+1]
