"""# 

### 谜题描述
There is a square grid of size n × n. Some cells are colored in black, all others are colored in white. In one operation you can select some rectangle and color all its cells in white. It costs min(h, w) to color a rectangle of size h × w. You are to make all cells white for minimum total cost.

The square is large, so we give it to you in a compressed way. The set of black cells is the union of m rectangles.

Input

The first line contains two integers n and m (1 ≤ n ≤ 10^{9}, 0 ≤ m ≤ 50) — the size of the square grid and the number of black rectangles.

Each of the next m lines contains 4 integers x_{i1} y_{i1} x_{i2} y_{i2} (1 ≤ x_{i1} ≤ x_{i2} ≤ n, 1 ≤ y_{i1} ≤ y_{i2} ≤ n) — the coordinates of the bottom-left and the top-right corner cells of the i-th black rectangle.

The rectangles may intersect.

Output

Print a single integer — the minimum total cost of painting the whole square in white.

Examples

Input


10 2
4 1 5 10
1 4 10 5


Output


4


Input


7 6
2 1 2 1
4 2 4 3
2 5 2 5
2 3 5 3
1 2 1 2
3 2 5 3


Output


3

Note

The examples and some of optimal solutions are shown on the pictures below.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int maxn = 10005;
const int inf = 0x3f3f3f3f;
int n, m, cnt = -1, p[maxn], dep[maxn], cur[maxn], s, t, zx[maxn], zy[maxn],
          cntx, cnty;
struct kkk {
  int lx, ly, rx, ry;
} a[maxn];
struct node {
  int v, next, cap, flow;
} e[maxn * 10];
void add(int u, int v, int cap, int flow) {
  cnt++;
  e[cnt].v = v;
  e[cnt].next = p[u];
  e[cnt].cap = cap;
  e[cnt].flow = flow;
  p[u] = cnt;
}
bool bfs() {
  queue<int> q;
  memset(dep, -1, sizeof(dep));
  q.push(s);
  dep[s] = 0;
  while (!q.empty()) {
    int u = q.front();
    q.pop();
    for (int i = p[u]; i != -1; i = e[i].next) {
      if (e[i].cap - e[i].flow > 0 && dep[e[i].v] == -1) {
        dep[e[i].v] = dep[u] + 1;
        q.push(e[i].v);
      }
    }
  }
  if (dep[t] == -1) return false;
  return true;
}
int dfs(int u, int maxflow) {
  if (u == t || maxflow == 0) return maxflow;
  int flow = 0;
  for (int &i = cur[u]; i != -1; i = e[i].next) {
    if (dep[e[i].v] == dep[u] + 1 && e[i].cap > e[i].flow) {
      int fl = dfs(e[i].v, min(maxflow, e[i].cap - e[i].flow));
      maxflow -= fl;
      flow += fl;
      e[i].flow += fl;
      e[i ^ 1].flow -= fl;
      if (maxflow == 0) break;
    }
  }
  return flow;
}
int dinic() {
  int ans = 0;
  while (bfs()) {
    for (int i = s; i <= t; i++) {
      cur[i] = p[i];
    }
    ans += dfs(s, inf);
  }
  return ans;
}
int main() {
  memset(p, -1, sizeof(p));
  int n, m;
  cin >> n >> m;
  s = 0, t = 201;
  for (int i = 1; i <= m; i++) {
    scanf(\"%d%d%d%d\", &a[i].lx, &a[i].ly, &a[i].rx, &a[i].ry);
    a[i].lx--;
    a[i].ly--;
    zx[++cntx] = a[i].lx;
    zx[++cntx] = a[i].rx;
    zy[++cnty] = a[i].ly;
    zy[++cnty] = a[i].ry;
  }
  sort(zx + 1, zx + cntx + 1);
  sort(zy + 1, zy + cnty + 1);
  cntx = unique(zx + 1, zx + cntx + 1) - zx - 1;
  cnty = unique(zy + 1, zy + cnty + 1) - zy - 1;
  for (int i = 2; i <= cntx; i++) {
    for (int j = 2; j <= cnty; j++) {
      for (int k = 1; k <= m; k++) {
        if (a[k].lx <= zx[i - 1] && a[k].rx >= zx[i] && a[k].ly <= zy[j - 1] &&
            a[k].ry >= zy[j]) {
          add(i, j + 100, inf, 0);
          add(j + 100, i, 0, 0);
          break;
        }
      }
    }
  }
  for (int i = 2; i <= cntx; i++)
    add(s, i, zx[i] - zx[i - 1], 0), add(i, s, 0, 0);
  for (int i = 2; i <= cnty; i++)
    add(i + 100, t, zy[i] - zy[i - 1], 0), add(t, i + 100, 0, 0);
  cout << dinic();
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

class Edge:
    def __init__(self, to, rev, capacity):
        self.to = to
        self.rev = rev
        self.capacity = capacity

class Dinic:
    def __init__(self, n):
        self.size = n
        self.graph = [[] for _ in range(n)]
    
    def add_edge(self, fr, to, cap):
        forward_rev = len(self.graph[to])
        forward = Edge(to, forward_rev, cap)
        self.graph[fr].append(forward)
        backward_rev = len(self.graph[fr]) - 1
        backward = Edge(fr, backward_rev, 0)
        self.graph[to].append(backward)
        forward.rev = len(self.graph[to]) - 1
    
    def bfs_level(self, s, t, level):
        queue = deque()
        level[:] = [-1] * self.size
        level[s] = 0
        queue.append(s)
        while queue:
            u = queue.popleft()
            for edge in self.graph[u]:
                if edge.capacity > 0 and level[edge.to] == -1:
                    level[edge.to] = level[u] + 1
                    queue.append(edge.to)
                    if edge.to == t:
                        return True
        return False
    
    def dfs_flow(self, u, t, flow, level, ptr):
        if u == t:
            return flow
        while ptr[u] < len(self.graph[u]):
            edge = self.graph[u][ptr[u]]
            if edge.capacity > 0 and level[u] < level[edge.to]:
                min_flow = min(flow, edge.capacity)
                pushed = self.dfs_flow(edge.to, t, min_flow, level, ptr)
                if pushed > 0:
                    edge.capacity -= pushed
                    self.graph[edge.to][edge.rev].capacity += pushed
                    return pushed
            ptr[u] += 1
        return 0
    
    def max_flow(self, s, t):
        flow = 0
        level = [-1] * self.size
        while self.bfs_level(s, t, level):
            ptr = [0] * self.size
            while True:
                pushed = self.dfs_flow(s, t, float('inf'), level, ptr)
                if pushed == 0:
                    break
                flow += pushed
            level = [-1] * self.size
        return flow

def calculate_min_cost(n, m, rectangles):
    if m == 0:
        return 0
    
    a = []
    zx = []
    zy = []
    for rect in rectangles:
        x1, y1, x2, y2 = rect
        a.append((x1-1, y1-1, x2, y2))  # 转换为0-based坐标
        zx.extend([x1-1, x2])
        zy.extend([y1-1, y2])
    
    zx = sorted(list(set(zx)))
    zy = sorted(list(set(zy)))
    
    # 处理空网格的特殊情况
    if len(zx) < 2 or len(zy) < 2:
        return 0
    
    R = len(zx) - 1
    C = len(zy) - 1
    
    s = 0
    t = R + C + 1
    dinic = Dinic(t + 1)
    
    # 添加行边
    for i in range(R):
        dinic.add_edge(s, i+1, zx[i+1] - zx[i])
    
    # 添加列边
    for j in range(C):
        dinic.add_edge(R+1 + j, t, zy[j+1] - zy[j])
    
    # 添加中间边
    for i in range(R):
        x_start, x_end = zx[i], zx[i+1]
        for j in range(C):
            y_start, y_end = zy[j], zy[j+1]
            # 检查是否被任何矩形覆盖
            for (lx, ly, rx, ry) in a:
                if lx <= x_start and rx >= x_end and ly <= y_start and ry >= y_end:
                    dinic.add_edge(i+1, R+1 + j, 10**18)
                    break
    
    return dinic.max_flow(s, t)

class Erectanglepainting2bootcamp(Basebootcamp):
    def __init__(self, max_n=1_000_000, max_m=50):
        self.max_n = max_n
        self.max_m = max_m
    
    def case_generator(self):
        n = random.randint(1, self.max_n)
        m = random.randint(0, min(self.max_m, 50))
        
        rectangles = []
        for _ in range(m):
            x1 = random.randint(1, n)
            x2 = random.randint(x1, n)
            y1 = random.randint(1, n)
            y2 = random.randint(y1, n)
            rectangles.append([x1, y1, x2, y2])
        
        return {
            'n': n,
            'm': m,
            'rectangles': rectangles,
            'correct_answer': calculate_min_cost(n, m, rectangles)
        }
    
    @staticmethod
    def prompt_func(question_case):
        lines = [
            "# 问题描述",
            f"你有一个{question_case['n']}×{question_case['n']}的网格，包含{question_case['m']}个黑色矩形区域。",
            "每个操作可以选择任意矩形区域将其全部染白，代价为该矩形高度和宽度的较小值。",
            "# 输入格式",
            f"第一行：{question_case['n']} {question_case['m']}",
            "接下来m行每行四个整数x1 y1 x2 y2，表示黑色矩形的坐标",
            "# 输出要求",
            "输出最小总代价，将最终答案包裹在[answer]和[/answer]标签内"
        ]
        
        if question_case['m'] > 0:
            lines.append("\n# 矩形坐标：")
            for rect in question_case['rectangles']:
                lines.append(f"{rect[0]} {rect[1]} {rect[2]} {rect[3]}")
        
        return '\n'.join(lines)
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](\d+)\[/answer\]', output)
        return int(matches[-1]) if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
