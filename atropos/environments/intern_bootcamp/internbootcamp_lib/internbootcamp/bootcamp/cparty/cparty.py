"""# 

### 谜题描述
Arseny likes to organize parties and invite people to it. However, not only friends come to his parties, but friends of his friends, friends of friends of his friends and so on. That's why some of Arseny's guests can be unknown to him. He decided to fix this issue using the following procedure.

At each step he selects one of his guests A, who pairwise introduces all of his friends to each other. After this action any two friends of A become friends. This process is run until all pairs of guests are friends.

Arseny doesn't want to spend much time doing it, so he wants to finish this process using the minimum number of steps. Help Arseny to do it.

Input

The first line contains two integers n and m (1 ≤ n ≤ 22; <image>) — the number of guests at the party (including Arseny) and the number of pairs of people which are friends.

Each of the next m lines contains two integers u and v (1 ≤ u, v ≤ n; u ≠ v), which means that people with numbers u and v are friends initially. It's guaranteed that each pair of friends is described not more than once and the graph of friendship is connected.

Output

In the first line print the minimum number of steps required to make all pairs of guests friends.

In the second line print the ids of guests, who are selected at each step.

If there are multiple solutions, you can output any of them.

Examples

Input

5 6
1 2
1 3
2 3
2 5
3 4
4 5


Output

2
2 3 

Input

4 4
1 2
1 3
1 4
3 4


Output

1
1 

Note

In the first test case there is no guest who is friend of all other guests, so at least two steps are required to perform the task. After second guest pairwise introduces all his friends, only pairs of guests (4, 1) and (4, 2) are not friends. Guest 3 or 5 can introduce them.

In the second test case guest number 1 is a friend of all guests, so he can pairwise introduce all guests in one step.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const long long INF = 1e9, MOD = INF + 7;
const int N = 22, M = (1 << N);
int adj[N], neigh[M];
int main() {
  ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
  int n, m;
  cin >> n >> m;
  for (int i = 0; i < m; i++) {
    int v, u;
    cin >> v >> u;
    v--, u--;
    adj[v] += 1 << u;
    adj[u] += 1 << v;
  }
  if (2 * m == n * n - n) return cout << 0 << \"\n\", 0;
  for (int i = 0; i < n; i++) adj[i] += 1 << i, neigh[1 << i] = adj[i];
  for (int mask = 0; mask < (1 << n); mask++)
    for (int i = 0; i < n; i++) {
      if (!(mask & (1 << i)) && (neigh[mask] & (1 << i))) {
        neigh[mask | (1 << i)] |= (neigh[mask] | adj[i]);
      }
    }
  int ans = (1 << n) - 1;
  for (int mask = 0; mask < (1 << n); mask++) {
    if (neigh[mask] == (1 << n) - 1 &&
        __builtin_popcount(mask) < __builtin_popcount(ans)) {
      ans = mask;
    }
  }
  cout << __builtin_popcount(ans) << \"\n\";
  for (int i = 0; i < n; i++)
    if (ans & (1 << i)) cout << i + 1 << ' ';
  cout << \"\n\";
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from itertools import combinations
from bootcamp import Basebootcamp

class Cpartybootcamp(Basebootcamp):
    def __init__(self, n_min=3, n_max=8):
        self.n_min = max(n_min, 1)  # 确保n≥1
        self.n_max = n_max
    
    def case_generator(self):
        """生成保证连通性的随机图并计算最优解"""
        n = random.randint(self.n_min, self.n_max)
        edges = self._generate_connected_graph(n)
        min_steps, solution = self._calculate_optimal_solution(n, edges)
        return {
            'n': n,
            'm': len(edges),
            'edges': edges,
            'min_steps': min_steps,
            'solution': solution
        }

    @staticmethod
    def prompt_func(case) -> str:
        """生成带明确格式要求的问题描述"""
        edges = '\n'.join(f"{u} {v}" for u, v in case['edges'])
        return f"""## 聚会好友问题

Arseny的聚会有{case['n']}位客人，初始好友关系如下：
{edges}

每次操作选择一位客人，使得他的所有当前好友互相成为直接好友。求达成全员互为好友所需的最少操作次数及对应选择顺序。

请按以下格式输出答案：
[answer]
步骤数
选择的客人序列（空格分隔）
[/answer]

示例：
[answer]
2
2 3
[/answer]"""

    @staticmethod
    def extract_output(output):
        """增强鲁棒性的答案提取"""
        matches = re.findall(r'\[answer\][\s]*((?:\d+[\s\n]*)+)[\s]*\[/answer\]', output, re.IGNORECASE)
        if not matches:
            return None
        
        # 取最后一个答案块并解析
        content = matches[-1].strip()
        parts = [p for p in re.split(r'\s+', content) if p]
        
        try:
            if len(parts) < 1:
                return None
            steps = int(parts[0])
            guests = list(map(int, parts[1:1+steps])) if steps >0 else []
            if len(guests) != steps:
                return None
            return (steps, guests)
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, case):
        """严格模拟验证的改进版本"""
        if not solution or solution[0] != case['min_steps']:
            return False
        
        n = case['n']
        edges = case['edges']
        selected = solution[1]

        # 构建初始邻接矩阵
        adj = [set() for _ in range(n+1)]  # 1-based索引
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # 初始自反关系
        for u in range(1, n+1):
            adj[u].add(u)
        
        # 模拟操作流程
        for guest in selected:
            friends = adj[guest].copy()
            # 将所有朋友两两连接
            new_edges = combinations(friends, 2)
            for u, v in new_edges:
                adj[u].add(v)
                adj[v].add(u)
        
        # 验证全连接
        full_set = set(range(1, n+1))
        return all(adj[u] == full_set for u in range(1, n+1))

    def _generate_connected_graph(self, n):
        """改进的连通图生成算法"""
        if n == 1:
            return []
        
        edges = set()
        nodes = list(range(1, n+1))
        visited = {nodes[0]}
        unvisited = set(nodes[1:])
        
        # Prim算法生成生成树
        while unvisited:
            u = random.choice(list(visited))
            v = random.choice(list(unvisited))
            edges.add(frozenset((u, v)))
            visited.add(v)
            unvisited.remove(v)
        
        # 添加随机边 (至少添加n-1条边)
        all_possible = {frozenset(e) for e in combinations(nodes, 2)}
        remaining = list(all_possible - edges)
        random.shuffle(remaining)
        
        extra = random.randint(0, len(remaining))
        edges.update(remaining[:extra])
        
        return sorted([sorted(list(e)) for e in edges])

    def _calculate_optimal_solution(self, n, edges):
        """基于位运算的高效算法（参考原题解）"""
        if n == 1:
            return 0, []
        
        # 转换为0-based邻接表
        adj = [0] * n
        for u, v in edges:
            u_idx = u - 1
            v_idx = v - 1
            adj[u_idx] |= 1 << v_idx
            adj[v_idx] |= 1 << u_idx
        
        # 添加自环
        for i in range(n):
            adj[i] |= 1 << i
        
        # 预处理覆盖关系
        full_mask = (1 << n) - 1
        if all(mask == full_mask for mask in adj):
            return 0, []
        
        # 初始化neigh数组
        max_mask = 1 << n
        coverage = [0] * max_mask
        for i in range(n):
            coverage[1 << i] = adj[i]
        
        # 预处理所有mask的覆盖关系
        for mask in range(max_mask):
            for i in range(n):
                if (mask & (1 << i)) and (coverage[mask ^ (1 << i)] & (1 << i)):
                    coverage[mask] = coverage[mask ^ (1 << i)] | adj[i]
        
        # 寻找最小集合
        best_mask = full_mask
        min_steps = n
        for mask in range(max_mask):
            if coverage[mask] == full_mask:
                cnt = bin(mask).count('1')
                if cnt < min_steps:
                    min_steps = cnt
                    best_mask = mask
        
        solution = [i+1 for i in range(n) if (best_mask & (1 << i))]
        return min_steps, solution
