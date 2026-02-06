"""# 

### 谜题描述
Denis came to Nastya and discovered that she was not happy to see him... There is only one chance that she can become happy. Denis wants to buy all things that Nastya likes so she will certainly agree to talk to him. 

The map of the city where they live has a lot of squares, some of which are connected by roads. There is exactly one way between each pair of squares which does not visit any vertex twice. It turns out that the graph of the city is a tree.

Denis is located at vertex 1 at the time 0. He wants to visit every vertex at least once and get back as soon as possible.

Denis can walk one road in 1 time. Unfortunately, the city is so large that it will take a very long time to visit all squares. Therefore, Denis took a desperate step. He pulled out his pocket time machine, which he constructed in his basement. With its help, Denis can change the time to any non-negative time, which is less than the current time.

But the time machine has one feature. If the hero finds himself in the same place and at the same time twice, there will be an explosion of universal proportions and Nastya will stay unhappy. Therefore, Denis asks you to find him a route using a time machine that he will get around all squares and will return to the first and at the same time the maximum time in which he visited any square will be minimal.

Formally, Denis's route can be represented as a sequence of pairs: \\{v_1, t_1\}, \\{v_2, t_2\}, \\{v_3, t_3\}, …, \\{v_k, t_k\}, where v_i is number of square, and t_i is time in which the boy is now.

The following conditions must be met:

  * The route starts on square 1 at time 0, i.e. v_1 = 1, t_1 = 0 and ends on the square 1, i.e. v_k = 1. 
  * All transitions are divided into two types: 
    1. Being in the square change the time: \{ v_i, t_i \} → \{ v_{i+1}, t_{i+1} \} : v_{i+1} = v_i, 0 ≤ t_{i+1} < t_i. 
    2. Walk along one of the roads: \{ v_i, t_i \} → \{ v_{i+1}, t_{i+1} \}. Herewith, v_i and v_{i+1} are connected by road, and t_{i+1} = t_i + 1 
  * All pairs \{ v_i, t_i \} must be different. 
  * All squares are among v_1, v_2, …, v_k. 



You need to find a route such that the maximum time in any square will be minimal, that is, the route for which max{(t_1, t_2, …, t_k)} will be the minimum possible.

Input

The first line contains a single integer n (1 ≤ n ≤ 10^5) — the number of squares in the city. 

The next n - 1 lines contain two integers u and v (1 ≤ v, u ≤ n, u ≠ v) - the numbers of the squares connected by the road. 

It is guaranteed that the given graph is a tree.

Output

In the first line output the integer k (1 ≤ k ≤ 10^6) — the length of the path of Denis.

In the next k lines output pairs v_i, t_i — pairs that describe Denis's route (as in the statement).

All route requirements described in the statements must be met.

It is guaranteed that under given restrictions there is at least one route and an answer whose length does not exceed 10^6. If there are several possible answers, print any.

Example

Input


5
1 2
2 3
2 4
4 5


Output


13
1 0
2 1
3 2
3 1
2 2
4 3
4 1
5 2
5 1
4 2
2 3
2 0
1 1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int mod = 998244353;
const int M = 1e6 + 10;
const int N = 1e5 + 10;
inline long long read() {
  long long b = 1, sum = 0;
  char c = getchar();
  while (!isdigit(c)) {
    if (c == '-') b = -1;
    c = getchar();
  }
  while (isdigit(c)) {
    sum = sum * 10 + c - '0';
    c = getchar();
  }
  return b * sum;
}
int a[N], in[N], d[N];
int n, m, k, T;
int vis[N];
vector<int> e[N];
vector<pair<int, int> > ans;
void dfs(int x, int _fa, int need) {
  int i, j;
  if (x == 1) {
    for (i = 0; i < e[x].size(); i++) {
      int y = e[x][i];
      ans.push_back({y, i + 1});
      dfs(y, x, i);
    }
    return;
  }
  int cha = need - in[x] + 1, sum = 0;
  if (in[x] == 1) {
    ans.push_back({x, need});
  } else if (cha >= 0) {
    ans.push_back({x, cha});
    for (i = 0; i < e[x].size(); i++) {
      int y = e[x][i];
      if (y == _fa) continue;
      ans.push_back({y, cha + sum + 1});
      dfs(y, x, cha + sum);
      sum++;
    }
  } else {
    int now = need + 1, sum = 0;
    for (i = 0; sum < -cha && i < e[x].size(); i++) {
      int y = e[x][i];
      if (y == _fa) continue;
      ans.push_back({y, now + 1});
      dfs(y, x, now);
      now++;
      sum++;
    }
    ans.push_back({x, 0});
    sum = 0;
    for (; i < e[x].size(); i++) {
      int y = e[x][i];
      if (y == _fa) continue;
      ans.push_back({y, sum + 1});
      dfs(y, x, sum);
      sum++;
    }
  }
  ans.push_back({_fa, need + 1});
}
int main() {
  int i, j;
  n = read();
  for (i = 1; i <= n - 1; i++) {
    int x = read(), y = read();
    e[x].push_back(y);
    e[y].push_back(x);
    in[x]++;
    in[y]++;
  }
  ans.push_back({1, 0});
  dfs(1, 0, 0);
  printf(\"%d\n\", ans.size());
  for (i = 0; i < ans.size(); i++) {
    printf(\"%d %d\n\", ans[i].first, ans[i].second);
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import defaultdict
from bootcamp import Basebootcamp

class Fnastyaandtimemachinebootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = params.copy()
        self.params.setdefault('n', 5)
        # 确保最小节点数为1
        if self.params['n'] < 1:
            self.params['n'] = 1
    
    def case_generator(self):
        n = self.params['n']
        edges = self._generate_tree(n)
        return {'n': n, 'edges': edges}
    
    @staticmethod
    def _generate_tree(n):
        if n == 1:
            return []
        
        # 优化后的树生成算法
        nodes = list(range(1, n+1))
        if n == 2:
            return [(nodes[0], nodes[1])]
        
        # 改进的Prufer序列生成
        prufer = [random.choice(nodes) for _ in range(n-2)]
        degree = defaultdict(int)
        for node in prufer:
            degree[node] += 1
        
        adj = defaultdict(list)
        # 阶段1：处理Prufer序列
        for p in prufer:
            for v in nodes:
                if degree[v] == 0 and (p != v or degree[p] > 0):
                    adj[p].append(v)
                    adj[v].append(p)
                    degree[p] -= 1
                    degree[v] -= 1
                    break
        
        # 阶段2：处理剩余节点
        leaves = [v for v in nodes if degree[v] == 0]
        while len(leaves) >= 2:
            u = leaves.pop()
            v = leaves.pop()
            adj[u].append(v)
            adj[v].append(u)
        
        # 去重并排序边
        seen = set()
        edges = []
        for u in adj:
            for v in adj[u]:
                if u < v and (u, v) not in seen:
                    edges.append((u, v))
                    seen.add((u, v))
        return sorted(edges, key=lambda x: (x[0], x[1]))
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        edges = question_case['edges']
        edge_lines = "\n".join(f"{u} {v}" for u, v in edges)
        return f"""Denis needs to visit all squares in the city and return to square 1 as quickly as possible. The city is structured as a tree with {n} squares. The squares are connected by the following roads:

Input:
{n}
{edge_lines}

Find a valid route that meets all requirements:
1. Starts at (1, 0) and ends at square 1
2. All transitions are either time jumps (same square, lower time) or road moves (adjacent square, time+1)
3. All (square, time) pairs must be unique
4. Visits all {n} squares
5. Minimizes the maximum time value

Format your answer with the route length first, then all (square, time) pairs enclosed in [answer] and [/answer] tags:

Example format for a 5-node case:
[answer]
13
1 0
2 1
3 2
3 1
2 2
4 3
4 1
5 2
5 1
4 2
2 3
2 0
1 1
[/answer]"""
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        content = matches[-1].strip()
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        if len(lines) < 1:
            return None
        
        try:
            k = int(lines[0])
        except ValueError:
            return None
        
        if len(lines) != k + 1:
            return None
        
        solution = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) != 2:
                return None
            try:
                v = int(parts[0])
                t = int(parts[1])
                solution.append((v, t))
            except ValueError:
                return None
        
        return solution
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        
        n = identity['n']
        edges = identity['edges']
        
        # 构建邻接表时增加双向验证
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # 验证起点和终点
        if solution[0] != (1, 0) or solution[-1][0] != 1:
            return False
        
        # 验证所有节点被访问
        visited_nodes = {v for v, _ in solution}
        if visited_nodes != set(range(1, n+1)):
            return False
        
        # 时空坐标唯一性验证
        seen = set()
        prev_v, prev_t = solution[0]
        seen.add((prev_v, prev_t))
        
        for v, t in solution[1:]:
            if (v, t) in seen:
                return False
            seen.add((v, t))
            
            # 转移类型验证
            if v == prev_v:
                # 时间跳跃
                if t >= prev_t:
                    return False
            else:
                # 移动验证
                if t != prev_t + 1:
                    return False
                if v not in adj[prev_v]:
                    return False
            
            prev_v, prev_t = v, t
        
        return True
