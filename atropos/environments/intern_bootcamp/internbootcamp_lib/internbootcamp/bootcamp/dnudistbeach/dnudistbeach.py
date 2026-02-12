"""# 

### 谜题描述
Nudist Beach is planning a military operation to attack the Life Fibers. In this operation, they will attack and capture several cities which are currently under the control of the Life Fibers.

There are n cities, labeled from 1 to n, and m bidirectional roads between them. Currently, there are Life Fibers in every city. In addition, there are k cities that are fortresses of the Life Fibers that cannot be captured under any circumstances. So, the Nudist Beach can capture an arbitrary non-empty subset of cities with no fortresses.

After the operation, Nudist Beach will have to defend the captured cities from counterattack. If they capture a city and it is connected to many Life Fiber controlled cities, it will be easily defeated. So, Nudist Beach would like to capture a set of cities such that for each captured city the ratio of Nudist Beach controlled neighbors among all neighbors of that city is as high as possible. 

More formally, they would like to capture a non-empty set of cities S with no fortresses of Life Fibers. The strength of a city <image> is defined as (number of neighbors of x in S) / (total number of neighbors of x). Here, two cities are called neighbors if they are connnected with a road. The goal is to maximize the strength of the weakest city in S.

Given a description of the graph, and the cities with fortresses, find a non-empty subset that maximizes the strength of the weakest city. 

Input

The first line of input contains three integers n, m, k (2 ≤ n ≤ 100 000, 1 ≤ m ≤ 100 000, 1 ≤ k ≤ n - 1).

The second line of input contains k integers, representing the cities with fortresses. These cities will all be distinct. 

The next m lines contain the roads. The i-th of these lines will have 2 integers ai, bi (1 ≤ ai, bi ≤ n, ai ≠ bi). Every city will have at least one road adjacent to it.

There is no more than one road between each pair of the cities.

Output

The first line should contain an integer r, denoting the size of an optimum set (1 ≤ r ≤ n - k). 

The second line should contain r integers, denoting the cities in the set. Cities may follow in an arbitrary order. This line should not contain any of the cities with fortresses.

If there are multiple possible answers, print any of them.

Examples

Input

9 8 4
3 9 6 8
1 2
1 3
1 4
1 5
2 6
2 7
2 8
2 9


Output

3
1 4 5


Input

10 8 2
2 9
1 3
2 9
4 5
5 6
6 7
7 8
8 10
10 4


Output

8
1 5 4 8 10 6 3 7

Note

The first example case achieves a strength of 1/2. No other subset is strictly better.

The second example case achieves a strength of 1. Note that the subset doesn't necessarily have to be connected.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int MAXN = 100005;
const int MOD = 1000000007;
const double EPS = 1e-9;
vector<int> g[MAXN];
int tot[MAXN], good[MAXN];
int p[MAXN], q[MAXN];
set<int> bad;
bool vis[MAXN];
vector<int> res;
int main() {
  int n, m, k;
  scanf(\"%d%d%d\", &n, &m, &k);
  for (int i = 0; i < k; ++i) {
    int val;
    scanf(\"%d\", &val);
    --val;
    bad.insert(val);
  }
  for (int i = 0; i < m; ++i) {
    int a, b;
    scanf(\"%d%d\", &a, &b);
    --a, --b;
    g[a].push_back(b);
    g[b].push_back(a);
  }
  for (int i = 0; i < n; ++i) {
    good[i] = tot[i] = (int)g[i].size();
  }
  for (int u : bad) {
    for (int v : g[u]) {
      --good[v];
    }
  }
  double low = 0, high = 1;
  for (int i = 0; i < 50; ++i) {
    double mid = 0.5 * (low + high);
    queue<int> Q;
    for (int j = 0; j < n; ++j) {
      p[j] = tot[j];
      q[j] = good[j];
      vis[j] = false;
    }
    for (int j = 0; j < n; ++j) {
      if (!bad.count(j)) {
        double val = 1.0 * q[j] / p[j];
        if (val <= mid - EPS) {
          Q.push(j);
          vis[j] = true;
        }
      }
    }
    int cur = n - k;
    while (!Q.empty()) {
      --cur;
      int u = Q.front();
      Q.pop();
      for (int v : g[u]) {
        if (!bad.count(v)) {
          --q[v];
          double val = 1.0 * q[v] / p[v];
          if ((val <= mid - EPS) && !vis[v]) {
            Q.push(v);
            vis[v] = true;
          }
        }
      }
    }
    if (cur > 0) {
      low = mid;
      res.clear();
      for (int j = 0; j < n; ++j) {
        if (!vis[j] && !bad.count(j)) {
          res.push_back(j);
        }
      }
    } else {
      high = mid;
    }
  }
  printf(\"%d\n\", (int)res.size());
  for (int i = 0; i + 1 < (int)res.size(); ++i) {
    printf(\"%d \", res[i] + 1);
  }
  printf(\"%d\n\", res.back() + 1);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import deque
from bootcamp import Basebootcamp

def generate_valid_graph(n, m):
    """生成每个节点度数至少为1的图（不要求连通）"""
    if n < 2:
        raise ValueError("n must be at least 2")
    if m < n//2:
        m = max(m, n//2)  # 保证足够的最小边数
    
    edges = set()
    nodes = list(range(1, n+1))
    random.shuffle(nodes)
    
    # 保证每个节点至少有一个边
    remaining = nodes.copy()
    while remaining:
        if len(remaining) == 1:
            # 最后一个节点随机连接到已有节点
            node = remaining.pop()
            candidates = [x for x in nodes if x != node]
            if not candidates:
                raise ValueError("Can't create valid graph")
            neighbor = random.choice(candidates)
            edge = tuple(sorted((node, neighbor)))
            edges.add(edge)
        else:
            a = remaining.pop()
            b = remaining.pop()
            edge = tuple(sorted((a, b)))
            edges.add(edge)
    
    # 添加剩余边
    possible_edges = [(i, j) for i in range(1, n+1) for j in range(i+1, n+1) if (i, j) not in edges]
    while len(edges) < m and possible_edges:
        edge = possible_edges.pop(random.randint(0, len(possible_edges)-1))
        edges.add(edge)
    
    return sorted(edges)[:m]

def solve_case(n, m, k, fortresses, roads):
    bad = {f-1 for f in fortresses}
    adj = [[] for _ in range(n)]
    
    for a, b in roads:
        a0, b0 = a-1, b-1
        adj[a0].append(b0)
        adj[b0].append(a0)
    
    total_degree = [len(neighbors) for neighbors in adj]
    good_degree = [len(neighbors) for neighbors in adj]
    
    for u in bad:
        for v in adj[u]:
            good_degree[v] -= 1
    
    low, high = 0.0, 1.0
    best_solution = []
    
    for _ in range(50):
        mid = (low + high) / 2
        removed = set()
        current_good = good_degree.copy()
        queue = deque()
        
        for city in range(n):
            if city not in bad and total_degree[city] > 0:
                ratio = current_good[city] / total_degree[city]
                if ratio <= mid - 1e-9:
                    queue.append(city)
                    removed.add(city)
        
        temp_removed = set(removed)
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v not in bad and v not in temp_removed:
                    current_good[v] -= 1
                    if current_good[v]/total_degree[v] <= mid - 1e-9:
                        queue.append(v)
                        temp_removed.add(v)
        
        valid_cities = [city for city in range(n) if city not in bad and city not in temp_removed]
        if valid_cities:
            low = mid
            best_solution = [c+1 for c in valid_cities]
        else:
            high = mid
    
    return best_solution

class Dnudistbeachbootcamp(Basebootcamp):
    def __init__(self, **params):
        default_params = {'n': 8, 'm': 10, 'k': 2}
        default_params.update(params)
        
        # 参数校验
        n = max(2, default_params['n'])
        m = max(n-1, default_params['m'])  # 确保足够的最小边数
        k = max(1, min(default_params['k'], n-1))
        
        self.params = {'n': n, 'm': m, 'k': k}
        self.n = n
        self.m = m
        self.k = k
    
    def case_generator(self):
        n, m, k = self.n, self.m, self.k
        fortresses = random.sample(range(1, n+1), k)
        
        # 生成保证每个城市有至少一条路的图
        roads = generate_valid_graph(n, m)
        
        # 计算正确答案
        correct_solution = solve_case(n, m, k, fortresses, roads)
        
        # 验证解决方案存在
        if not correct_solution:
            raise RuntimeError("Failed to generate valid solution")
        
        # 计算最小强度值
        adj = {city: set() for city in range(1, n+1)}
        for a, b in roads:
            adj[a].add(b)
            adj[b].add(a)
        
        min_strength = min(
            sum(1 for neighbor in adj[city] if neighbor in correct_solution) / len(adj[city])
            for city in correct_solution
        )
        
        return {
            'n': n, 'm': m, 'k': k,
            'fortresses': sorted(fortresses),
            'roads': roads,
            'correct_answer': correct_solution,
            'max_min_strength': min_strength
        }
    
    @staticmethod
    def prompt_func(question_case):
        fortress_list = ', '.join(map(str, question_case['fortresses']))
        road_list = '\n'.join(f'{a} {b}' for a, b in question_case['roads'])
        return f"""Nudist Beach军事行动需要选择占领城市集合。已知：
- 总城市数：{question_case['n']}个（编号1~{question_case['n']}）
- 道路数量：{question_case['m']}条
- 禁城列表：{fortress_list}

规则：
1. 必须选择非空的非禁城集合
2. 每个被占城市的强度 = (被占邻居数)/(总邻居数)
3. 目标：最大化集合中最小的强度值

道路连接：
{road_list}

请给出最优解。答案格式：
[answer]
第一行：城市数量r
第二行：用空格分隔的r个城市编号
[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        
        content = matches[-1].strip()
        try:
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            r = int(lines[0])
            cities = list(map(int, lines[1].split()))
            if r != len(cities) or r < 1:
                return None
            if len(set(cities)) != len(cities):
                return None
            return {'r': r, 'cities': cities}
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        
        cities = solution['cities']
        r = solution['r']
        
        # 基础校验
        if r != len(cities) or r < 1:
            return False
        if any(c in identity['fortresses'] for c in cities):
            return False
        if any(c < 1 or c > identity['n'] for c in cities):
            return False
        if len(set(cities)) != len(cities):
            return False
        
        # 构建邻接表
        adj = {c: set() for c in range(1, identity['n']+1)}
        for a, b in identity['roads']:
            adj[a].add(b)
            adj[b].add(a)
        
        # 计算实际最小强度值
        current_min = float('inf')
        for city in cities:
            neighbors = adj[city]
            in_solution = sum(1 for n in neighbors if n in cities)
            strength = in_solution / len(neighbors)
            current_min = min(current_min, strength)
        
        # 允许浮点误差
        return abs(current_min - identity['max_min_strength']) < 1e-6
