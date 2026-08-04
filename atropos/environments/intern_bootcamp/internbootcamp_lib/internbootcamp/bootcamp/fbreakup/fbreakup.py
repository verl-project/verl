"""# 

### 谜题描述
Again, there are hard times in Berland! Many towns have such tensions that even civil war is possible. 

There are n towns in Reberland, some pairs of which connected by two-way roads. It is not guaranteed that it is possible to reach one town from any other town using these roads. 

Towns s and t announce the final break of any relationship and intend to rule out the possibility of moving between them by the roads. Now possibly it is needed to close several roads so that moving from s to t using roads becomes impossible. Each town agrees to spend money on closing no more than one road, therefore, the total number of closed roads will be no more than two.

Help them find set of no more than two roads such that there will be no way between s and t after closing these roads. For each road the budget required for its closure was estimated. Among all sets find such that the total budget for the closure of a set of roads is minimum.

Input

The first line of the input contains two integers n and m (2 ≤ n ≤ 1000, 0 ≤ m ≤ 30 000) — the number of towns in Berland and the number of roads.

The second line contains integers s and t (1 ≤ s, t ≤ n, s ≠ t) — indices of towns which break up the relationships.

Then follow m lines, each of them contains three integers xi, yi and wi (1 ≤ xi, yi ≤ n, 1 ≤ wi ≤ 109) — indices of towns connected by the i-th road, and the budget on its closure.

All roads are bidirectional. It is allowed that the pair of towns is connected by more than one road. Roads that connect the city to itself are allowed. 

Output

In the first line print the minimum budget required to break up the relations between s and t, if it is allowed to close no more than two roads.

In the second line print the value c (0 ≤ c ≤ 2) — the number of roads to be closed in the found solution.

In the third line print in any order c diverse integers from 1 to m — indices of closed roads. Consider that the roads are numbered from 1 to m in the order they appear in the input. 

If it is impossible to make towns s and t disconnected by removing no more than 2 roads, the output should contain a single line -1. 

If there are several possible answers, you may print any of them.

Examples

Input

6 7
1 6
2 1 6
2 3 5
3 4 9
4 6 4
4 6 5
4 5 1
3 1 3


Output

8
2
2 7


Input

6 7
1 6
2 3 1
1 2 2
1 3 3
4 5 4
3 6 5
4 6 6
1 5 7


Output

9
2
4 5


Input

5 4
1 5
2 1 3
3 2 1
3 4 4
4 5 2


Output

1
1
2


Input

2 3
1 2
1 2 734458840
1 2 817380027
1 2 304764803


Output

-1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
inline void down(int &a, const int &b) {
  if (a > b) a = b;
}
const int maxn = 2100;
const int maxm = 110000;
int n, m, S, T;
int e[maxm][3], ok[maxm];
int t[maxm], tp;
struct edge {
  int y, i, nex;
} a[maxm];
int len, fir[maxn];
inline void ins(const int x, const int y, const int i) {
  a[++len] = (edge){y, i, fir[x]};
  fir[x] = len;
}
int dfn[maxn], low[maxn], dfi, ib[maxm];
int fa[maxn], fai[maxn];
void init() {
  for (int i = 1; i <= n; i++) dfn[i] = fa[i] = fai[i] = 0;
  for (int i = 1; i <= m; i++) ib[i] = 0;
  dfi = 0;
}
void tarjan(const int x) {
  dfn[x] = low[x] = ++dfi;
  for (int k = fir[x], y = a[k].y; k; k = a[k].nex, y = a[k].y)
    if (a[k].i != fai[x] && !ok[a[k].i]) {
      if (!dfn[y]) {
        fai[y] = a[k].i, fa[y] = x;
        tarjan(y);
        down(low[x], low[y]);
        if (low[y] == dfn[y]) ib[a[k].i] = 1;
      } else
        down(low[x], dfn[y]);
    }
}
int ans, ansn, re[10];
int main() {
  scanf(\"%d%d\", &n, &m);
  scanf(\"%d%d\", &S, &T);
  for (int i = 1; i <= m; i++) {
    scanf(\"%d%d%d\", &e[i][0], &e[i][1], &e[i][2]);
    ins(e[i][0], e[i][1], i);
    ins(e[i][1], e[i][0], i);
  }
  init();
  tarjan(S);
  if (!dfn[T]) {
    puts(\"0\");
    puts(\"0\");
    putchar('\n');
    return 0;
  }
  ans = INT_MAX;
  for (int i = T; i != S; i = fa[i]) {
    int k = fai[i];
    if (ib[k]) {
      if (ans > e[k][2]) ans = e[k][2], re[ansn = 1] = k;
    } else
      t[++tp] = k;
  }
  for (int i = 1; i <= tp; i++) {
    ok[t[i]] = 1;
    init();
    tarjan(S);
    for (int j = T; j != S; j = fa[j]) {
      int k = fai[j];
      if (ib[k]) {
        if (ans > e[t[i]][2] + e[k][2])
          ans = e[t[i]][2] + e[k][2], ansn = 2, re[1] = t[i], re[2] = k;
      }
    }
    ok[t[i]] = 0;
  }
  if (ans == INT_MAX)
    puts(\"-1\");
  else {
    printf(\"%d\n\", ans);
    printf(\"%d\n\", ansn);
    for (int i = 1; i <= ansn; i++) printf(\"%d \", re[i]);
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import deque
from bootcamp import Basebootcamp

class Fbreakupbootcamp(Basebootcamp):
    def __init__(self, node_range=(4,8), budget_range=(10, 100)):
        """
        参数说明:
        node_range: 生成图的节点数量范围 (min, max)
        budget_range: 道路关闭预算范围 (min, max)
        """
        self.min_nodes, self.max_nodes = node_range
        self.min_budget, self.max_budget = budget_range
    
    def case_generator(self):
        # 随机选择案例类型：单边解/双边解/无解
        case_type = random.choices(
            ['single', 'double', 'impossible'], 
            weights=[45, 45, 10], 
            k=1
        )[0]

        if case_type == 'single':
            return self._generate_single_case()
        elif case_type == 'double':
            return self._generate_double_case()
        else:
            return self._generate_impossible_case()

    def _generate_single_case(self):
        """生成需要切断一条边的案例（链式结构）"""
        n = random.randint(self.min_nodes, self.max_nodes)
        s, t = 1, n
        roads = []
        for i in range(n-1):
            w = random.randint(self.min_budget, self.max_budget)
            roads.append( (i+1, i+2, w) )
        
        # 找预算最小的边
        min_idx, (x,y,min_w) = min(enumerate(roads), key=lambda x: x[1][2])
        return {
            'n': n,
            'm': n-1,
            's': s,
            't': t,
            'roads': roads,
            'expected': {
                'min_budget': min_w,
                'c': 1,
                'roads': [min_idx+1]  # 道路编号从1开始
            }
        }

    def _generate_double_case(self):
        """生成需要切断两条边的案例（并行双路径结构）"""
        # s=1, t=4
        roads = [
            (1,2, random.randint(10,50)),  # 路径1-边1
            (2,4, random.randint(10,50)),  # 路径1-边2
            (1,3, random.randint(10,50)),  # 路径2-边1
            (3,4, random.randint(10,50)),  # 路径2-边2
            (2,3, self.max_budget*2)       # 高成本边，不应被选
        ]
        # 最优解为选两条路径各一个最低成本边
        path1 = [roads[0][2], roads[1][2]]
        path2 = [roads[2][2], roads[3][2]]
        min1 = min(path1)
        min2 = min(path2)
        solution = {
            'min_budget': min1 + min2,
            'c': 2,
            'roads': [
                roads.index(r)+1 for r in roads 
                if r[2] in (min1, min2)
            ]
        }
        return {
            'n': 4,
            'm': 5,
            's': 1,
            't': 4,
            'roads': roads,
            'expected': solution
        }

    def _generate_impossible_case(self):
        """生成无法断开连接的案例"""
        return {
            'n': 3,
            'm': 4,
            's': 1,
            't': 3,
            'roads': [
                (1,2, 10), (2,3, 20),
                (1,3, 30), (1,3, 40)
            ],
            'expected': -1
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        roads = "\n".join(
            f"Road {i+1}: {x}-{y} (closure cost {w})" 
            for i, (x,y,w) in enumerate(question_case['roads'])
        )
        return f"""Cities {question_case['s']} and {question_case['t']} need to disconnect. 
Find up to 2 roads to close with minimal total cost.

Cities: {question_case['n']}
Roads:
{roads}

Output format:
[answer]
<total_cost>
<road_count>
<road_numbers>
[/answer]"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_answer = matches[-1].strip()
        
        lines = [l.strip() for l in last_answer.split('\n') if l.strip()]
        if len(lines) < 2:
            return None
        
        try:
            # 处理无解情况
            if lines[0] == '-1':
                return {'min_budget': -1, 'c': 0, 'roads': []}
            
            # 解析正常情况
            total_cost = int(lines[0])
            road_count = int(lines[1])
            if road_count == 0:
                return {'min_budget': total_cost, 'c': 0, 'roads': []}
            
            if len(lines) < 3:
                return None
            road_indices = list(map(int, lines[2].split()))
            if len(road_indices) != road_count or road_count not in (1,2):
                return None
            
            return {
                'min_budget': total_cost,
                'c': road_count,
                'roads': road_indices
            }
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 处理无解情况
        if identity['expected'] == -1:
            return solution.get('min_budget') == -1
        
        # 验证数值正确性
        expected = identity['expected']
        if (solution['min_budget'] != expected['min_budget'] or
            solution['c'] != expected['c']):
            return False
        
        # 验证边集合是否匹配（顺序无关）
        if set(solution['roads']) != set(expected['roads']):
            return False
        
        # 验证图是否真正断开
        return cls._is_disconnected(
            identity['n'], 
            identity['roads'],
            identity['s'],
            identity['t'],
            solution['roads']
        )

    @staticmethod
    def _is_disconnected(n, roads, s, t, deleted_roads):
        """判断删除指定边后是否断开连接"""
        deleted = set(deleted_roads)
        adj = [[] for _ in range(n+1)]
        for idx, (x,y,w) in enumerate(roads):
            if (idx+1) not in deleted:
                adj[x].append(y)
                adj[y].append(x)
        
        # BFS检查连通性
        visited = [False]*(n+1)
        queue = deque([s])
        visited[s] = True
        while queue:
            u = queue.popleft()
            if u == t:
                return False
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        return True
