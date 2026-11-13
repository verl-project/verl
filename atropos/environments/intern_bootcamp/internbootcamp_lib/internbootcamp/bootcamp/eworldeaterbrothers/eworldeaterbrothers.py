"""# 

### 谜题描述
You must have heard of the two brothers dreaming of ruling the world. With all their previous plans failed, this time they decided to cooperate with each other in order to rule the world. 

As you know there are n countries in the world. These countries are connected by n - 1 directed roads. If you don't consider direction of the roads there is a unique path between every pair of countries in the world, passing through each road at most once. 

Each of the brothers wants to establish his reign in some country, then it's possible for him to control the countries that can be reached from his country using directed roads. 

The brothers can rule the world if there exists at most two countries for brothers to choose (and establish their reign in these countries) so that any other country is under control of at least one of them. In order to make this possible they want to change the direction of minimum number of roads. Your task is to calculate this minimum number of roads.

Input

The first line of input contains an integer n (1 ≤ n ≤ 3000). Each of the next n - 1 lines contains two space-separated integers ai and bi (1 ≤ ai, bi ≤ n; ai ≠ bi) saying there is a road from country ai to country bi.

Consider that countries are numbered from 1 to n. It's guaranteed that if you don't consider direction of the roads there is a unique path between every pair of countries in the world, passing through each road at most once.

Output

In the only line of output print the minimum number of roads that their direction should be changed so that the brothers will be able to rule the world.

Examples

Input

4
2 1
3 1
4 1


Output

1


Input

5
2 1
2 3
4 3
4 5


Output

0

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int maxn = 3000 + 10;
int n;
struct Tside {
  int x, y, v;
  int num;
  Tside *next;
} * h[maxn];
int f[maxn], floors[maxn];
int res = 1000000, res1, res2;
int tx, ty, tv;
void swap(int &a, int &b) {
  int tmp = a;
  a = b;
  b = tmp;
}
void ins(int x, int y, int num) {
  Tside *tmp = h[x];
  h[x] = new (Tside);
  h[x]->x = x;
  h[x]->y = y;
  h[x]->v = 0;
  h[x]->num = num;
  h[x]->next = tmp;
  tmp = h[y];
  h[y] = new (Tside);
  h[y]->x = y;
  h[y]->y = x;
  h[y]->v = 1;
  h[y]->num = num;
  h[y]->next = tmp;
}
void dfs1(int s, int fa) {
  floors[s] = floors[fa] + 1;
  f[s] = 0;
  Tside *tmp;
  tmp = h[s];
  while (tmp != NULL) {
    if (tmp->y != fa) {
      dfs1(tmp->y, s);
      f[s] += f[tmp->y] + tmp->v;
    }
    tmp = tmp->next;
  }
}
void dfs2(int s, int fa, int num, int ans) {
  Tside *tmp = h[s];
  if (num == 0) {
    if (ans < res2) res2 = ans;
  } else {
    if (ans < res1) res1 = ans;
  }
  int tmp_ans;
  while (tmp != NULL) {
    if (tmp->y != fa && tmp->num != num) {
      if (tmp->v == 1)
        tmp_ans = ans - 1;
      else
        tmp_ans = ans + 1;
      dfs2(tmp->y, s, num, tmp_ans);
    }
    tmp = tmp->next;
  }
}
int main() {
  cin >> n;
  for (int i = 1; i <= n; i++) h[i] = NULL;
  for (int i = 1; i < n; i++) {
    cin >> tx >> ty;
    ins(tx, ty, i);
  }
  floors[0] = 0;
  dfs1(1, 0);
  Tside *tmp;
  for (int i = 1; i <= n; i++) {
    tmp = h[i];
    while (tmp != NULL) {
      tx = tmp->x;
      ty = tmp->y;
      tv = tmp->v;
      if (floors[tx] > floors[ty]) {
        swap(tx, ty);
        tv = 1 - tv;
      }
      res1 = f[1] - f[ty] - tv;
      dfs2(1, 0, tmp->num, res1);
      res2 = f[ty];
      dfs2(ty, tx, 0, res2);
      if (res > res1 + res2) res = res1 + res2;
      tmp = tmp->next;
    }
  }
  if (res == 1000000) res = 0;
  cout << res << endl;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
from random import randint, choice
from collections import defaultdict
from bootcamp import Basebootcamp

class Eworldeaterbrothersbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', 4)
        self.max_retry = params.get('max_retry', 3)  # 添加容错机制
    
    def case_generator(self):
        for _ in range(self.max_retry):
            try:
                n = self.n
                if n == 1:
                    return {'n':1, 'edges':[], 'expected':0}
                
                # 生成随机树结构（保证连通无环）
                parents = {}
                undirected_edges = []
                for i in range(2, n+1):
                    parents[i] = randint(1, i-1)
                    undirected_edges.append((parents[i], i))
                
                # 随机分配方向
                directed_edges = []
                for a, b in undirected_edges:
                    if choice([True, False]):
                        directed_edges.append((a, b))
                    else:
                        directed_edges.append((b, a))
                
                # 计算正确答案
                expected = self._calculate_min_reversals(n, directed_edges)
                return {
                    'n': n,
                    'edges': directed_edges,
                    'expected': expected
                }
            except Exception as e:
                continue
        raise RuntimeError("Case generation failed after retries")

    @staticmethod
    def prompt_func(question_case):
        edges = question_case['edges']
        edges_str = '\n'.join([f"{a} {b}" for a, b in edges])
        return f"""The two brothers aim to control all countries by directing roads. Find the minimal road reversals needed.

Input:
n = {question_case['n']}
Roads (directed from a to b):
{edges_str}

Output the minimal number using [answer] and [/answer]. Example: [answer]0[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return int(solution) == identity['expected']
        except:
            return False

    @staticmethod
    def _calculate_min_reversals(n, edges):
        if n <= 1:
            return 0
        
        # 构建双向邻接表
        h = defaultdict(list)
        edge_dict = {}
        for idx, (a, b) in enumerate(edges):
            num = idx + 1
            h[a].append({'y':b, 'v':0, 'num':num})
            h[b].append({'y':a, 'v':1, 'num':num})
            edge_dict[num] = (a, b)

        # 第一遍DFS计算层级和初始cost
        floors = [0]*(n+1)
        f = [0]*(n+1)
        stack = [(1, 0, False)]
        while stack:
            node, parent, visited = stack.pop()
            if not visited:
                floors[node] = floors[parent] + 1
                stack.append((node, parent, True))
                # 按随机顺序处理子节点（避免生成链式结构）
                children = [edge for edge in h[node] if edge['y'] != parent]
                for edge in reversed(children):
                    stack.append((edge['y'], node, False))
            else:
                f[node] = 0
                for edge in h[node]:
                    if edge['y'] != parent:
                        f[node] += f[edge['y']] + edge['v']

        min_flips = float('inf')
        processed = set()

        # 遍历所有可能的切割边
        for num in edge_dict:
            if num in processed:
                continue
            processed.add(num)
            
            a, b = edge_dict[num]
            # 确定父子关系
            if floors[a] > floors[b]:
                parent, child = b, a
                original_dir = 1  # 当前方向是child->parent
            else:
                parent, child = a, b
                original_dir = 0  # 当前方向是parent->child

            # 计算上半部分的最小翻转
            upper_min = f[1] - f[child] - original_dir
            stack = [(1, 0, upper_min)]
            current_min = upper_min
            while stack:
                node, father, cost = stack.pop()
                current_min = min(current_min, cost)
                for edge in h[node]:
                    if edge['y'] != father and edge['num'] != num:
                        new_cost = cost - 1 if edge['v'] else cost + 1
                        stack.append((edge['y'], node, new_cost))
            
            # 计算下半部分的最小翻转
            lower_min = f[child]
            stack = [(child, parent, lower_min)]
            current_lower = lower_min
            while stack:
                node, father, cost = stack.pop()
                current_lower = min(current_lower, cost)
                for edge in h[node]:
                    if edge['y'] != father and edge['num'] != num:
                        new_cost = cost - 1 if edge['v'] else cost + 1
                        stack.append((edge['y'], node, new_cost))
            
            min_flips = min(min_flips, current_min + current_lower)

        return min_flips if min_flips != float('inf') else 0
