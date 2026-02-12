"""# 

### 谜题描述
One remarkable day company \"X\" received k machines. And they were not simple machines, they were mechanical programmers! This was the last unsuccessful step before switching to android programmers, but that's another story.

The company has now n tasks, for each of them we know the start time of its execution si, the duration of its execution ti, and the company profit from its completion ci. Any machine can perform any task, exactly one at a time. If a machine has started to perform the task, it is busy at all moments of time from si to si + ti - 1, inclusive, and it cannot switch to another task.

You are required to select a set of tasks which can be done with these k machines, and which will bring the maximum total profit.

Input

The first line contains two integer numbers n and k (1 ≤ n ≤ 1000, 1 ≤ k ≤ 50) — the numbers of tasks and machines, correspondingly.

The next n lines contain space-separated groups of three integers si, ti, ci (1 ≤ si, ti ≤ 109, 1 ≤ ci ≤ 106), si is the time where they start executing the i-th task, ti is the duration of the i-th task and ci is the profit of its execution.

Output

Print n integers x1, x2, ..., xn. Number xi should equal 1, if task i should be completed and otherwise it should equal 0.

If there are several optimal solutions, print any of them.

Examples

Input

3 1
2 7 5
1 3 3
4 1 3


Output

0 1 1


Input

5 2
1 5 4
1 4 5
1 3 2
4 1 2
5 6 1


Output

1 1 0 0 1

Note

In the first sample the tasks need to be executed at moments of time 2 ... 8, 1 ... 3 and 4 ... 4, correspondingly. The first task overlaps with the second and the third ones, so we can execute either task one (profit 5) or tasks two and three (profit 6).

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
priority_queue<pair<int, int> > que;
int size, n, k, S, T, dis[2010000], i, j, g[2010000], p[2010000], flow[2010000],
    num[2010000], tot, ex[2010000], cnt, vis[2010000], h[2010000];
struct node {
  int to, next, f, v;
} e[2010000];
void add1(int o, int p, int q, int w) {
  e[++size].to = p, e[size].next = g[o], g[o] = size, e[size].f = q,
  e[size].v = w;
}
void add(int o, int p, int q, int w) { add1(o, p, q, w), add1(p, o, 0, -w); }
struct node1 {
  int s, t, val;
} a[2010000];
bool check(int i, int j) {
  int tmp = a[i].s + a[i].t - 1;
  return tmp < a[j].s;
}
void init() {
  sort(ex + 1, ex + 1 + cnt);
  tot = unique(ex + 1, ex + 1 + cnt) - ex - 1;
  for (i = 1; i <= n; i++) {
    a[i].s = lower_bound(ex + 1, ex + 1 + tot, a[i].s) - ex;
    a[i].t = lower_bound(ex + 1, ex + 1 + tot, a[i].t) - ex;
  }
}
void mcmf() {
  for (i = 1; i <= T; i++) {
    for (int x = S; x <= T; x++)
      for (int k = g[x]; k; k = e[k].next) {
        if (e[k].f == 0) continue;
        int y = e[k].to;
        if (h[y] < h[x] + e[k].v) h[y] = h[x] + e[k].v;
      }
  }
  while (1) {
    for (i = S; i <= T; i++) dis[i] = -2000000000, vis[i] = 0;
    dis[S] = 0;
    que.push(make_pair(0, S));
    flow[S] = 2000000000;
    while (!que.empty()) {
      int x = que.top().second;
      que.pop();
      if (vis[x] == 1) continue;
      vis[x] = 1;
      for (int k = g[x]; k; k = e[k].next) {
        int y = e[k].to, cost = e[k].v + h[x] - h[y];
        if (e[k].f && dis[y] < dis[x] + cost) {
          dis[y] = dis[x] + cost;
          flow[y] = min(flow[x], e[k].f);
          p[y] = k;
          que.push(make_pair(dis[y], y));
        }
      }
    }
    if (vis[T] == 0) break;
    for (i = S; i <= T; i++) h[i] += dis[i];
    int now = p[T];
    while (now) {
      e[now].f -= flow[T], e[now ^ 1].f += flow[T], now = p[e[now ^ 1].to];
    };
  }
}
int main() {
  scanf(\"%d %d\", &n, &k);
  size = 1;
  for (i = 1; i <= n; i++) {
    scanf(\"%d %d %d\", &a[i].s, &a[i].t, &a[i].val);
    a[i].t += a[i].s;
    ex[++cnt] = a[i].s, ex[++cnt] = a[i].t;
  }
  init();
  S = 0, T = tot + 1;
  for (i = 0; i <= tot; i++) add(i, i + 1, k, 0);
  for (i = 1; i <= n; i++) add(a[i].s, a[i].t, 1, a[i].val), num[i] = size - 1;
  mcmf();
  for (i = 1; i <= n; i++) printf(\"%d \", 1 - e[num[i]].f);
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cmachineprogrammingbootcamp(Basebootcamp):
    def __init__(self, max_n=8, max_k=3, time_max=20, **params):
        super().__init__()
        self.max_n = max_n
        self.max_k = max_k
        self.time_max = time_max

    def case_generator(self):
        n = random.randint(3, self.max_n)
        k = random.randint(1, self.max_k)
        
        tasks = []
        for _ in range(n):
            si = random.randint(1, self.time_max)
            ti = random.randint(1, self.time_max//2)
            ci = random.randint(1, 100)
            tasks.append({
                'si': si,
                'ti': ti,
                'ci': ci
            })
        
        best_profit = self.calculate_optimal(tasks, k)
        
        return {
            'n': n,
            'k': k,
            'tasks': tasks,
            'optimal_profit': best_profit
        }
    
    @staticmethod
    def calculate_optimal(tasks, k):
        if not tasks:
            return 0
            
        sorted_tasks = sorted(tasks, key=lambda x: x['si'] + x['ti'])
        n = len(sorted_tasks)
        dp = [[0]*(k+1) for _ in range(n+1)]
        
        for i in range(1, n+1):
            current = sorted_tasks[i-1]
            s_i = current['si']
            end_i = s_i + current['ti']
            
            j = i-2
            while j >= 0 and (sorted_tasks[j]['si'] + sorted_tasks[j]['ti']) > s_i:
                j -= 1
                
            for m in range(1, k+1):
                include_profit = current['ci']
                if j >= 0:
                    include_profit += dp[j+1][m-1]
                dp[i][m] = max(dp[i-1][m], include_profit)
        
        return max(dp[n])

    @staticmethod
    def calculate_overlap(solution_tasks, k):
        timeline = []
        for task in solution_tasks:
            start = task['si']
            end = start + task['ti']
            timeline.append((start, 1))
            timeline.append((end, -1))
        
        timeline.sort()
        current = 0
        peak = 0
        for t, delta in timeline:
            current += delta
            peak = max(peak, current)
        return peak <= k

    @staticmethod
    def prompt_func(question_case):
        tasks = question_case['tasks']
        n = question_case['n']
        k = question_case['k']
        problem = f"Company X has {k} machine{'s' if k>1 else ''} and {n} tasks:\n"
        problem += "Each task has [start time, duration, profit]:\n"
        for i, t in enumerate(tasks, 1):
            problem += f"Task {i}: {t['si']} {t['ti']} {t['ci']}\n"
        problem += "\nSelect tasks to maximize profit without overlapping.\n"
        problem += "Output format: 0/1 sequence like: [answer]1 0 1[/answer]"
        return problem

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return list(map(int, last_match.split()))
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 格式验证
        if not solution or len(solution) != identity['n']:
            return False
        if any(bit not in (0,1) for bit in solution):
            return False
        
        # 选择的任务列表
        selected = [t for t, bit in zip(identity['tasks'], solution) if bit]
        
        # 计算实际利润
        actual_profit = sum(t['ci'] for t in selected)
        
        # 验证最优性
        if actual_profit != identity['optimal_profit']:
            return False
        
        # 验证机器约束
        return cls.calculate_overlap(selected, identity['k'])
