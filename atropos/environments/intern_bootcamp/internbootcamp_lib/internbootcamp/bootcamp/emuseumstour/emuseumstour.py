"""# 

### 谜题描述
In the country N, there are n cities connected by m one-way roads. Although this country seems unremarkable, there are two interesting facts about it. At first, a week lasts d days here. At second, there is exactly one museum in each city of the country N.

Travel agency \"Open museums\" is developing a new program for tourists interested in museums. Agency's employees know which days each of the museums is open. The tour should start in the capital — the city number 1, and the first day of the tour must be on the first day of a week. Each day a tourist will be in some city, watching the exposition in its museum (in case museum is open today), and by the end of the day, the tour either ends or the tourist goes into another city connected by a road with the current one. The road system of N is designed in such a way that traveling by a road always takes one night and also all the roads are one-way. It's allowed to visit a city multiple times during the trip.

You should develop such route for the trip that the number of distinct museums, possible to visit during it, is maximum.

Input

The first line contains three integers n, m and d (1 ≤ n ≤ 100 000, 0 ≤ m ≤ 100 000, 1 ≤ d ≤ 50), the number of cities, the number of roads and the number of days in a week.

Each of next m lines contains two integers u_i and v_i (1 ≤ u_i, v_i ≤ n, u_i ≠ v_i), denoting a one-way road from the city u_i to the city v_i.

The next n lines contain the museums' schedule. The schedule of the museum located in the i-th city is described in the i-th of these lines. Each line consists of exactly d characters \"0\" or \"1\", the j-th character of the string equals to \"1\" if the museum is open at the j-th day of a week, and \"0\", otherwise.

It's guaranteed that for each pair of cities (u, v) there exists no more than one road leading from u to v.

Output

Print a single integer — the maximum number of distinct museums, that it's possible to visit, starting a trip in the first city on the first day of the week.

Examples

Input


4 5 3
3 1
1 2
2 4
4 1
2 3
011
110
111
001


Output


3


Input


3 3 7
1 2
1 3
2 3
1111111
0000000
0111111


Output


2

Note

Explanation of the first example <image>

The maximum number of distinct museums to visit is 3. It's possible to visit 3 museums, for example, in the way described below.

  * Day 1. Now it's the 1st day of a week, and the tourist is in the city 1. The museum there is closed. At night the tourist goes to the city number 2. 
  * Day 2. Now it's the 2nd day of a week, and the tourist is in the city 2. The museum there is open, and the tourist visits it. At night the tourist goes to the city number 4. 
  * Day 3. Now it's the 3rd day of a week, and the tourist is in the city 4. The museum there is open, and the tourist visits it. At night the tourist goes to the city number 1. 
  * Day 4. Now it's the 1st day of a week, and the tourist is in the city 1. The museum there is closed. At night the tourist goes to the city number 2. 
  * Day 5. Now it's the 2nd of a week number 2, and the tourist is in the city 2. The museum there is open, but the tourist has already visited it. At night the tourist goes to the city number 3. 
  * Day 6. Now it's the 3rd day of a week, and the tourist is in the city 3. The museum there is open, and the tourist visits it. After this, the tour is over. 

Explanation of the second example <image>

The maximum number of distinct museums to visit is 2. It's possible to visit 2 museums, for example, in the way described below.

  * Day 1. Now it's the 1st day of a week, and the tourist is in the city 1. The museum there is open, and the tourist visits it. At night the tourist goes to the city number 2. 
  * Day 2. Now it's the 2nd day of a week, and the tourist is in the city 2. The museum there is closed. At night the tourist goes to the city number 3. 
  * Day 3. Now it's the 3rd day of a week, and the tourist is in the city 3. The museum there is open, and the tourist visits it. After this, the tour is over. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
ostream& operator<<(ostream& cerr, vector<long long> aux) {
  cerr << \"[\";
  for (auto e : aux) cerr << e << ' ';
  cerr << \"]\";
  return cerr;
}
struct state {
  int x, y;
  int where;
};
const int maxN = 100011;
const int maxD = 54;
const int init = 1000000000;
int n, m, d, x, y;
vector<int> adj[maxN], rev[maxN];
char open[maxN][maxD];
bool us[maxN][maxD];
int best[maxN][maxD];
stack<pair<int, int> > S;
int nxt[maxD], previous[maxD];
vector<pair<int, int> > here;
int best_here;
int cnt_vis;
bool vis[maxN];
int sol;
stack<state> dfStack;
void dfs(int x, int y) {
  int wh;
  dfStack.push({x, y, 0});
  while (!dfStack.empty()) {
    auto act = dfStack.top();
    dfStack.pop();
    x = act.x;
    y = act.y;
    wh = act.where;
    us[x][y] = true;
    while (wh < adj[x].size() && us[adj[x][wh]][nxt[y]]) wh++;
    if (wh == adj[x].size()) {
      S.push(make_pair(x, y));
      continue;
    } else {
      dfStack.push({x, y, wh + 1});
      dfStack.push({adj[x][wh], nxt[y], 0});
    }
  }
}
void dfs2(int x, int y) {
  int wh;
  dfStack.push({x, y, 0});
  while (!dfStack.empty()) {
    auto act = dfStack.top();
    dfStack.pop();
    x = act.x;
    y = act.y;
    wh = act.where;
    us[x][y] = true;
    if (wh == 0) {
      here.push_back(make_pair(x, y));
      best_here = max(best_here, best[x][y]);
      if (open[x][y] == '1' && vis[x] == false) {
        vis[x] = true;
        cnt_vis++;
      }
    }
    while (wh < rev[x].size() && us[rev[x][wh]][previous[y]]) wh++;
    if (wh == rev[x].size()) {
      continue;
    } else {
      dfStack.push({x, y, wh + 1});
      dfStack.push({rev[x][wh], previous[y], 0});
    }
  }
}
int main() {
  scanf(\"%d%d%d\n\", &n, &m, &d);
  for (int i = 1; i <= m; i++) {
    scanf(\"%d%d\n\", &x, &y);
    adj[x].push_back(y);
    rev[y].push_back(x);
  }
  for (int i = 1; i <= n; i++) scanf(\"%s\n\", open[i]);
  for (int i = 0; i < d; i++) {
    nxt[i] = (i + 1) % d;
    previous[i] = (i - 1 + d) % d;
  }
  for (int i = 1; i <= n; i++) {
    for (int j = 0; j < d; j++) {
      if (us[i][j]) continue;
      dfs(i, j);
    }
  }
  memset(us, 0, sizeof(us));
  best[1][0] = init;
  while (!S.empty()) {
    auto act = S.top();
    S.pop();
    x = act.first;
    y = act.second;
    if (us[x][y]) continue;
    here.clear();
    best_here = 0;
    cnt_vis = 0;
    dfs2(x, y);
    for (auto e : here) {
      x = e.first;
      y = e.second;
      best[x][y] = best_here + cnt_vis;
      for (auto to : adj[x]) {
        best[to][nxt[y]] = max(best[to][nxt[y]], best[x][y]);
      }
      vis[x] = false;
    }
    sol = max(sol, best_here + cnt_vis);
    continue;
    cerr << best_here + cnt_vis << '\n';
    for (auto e : here) cerr << e.first << \" \" << e.second << '\n';
  }
  printf(\"%d\", sol - init);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp
import re

class Emuseumstourbootcamp(Basebootcamp):
    def __init__(self, max_n=4, max_m=5, max_d=3):
        self.max_n = max_n
        self.max_m = max_m
        self.max_d = max_d

    def case_generator(self):
        """生成完全符合题目约束的测试案例"""
        # 生成城市数量（包含n=1的边界情况）
        n = random.randint(1, self.max_n)
        
        # 动态计算最大可能道路数
        max_possible_roads = n * (n - 1)
        available_m_upper = min(max_possible_roads, self.max_m)
        m = random.randint(0, available_m_upper) if n > 1 else 0  # n=1时强制m=0
        
        # 生成合法道路集合
        possible_edges = []
        if n > 1:
            possible_edges = [(u, v) for u in range(1, n+1) for v in range(1, n+1) if u != v]
            roads = random.sample(possible_edges, k=m) if m > 0 else []
        else:
            roads = []
        
        # 生成合法博物馆开放时间（保证每个馆至少开放一天）
        d = random.randint(1, self.max_d)
        museums = []
        for _ in range(n):
            while True:
                schedule = ''.join(random.choice('01') for _ in range(d))
                if '1' in schedule:
                    break
            museums.append(schedule)
        
        return {
            'n': n,
            'm': m,
            'd': d,
            'roads': roads,
            'museums': museums,
            'correct_answer': self.compute_solution(n, m, d, roads, museums)
        }

    @staticmethod
    def compute_solution(n, m, d, roads, museums):
        """严格实现原题参考算法逻辑"""
        # 邻接表初始化（1-based）
        adj = [[] for _ in range(n+1)]
        rev = [[] for _ in range(n+1)]
        for u, v in roads:
            adj[u].append(v)
            rev[v].append(u)

        # 日期循环处理
        nxt = [(i+1)%d for i in range(d)]
        prev = [(i-1+d)%d for i in range(d)]

        # 第一次DFS确定处理顺序
        visited = [[False]*d for _ in range(n+1)]
        process_stack = []
        
        for city in range(1, n+1):
            for day in range(d):
                if not visited[city][day]:
                    stack = [(city, day, False)]
                    while stack:
                        x, y, processed = stack.pop()
                        if processed:
                            process_stack.append((x, y))
                            continue
                        if visited[x][y]:
                            continue
                        visited[x][y] = True
                        stack.append((x, y, True))  # 标记为已处理
                        # 处理相邻节点
                        for v in adj[x]:
                            ny = nxt[y]
                            if not visited[v][ny]:
                                stack.append((v, ny, False))

        # 逆向处理强连通分量
        visited = [[False]*d for _ in range(n+1)]
        best = [[0]*d for _ in range(n+1)]
        INIT = 10**9
        best[1][0] = INIT
        max_result = 0

        while process_stack:
            x, y = process_stack.pop()
            if visited[x][y]:
                continue
            
            component = []
            component_best = 0
            unique_museums = set()
            dfs_stack = [(x, y)]
            
            # 收集强连通分量节点
            while dfs_stack:
                cx, cy = dfs_stack.pop()
                if visited[cx][cy]:
                    continue
                visited[cx][cy] = True
                component.append((cx, cy))
                component_best = max(component_best, best[cx][cy])
                
                # 记录未访问的开放博物馆
                if museums[cx-1][cy] == '1' and cx not in unique_museums:
                    unique_museums.add(cx)
                
                # 逆向遍历
                for u in rev[cx]:
                    py = prev[cy]
                    if not visited[u][py]:
                        dfs_stack.append((u, py))

            # 计算结果
            total = component_best + len(unique_museums)
            for (cx, cy) in component:
                best[cx][cy] = total
                # 更新邻接节点状态
                for v in adj[cx]:
                    nd = nxt[cy]
                    if best[v][nd] < total:
                        best[v][nd] = total
            max_result = max(max_result, total)

        return max_result - INIT

    @staticmethod
    def prompt_func(question_case):
        """生成符合题目要求的详细描述"""
        input_lines = [
            f"{question_case['n']} {question_case['m']} {question_case['d']}",
            *[f"{u} {v}" for (u, v) in question_case['roads']],
            *question_case['museums']
        ]
        problem_desc = (
            "You are a tourist in country N with the following configuration:\n"
            f"- Cities: {question_case['n']}\n"
            f"- One-way roads: {question_case['m']}\n"
            f"- Week length: {question_case['d']} days\n\n"
            "Road list:\n" + '\n'.join(f"{u} → {v}" for u, v in question_case['roads']) + "\n\n"
            "Museum schedules (city 1 to n):\n" + '\n'.join(
                f"City {i+1}: {s}" for i, s in enumerate(question_case['museums'])
            ) + "\n\n"
            "Find the maximum distinct museums visitable starting from city 1 on day 1.\n"
            "Format your final answer as [answer]N[/answer] where N is the number."
        )
        return problem_desc

    @staticmethod
    def extract_output(output):
        """增强答案提取鲁棒性"""
        matches = re.findall(r'\[answer\s*\](.*?)\[/answer\s*\]', output, re.DOTALL|re.IGNORECASE)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """直接比对预计算结果"""
        return solution == identity['correct_answer']
