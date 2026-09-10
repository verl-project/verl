"""# 

### 谜题描述
If the girl doesn't go to Denis, then Denis will go to the girl. Using this rule, the young man left home, bought flowers and went to Nastya. 

On the way from Denis's house to the girl's house is a road of n lines. This road can't be always crossed in one green light. Foreseeing this, the good mayor decided to place safety islands in some parts of the road. Each safety island is located after a line, as well as at the beginning and at the end of the road. Pedestrians can relax on them, gain strength and wait for a green light.

Denis came to the edge of the road exactly at the moment when the green light turned on. The boy knows that the traffic light first lights up g seconds green, and then r seconds red, then again g seconds green and so on.

Formally, the road can be represented as a segment [0, n]. Initially, Denis is at point 0. His task is to get to point n in the shortest possible time.

He knows many different integers d_1, d_2, …, d_m, where 0 ≤ d_i ≤ n — are the coordinates of points, in which the safety islands are located. Only at one of these points, the boy can be at a time when the red light is on.

Unfortunately, Denis isn't always able to control himself because of the excitement, so some restrictions are imposed:

  * He must always move while the green light is on because it's difficult to stand when so beautiful girl is waiting for you. Denis can change his position by ± 1 in 1 second. While doing so, he must always stay inside the segment [0, n]. 
  * He can change his direction only on the safety islands (because it is safe). This means that if in the previous second the boy changed his position by +1 and he walked on a safety island, then he can change his position by ± 1. Otherwise, he can change his position only by +1. Similarly, if in the previous second he changed his position by -1, on a safety island he can change position by ± 1, and at any other point by -1. 
  * At the moment when the red light is on, the boy must be on one of the safety islands. He can continue moving in any direction when the green light is on. 



Denis has crossed the road as soon as his coordinate becomes equal to n.

This task was not so simple, because it's possible that it is impossible to cross the road. Since Denis has all thoughts about his love, he couldn't solve this problem and asked us to help him. Find the minimal possible time for which he can cross the road according to these rules, or find that it is impossible to do.

Input

The first line contains two integers n and m (1 ≤ n ≤ 10^6, 2 ≤ m ≤ min(n + 1, 10^4)) — road width and the number of safety islands.

The second line contains m distinct integers d_1, d_2, …, d_m (0 ≤ d_i ≤ n) — the points where the safety islands are located. It is guaranteed that there are 0 and n among them.

The third line contains two integers g, r (1 ≤ g, r ≤ 1000) — the time that the green light stays on and the time that the red light stays on.

Output

Output a single integer — the minimum time for which Denis can cross the road with obeying all the rules.

If it is impossible to cross the road output -1.

Examples

Input


15 5
0 3 7 14 15
11 11


Output


45

Input


13 4
0 3 7 13
9 9


Output


-1

Note

In the first test, the optimal route is: 

  * for the first green light, go to 7 and return to 3. In this case, we will change the direction of movement at the point 7, which is allowed, since there is a safety island at this point. In the end, we will be at the point of 3, where there is also a safety island. The next 11 seconds we have to wait for the red light. 
  * for the second green light reaches 14. Wait for the red light again. 
  * for 1 second go to 15. As a result, Denis is at the end of the road. 



In total, 45 seconds are obtained.

In the second test, it is impossible to cross the road according to all the rules.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
using pii = pair<int, int>;
using pi3 = pair<int, pii>;
using INT = long long;
const int inf = 1e9;
const int MM = 10101;
const int KK = 1010;
int dp[MM][KK];
int a[MM];
int n, m;
int out(int first) { return first < 1 || first > m; }
int dx[] = {-1, 1};
int main() {
  cin >> n >> m;
  for (int i = 1; i <= m; i++) scanf(\"%d\", a + i);
  sort(a + 1, a + m + 1);
  int R, G;
  cin >> G >> R;
  for (int i = 0; i <= m; i++) {
    for (int j = 0; j <= G; j++) {
      dp[i][j] = 1e9;
    }
  }
  a[m + 1] = n;
  for (int i = 1; i <= m + 1; i++) {
    if (a[i] - a[i - 1] > G) {
      puts(\"-1\");
      return 0;
    }
  }
  dp[0][0] = 0;
  if (a[1] < G)
    dp[1][a[1]] = 0;
  else
    dp[1][0] = 1;
  priority_queue<pii, vector<pii>, greater<pii> > st;
  if (a[1] < G)
    st.push(pii(0, G + a[1]));
  else
    st.push(pii(1, G));
  while (!st.empty()) {
    auto p = st.top();
    st.pop();
    int dis = p.first, first = p.second;
    int u = first / G;
    int tp = first % G;
    for (int i = 0; i < 2; i++) {
      int v = u + dx[i];
      if (out(v)) continue;
      int dst = tp + abs(a[u] - a[v]);
      if (dst > G) continue;
      if (dst == G) {
        if (dp[v][0] > dp[u][tp] + 1) {
          dp[v][0] = dp[u][tp] + 1;
          st.push(pii(dp[v][0], v * G));
        }
        continue;
      }
      if (dp[v][dst] > dp[u][tp]) {
        dp[v][dst] = dp[u][tp];
        st.push(pii(dp[v][dst], v * G + dst));
      }
    }
  }
  INT ans = inf;
  for (int i = 1; i <= m; i++) {
    if (a[i] + G < n) continue;
    if (dp[i][0] > inf) continue;
    ans = min(ans, 1ll * dp[i][0] * (G + R) + n - a[i]);
  }
  if (ans == inf) ans = -1;
  cout << ans;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def correct_solution(n, m, d, g, r):
    d_sorted = sorted(d)
    # Check if any adjacent islands exceed g distance
    for i in range(1, len(d_sorted)):
        if d_sorted[i] - d_sorted[i-1] > g:
            return -1
    m = len(d_sorted)
    INF = float('inf')
    dp = [[INF] * (g + 1) for _ in range(m)]
    dp[0][0] = 0
    heap = []
    import heapq
    heapq.heappush(heap, (0, 0, 0))  # (cycles, u, rem)

    while heap:
        cycles, u, rem = heapq.heappop(heap)
        if cycles > dp[u][rem]:
            continue
        for dv in [-1, 1]:
            v = u + dv
            if 0 <= v < m:
                distance = abs(d_sorted[u] - d_sorted[v])
                new_rem = rem + distance
                if new_rem > g:
                    continue
                if new_rem == g:
                    new_cycles = cycles + 1
                    new_r = 0
                else:
                    new_cycles = cycles
                    new_r = new_rem
                if dp[v][new_r] > new_cycles:
                    dp[v][new_r] = new_cycles
                    heapq.heappush(heap, (new_cycles, v, new_r))
    min_time = INF
    for i in range(m):
        time_needed = n - d_sorted[i]
        if time_needed <= g and dp[i][0] != INF:
            total_time = dp[i][0] * (g + r) + time_needed
            if total_time < min_time:
                min_time = total_time
    return min_time if min_time != INF else -1

class Enastyaandunexpectedguestbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = {
            'n_range': (1, 100),
            'm_range_low': 2,
            'm_range_high': 100,
            'g_range': (1, 1000),
            'r_range': (1, 1000),
        }
        self.params.update(params)

    def case_generator(self):
        n = random.randint(*self.params['n_range'])
        m_high = min(n + 1, self.params['m_range_high'])
        m = random.randint(self.params['m_range_low'], m_high)
        d = [0, n]
        while len(d) < m:
            new_point = random.randint(0, n)
            if new_point not in d:
                d.append(new_point)
        d = sorted(d)
        # Allow g to be smaller than gaps to generate impossible cases
        g = random.randint(*self.params['g_range'])
        r = random.randint(*self.params['r_range'])
        return {
            'n': n,
            'm': m,
            'd': d,
            'g': g,
            'r': r,
        }

    @staticmethod
    def prompt_func(question_case):
        case = question_case
        d_str = ' '.join(map(str, case['d']))
        prompt = f"""你是编程竞赛选手，需要解决以下问题。请仔细阅读问题描述并按要求输出答案。

问题描述：
Enastyaandunexpectedguest必须穿过一条宽为{case['n']}米的马路。马路的安全岛位于多个坐标点，包括0米和{case['n']}米。交通灯先绿{case['g']}秒，后红{case['r']}秒，反复循环。Enastyaandunexpectedguest从坐标0出发，必须在绿灯期间移动，每秒移动±1米，且只能在安全岛改变方向。红灯期间必须停留在安全岛。求到达坐标{case['n']}的最短时间，若不可能输出-1。

输入格式：
第一行包含两个整数n和m：{case['n']} {case['m']}
第二行包含{case['m']}个不同的整数，按递增顺序排列：{d_str}
第三行包含两个整数g和r：{case['g']} {case['r']}

输出格式：
输出一个整数，表示最短时间或-1。

请将你的答案放置在[answer]标签内。例如：[answer]45[/answer] 或 [answer]-1[/answer]。"""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        numbers = re.findall(r'-?\d+', last_match)
        if not numbers:
            return None
        try:
            return int(numbers[-1])
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        m = identity['m']
        d = identity['d']
        g = identity['g']
        r = identity['r']
        correct = correct_solution(n, m, d, g, r)
        return solution == correct
