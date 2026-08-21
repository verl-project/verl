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
const int N = 100005, D = 50;
const int V = N * D;
vector<int> g[N];
int S[V], ssz;
int low[V], num[V], cnt, vis[V];
int scc[V], scc_cnt;
int all[V], sz;
int n, m, d;
long long val[N];
void push(int i, int j) {
  if (val[i] >> j & 1) {
    all[sz++] = i;
    while (sz >= V) sz++;
  }
}
int dp[V];
void tarjanSCC(int u) {
  low[u] = num[u] = ++cnt;
  vis[u] = 1;
  S[ssz++] = u;
  while (ssz >= V) {
    ssz++;
  }
  for (int w : g[u / d]) {
    int v = w * d + (u % d + 1) % d;
    while (v >= V) v++;
    if (!num[v]) tarjanSCC(v);
    if (vis[v]) low[u] = min(low[u], low[v]);
  }
  if (low[u] == num[u]) {
    scc[u] = ++scc_cnt;
    int v;
    sz = 0;
    push(u / d, u % d);
    do {
      v = S[--ssz];
      vis[v] = 0;
      push(v / d, v % d);
      scc[v] = scc_cnt;
      for (int wver : g[v / d]) {
        int ver = wver * d + (v % d + 1) % d;
        if (scc[ver]) {
          dp[scc_cnt] = max(dp[scc_cnt], dp[scc[ver]]);
        }
      }
    } while (u != v);
    sort(all, all + sz);
    dp[scc_cnt] += int(unique(all, all + sz) - all);
  }
}
int main() {
  scanf(\"%d %d %d\", &n, &m, &d);
  for (int i = 0; i < m; i++) {
    int u, v;
    scanf(\"%d %d\", &u, &v);
    u--, v--;
    g[u].push_back(v);
  }
  char str[100];
  for (int i = 0; i < n; i++) {
    scanf(\" %s\", str);
    for (int j = 0; j < d; j++) {
      if (str[j] == '1') {
        val[i] |= 1LL << j;
      }
    }
  }
  tarjanSCC(0);
  printf(\"%d\n\", dp[scc[0]]);
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
from collections import deque

def calculate_answer(n, m, d, roads, schedules):
    adj = [[] for _ in range(n)]
    for u, v in roads:
        adj[u].append(v)
    open_table = [ [c == '1' for c in s] for s in schedules ]

    max_museums = 0

    visited = {}  # (current city, day in week) -> max museums count

    initial_museums = 0
    if open_table[0][0]:
        initial_museums = 1

    queue = deque()
    # State: (city, day, visited_museums_bitmask)
    initial_state = (0, 0, initial_museums, 1 << 0 if open_table[0][0] else 0)
    queue.append(initial_state)
    visited[(0, 0)] = (initial_museums, initial_state[3])

    max_museums = initial_museums

    while queue:
        u, t, count, mask = queue.popleft()

        next_t = (t + 1) % d

        for v in adj[u]:
            new_mask = mask
            new_count = count
            # Check if we can visit v's museum at next_t day
            if open_table[v][next_t] and not (mask & (1 << v)):
                new_count += 1
                new_mask |= 1 << v
            key = (v, next_t)
            if key not in visited or visited[key][0] < new_count or (visited[key][0] == new_count and visited[key][1] | new_mask != visited[key][1]):
                visited[key] = (new_count, new_mask)
                queue.append((v, next_t, new_count, new_mask))
                if new_count > max_museums:
                    max_museums = new_count

    return max_museums

class Cmuseumstourbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = {
            'n': params.get('n', 4),
            'm': params.get('m', 5),
            'd': params.get('d', 3),
        }
        # Ensure parameters are within BFS processing limits
        max_n = 10  # Adjust based on performance testing
        max_d = 7
        self.params['n'] = min(self.params['n'], max_n)
        self.params['d'] = min(self.params['d'], max_d)
        # Ensure m does not exceed possible roads
        max_possible_m = self.params['n'] * (self.params['n'] - 1)
        self.params['m'] = min(self.params['m'], max_possible_m)

    def case_generator(self):
        n = self.params['n']
        m = self.params['m']
        d = self.params['d']

        # Generate all possible valid roads
        possible_roads = []
        for u in range(n):
            for v in range(n):
                if u != v:
                    possible_roads.append((u, v))
        
        # Adjust m if it exceeds possible roads
        if not possible_roads:
            m = 0
        else:
            m = min(m, len(possible_roads))
        
        # Randomly sample unique roads
        roads = random.sample(possible_roads, m) if possible_roads else []

        # Generate schedules ensuring capital has at least one open day
        schedules = []
        for i in range(n):
            if i == 0:  # Capital city
                while True:
                    s = ''.join(random.choice(['0', '1']) for _ in range(d))
                    if '1' in s:
                        break
                schedules.append(s)
            else:
                s = ''.join(random.choice(['0', '1']) for _ in range(d))
                schedules.append(s)

        # Calculate correct answer with retry logic
        max_retry = 3
        correct_answer = 0
        for _ in range(max_retry):
            try:
                correct_answer = calculate_answer(n, m, d, roads, schedules)
                break
            except:
                continue

        return {
            'n': n,
            'm': m,
            'd': d,
            'roads': roads,
            'schedules': schedules,
            'correct_answer': correct_answer
        }

    @staticmethod
    def prompt_func(question_case):
        input_desc = [
            f"{question_case['n']} {question_case['m']} {question_case['d']}",
            '\n'.join(f"{u+1} {v+1}" for u, v in question_case['roads']),
            '\n'.join(question_case['schedules'])
        ]
        input_example = '\n'.join(input_desc)

        problem_text = f"""\
You are planning a museum tour in a country with one-way roads. Starting in city 1 on week day 1, maximize the number of distinct museums visited. Each road takes one night to travel. Museums have weekly schedules.

Input format:
- First line: n m d
- Next m lines: u v (one-way roads)
- Next n lines: d digits (0/1) per city's schedule

Write your answer as [answer]N[/answer], replacing N with the maximum number.

Input:
{input_example}

What's the maximum number of distinct museums you can visit?"""
        return problem_text

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
