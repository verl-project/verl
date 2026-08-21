"""# 

### 谜题描述
Petya loves volleyball very much. One day he was running late for a volleyball match. Petya hasn't bought his own car yet, that's why he had to take a taxi. The city has n junctions, some of which are connected by two-way roads. The length of each road is defined by some positive integer number of meters; the roads can have different lengths.

Initially each junction has exactly one taxi standing there. The taxi driver from the i-th junction agrees to drive Petya (perhaps through several intermediate junctions) to some other junction if the travel distance is not more than ti meters. Also, the cost of the ride doesn't depend on the distance and is equal to ci bourles. Taxis can't stop in the middle of a road. Each taxi can be used no more than once. Petya can catch taxi only in the junction, where it stands initially.

At the moment Petya is located on the junction x and the volleyball stadium is on the junction y. Determine the minimum amount of money Petya will need to drive to the stadium.

Input

The first line contains two integers n and m (1 ≤ n ≤ 1000, 0 ≤ m ≤ 1000). They are the number of junctions and roads in the city correspondingly. The junctions are numbered from 1 to n, inclusive. The next line contains two integers x and y (1 ≤ x, y ≤ n). They are the numbers of the initial and final junctions correspondingly. Next m lines contain the roads' description. Each road is described by a group of three integers ui, vi, wi (1 ≤ ui, vi ≤ n, 1 ≤ wi ≤ 109) — they are the numbers of the junctions connected by the road and the length of the road, correspondingly. The next n lines contain n pairs of integers ti and ci (1 ≤ ti, ci ≤ 109), which describe the taxi driver that waits at the i-th junction — the maximum distance he can drive and the drive's cost. The road can't connect the junction with itself, but between a pair of junctions there can be more than one road. All consecutive numbers in each line are separated by exactly one space character.

Output

If taxis can't drive Petya to the destination point, print \"-1\" (without the quotes). Otherwise, print the drive's minimum cost.

Please do not use the %lld specificator to read or write 64-bit integers in С++. It is preferred to use cin, cout streams or the %I64d specificator.

Examples

Input

4 4
1 3
1 2 3
1 4 1
2 4 1
2 3 5
2 7
7 2
1 2
7 7


Output

9

Note

An optimal way — ride from the junction 1 to 2 (via junction 4), then from 2 to 3. It costs 7+2=9 bourles.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
long long n, m, x, y;
long long dis[1001], t[1001], c[1001];
vector<pair<long long, long long> > adj[1001], fadj[1001];
void dijkstra(int S) {
  fill(dis, dis + 1001, (1LL << 61));
  priority_queue<pair<long long, long long> > q;
  q.push(make_pair(S, 0));
  dis[S] = 0;
  while (!q.empty()) {
    pair<long long, long long> x = q.top();
    q.pop();
    for (int i = 0; i < adj[x.first].size(); i++) {
      if (dis[x.first] + adj[x.first][i].second < dis[adj[x.first][i].first]) {
        dis[adj[x.first][i].first] = dis[x.first] + adj[x.first][i].second;
        q.push(adj[x.first][i]);
      }
    }
  }
}
int main() {
  cin >> n >> m >> x >> y;
  for (int i = 0; i < m; i++) {
    long long a, b, w;
    cin >> a >> b >> w;
    adj[a].push_back(make_pair(b, w));
    adj[b].push_back(make_pair(a, w));
  }
  for (int i = 1; i <= m; i++) cin >> t[i] >> c[i];
  for (int i = 1; i <= n; i++) {
    dijkstra(i);
    for (int j = 1; j <= n; j++) {
      if (i == j) continue;
      if (t[i] >= dis[j]) fadj[i].push_back(make_pair(j, c[i]));
    }
  }
  for (int i = 1; i <= n; i++) adj[i] = fadj[i];
  dijkstra(x);
  cout << (dis[y] == (1LL << 61) ? -1 : dis[y]);
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import json
import heapq
import re
from bootcamp import Basebootcamp

class Dvolleyballbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__()
        self.params = {
            'n': params.get('n', 4),
            'm': params.get('m', 4),
            'x': params.get('x', 1),
            'y': params.get('y', 3),
            'roads': params.get('roads', []),
            'taxis': params.get('taxis', []),
            'correct_answer': params.get('correct_answer', 0)
        }
    
    def case_generator(self):
        while True:
            n = random.randint(1, 1000)
            m = random.randint(0, 1000)
            x = random.randint(1, n)
            y = random.randint(1, n)
            
            roads = []
            for _ in range(m):
                u = random.randint(1, n)
                v = random.randint(1, n)
                if u == v:
                    continue
                w = random.randint(1, 10**9)
                roads.append((u, v, w))
            
            taxis = []
            for _ in range(n):
                ti = random.randint(1, 10**9)
                ci = random.randint(1, 10**9)
                taxis.append((ti, ci))
            
            adj = [[] for _ in range(n+1)]
            for u, v, w in roads:
                adj[u].append((v, w))
                adj[v].append((u, w))
            
            def dijkstra(start):
                dist = [float('inf')] * (n + 1)
                dist[start] = 0
                heap = []
                heapq.heappush(heap, (0, start))
                while heap:
                    current_dist, u = heapq.heappop(heap)
                    if current_dist > dist[u]:
                        continue
                    for v, w in adj[u]:
                        if dist[v] > current_dist + w:
                            dist[v] = current_dist + w
                            heapq.heappush(heap, (dist[v], v))
                return dist
            
            shortest = {}
            for i in range(1, n+1):
                shortest[i] = dijkstra(i)
            
            new_adj = [[] for _ in range(n+1)]
            for i in range(1, n+1):
                ti, ci = taxis[i-1]
                for j in range(1, n+1):
                    if i != j and shortest[i][j] <= ti:
                        new_adj[i].append((j, ci))
            
            def dijkstra_cost(start):
                dist = [float('inf')] * (n + 1)
                dist[start] = 0
                heap = []
                heapq.heappush(heap, (0, start))
                while heap:
                    current_cost, u = heapq.heappop(heap)
                    if current_cost > dist[u]:
                        continue
                    for v, cost in new_adj[u]:
                        if dist[v] > current_cost + cost:
                            dist[v] = current_cost + cost
                            heapq.heappush(heap, (dist[v], v))
                return dist
            
            min_cost = dijkstra_cost(x)[y]
            if min_cost != float('inf'):
                case = {
                    'n': n,
                    'm': m,
                    'x': x,
                    'y': y,
                    'roads': roads,
                    'taxis': taxis,
                    'correct_answer': min_cost
                }
                return case
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        x = question_case['x']
        y = question_case['y']
        roads = question_case['roads']
        taxis = question_case['taxis']
        
        prompt = f"Petya needs to get from junction {x} to junction {y} in a city with {n} junctions and {m} roads. Each road connects two junctions and has a certain length. Each junction has a taxi that can drive Petya to another junction if the distance is within its limit, and the cost is fixed regardless of the distance.\n\n"
        prompt += "The roads are as follows:\n"
        for i, (u, v, w) in enumerate(roads, 1):
            prompt += f"Road {i}: connects {u} and {v}, length {w} meters\n"
        
        prompt += "\nThe taxis at each junction have the following limits and costs:\n"
        for i in range(n):
            ti, ci = taxis[i]
            prompt += f"Junction {i+1}: maximum distance {ti} meters, cost {ci} bourles\n"
        
        prompt += "\nWhat is the minimum cost for Petya to get from junction {x} to junction {y}? Please provide your answer within [answer] tags.\n"
        prompt += "For example, if the minimal cost is 9, write [answer]9[/answer].\n"
        
        return prompt
    
    @staticmethod
    def extract_output(output):
        pattern = r'\[answer\](.*?)\[\/answer\]'
        matches = re.findall(pattern, output)
        if matches:
            return matches[-1].strip()
        else:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False
        try:
            solution = int(solution)
        except:
            return False
        correct_answer = identity['correct_answer']
        return solution == correct_answer
