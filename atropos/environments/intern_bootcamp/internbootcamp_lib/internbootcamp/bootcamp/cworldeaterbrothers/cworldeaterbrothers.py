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
int const N = 3000 + 20;
int n, dp[N], mx1, mx2, ans;
vector<pair<int, bool>> adj[N];
void dfs1(int v, int p = -1) {
  for (auto x : adj[v]) {
    int u = x.first, f = x.second;
    if (u != p) {
      dfs1(u, v);
      dp[v] += dp[u] + (!f);
    }
  }
  return;
}
void dfs2(int v, int p = -1) {
  for (auto x : adj[v]) {
    int u = x.first, f = x.second;
    if (u != p) {
      dp[u] = dp[v];
      f ? dp[u]++ : dp[u]--;
      dfs2(u, v);
    }
  }
  return;
}
pair<int, int> dfs3(int v, int p = -1) {
  int m1 = 0, m2 = 0;
  for (auto x : adj[v]) {
    int u = x.first, f = x.second;
    if (u != p) {
      m2 = max(m2, dfs3(u, v).first + (f ? -1 : 1));
      if (m2 > m1) swap(m1, m2);
    }
  }
  return {m1, m2};
}
int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);
  cin >> n;
  ans = n - 1;
  for (int i = 0; i < n - 1; i++) {
    int x, y;
    cin >> x >> y;
    x--, y--;
    adj[x].push_back({y, 1});
    adj[y].push_back({x, 0});
  }
  dfs1(0);
  dfs2(0);
  for (int i = 0; i < n; i++) {
    pair<int, int> cur = dfs3(i);
    mx1 = cur.first, mx2 = cur.second;
    ans = min(ans, dp[i] - mx1 - mx2);
  }
  return cout << ans, 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re
import sys

class Cworldeaterbrothersbootcamp(Basebootcamp):
    def __init__(self, n=5):
        self.n = n
    
    def case_generator(self):
        n = self.n
        if n < 1:
            raise ValueError("n must be at least 1")
        
        if n == 1:
            edges = []
            correct_answer = 0
        else:
            undirected_edges = self.generate_tree(n)
            directed_edges = []
            for u, v in undirected_edges:
                if random.choice([True, False]):
                    directed_edges.append([u + 1, v + 1])
                else:
                    directed_edges.append([v + 1, u + 1])
            
            adj = [[] for _ in range(n)]
            for ai, bi in directed_edges:
                x = ai - 1
                y = bi - 1
                adj[x].append((y, 1))
                adj[y].append((x, 0))
            
            correct_answer = self.compute_answer(n, adj)
        
        return {
            "n": n,
            "edges": directed_edges if n > 1 else [],
            "correct_answer": correct_answer
        }
    
    @staticmethod
    def generate_tree(n):
        if n == 1:
            return []
        parents = [random.randint(0, i-1) for i in range(1, n)]
        edges = []
        for i in range(1, n):
            parent = parents[i-1]
            edges.append((parent, i))
        return edges
    
    @staticmethod
    def compute_answer(n, adj):
        if n == 1:
            return 0
        
        dp = [0] * n
        ans = n - 1
        
        def dfs1(v, p):
            for u, f in adj[v]:
                if u == p:
                    continue
                dfs1(u, v)
                dp[v] += dp[u] + (1 - f)
        
        dfs1(0, -1)
        
        def dfs2(v, p):
            for u, f in adj[v]:
                if u == p:
                    continue
                dp[u] = dp[v] + (1 if f else -1)
                dfs2(u, v)
        
        dfs2(0, -1)
        
        def dfs3(v, p):
            m1, m2 = 0, 0
            for u, f in adj[v]:
                if u == p:
                    continue
                cm1, cm2 = dfs3(u, v)
                current = cm1 + (0 if f else 1)
                if current >= m1:
                    m2 = m1
                    m1 = current
                elif current > m2:
                    m2 = current
            return (m1, m2)
        
        for i in range(n):
            mx1, mx2 = dfs3(i, -1)
            current_ans = dp[i] - mx1 - mx2
            ans = min(ans, current_ans)
        
        return ans
    
    @staticmethod
    def prompt_func(question_case):
        edges = question_case['edges']
        edges_str = "\n".join(f"{a} {b}" for a, b in edges)
        prompt = f"""The world consists of {question_case['n']} countries connected by {question_case['n']-1} directed roads, forming a tree when directions are ignored. Two brothers want to choose up to two countries such that every other country is reachable from at least one of them via directed roads. Find the minimum number of road reversals needed.

Input:
{question_case['n']}
{edges_str if edges else ''}

Please provide your answer as an integer enclosed within [answer] and [/answer] tags. For example: [answer]5[/answer]."""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        if last_match.isdigit():
            return int(last_match)
        else:
            try:
                return int(float(last_match))
            except:
                return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
