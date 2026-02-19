"""# 

### 谜题描述
Konrad is a Human Relations consultant working for VoltModder, a large electrical equipment producer. Today, he has been tasked with evaluating the level of happiness in the company.

There are n people working for VoltModder, numbered from 1 to n. Each employee earns a different amount of money in the company — initially, the i-th person earns i rubles per day.

On each of q following days, the salaries will be revised. At the end of the i-th day, employee v_i will start earning n+i rubles per day and will become the best-paid person in the company. The employee will keep his new salary until it gets revised again.

Some pairs of people don't like each other. This creates a great psychological danger in the company. Formally, if two people a and b dislike each other and a earns more money than b, employee a will brag about this to b. A dangerous triple is a triple of three employees a, b and c, such that a brags to b, who in turn brags to c. If a dislikes b, then b dislikes a.

At the beginning of each day, Konrad needs to evaluate the number of dangerous triples in the company. Can you help him do it?

Input

The first line contains two integers n and m (1 ≤ n ≤ 100 000, 0 ≤ m ≤ 100 000) — the number of employees in the company and the number of pairs of people who don't like each other. Each of the following m lines contains two integers a_i, b_i (1 ≤ a_i, b_i ≤ n, a_i ≠ b_i) denoting that employees a_i and b_i hate each other (that is, a_i dislikes b_i and b_i dislikes a_i). Each such relationship will be mentioned exactly once.

The next line contains an integer q (0 ≤ q ≤ 100 000) — the number of salary revisions. The i-th of the following q lines contains a single integer v_i (1 ≤ v_i ≤ n) denoting that at the end of the i-th day, employee v_i will earn the most.

Output

Output q + 1 integers. The i-th of them should contain the number of dangerous triples in the company at the beginning of the i-th day.

Examples

Input


4 5
1 2
2 4
1 3
3 4
2 3
2
2
3


Output


4
3
2


Input


3 3
1 2
2 3
1 3
5
1
2
2
1
3


Output


1
1
1
1
1
1

Note

Consider the first sample test. The i-th row in the following image shows the structure of the company at the beginning of the i-th day. A directed edge from a to b denotes that employee a brags to employee b. The dangerous triples are marked by highlighted edges.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
long long arr[100005];
vector<long long> G[100005];
long long deg[100005];
long long cnt[100005];
vector<long long> gtrs[100005];
inline long long cal(long long u) {
  return deg[u] > 320
             ? (deg[u] - (long long)gtrs[u].size()) * (long long)gtrs[u].size()
             : (deg[u] - cnt[u]) * cnt[u];
}
int32_t main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  long long n, m;
  cin >> n >> m;
  for (long long i = 1, u, v; i <= m; i++) {
    cin >> u >> v;
    G[u].emplace_back(v);
    G[v].emplace_back(u);
  }
  iota(arr + 1, arr + 1 + n, 1);
  long long ans = 0;
  for (long long u = 1; u <= n; u++) {
    deg[u] = (long long)G[u].size();
    if (deg[u] > 320) {
      for (long long v : G[u])
        if (arr[v] > arr[u]) gtrs[u].emplace_back(v);
    } else {
      for (long long v : G[u])
        if (arr[v] > arr[u]) cnt[u]++;
    }
    ans += cal(u);
  }
  cout << ans << '\n';
  long long q;
  cin >> q;
  for (long long t = 1; t <= q; t++) {
    long long u;
    cin >> u;
    ans -= cal(u);
    vector<long long>& tem = (deg[u] > 320 ? gtrs[u] : G[u]);
    for (long long v : tem)
      if (arr[u] < arr[v]) {
        ans -= cal(v);
        if (deg[v] > 320)
          gtrs[v].emplace_back(u);
        else
          cnt[v]++;
        ans += cal(v);
      }
    if (deg[u] > 320)
      gtrs[u].clear();
    else
      cnt[u] = 0;
    arr[u] = n + t;
    cout << ans << '\n';
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

def compute_correct_output(n, m, edges, updates):
    G = [[] for _ in range(n + 1)]
    for a, b in edges:
        G[a].append(b)
        G[b].append(a)
    
    arr = list(range(n + 1))
    deg = [len(g) for g in G]
    gtrs = [[] for _ in range(n + 1)]
    cnt = [0] * (n + 1)
    
    ans = 0
    for u in range(1, n + 1):
        if deg[u] > 320:
            gtrs[u] = [v for v in G[u] if arr[v] > arr[u]]
            ans += (deg[u] - len(gtrs[u])) * len(gtrs[u])
        else:
            cnt[u] = sum(1 for v in G[u] if arr[v] > arr[u])
            ans += (deg[u] - cnt[u]) * cnt[u]
    
    correct_outputs = [ans]
    q = len(updates)
    
    for t in range(1, q + 1):
        u = updates[t - 1]
        ans -= (deg[u] - (len(gtrs[u]) if deg[u] > 320 else cnt[u])) * (len(gtrs[u]) if deg[u] > 320 else cnt[u])
        
        candidates = gtrs[u] if deg[u] > 320 else G[u]
        processed = [v for v in candidates if arr[u] < arr[v]]
        
        for v in processed:
            ans_before = (deg[v] - (len(gtrs[v]) if deg[v] > 320 else cnt[v])) * (len(gtrs[v]) if deg[v] > 320 else cnt[v])
            if deg[v] > 320:
                gtrs[v].append(u)
            else:
                cnt[v] += 1
            ans_after = (deg[v] - (len(gtrs[v]) if deg[v] > 320 else cnt[v])) * (len(gtrs[v]) if deg[v] > 320 else cnt[v])
            ans += (ans_after - ans_before)
        
        if deg[u] > 320:
            gtrs[u].clear()
        else:
            cnt[u] = 0
        
        arr[u] = n + t
        correct_outputs.append(ans)
    
    return correct_outputs

class Dkonradandcompanyevaluationbootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_m=20, max_q=5):
        self.max_n = max(max_n, 1)  # 确保最小n=1
        self.max_m = max_m
        self.max_q = max_q
    
    def case_generator(self):
        n = random.randint(1, self.max_n)
        max_possible_m = n * (n - 1) // 2
        m = random.randint(0, min(self.max_m, max_possible_m))
        
        edges_set = set()
        available_pairs = [(i, j) for i in range(1, n+1) for j in range(i+1, n+1)] if n >= 2 else []
        if available_pairs and m > 0:
            edges_set.update(random.sample(available_pairs, k=min(m, len(available_pairs))))
        
        edges = [list(pair) for pair in edges_set]
        q = random.randint(0, self.max_q)
        updates = [random.randint(1, n) for _ in range(q)]
        
        correct_outputs = compute_correct_output(n, m, edges, updates)
        
        return {
            'n': n,
            'm': m,
            'edges': edges,
            'q': q,
            'updates': updates,
            'correct_outputs': correct_outputs
        }
    
    @staticmethod
    def prompt_func(question_case):
        input_lines = [f"{question_case['n']} {question_case['m']}"]
        input_lines.extend(f"{a} {b}" for a, b in question_case['edges'])
        input_lines.append(str(question_case['q']))
        input_lines.extend(map(str, question_case['updates']))
        input_str = '\n'.join(input_lines)
        
        prompt = f"""Konrad needs to calculate dangerous triples in VoltModder. A dangerous triple is a sequence of three employees a->b->c where a dislikes b, b dislikes c, and salaries satisfy a > b > c. Each day an employee's salary is updated to be the highest. Output the number of dangerous triples before each day.

Input format:
n m
a1 b1
...
am bm
q
v1
...
vq

Output q+1 integers. Put your answer between [answer] and [/answer]. Example:
[answer]
0
1
2
[/answer]

Current Input:
{input_str}"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        
        numbers = []
        for line in matches[-1].strip().splitlines():
            stripped = line.strip()
            if stripped.isdigit() or (stripped.startswith('-') and stripped[1:].isdigit()):
                numbers.append(int(stripped))
        return numbers if len(numbers) > 0 else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity.get('correct_outputs', [])
        if solution is None:
            return False
        return solution == expected
