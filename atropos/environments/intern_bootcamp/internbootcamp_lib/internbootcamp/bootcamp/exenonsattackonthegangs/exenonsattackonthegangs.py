"""# 

### 谜题描述
[INSPION FullBand Master - INSPION](https://www.youtube.com/watch?v=kwsciXm_7sA)

[INSPION - IOLITE-SUNSTONE](https://www.youtube.com/watch?v=kwsciXm_7sA)

On another floor of the A.R.C. Markland-N, the young man Simon \"Xenon\" Jackson, takes a break after finishing his project early (as always). Having a lot of free time, he decides to put on his legendary hacker \"X\" instinct and fight against the gangs of the cyber world.

His target is a network of n small gangs. This network contains exactly n - 1 direct links, each of them connecting two gangs together. The links are placed in such a way that every pair of gangs is connected through a sequence of direct links.

By mining data, Xenon figured out that the gangs used a form of cross-encryption to avoid being busted: every link was assigned an integer from 0 to n - 2 such that all assigned integers are distinct and every integer was assigned to some link. If an intruder tries to access the encrypted data, they will have to surpass S password layers, with S being defined by the following formula:

$$$S = ∑_{1 ≤ u < v ≤ n} mex(u, v)$$$

Here, mex(u, v) denotes the smallest non-negative integer that does not appear on any link on the unique simple path from gang u to gang v.

Xenon doesn't know the way the integers are assigned, but it's not a problem. He decides to let his AI's instances try all the passwords on his behalf, but before that, he needs to know the maximum possible value of S, so that the AIs can be deployed efficiently.

Now, Xenon is out to write the AI scripts, and he is expected to finish them in two hours. Can you find the maximum possible S before he returns?

Input

The first line contains an integer n (2 ≤ n ≤ 3000), the number of gangs in the network.

Each of the next n - 1 lines contains integers u_i and v_i (1 ≤ u_i, v_i ≤ n; u_i ≠ v_i), indicating there's a direct link between gangs u_i and v_i.

It's guaranteed that links are placed in such a way that each pair of gangs will be connected by exactly one simple path.

Output

Print the maximum possible value of S — the number of password layers in the gangs' network.

Examples

Input


3
1 2
2 3


Output


3


Input


5
1 2
1 3
1 4
3 5


Output


10

Note

In the first example, one can achieve the maximum S with the following assignment:

<image>

With this assignment, mex(1, 2) = 0, mex(1, 3) = 2 and mex(2, 3) = 1. Therefore, S = 0 + 2 + 1 = 3.

In the second example, one can achieve the maximum S with the following assignment:

<image>

With this assignment, all non-zero mex value are listed below: 

  * mex(1, 3) = 1 
  * mex(1, 5) = 2 
  * mex(2, 3) = 1 
  * mex(2, 5) = 2 
  * mex(3, 4) = 1 
  * mex(4, 5) = 3 



Therefore, S = 1 + 2 + 1 + 2 + 1 + 3 = 10.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int maxn = 3000 + 50;
vector<int> edge[maxn];
int sub[maxn][maxn], fa[maxn][maxn];
long long n, ans, dp[maxn][maxn];
void dfs(int root, int k, int dad) {
  sub[root][k]++;
  fa[root][k] = dad;
  for (auto t : edge[k]) {
    if (t != dad) {
      dfs(root, t, k);
      sub[root][k] += sub[root][t];
    }
  }
}
long long DP(int a, int b) {
  if (a == b) return 0;
  if (dp[a][b] != -1) return dp[a][b];
  return dp[a][b] =
             sub[a][b] * sub[b][a] + max(DP(fa[a][b], a), DP(fa[b][a], b));
}
int main() {
  cin >> n;
  memset(dp, -1, sizeof(dp));
  for (int i = 1; i < n; i++) {
    int u, v;
    cin >> u >> v;
    edge[u].push_back(v);
    edge[v].push_back(u);
  }
  for (int i = 1; i <= n; i++) dfs(i, i, -1);
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n; j++)
      if (i != j) ans = max(ans, DP(i, j));
  cout << ans << endl;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random

class Exenonsattackonthegangsbootcamp(Basebootcamp):
    def __init__(self, n=5):
        self.n = n
    
    def case_generator(self):
        edges = self.generate_random_tree(self.n)
        return {
            'n': self.n,
            'edges': edges
        }
    
    @staticmethod
    def generate_random_tree(n):
        if n == 1:
            return []
        edges = []
        nodes = list(range(1, n+1))
        random.shuffle(nodes)
        for i in range(1, n):
            u = nodes[i]
            v = nodes[random.randint(0, i-1)]
            edges.append((min(u, v), max(u, v)))
        return edges
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        edges = question_case['edges']
        edge_lines = "\n".join([f"{u} {v}" for u, v in edges])
        prompt = f"""You are Xenon, a cybersecurity expert. A network of {n} gangs forms a tree structure. Each edge must be assigned a distinct integer from 0 to {n-2}. The password layers S is the sum of mex(u, v) for all pairs u < v, where mex is the minimum non-negative integer not present on the path between u and v. Find the maximum possible S.

Input:
{n}
{edge_lines}

Output the maximum S. Put your final answer within [answer] and [/answer], like [answer]5[/answer]."""
        return prompt
    
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
        n = identity['n']
        edges = identity['edges']
        adj = [[] for _ in range(n+1)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        sub = {}
        fa = {}
        for root in range(1, n+1):
            sub[root] = {}
            fa[root] = {}
            stack = [(root, -1)]  # (node, parent)
            sub[root][root] = 1
            fa[root][root] = -1  # Mark root's parent as -1
            while stack:
                node, parent = stack.pop()
                for neighbor in adj[node]:
                    if neighbor != parent:
                        fa[root][neighbor] = node
                        stack.append((neighbor, node))
            # Calculate subtree sizes using BFS from leaves
            post_order = []
            stack = [(root, -1, False)]
            while stack:
                node, parent, visited = stack.pop()
                if visited:
                    post_order.append(node)
                    total = 1
                    for neighbor in adj[node]:
                        if neighbor != parent and fa[root].get(neighbor, -1) == node:
                            total += sub[root][neighbor]
                    sub[root][node] = total
                else:
                    stack.append((node, parent, True))
                    for neighbor in adj[node]:
                        if neighbor != parent:
                            stack.append((neighbor, node, False))
        
        memo = {}
        pairs = [(i, j) for i in range(1, n+1) for j in range(1, n+1) if i != j]
        stack = pairs.copy()
        processed = set()
        
        while stack:
            a, b = stack.pop()
            if (a, b) in processed:
                continue
            if a == b:
                memo[(a, b)] = 0
                processed.add((a, b))
                continue
            
            fa_a_b = fa[a].get(b, -1)
            fa_b_a = fa[b].get(a, -1)
            deps = []
            if fa_a_b != -1 and (fa_a_b, a) not in memo:
                deps.append((fa_a_b, a))
            if fa_b_a != -1 and (fa_b_a, b) not in memo:
                deps.append((fa_b_a, b))
            
            if deps:
                stack.append((a, b))
                stack.extend(deps)
                continue
            
            opt1 = memo.get((fa_a_b, a), 0) if fa_a_b != -1 else 0
            opt2 = memo.get((fa_b_a, b), 0) if fa_b_a != -1 else 0
            memo[(a, b)] = sub[a][b] * sub[b][a] + max(opt1, opt2)
            processed.add((a, b))
        
        max_s = max(memo.values()) if memo else 0
        return solution == max_s
