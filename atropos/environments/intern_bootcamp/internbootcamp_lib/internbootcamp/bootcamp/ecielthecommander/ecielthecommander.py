"""# 

### 谜题描述
Now Fox Ciel becomes a commander of Tree Land. Tree Land, like its name said, has n cities connected by n - 1 undirected roads, and for any two cities there always exists a path between them.

Fox Ciel needs to assign an officer to each city. Each officer has a rank — a letter from 'A' to 'Z'. So there will be 26 different ranks, and 'A' is the topmost, so 'Z' is the bottommost.

There are enough officers of each rank. But there is a special rule must obey: if x and y are two distinct cities and their officers have the same rank, then on the simple path between x and y there must be a city z that has an officer with higher rank. The rule guarantee that a communications between same rank officers will be monitored by higher rank officer.

Help Ciel to make a valid plan, and if it's impossible, output \"Impossible!\".

Input

The first line contains an integer n (2 ≤ n ≤ 105) — the number of cities in Tree Land.

Each of the following n - 1 lines contains two integers a and b (1 ≤ a, b ≤ n, a ≠ b) — they mean that there will be an undirected road between a and b. Consider all the cities are numbered from 1 to n.

It guaranteed that the given graph will be a tree.

Output

If there is a valid plane, output n space-separated characters in a line — i-th character is the rank of officer in the city with number i. 

Otherwise output \"Impossible!\".

Examples

Input

4
1 2
1 3
1 4


Output

A B B B


Input

10
1 2
2 3
3 4
4 5
5 6
6 7
7 8
8 9
9 10


Output

D C B A D C B D C D

Note

In the first example, for any two officers of rank 'B', an officer with rank 'A' will be on the path between them. So it is a valid solution.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int SIZE = 100009;
vector<int> adj[SIZE];
int N;
bool pass[SIZE];
char ans[SIZE];
int table[SIZE];
void assign(int, char, int);
int build(int n, int p, char c, int total) {
  table[n] = 1;
  bool check = true;
  for (vector<int>::iterator it = adj[n].begin(); it != adj[n].end(); it++)
    if (*it != p && !pass[*it]) {
      if (build(*it, n, c, total) == -1) return -1;
      if (table[*it] > total / 2) check = false;
      table[n] += table[*it];
    }
  if (total - table[n] > total / 2) check = false;
  if (check) {
    ans[n] = c;
    pass[n] = true;
    for (vector<int>::iterator it = adj[n].begin(); it != adj[n].end(); it++)
      if (*it != p && !pass[*it]) assign(*it, c + 1, table[*it]);
    if (p) assign(p, c + 1, total - table[n]);
    return -1;
  }
  return table[n];
}
void assign(int n, char c, int total) { build(n, 0, c, total); }
int main() {
  scanf(\"%d\", &N);
  for (int i = 1; i < N; i++) {
    int a, b;
    scanf(\"%d%d\", &a, &b);
    adj[a].push_back(b);
    adj[b].push_back(a);
  }
  assign(1, 'A', N);
  for (int i = 1; i <= N; i++) {
    if (i > 1) printf(\" \");
    printf(\"%c\", ans[i]);
  }
  printf(\"\n\");
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import defaultdict, deque
from bootcamp import Basebootcamp

def generate_tree(n):
    """Generate a random tree using Prüfer sequence with shuffled node labels."""
    if n == 1: return []
    labels = list(range(1, n+1))
    random.shuffle(labels)
    
    if n == 2: return [(labels[0], labels[1])]
    
    prufer = [random.randint(0, n-2) for _ in range(n-2)]
    node_count = [0] * n
    for node in prufer: node_count[node] += 1
    
    edges = []
    leaf = None
    for node in prufer:
        if leaf is None:
            for i in range(n):
                if node_count[i] == 0 and i != node:
                    leaf = i
                    break
        edges.append((leaf, node))
        node_count[leaf] = -1
        node_count[node] -= 1
        if node_count[node] == 0 and leaf > node:
            leaf = node
        else:
            leaf = None
    
    last_nodes = [i for i in range(n) if node_count[i] != -1]
    edges.append((last_nodes[0], last_nodes[1]))
    
    return [(labels[a], labels[b]) for a, b in edges]

class SolutionValidator:
    def __init__(self, n, edges, solution):
        self.n = n
        self.adj = [[] for _ in range(n+1)]
        for a, b in edges:
            self.adj[a].append(b)
            self.adj[b].append(a)
        self.rank = solution.split() if solution != "Impossible!" else []
        self.parent = [0]*(n+1)
        self.depth = [0]*(n+1)
        self._build_lca(1, 0)

    def _build_lca(self, u, p):
        stack = [(u, p, False)]
        while stack:
            u, p, visited = stack.pop()
            if visited:
                for v in self.adj[u]:
                    if v != p and v != self.parent[v]:
                        self.depth[v] = self.depth[u] + 1
                        self.parent[v] = u
            else:
                stack.append((u, p, True))
                for v in self.adj[u]:
                    if v != p:
                        stack.append((v, u, False))

    def _lca(self, u, v):
        while u != v:
            if self.depth[u] > self.depth[v]:
                u = self.parent[u]
            else:
                v = self.parent[v]
        return u

    def validate(self):
        if self.rank == ["Impossible!"]:
            return self._validate_impossible()
        
        if len(self.rank) != self.n:
            return False
        ranks = {}
        for i, r in enumerate(self.rank):
            if len(r) != 1 or not r.isupper():
                return False
            ranks[i+1] = r

        # Check all pairs with same rank
        rank_map = defaultdict(list)
        for node in range(1, self.n+1):
            rank_map[ranks[node]].append(node)

        for r, nodes in rank_map.items():
            if len(nodes) < 2: 
                continue
            # Check all pairs
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    a, b = nodes[i], nodes[j]
                    lca = self._lca(a, b)
                    path = []
                    while a != lca:
                        path.append(a)
                        a = self.parent[a]
                    path.append(lca)
                    temp = []
                    while b != lca:
                        temp.append(b)
                        b = self.parent[b]
                    path += reversed(temp)
                    # Check path
                    has_higher = False
                    for node in path:
                        if ranks[node] < r:
                            has_higher = True
                            break
                    if not has_higher:
                        return False
        return True

    def _validate_impossible(self):
        try:
            gen = SolutionGenerator(self.n, self.adj[1:])
            solution = gen.generate()
            return solution == "Impossible!"
        except:
            return False

class Ecielthecommanderbootcamp(Basebootcamp):
    def __init__(self, max_n=15, min_n=2):
        self.max_n = max_n
        self.min_n = min_n

    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        edges = generate_tree(n)
        return {'n': n, 'edges': edges}

    @staticmethod
    def prompt_func(case):
        n = case['n']
        edges = case['edges']
        edge_lines = '\n'.join(f"{a} {b}" for a, b in edges)
        return f"""As the commander of Tree Land with {n} cities connected in a tree structure:
{edge_lines}

Assign A-Z ranks to each city such that:
- Any two cities with the same rank must have a higher-ranked city on their connecting path

Output format: Either "Impossible!" or {n} space-separated uppercase letters.
Enclose your final answer within [answer] and [/answer] tags."""

    @staticmethod
    def extract_output(output):
        import re
        answers = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answers:
            return None
        answer = answers[-1].strip()
        if answer.upper() == "IMPOSSIBLE!":
            return "Impossible!"
        return answer

    @classmethod
    def _verify_correction(cls, solution, identity):
        validator = SolutionValidator(
            identity['n'],
            identity['edges'],
            solution
        )
        return validator.validate()
