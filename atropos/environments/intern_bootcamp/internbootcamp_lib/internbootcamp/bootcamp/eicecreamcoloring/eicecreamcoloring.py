"""# 

### 谜题描述
Isart and Modsart were trying to solve an interesting problem when suddenly Kasra arrived. Breathless, he asked: \"Can you solve a problem I'm stuck at all day?\"

We have a tree T with n vertices and m types of ice cream numerated from 1 to m. Each vertex i has a set of si types of ice cream. Vertices which have the i-th (1 ≤ i ≤ m) type of ice cream form a connected subgraph. We build a new graph G with m vertices. We put an edge between the v-th and the u-th (1 ≤ u, v ≤ m, u ≠ v) vertices in G if and only if there exists a vertex in T that has both the v-th and the u-th types of ice cream in its set. The problem is to paint the vertices of G with minimum possible number of colors in a way that no adjacent vertices have the same color.

Please note that we consider that empty set of vertices form a connected subgraph in this problem.

As usual, Modsart don't like to abandon the previous problem, so Isart wants you to solve the new problem.

Input

The first line contains two integer n and m (1 ≤ n, m ≤ 3·105) — the number of vertices in T and the number of ice cream types.

n lines follow, the i-th of these lines contain single integer si (0 ≤ si ≤ 3·105) and then si distinct integers, each between 1 and m — the types of ice cream in the i-th vertex. The sum of si doesn't exceed 5·105.

n - 1 lines follow. Each of these lines describes an edge of the tree with two integers u and v (1 ≤ u, v ≤ n) — the indexes of connected by this edge vertices.

Output

Print single integer c in the first line — the minimum number of colors to paint the vertices in graph G.

In the second line print m integers, the i-th of which should be the color of the i-th vertex. The colors should be between 1 and c. If there are some answers, print any of them.

Examples

Input

3 3
1 1
2 2 3
1 2
1 2
2 3


Output

2
1 1 2 

Input

4 5
0
1 1
1 3
3 2 4 5
2 1
3 2
4 3


Output

3
1 1 1 2 3 

Note

In the first example the first type of ice cream is present in the first vertex only, so we can color it in any color. The second and the third ice cream are both presented in the second vertex, so we should paint them in different colors.

In the second example the colors of the second, the fourth and the fifth ice cream should obviously be distinct.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int col[300010], flag[300010];
vector<int> ice[300010];
vector<int> edge[300010];
int dfn;
void dfs(int u, int f) {
  for (int ic : ice[u]) {
    if (col[ic]) {
      flag[col[ic]] = dfn;
    }
  }
  int cur = 1;
  for (int ic : ice[u]) {
    if (!col[ic]) {
      while (flag[cur] == dfn) cur++;
      col[ic] = cur;
      cur++;
    }
  }
  dfn++;
  for (int v : edge[u]) {
    if (v != f) {
      dfs(v, u);
    }
  }
}
int main() {
  int n, m, s;
  scanf(\"%d%d\", &n, &m);
  int ret = 1;
  for (int i = 1; i <= n; i++) {
    scanf(\"%d\", &s);
    ret = max(ret, s);
    int t;
    for (int j = 0; j < s; j++) {
      scanf(\"%d\", &t);
      ice[i].push_back(t);
    }
  }
  int u, v;
  for (int i = 1; i < n; i++) {
    scanf(\"%d%d\", &u, &v);
    edge[u].push_back(v);
    edge[v].push_back(u);
  }
  dfn = 1;
  dfs(1, -1);
  printf(\"%d\n\", ret);
  for (int i = 1; i <= m; i++) {
    if (!col[i]) col[i] = 1;
    printf(\"%d \", col[i]);
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import defaultdict

from bootcamp import Basebootcamp

class Eicecreamcoloringbootcamp(Basebootcamp):
    def __init__(self, n=10, m=10, max_si=5):
        self.n = n
        self.m = m
        self.max_si = max_si
        self.tree_edges = []
        self.node_types = defaultdict(list)
        self.g_edges = set()

    def case_generator(self):
        # Generate a tree structure
        self.tree_edges = []
        # Create a more complex tree structure
        parent = [i for i in range(self.n)]
        for i in range(1, self.n):
            # Choose a random parent for node i+1
            p = random.randint(1, i)
            self.tree_edges.append((p, i+1))
        
        # Assign each type to a node ensuring connected subgraph
        self.node_types = defaultdict(list)
        for type_id in range(1, self.m+1):
            # Choose a root node randomly
            root = random.randint(1, self.n)
            # Perform BFS to select a connected subset
            visited = set()
            queue = [root]
            visited.add(root)
            while queue and random.random() < 0.7:  # Control the size of the subset
                current = queue.pop(0)
                self.node_types[current].append(type_id)
                # Add neighbors to queue
                for neighbor in self.get_neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
        
        # Build the graph G
        self.g_edges = set()
        for node in self.node_types.values():
            # Generate all pairs of types in this node
            for i in range(len(node)):
                for j in range(i+1, len(node)):
                    u, v = node[i], node[j]
                    if u < v:
                        self.g_edges.add((u, v))
                    else:
                        self.g_edges.add((v, u))
        
        # Construct the problem case
        case = {
            'n': self.n,
            'm': self.m,
            'node_types': {k: v for k, v in self.node_types.items()},
            'tree_edges': self.tree_edges,
            'g_edges': list(self.g_edges)
        }
        return case

    def get_neighbors(self, node):
        # Helper function to get neighbors of a node in the tree
        neighbors = []
        for u, v in self.tree_edges:
            if u == node:
                neighbors.append(v)
            if v == node:
                neighbors.append(u)
        return neighbors

    @staticmethod
    def prompt_func(question_case):
        prompt = (
            "We have a tree T with {} vertices and {} types of ice cream. Each vertex contains certain types of ice cream as follows:\n"
            "Vertex Types:\n"
        ).format(question_case['n'], question_case['m'])
        
        for node, types in question_case['node_types'].items():
            prompt += f"Vertex {node}: {types}\n"
        
        prompt += (
            "The tree T has the following edges:\n"
            "Tree Edges:\n"
        )
        for u, v in question_case['tree_edges']:
            prompt += f"{u} {v}\n"
        
        prompt += (
            "We need to construct a graph G where each vertex represents an ice cream type. There is an edge between two types if they appear together in at least one vertex of T.\n"
            "The task is to color the vertices of G using the minimum number of colors such that no two adjacent vertices share the same color.\n"
            "Please provide the minimum number of colors and the color assignment for each type. The colors should be integers starting from 1.\n"
            "Format your answer as follows:\n"
            "[answer]\n"
            "c\n"
            "c1 c2 c3 ... cm\n"
            "[/answer]\n"
        )
        
        return prompt

    @staticmethod
    def extract_output(output):
        pattern = r'\[answer\](.*?)\[\/answer\]'
        matches = re.findall(pattern, output, re.DOTALL)
        if not matches:
            return None
        last_answer = matches[-1].strip()
        lines = last_answer.split('\n')
        if len(lines) < 2:
            return None
        try:
            c = int(lines[0].strip())
            colors = list(map(int, lines[1].strip().split()))
            return (c, colors)
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False
        c, colors = solution
        if len(colors) != identity['m']:
            return False
        if any(color < 1 or color > c for color in colors):
            return False
        for u, v in identity['g_edges']:
            if colors[u-1] == colors[v-1]:
                return False
        return True
