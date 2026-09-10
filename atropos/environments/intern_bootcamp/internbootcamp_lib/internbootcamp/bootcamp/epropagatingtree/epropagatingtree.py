"""# 

### 谜题描述
Iahub likes trees very much. Recently he discovered an interesting tree named propagating tree. The tree consists of n nodes numbered from 1 to n, each node i having an initial value ai. The root of the tree is node 1.

This tree has a special property: when a value val is added to a value of node i, the value -val is added to values of all the children of node i. Note that when you add value -val to a child of node i, you also add -(-val) to all children of the child of node i and so on. Look an example explanation to understand better how it works.

This tree supports two types of queries:

  * \"1 x val\" — val is added to the value of node x; 
  * \"2 x\" — print the current value of node x. 



In order to help Iahub understand the tree better, you must answer m queries of the preceding type.

Input

The first line contains two integers n and m (1 ≤ n, m ≤ 200000). The second line contains n integers a1, a2, ..., an (1 ≤ ai ≤ 1000). Each of the next n–1 lines contains two integers vi and ui (1 ≤ vi, ui ≤ n), meaning that there is an edge between nodes vi and ui.

Each of the next m lines contains a query in the format described above. It is guaranteed that the following constraints hold for all queries: 1 ≤ x ≤ n, 1 ≤ val ≤ 1000.

Output

For each query of type two (print the value of node x) you must print the answer to the query on a separate line. The queries must be answered in the order given in the input.

Examples

Input

5 5
1 2 1 1 2
1 2
1 3
2 4
2 5
1 2 3
1 1 2
2 1
2 2
2 4


Output

3
3
0

Note

The values of the nodes are [1, 2, 1, 1, 2] at the beginning.

Then value 3 is added to node 2. It propagates and value -3 is added to it's sons, node 4 and node 5. Then it cannot propagate any more. So the values of the nodes are [1, 5, 1, - 2, - 1].

Then value 2 is added to node 1. It propagates and value -2 is added to it's sons, node 2 and node 3. From node 2 it propagates again, adding value 2 to it's sons, node 4 and node 5. Node 3 has no sons, so it cannot propagate from there. The values of the nodes are [3, 3, - 1, 0, 1].

You can see all the definitions about the tree at the following link: http://en.wikipedia.org/wiki/Tree_(graph_theory)

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int c = 2e5 + 5;
int strength[c], level[c], size[c], visited[c], tree_even[4 * c],
    tree_odd[4 * c], M;
std::vector<int> v[c], preorder;
map<int, int> Map;
int dfs(int vertex, int lev) {
  if (visited[vertex]) return 0;
  visited[vertex] = 1;
  level[vertex] = lev;
  size[vertex] = 1;
  preorder.push_back(vertex);
  Map[vertex] = (int)preorder.size() - 1;
  for (int i = 0; i < (int)v[vertex].size(); i++) {
    size[vertex] += dfs(v[vertex][i], lev + 1);
  }
  return size[vertex];
}
void update(int index, int s, int e, int l, int r, int val) {
  if (e < l || s > r) return;
  if (s >= l && e <= r) {
    if (M == 0)
      tree_even[index] += val;
    else
      tree_odd[index] += val;
    return;
  }
  int mid = (s + e) / 2;
  update(2 * index, s, mid, l, r, val);
  update(2 * index + 1, mid + 1, e, l, r, val);
  return;
}
int sum;
int query(int index, int l, int r, int i) {
  if (i < l || i > r) return 0;
  if (l <= i && i <= r) {
    if (M == 0)
      sum += tree_even[index] - tree_odd[index];
    else
      sum += tree_odd[index] - tree_even[index];
  }
  if (l == r) return 0;
  int mid = (l + r) / 2;
  query(2 * index, l, mid, i);
  query(2 * index + 1, mid + 1, r, i);
  return 0;
}
int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL), cout.tie(NULL);
  int n, m, v1, v2, type, node, upd, start;
  cin >> n >> m;
  for (int i = 1; i < n + 1; i++) cin >> strength[i];
  for (int i = 0; i < n - 1; i++) {
    cin >> v1 >> v2;
    v[v1].push_back(v2);
    v[v2].push_back(v1);
  }
  preorder.push_back(0);
  dfs(1, 0);
  while (m--) {
    cin >> type;
    if (type == 1) {
      cin >> node >> upd;
      start = Map[node];
      M = level[node] % 2;
      update(1, 1, n, start, start + size[node] - 1, upd);
    } else {
      cin >> node;
      start = Map[node];
      M = level[node] % 2;
      sum = strength[node];
      query(1, 1, n, start);
      cout << sum << \"\n\";
    }
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from collections import deque

class Epropagatingtreebootcamp(Basebootcamp):
    def __init__(self, n=5, m=5):
        self.n = n
        self.m = m

    def case_generator(self):
        n = self.n
        edges, parent = self.generate_tree(n)
        level = self.compute_levels(parent, n)
        children = self.build_children_dict(parent)
        subtree_nodes = {x: self.get_subtree_nodes(x, children) for x in range(1, n+1)}
        initial_a = [random.randint(1, 1000) for _ in range(n)]
        
        current_values = initial_a.copy()
        values_history = [current_values.copy()]
        queries = []
        correct_outputs = []
        
        # Generate all queries first
        for _ in range(self.m):
            if len(correct_outputs) < self.m // 2 or random.random() < 0.5:
                # Generate type 1 query
                x = random.randint(1, n)
                val = random.randint(1, 1000)
                queries.append(('1', x, val))
                
                # Record state before applying
                values_history.append(current_values.copy())
                # Apply changes
                x_level = level[x]
                for y in subtree_nodes[x]:
                    delta = val * ((-1) ** (level[y] - x_level))
                    current_values[y-1] += delta
            else:
                # Generate type 2 query
                x = random.randint(1, n)
                queries.append(('2', x))
                correct_outputs.append(current_values[x-1])
                values_history.append(current_values.copy())
        
        # Ensure at least one type 2 query and correct outputs
        if not correct_outputs:
            # Find last type 1 query to replace
            for i in reversed(range(len(queries))):
                if queries[i][0] == '1':
                    # Get state before this query
                    prev_values = values_history[i]
                    x = random.randint(1, n)
                    # Replace with type 2 query
                    queries[i] = ('2', x)
                    correct_outputs.insert(i - sum(1 for q in queries[:i] if q[0] == '2'), prev_values[x-1])
                    break
        
        # Format queries to strings
        formatted_queries = []
        for q in queries:
            if q[0] == '1':
                formatted_queries.append(f'1 {q[1]} {q[2]}')
            else:
                formatted_queries.append(f'2 {q[1]}')
        
        case = {
            'n': n,
            'm': self.m,
            'a': initial_a,
            'edges': [[u, v] for u, v in edges],
            'queries': formatted_queries,
            'correct_outputs': correct_outputs
        }
        return case

    @staticmethod
    def generate_tree(n):
        if n == 1:
            return [], {1: None}
        parent = {1: None}
        edges = []
        available = [1]
        for i in range(2, n+1):
            p = random.choice(available)
            parent[i] = p
            edges.append((p, i))
            available.append(i)
        return edges, parent

    @staticmethod
    def compute_levels(parent, n):
        level = {1: 0}
        for i in range(2, n+1):
            level[i] = level[parent[i]] + 1
        return level

    @staticmethod
    def build_children_dict(parent):
        children = {}
        for child in parent:
            p = parent.get(child)
            if p is not None:
                children.setdefault(p, []).append(child)
        return children

    @staticmethod
    def get_subtree_nodes(x, children):
        subtree = []
        stack = [x]
        while stack:
            node = stack.pop()
            subtree.append(node)
            stack.extend(reversed(children.get(node, [])))  # Maintain order
        return subtree

    @staticmethod
    def prompt_func(question_case) -> str:
        input_lines = [
            f"{question_case['n']} {question_case['m']}",
            ' '.join(map(str, question_case['a']))
        ]
        input_lines += [' '.join(map(str, e)) for e in question_case['edges']]
        input_lines += question_case['queries']
        joined_input = '\n'.join(input_lines)
        
        problem_text = f"""Iahub discovered a propagating tree with a special property. When a value is added to a node, it propagates to children with alternating signs. 

You must process the following queries and output answers for type 2 queries. Format each answer on a separate line within [answer] and [/answer] tags.

Input:
{joined_input}

Provide your answers for type 2 queries in order, each enclosed in [answer] tags:"""
        return problem_text

    @staticmethod
    def extract_output(output):
        answers = []
        matches = re.findall(r'\[answer\]\s*(-?\d+)\s*\[/answer\]', output, re.IGNORECASE)
        return list(map(int, matches)) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity.get('correct_outputs', [])
        return isinstance(solution, list) and solution == expected
