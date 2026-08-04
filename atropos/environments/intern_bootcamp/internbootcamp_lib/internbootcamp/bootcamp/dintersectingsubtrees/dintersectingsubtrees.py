"""# 

### 谜题描述
You are playing a strange game with Li Chen. You have a tree with n nodes drawn on a piece of paper. All nodes are unlabeled and distinguishable. Each of you independently labeled the vertices from 1 to n. Neither of you know the other's labelling of the tree.

You and Li Chen each chose a subtree (i.e., a connected subgraph) in that tree. Your subtree consists of the vertices labeled x_1, x_2, …, x_{k_1} in your labeling, Li Chen's subtree consists of the vertices labeled y_1, y_2, …, y_{k_2} in his labeling. The values of x_1, x_2, …, x_{k_1} and y_1, y_2, …, y_{k_2} are known to both of you.

<image> The picture shows two labelings of a possible tree: yours on the left and Li Chen's on the right. The selected trees are highlighted. There are two common nodes.

You want to determine whether your subtrees have at least one common vertex. Luckily, your friend Andrew knows both labelings of the tree. You can ask Andrew at most 5 questions, each of which is in one of the following two forms: 

  * A x: Andrew will look at vertex x in your labeling and tell you the number of this vertex in Li Chen's labeling. 
  * B y: Andrew will look at vertex y in Li Chen's labeling and tell you the number of this vertex in your labeling. 



Determine whether the two subtrees have at least one common vertex after asking some questions. If there is at least one common vertex, determine one of your labels for any of the common vertices.

Interaction

Each test consists of several test cases.

The first line of input contains a single integer t (1 ≤ t ≤ 100) — the number of test cases.

For each testcase, your program should interact in the following format.

The first line contains a single integer n (1 ≤ n ≤ 1 000) — the number of nodes in the tree.

Each of the next n-1 lines contains two integers a_i and b_i (1≤ a_i, b_i≤ n) — the edges of the tree, indicating an edge between node a_i and b_i according to your labeling of the nodes.

The next line contains a single integer k_1 (1 ≤ k_1 ≤ n) — the number of nodes in your subtree.

The next line contains k_1 distinct integers x_1,x_2,…,x_{k_1} (1 ≤ x_i ≤ n) — the indices of the nodes in your subtree, according to your labeling. It is guaranteed that these vertices form a subtree.

The next line contains a single integer k_2 (1 ≤ k_2 ≤ n) — the number of nodes in Li Chen's subtree.

The next line contains k_2 distinct integers y_1, y_2, …, y_{k_2} (1 ≤ y_i ≤ n) — the indices (according to Li Chen's labeling) of the nodes in Li Chen's subtree. It is guaranteed that these vertices form a subtree according to Li Chen's labelling of the tree's nodes.

Test cases will be provided one by one, so you must complete interacting with the previous test (i.e. by printing out a common node or -1 if there is not such node) to start receiving the next one.

You can ask the Andrew two different types of questions. 

  * You can print \"A x\" (1 ≤ x ≤ n). Andrew will look at vertex x in your labeling and respond to you with the number of this vertex in Li Chen's labeling. 
  * You can print \"B y\" (1 ≤ y ≤ n). Andrew will look at vertex y in Li Chen's labeling and respond to you with the number of this vertex in your labeling. 



You may only ask at most 5 questions per tree.

When you are ready to answer, print \"C s\", where s is your label of a vertex that is common to both subtrees, or -1, if no such vertex exists. Printing the answer does not count as a question. Remember to flush your answer to start receiving the next test case. 

After printing a question do not forget to print end of line and flush the output. Otherwise, you will get Idleness limit exceeded. To do this, use:

  * fflush(stdout) or cout.flush() in C++; 
  * System.out.flush() in Java; 
  * flush(output) in Pascal; 
  * stdout.flush() in Python; 
  * see documentation for other languages. 



If the judge responds with -1, it means that you asked more queries than allowed, or asked an invalid query. Your program should immediately terminate (for example, by calling exit(0)). You will receive Wrong Answer; it means that you asked more queries than allowed, or asked an invalid query. If you ignore this, you can get other verdicts since your program will continue to read from a closed stream.

Hack Format

To hack, use the following format. Note that you can only hack with one test case.

The first line should contain a single integer t (t=1).

The second line should contain a single integer n (1 ≤ n ≤ 1 000).

The third line should contain n integers p_1, p_2, …, p_n (1≤ p_i≤ n) — a permutation of 1 to n. This encodes the labels that Li Chen chose for his tree. In particular, Li Chen chose label p_i for the node you labeled i.

Each of the next n-1 lines should contain two integers a_i and b_i (1≤ a_i, b_i≤ n). These edges should form a tree.

The next line should contain a single integer k_1 (1 ≤ k_1 ≤ n).

The next line should contain k_1 distinct integers x_1,x_2,…,x_{k_1} (1 ≤ x_i ≤ n). These vertices should form a subtree.

The next line should contain a single integer k_2 (1 ≤ k_2 ≤ n).

The next line should contain k_2 distinct integers y_1, y_2, …, y_{k_2} (1 ≤ y_i ≤ n). These vertices should form a subtree in Li Chen's tree according to the permutation above.

Examples

Input

1
3
1 2
2 3
1
1
1
2
2
1


Output

A 1
B 2
C 1


Input

2
6
1 2
1 3
1 4
4 5
4 6
4
1 3 4 5
3
3 5 2
3
6
1 2
1 3
1 4
4 5
4 6
3
1 2 3
3
4 1 6
5


Output

B 2
C 1
A 1
C -1

Note

For the first sample, Li Chen's hidden permutation is [2, 3, 1], and for the second, his hidden permutation is [5, 3, 2, 4, 1, 6] for both cases.

In the first sample, there is a tree with three nodes in a line. On the top, is how you labeled the tree and the subtree you chose, and the bottom is how Li Chen labeled the tree and the subtree he chose: 

<image>

In the first question, you ask Andrew to look at node 1 in your labelling and tell you the label of it in Li Chen's labelling. Andrew responds with 2. At this point, you know that both of your subtrees contain the same node (i.e. node 1 according to your labeling), so you can output \"C 1\" and finish. However, you can also ask Andrew to look at node 2 in Li Chen's labelling and tell you the label of it in your labelling. Andrew responds with 1 (this step was given with the only reason — to show you how to ask questions).

For the second sample, there are two test cases. The first looks is the one from the statement: 

<image>

We first ask \"B 2\", and Andrew will tell us 3. In this case, we know 3 is a common vertex, and moreover, any subtree with size 3 that contains node 3 must contain node 1 as well, so we can output either \"C 1\" or \"C 3\" as our answer.

In the second case in the second sample, the situation looks as follows: 

<image>

In this case, you know that the only subtree of size 3 that doesn't contain node 1 is subtree 4,5,6. You ask Andrew for the label of node 1 in Li Chen's labelling and Andrew says 5. In this case, you know that Li Chen's subtree doesn't contain node 1, so his subtree must be consist of the nodes 4,5,6 (in your labelling), thus the two subtrees have no common nodes.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
using namespace std;
void enumerateSubmasks(long long m) {
  for (long long s = m;; s = (s - 1) & m) {
    if (s == 0) {
      break;
    }
  }
}
long long mpow(long long a, long long b, long long m) {
  if (b == 0) return 1;
  long long x = mpow(a, b / 2, m);
  x = (x * x) % m;
  if (b % 2) {
    x = (x * a) % m;
  }
  return x;
}
void update(long long s, long long e, long long qs, long long qe,
            vector<long long> &seg, vector<long long> &lazy, long long index,
            long long value) {
  if (lazy[index] != -1) {
    seg[index] = max(seg[index], lazy[index]);
    if (s != e) {
      if (lazy[2 * index] == -1)
        lazy[2 * index] = lazy[index];
      else
        lazy[2 * index] = max(lazy[2 * index], lazy[index]);
      if (lazy[2 * index + 1] == -1)
        lazy[2 * index + 1] = lazy[index];
      else
        lazy[2 * index + 1] = max(lazy[2 * index + 1], lazy[index]);
    }
    lazy[index] = -1;
  }
  if (qs > e || qe < s) return;
  if (s >= qs && e <= qe) {
    seg[index] = max(seg[index], value);
    if (s != e) {
      if (lazy[2 * index] == -1)
        lazy[2 * index] = value;
      else
        lazy[2 * index] = max(lazy[2 * index], value);
      if (lazy[2 * index + 1] == -1)
        lazy[2 * index + 1] = value;
      else
        lazy[2 * index + 1] = max(lazy[2 * index + 1], value);
    }
    return;
  }
  long long mid = (s + e) / 2;
  update(s, mid, qs, qe, seg, lazy, 2 * index, value);
  update(mid + 1, e, qs, qe, seg, lazy, 2 * index + 1, value);
}
long long query(long long s, long long e, long long qs, long long qe,
                vector<long long> &seg, vector<long long> &lazy,
                long long index) {
  if (lazy[index] != -1) {
    seg[index] = max(seg[index], lazy[index]);
    if (s != e) {
      if (lazy[2 * index] == -1)
        lazy[2 * index] = lazy[index];
      else
        lazy[2 * index] = max(lazy[2 * index], lazy[index]);
      if (lazy[2 * index + 1] == -1)
        lazy[2 * index + 1] = lazy[index];
      else
        lazy[2 * index + 1] = max(lazy[2 * index + 1], lazy[index]);
    }
    lazy[index] = -1;
  }
  if (qs > e || qe < s) return LLONG_MIN;
  if (s >= qs && e <= qe) {
    return seg[index];
  }
  long long mid = (s + e) / 2;
  long long a = query(s, mid, qs, qe, seg, lazy, 2 * index);
  long long b = query(mid + 1, e, qs, qe, seg, lazy, 2 * index + 1);
  return max(a, b);
}
void printBinaryString(long long n) {
  vector<long long> temp;
  while (n) {
    if (n & 1)
      temp.push_back(1);
    else
      temp.push_back(0);
    n = n >> 1;
  }
  reverse(temp.begin(), temp.end());
  for (auto node : temp) cout << node << \" \";
  cout << endl;
}
void readVector(vector<long long> &a) {
  long long n = a.size();
  for (long long i = 0; i < n; ++i) cin >> a[i];
}
struct node {
  long long id;
  long long val;
  char dir;
};
map<long long, list<long long>> adj;
map<long long, long long> par;
map<long long, bool> x;
map<long long, bool> y;
long long k1, k2;
long long answer;
long long interactA(long long x) {
  cout << \"A \" << x << endl;
  long long ret;
  cin >> ret;
  fflush(stdout);
  return ret;
}
long long interactB(long long x) {
  cout << \"B \" << x << endl;
  long long ret;
  cin >> ret;
  fflush(stdout);
  return ret;
}
pair<long long, bool> solve(long long node, long long par, long long k) {
  long long totalInSubtree = 1;
  for (auto child : adj[node]) {
    if (child == par) continue;
    auto ret = solve(child, node, k);
    bool mila = ret.second;
    if (mila) return {0, true};
    totalInSubtree += ret.first;
  }
  if (totalInSubtree < k) return {totalInSubtree, false};
  if (x[node] == false) return {0, false};
  long long bLabel = interactA(node);
  if (y[bLabel]) {
    answer = node;
    return {0, true};
  } else
    return {0, false};
}
int32_t main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  long long tc;
  cin >> tc;
  while (tc--) {
    answer = -1;
    x.clear();
    y.clear();
    adj.clear();
    par.clear();
    long long n;
    cin >> n;
    for (long long i = 0; i < n - 1; ++i) {
      long long u, v;
      cin >> u >> v;
      adj[u].push_back(v);
      adj[v].push_back(u);
    }
    cin >> k1;
    for (long long i = 0; i < k1; ++i) {
      long long temp;
      cin >> temp;
      x[temp] = true;
    }
    long long st;
    cin >> k2;
    for (long long i = 0; i < k2; ++i) {
      long long temp;
      cin >> temp;
      st = temp;
      y[temp] = true;
    }
    long long start = interactB(st);
    queue<long long> bfs;
    long long toCompare;
    bfs.push(start);
    map<long long, bool> visited;
    while (!bfs.empty()) {
      auto node = bfs.front();
      bfs.pop();
      visited[node] = true;
      if (x[node]) {
        toCompare = node;
        break;
      }
      for (auto child : adj[node]) {
        if (visited[child]) continue;
        bfs.push(child);
      }
    }
    long long temp = interactA(toCompare);
    if (y[temp])
      cout << \"C \" << toCompare << endl;
    else
      cout << \"C -1\" << endl;
    fflush(stdout);
  }
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
from collections import deque
import re

class Dintersectingsubtreesbootcamp(Basebootcamp):
    def __init__(self, n=5, sol_prob=0.5, k1_range=(1, 3), k2_range=(1, 3)):
        self.n = n
        self.sol_prob = sol_prob
        self.k1_range = k1_range
        self.k2_range = k2_range

    def generate_random_tree(self, n):
        """生成n节点的随机树"""
        if n == 1:
            return {1: []}
        parent = {}
        for i in range(2, n + 1):
            parent[i] = random.randint(1, i - 1)
        adj = {i: [] for i in range(1, n + 1)}
        for child in parent:
            p = parent[child]
            adj[p].append(child)
            adj[child].append(p)
        return adj

    def tree_adj_to_edges(self, adj):
        """邻接表转换为边的列表"""
        edges = set()
        for node in adj:
            for neighbor in adj[node]:
                if node < neighbor:
                    edges.add((node, neighbor))
        return sorted(edges)

    def generate_connected_subset(self, adj, k, must_include=None):
        """生成严格包含k个节点的连通子集"""
        if k == 0 or not adj:
            return None
        candidates = [must_include] if must_include else list(adj.keys())
        for start in candidates:
            if start not in adj:
                continue
            visited = set()
            queue = deque([start])
            visited.add(start)
            while queue and len(visited) < k:
                current = queue.popleft()
                for neighbor in adj.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        if len(visited) == k:
                            break
                if len(visited) == k:
                    return sorted(list(visited))
            if len(visited) == k:
                return sorted(list(visited))
        return None

    def case_generator(self):
        """生成符合所有约束的案例"""
        for _ in range(20):  # 增加生成尝试次数
            adj = self.generate_random_tree(self.n)
            edges = self.tree_adj_to_edges(adj)
            permutation = list(range(1, self.n + 1))
            random.shuffle(permutation)
            have_solution = random.random() < self.sol_prob

            k1 = random.randint(*self.k1_range)
            k2 = random.randint(*self.k2_range)
            k1 = min(k1, self.n)
            k2 = min(k2, self.n)

            # 生成用户子树
            x_list = self.generate_connected_subset(adj, k1)
            if not x_list or len(x_list) != k1:
                continue

            # 构建李晨的树结构
            lich_adj = {}
            for u in adj:
                p_u = permutation[u-1]
                lich_adj[p_u] = [permutation[v-1] for v in adj[u]]

            # 生成李晨子树
            if have_solution:
                c = random.choice(x_list)
                p_c = permutation[c-1]
                y_list = self.generate_connected_subset(lich_adj, k2, p_c)
            else:
                y_list = self.generate_connected_subset(lich_adj, k2)
            
            if not y_list or len(y_list) != k2:
                continue

            # 转换为用户标签并验证约束
            inv_perm = {v: i+1 for i, v in enumerate(permutation)}
            q_list = [inv_perm[y] for y in y_list]
            is_valid_solution = not set(x_list).isdisjoint(q_list)
            
            if have_solution != is_valid_solution:
                continue  # 解决方案状态不匹配，重新生成

            return {
                'n': self.n,
                'edges': edges,
                'permutation': permutation,
                'x_list': x_list,
                'y_list': y_list
            }

        # 降级案例
        return {
            'n': 3,
            'edges': [(1,2), (2,3)],
            'permutation': [2,3,1],
            'x_list': [1],
            'y_list': [2]
        }

    @staticmethod
    def prompt_func(question_case):
        edges = '\n'.join(f"{a} {b}" for a, b in question_case['edges'])
        x_list = ' '.join(map(str, question_case['x_list']))
        y_list = ' '.join(map(str, question_case['y_list']))
        return f"""You are solving a tree puzzle. The tree has {question_case['n']} nodes connected as:
{edges}
Your subtree nodes: {x_list}
Li Chen's subtree nodes (his labels): {y_list}

Ask up to 5 questions (A X or B Y) to determine a common node. Final answer as 'C s' within [answer] tags."""

    @staticmethod
    def extract_output(output):
        answers = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if answers:
            last = answers[-1].strip()
            match = re.search(r'C\s*(-1|\d+)', last, re.I)
            if match:
                return int(match.group(1)) if match.group(1) != '-1' else -1
        matches = re.findall(r'C\s*(-1|\d+)\b', output, re.I)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        x_nodes = set(identity['x_list'])
        perm = identity['permutation']
        y_labels = set(identity['y_list'])
        
        if solution == -1:
            lich_nodes = {inv_perm[y] for y in y_labels}
            return x_nodes.isdisjoint(lich_nodes)
        
        return solution in x_nodes and perm[solution-1] in y_labels
