"""# 

### 谜题描述
Note that the only difference between the easy and hard version is the constraint on the number of queries. You can make hacks only if all versions of the problem are solved.

This is an interactive problem.

You are given a tree consisting of n nodes numbered with integers from 1 to n. Ayush and Ashish chose two secret distinct nodes in the tree. You need to find out both the nodes. You can make the following query: 

  * Provide a list of nodes and you will receive a node from that list whose sum of distances to both the hidden nodes is minimal (if there are multiple such nodes in the list, you will receive any one of them). You will also get the sum of distances of that node to the hidden nodes. 



Recall that a tree is a connected graph without cycles. The distance between two nodes is defined as the number of edges in the simple path between them.

More formally, let's define two hidden nodes as s and f. In one query you can provide the set of nodes \\{a_1, a_2, …, a_c\} of the tree. As a result, you will get two numbers a_i and dist(a_i, s) + dist(a_i, f). The node a_i is any node from the provided set, for which the number dist(a_i, s) + dist(a_i, f) is minimal.

You can ask no more than 11 queries.

Input

The first line contains a single integer t (1 ≤ t ≤ 10) — the number of test cases. Please note, how the interaction process is organized.

The first line of each test case consists of a single integer n (2 ≤ n ≤ 1000) — the number of nodes in the tree.

The next n - 1 lines consist of two integers u, v (1 ≤ u, v ≤ n, u ≠ v) — the edges of the tree.

Interaction

To ask a query print a single line: 

  * In the beginning print \"? c \" (without quotes) where c (1 ≤ c ≤ n) denotes the number of nodes being queried, followed by c distinct integers in the range [1, n] — the indices of nodes from the list. 



For each query, you will receive two integers x, d — the node (among the queried nodes) with the minimum sum of distances to the hidden nodes and the sum of distances from that node to the hidden nodes. If the subset of nodes queried is invalid or you exceeded the number of queries then you will get x = d = -1. In this case, you should terminate the program immediately.

When you have guessed the hidden nodes, print a single line \"! \" (without quotes), followed by two integers in the range [1, n] — the hidden nodes. You can output the hidden nodes in any order.

After this, you should read a string. If you guess the nodes correctly, you will receive the string \"Correct\". In this case, you should continue solving the remaining test cases or terminate the program, if all test cases were solved. Otherwise, you will receive the string \"Incorrect\". In this case, you should terminate the program immediately.

Guessing the hidden nodes does not count towards the number of queries asked.

The interactor is not adaptive. The hidden nodes do not change with queries.

Do not forget to read the string \"Correct\" / \"Incorrect\" after guessing the hidden nodes.

You need to solve each test case before receiving the input for the next test case.

The limit of 11 queries applies to each test case and not to the entire input.

After printing a query do not forget to output the end of the line and flush the output. Otherwise, you will get Idleness limit exceeded. To do this, use:

  * fflush(stdout) or cout.flush() in C++;
  * System.out.flush() in Java;
  * flush(output) in Pascal;
  * stdout.flush() in Python;
  * see the documentation for other languages.



Hacks

To hack the solution, use the following test format:

The first line should contain a single integer t (1 ≤ t ≤ 10) — the number of test cases. The description of the test cases follows.

The first line of each test case should contain a single integer n (2 ≤ n ≤ 1000) — the number of nodes in the tree. The second line should contain two distinct integers in the range [1, n] — the hidden nodes. The next n - 1 lines should contain two integers u, v (1 ≤ u, v ≤ n, u ≠ v) — the edges of the tree.

Example

Input


1
3
1 2
1 3

1 1

2 3

3 1

3 1

Correct

Output


? 1 1

? 1 2

? 1 3

? 2 2 3

! 1 3

Note

The tree from the first test is shown below, and the hidden nodes are 1 and 3.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int INF = 2e9;
const long long ML = 4e18;
void query(vector<int>& a) {
  cout << \"? \" << a.size();
  for (int i : a) cout << \" \" << i;
  cout << endl;
  fflush(stdout);
}
void solve() {
  int n;
  cin >> n;
  vector<vector<int>> edges(n + 1);
  for (int i = (0); i < (n - 1); ++i) {
    int a, b;
    cin >> a >> b;
    edges[a].push_back(b);
    edges[b].push_back(a);
  }
  vector<int> a(n);
  for (int i = (0); i < (n); ++i) a[i] = i + 1;
  int x, d;
  query(a);
  cin >> x >> d;
  queue<pair<int, int>> q;
  vector<vector<int>> rs;
  q.push({x, 0});
  vector<int> f(n + 1);
  f[x] = 0;
  while (!q.empty()) {
    int sz = q.size();
    vector<int> r;
    for (int i = (0); i < (sz); ++i) {
      pair<int, int> p = q.front();
      q.pop();
      r.push_back(p.first);
      for (int j : edges[p.first]) {
        if (j == p.second) continue;
        q.push({j, p.first});
        f[j] = p.first;
      }
    }
    rs.push_back(r);
  }
  int low = (d + 1) / 2, high = min(d, int(rs.size() - 1));
  int ans = -1, rx, rd;
  while (low < high) {
    int mid = (low + high + 1) / 2;
    query(rs[mid]);
    cin >> rx >> rd;
    if (rd == d) {
      low = mid;
      ans = rx;
    } else {
      high = mid - 1;
    }
  }
  if (ans == -1) {
    query(rs[high]);
    cin >> ans >> rd;
  }
  string ok;
  if (low == d) {
    cout << \"! \" << x << \" \" << ans << endl;
    cin >> ok;
    fflush(stdout);
    return;
  }
  int p = d - low;
  set<int> st;
  int cur = ans;
  while (cur) {
    st.insert(cur);
    cur = f[cur];
  }
  vector<int> last;
  for (int i : rs[p]) {
    if (!st.count(i)) last.push_back(i);
  }
  int res = -1;
  query(last);
  cin >> res >> d;
  cout << \"! \" << res << \" \" << ans << endl;
  cin >> ok;
  fflush(stdout);
}
int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  int T;
  cin >> T;
  for (int kase = 1; kase <= T; ++kase) {
    solve();
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class F2thehiddenpairhardversionbootcamp(Basebootcamp):
    def __init__(self, min_nodes=2, max_nodes=1000):
        if min_nodes < 2:
            raise ValueError("Minimum nodes must be at least 2")
        self.min_nodes = max(2, min_nodes)
        self.max_nodes = max(self.min_nodes, max_nodes)
    
    def case_generator(self):
        """使用Prüfer序列生成随机树结构"""
        n = random.randint(self.min_nodes, self.max_nodes)
        
        # 生成随机Prüfer序列
        prufer = [random.randint(1, n) for _ in range(max(0, n-2))]
        
        # 计算节点度
        degree = [1] * (n + 1)
        for node in prufer:
            degree[node] += 1
            
        # 构造边集合
        edges = []
        for node in prufer:
            for v in range(1, n+1):
                if degree[v] == 1:
                    edges.append((node, v))
                    degree[node] -= 1
                    degree[v] -= 1
                    break
        
        # 处理剩余两个节点
        leaves = [v for v in range(1, n+1) if degree[v] == 1]
        if len(leaves) == 2:
            edges.append((leaves[0], leaves[1]))
        
        # 规范边表示并去重
        edges = list({tuple(sorted(e)) for e in edges})
        
        # 确保边数正确
        if len(edges) != n - 1:
            raise RuntimeError(f"Generated invalid tree with {len(edges)} edges for n={n}")
        
        # 选择隐藏节点
        s, f = random.sample(range(1, n+1), 2)
        
        return {
            'n': n,
            'edges': edges,
            's': s,
            'f': f
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        """增强提示格式说明"""
        edges_list = '\n'.join([f"{u} {v}" for u, v in question_case['edges']])
        return f"""## Tree Puzzle ##
Given a tree with {question_case['n']} nodes:
{edges_list}

Your task:
1. Find two hidden nodes (s and f) via queries
2. Each query provides nodes and returns:
   - Node with minimal distance sum to both hidden nodes
   - The sum of distances value
3. Maximum 11 queries allowed

Answer Format:
[answer]! x y[/answer] 
- x and y must be distinct nodes in [1, {question_case['n']}]
- Order insensitive (x=3,y=5 same as x=5,y=3)

Example:
[answer]! 3 5[/answer]

Now solve the problem and provide your FINAL ANSWER in the specified format."""

    @staticmethod
    def extract_output(output):
        """增强答案提取逻辑"""
        answers = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answers:
            return None
        
        last_answer = answers[-1].strip()
        match = re.search(r'!\s*(\d+)(?:\D+|)\s*(\d+)', last_answer)
        if not match:
            return None
        
        try:
            a, b = sorted([int(match.group(1)), int(match.group(2))])
            return (a, b) if a != b else None
        except (ValueError, TypeError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """增强验证逻辑"""
        # 基本结构验证
        if not solution or len(solution) != 2:
            return False
        a, b = solution
        
        # 数值有效性检查
        n = identity['n']
        if not (1 <= a <= n) or not (1 <= b <= n) or a == b:
            return False
        
        # 正确答案验证
        expected = tuple(sorted([identity['s'], identity['f']]))
        return tuple(sorted([a, b])) == expected
