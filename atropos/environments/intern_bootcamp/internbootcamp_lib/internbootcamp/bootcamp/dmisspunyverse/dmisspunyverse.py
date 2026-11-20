"""# 

### 谜题描述
The Oak has n nesting places, numbered with integers from 1 to n. Nesting place i is home to b_i bees and w_i wasps.

Some nesting places are connected by branches. We call two nesting places adjacent if there exists a branch between them. A simple path from nesting place x to y is given by a sequence s_0, …, s_p of distinct nesting places, where p is a non-negative integer, s_0 = x, s_p = y, and s_{i-1} and s_{i} are adjacent for each i = 1, …, p. The branches of The Oak are set up in such a way that for any two pairs of nesting places x and y, there exists a unique simple path from x to y. Because of this, biologists and computer scientists agree that The Oak is in fact, a tree.

A village is a nonempty set V of nesting places such that for any two x and y in V, there exists a simple path from x to y whose intermediate nesting places all lie in V. 

A set of villages \cal P is called a partition if each of the n nesting places is contained in exactly one of the villages in \cal P. In other words, no two villages in \cal P share any common nesting place, and altogether, they contain all n nesting places.

The Oak holds its annual Miss Punyverse beauty pageant. The two contestants this year are Ugly Wasp and Pretty Bee. The winner of the beauty pageant is determined by voting, which we will now explain. Suppose P is a partition of the nesting places into m villages V_1, …, V_m. There is a local election in each village. Each of the insects in this village vote for their favorite contestant. If there are strictly more votes for Ugly Wasp than Pretty Bee, then Ugly Wasp is said to win in that village. Otherwise, Pretty Bee wins. Whoever wins in the most number of villages wins.

As it always goes with these pageants, bees always vote for the bee (which is Pretty Bee this year) and wasps always vote for the wasp (which is Ugly Wasp this year). Unlike their general elections, no one abstains from voting for Miss Punyverse as everyone takes it very seriously.

Mayor Waspacito, and his assistant Alexwasp, wants Ugly Wasp to win. He has the power to choose how to partition The Oak into exactly m villages. If he chooses the partition optimally, determine the maximum number of villages in which Ugly Wasp wins.

Input

The first line of input contains a single integer t (1 ≤ t ≤ 100) denoting the number of test cases. The next lines contain descriptions of the test cases. 

The first line of each test case contains two space-separated integers n and m (1 ≤ m ≤ n ≤ 3000). The second line contains n space-separated integers b_1, b_2, …, b_n (0 ≤ b_i ≤ 10^9). The third line contains n space-separated integers w_1, w_2, …, w_n (0 ≤ w_i ≤ 10^9). The next n - 1 lines describe the pairs of adjacent nesting places. In particular, the i-th of them contains two space-separated integers x_i and y_i (1 ≤ x_i, y_i ≤ n, x_i ≠ y_i) denoting the numbers of two adjacent nesting places. It is guaranteed that these pairs form a tree.

It is guaranteed that the sum of n in a single file is at most 10^5.

Output

For each test case, output a single line containing a single integer denoting the maximum number of villages in which Ugly Wasp wins, among all partitions of The Oak into m villages.

Example

Input


2
4 3
10 160 70 50
70 111 111 0
1 2
2 3
3 4
2 1
143 420
214 349
2 1


Output


2
0

Note

In the first test case, we need to partition the n = 4 nesting places into m = 3 villages. We can make Ugly Wasp win in 2 villages via the following partition: \{\{1, 2\}, \{3\}, \{4\}\}. In this partition,

  * Ugly Wasp wins in village \{1, 2\}, garnering 181 votes as opposed to Pretty Bee's 170; 
  * Ugly Wasp also wins in village \{3\}, garnering 111 votes as opposed to Pretty Bee's 70; 
  * Ugly Wasp loses in the village \{4\}, garnering 0 votes as opposed to Pretty Bee's 50. 



Thus, Ugly Wasp wins in 2 villages, and it can be shown that this is the maximum possible number.

In the second test case, we need to partition the n = 2 nesting places into m = 1 village. There is only one way to do this: \{\{1, 2\}\}. In this partition's sole village, Ugly Wasp gets 563 votes, and Pretty Bee also gets 563 votes. Ugly Wasp needs strictly more votes in order to win. Therefore, Ugly Wasp doesn't win in any village.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
template <typename T1, typename T2>
inline void minaj(T1 &x, T2 y) {
  x = (x > y ? y : x);
}
template <typename T1, typename T2>
inline void maxaj(T1 &x, T2 y) {
  x = (x < y ? y : x);
}
const int MAXN = 3005;
int sz[MAXN];
vector<int> e[MAXN];
int a[MAXN];
pair<int, long long> dp[MAXN][MAXN];
pair<int, long long> ndp[MAXN];
int m;
void dfs(int u, int v) {
  sz[v] = 1;
  dp[v][1] = pair<int, long long>(0, a[v]);
  for (int w : e[v]) {
    if (u != w) {
      dfs(v, w);
      for (int i = 0; i < sz[v] + sz[w] + 1; ++i) ndp[i] = {-1, 0};
      for (int i = 1; i < min(sz[v], m) + 1; ++i)
        for (int j = 1; j < min(sz[w], m) + 1; ++j) {
          long long v1 = dp[v][i].second;
          long long v2 = dp[w][j].second;
          maxaj(ndp[i + j - 1],
                pair<int, long long>(dp[v][i].first + dp[w][j].first, v1 + v2));
          maxaj(ndp[i + j],
                pair<int, long long>(dp[v][i].first + dp[w][j].first + (v2 > 0),
                                     v1));
        }
      for (int i = 0; i < sz[v] + sz[w] + 1; ++i) dp[v][i] = ndp[i];
      sz[v] += sz[w];
    }
  }
}
void solve() {
  int n;
  cin >> n >> m;
  for (int i = 0; i < n; ++i) {
    cin >> a[i];
  }
  for (int i = 0; i < n; ++i) {
    int x;
    cin >> x;
    a[i] -= x;
    a[i] *= -1;
    e[i].clear();
  }
  for (int i = 0; i < n - 1; ++i) {
    int u, v;
    cin >> u >> v;
    u--;
    v--;
    e[u].push_back(v);
    e[v].push_back(u);
  }
  dfs(0, 0);
  cout << dp[0][m].first + (dp[0][m].second > 0) << endl;
}
int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(0);
  int t;
  cin >> t;
  while (t--) solve();
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from sys import setrecursionlimit
from bootcamp import Basebootcamp

class Dmisspunyversebootcamp(Basebootcamp):
    def __init__(self, **params):
        self.max_n = params.get('max_n', 10)
        self.min_n = params.get('min_n', 2)
        self.max_value = params.get('max_value', 100)
        setrecursionlimit(1000000)  # 统一设置递归深度

    def generate_tree_edges(self, n):
        """生成合法的树结构，节点编号从1开始"""
        if n == 1:
            return []
        edges = []
        parents = list(range(n))
        for i in range(1, n):
            parents[i] = random.randint(0, i-1)
        # 转换为1-based节点编号
        for i in range(1, n):
            u = parents[i] + 1
            v = i + 1
            edges.append((u, v))
        random.shuffle(edges)
        return edges

    def case_generator(self):
        """生成合法测试用例，确保m <= n"""
        while True:
            n = random.randint(self.min_n, self.max_n)
            m = random.randint(1, n)
            b = [random.randint(0, self.max_value) for _ in range(n)]
            w = [random.randint(0, self.max_value) for _ in range(n)]
            edges = self.generate_tree_edges(n)
            
            try:
                # 使用类方法调用避免实例属性访问
                correct_answer = self.solve_case(n, m, b, w, edges)
                return {
                    'n': n,
                    'm': m,
                    'b': b,
                    'w': w,
                    'edges': edges,
                    'correct_answer': correct_answer
                }
            except Exception as e:
                # 打印调试信息
                print(f"Error generating case: {e}")
                continue

    @staticmethod
    def solve_case(n, m, b, w, edges):
        """树形DP实现，修复状态初始化问题"""
        # 构建邻接表 (0-based)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u-1].append(v-1)
            adj[v-1].append(u-1)

        a = [w[i] - b[i] for i in range(n)]
        
        # DP状态数组 (max_m+2防止越界)
        max_m = m
        dp = [[(-1, -float('inf'))] * (max_m + 2) for _ in range(n)]
        sz = [0] * n

        def dfs(parent, u):
            sz[u] = 1
            dp[u][1] = (0, a[u])  # 初始状态
            
            for v in adj[u]:
                if v == parent:
                    continue
                dfs(u, v)
                
                # 合并子树状态
                current_max = min(sz[u], max_m)
                child_max = min(sz[v], max_m)
                ndp = [(-1, -float('inf'))] * (current_max + child_max + 1)
                
                for i in range(1, current_max + 1):
                    if dp[u][i][0] == -1:
                        continue
                    for j in range(1, child_max + 1):
                        if dp[v][j][0] == -1:
                            continue
                        
                        # 合并分支选项
                        merged_k = i + j - 1
                        if merged_k <= max_m:
                            total_win = dp[u][i][0] + dp[v][j][0]
                            total_sum = dp[u][i][1] + dp[v][j][1]
                            if (total_win > ndp[merged_k][0]) or \
                               (total_win == ndp[merged_k][0] and total_sum > ndp[merged_k][1]):
                                ndp[merged_k] = (total_win, total_sum)
                        
                        # 独立分支选项
                        separate_k = i + j
                        if separate_k <= max_m:
                            add_win = 1 if dp[v][j][1] > 0 else 0
                            total_win = dp[u][i][0] + dp[v][j][0] + add_win
                            total_sum = dp[u][i][1]
                            if (total_win > ndp[separate_k][0]) or \
                               (total_win == ndp[separate_k][0] and total_sum > ndp[separate_k][1]):
                                ndp[separate_k] = (total_win, total_sum)
                
                # 更新状态数组
                for k in range(len(ndp)):
                    if k > max_m:
                        continue
                    if ndp[k][0] > dp[u][k][0] or \
                       (ndp[k][0] == dp[u][k][0] and ndp[k][1] > dp[u][k][1]):
                        dp[u][k] = ndp[k]
                sz[u] += sz[v]

        dfs(-1, 0)
        max_win, sum_total = dp[0][m]
        
        # 处理根节点的剩余值
        if sum_total > 0:
            max_win += 1
        return max(max_win, 0)  # 保证非负

    @staticmethod
    def prompt_func(question_case) -> str:
        """增强提示模板，明确答案格式"""
        input_lines = [
            f"{question_case['n']} {question_case['m']}",
            ' '.join(map(str, question_case['b'])),
            ' '.join(map(str, question_case['w'])),
            '\n'.join(f"{u} {v}" for u, v in question_case['edges'])
        ]
        return f"""Solve the tree partition problem. You are given:
- A tree with {question_case['n']} nodes
- Bees and wasps counts for each node
- Need to partition into exactly {question_case['m']} villages

Calculate the maximum villages where wasps win (strict majority).

Input format:
{input_lines[0]}
{input_lines[1]}
{input_lines[2]}
{input_lines[3]}

Output a single integer within [answer]...[/answer]. Example: [answer]2[/answer]"""

    @staticmethod
    def extract_output(output):
        """增强答案提取，处理多标签情况"""
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except (ValueError, TypeError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """直接验证数值正确性"""
        return solution == identity['correct_answer']
