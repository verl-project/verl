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
const int MAXN = 3005;
const long long INF = 1e16;
const int MOD = 998244353;
long long b[MAXN], w[MAXN], size[MAXN];
pair<long long, long long> dp[MAXN][MAXN], tmp[MAXN];
int n, m, T;
vector<int> e[MAXN];
void dfs(int u, int fa) {
  dp[u][0] = make_pair(0, b[u]);
  size[u] = 1;
  for (int i = 0; i < e[u].size(); ++i) {
    int v = e[u][i];
    if (v == fa) continue;
    dfs(v, u);
    for (int j = 0; j <= min((long long)m, size[u] + size[v]); ++j)
      tmp[j] = make_pair(0, -INF);
    for (int j = 0; j <= min((long long)m, size[u]); ++j)
      for (int k = 0; k <= min((long long)(m - j), size[v]); ++k) {
        tmp[k + j] =
            max(tmp[k + j], make_pair(dp[v][k].first + dp[u][j].first,
                                      dp[v][k].second + dp[u][j].second));
        tmp[k + j + 1] = max(
            tmp[k + j + 1],
            make_pair(dp[v][k].first + dp[u][j].first + (dp[v][k].second > 0),
                      dp[u][j].second));
      }
    size[u] += size[v];
    for (int j = 0; j <= min((long long)m, size[u]); ++j) dp[u][j] = tmp[j];
  }
}
int main() {
  scanf(\"%d\", &T);
  while (T) {
    --T;
    scanf(\"%d%d\", &n, &m);
    for (int i = 1; i <= n; ++i)
      size[i] = 0, tmp[i] = make_pair(0, 0), e[i].clear();
    for (int i = 1; i <= n; ++i)
      for (int j = 1; j <= m; ++j) dp[i][j] = make_pair(0, 0);
    for (int i = 1; i <= n; ++i) cin >> w[i];
    for (int i = 1; i <= n; ++i) {
      cin >> b[i];
      b[i] -= w[i];
    }
    for (int i = 1; i < n; ++i) {
      int u, v;
      scanf(\"%d%d\", &u, &v);
      e[u].push_back(v), e[v].push_back(u);
    }
    dfs(1, 0);
    cout << dp[1][m - 1].first + (dp[1][m - 1].second > 0) << endl;
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import json
import re
import sys
from random import randint
from bootcamp import Basebootcamp

class Fmisspunyversebootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = {
            'max_t': params.get('max_t', 2),
            'max_n': params.get('max_n', 10),
            'max_m': params.get('max_m', 5),
            'value_range': (0, 200),
        }

    def generate_tree(self, n):
        """Generate a random tree using Prufer sequence."""
        if n == 1:
            return []
        prufer = [randint(1, n) for _ in range(n-2)]
        degree = [1] * (n + 1)
        for node in prufer:
            degree[node] += 1

        edges = []
        for node in prufer:
            for v in range(1, n + 1):
                if degree[v] == 1:
                    edges.append((node, v))
                    degree[node] -= 1
                    degree[v] -= 1
                    break

        leaves = [i for i in range(1, n+1) if degree[i] == 1]
        edges.append((leaves[0], leaves[1]))
        return edges

    def solve_test_case(self, n, m, deltas, edges):
        """Correctly compute the maximum number of villages where Ugly Wasp wins."""
        sys.setrecursionlimit(1 << 25)
        adj = [[] for _ in range(n+1)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Initialize DP table
        dp = [[(-1, -float('inf'))] * (m+1) for _ in range(n+1)]
        size = [0]*(n+1)

        def dfs(u, parent):
            dp[u][0] = (0, deltas[u-1])  # deltas: w_i - b_i
            size[u] = 1
            for v in adj[u]:
                if v == parent:
                    continue
                dfs(v, u)

                # Temporary DP for merging
                tmp = [(-1, -float('inf')) for _ in range(min(size[u] + size[v], m) + 1)]
                for j in range(size[u] + 1):
                    if dp[u][j][0] == -1:
                        continue
                    for k in range(size[v] + 1):
                        if j + k > m:
                            continue
                        if dp[v][k][0] == -1:
                            continue
                        # Merge without splitting
                        total_win = dp[u][j][0] + dp[v][k][0]
                        total_delta = dp[u][j][1] + dp[v][k][1]
                        if (total_win > tmp[j+k][0]) or (total_win == tmp[j+k][0] and total_delta > tmp[j+k][1]):
                            tmp[j+k] = (total_win, total_delta)
                        # Merge with splitting
                        if j + k + 1 > m:
                            continue
                        new_win = dp[u][j][0] + dp[v][k][0] + (1 if dp[v][k][1] > 0 else 0)
                        new_delta = dp[u][j][1]
                        if (new_win > tmp[j+k+1][0]) or (new_win == tmp[j+k+1][0] and new_delta > tmp[j+k+1][1]):
                            tmp[j+k+1] = (new_win, new_delta)
                
                # Update DP state
                new_size = min(size[u] + size[v], m)
                for j in range(new_size + 1):
                    dp[u][j] = tmp[j]
                size[u] = new_size

        dfs(1, -1)
        max_wins = dp[1][m-1][0] + (1 if dp[1][m-1][1] > 0 else 0)
        return max_wins

    def case_generator(self):
        """Generate valid test cases with correct solutions."""
        t = randint(1, self.params['max_t'])
        test_cases = []
        for _ in range(t):
            n = randint(1, self.params['max_n'])
            m = randint(1, min(n, self.params['max_m']))
            # Ensure non-negative delta to avoid negative total in villages
            b = [randint(*self.params['value_range']) for _ in range(n)]
            w = [randint(b[i], self.params['value_range'][1]) for i in range(n)]  # Ensure w >= b to potentially have positive delta
            edges = self.generate_tree(n)
            deltas = [w[i] - b[i] for i in range(n)]  # Correct delta calculation
            correct = self.solve_test_case(n, m, deltas, edges)
            test_cases.append({
                'n': n,
                'm': m,
                'b': b,
                'w': w,
                'edges': edges,
                'correct_output': correct
            })
        return {'test_cases': test_cases}

    @staticmethod
    def prompt_func(question_case) -> str:
        """Generate problem description with exact input format."""
        input_lines = [f"{len(question_case['test_cases'])}"]
        for tc in question_case['test_cases']:
            input_lines.append(f"{tc['n']} {tc['m']}")
            input_lines.append(' '.join(map(str, tc['b'])))
            input_lines.append(' '.join(map(str, tc['w'])))
            for u, v in tc['edges']:
                input_lines.append(f"{u} {v}")

        problem_text = (
            "Determine the maximum number of villages where Ugly Wasp wins.\n"
            "Rules:\n"
            "- Partition the tree into exactly m connected villages.\n"
            "- Ugly Wasp wins a village if wasps > bees.\n"
            "Input format:\n"
            f"{len(question_case['test_cases'])}\n"
            + '\n'.join(input_lines[1:]) +
            "\n\nOutput each answer on a new line within [answer] tags."
        )
        return problem_text

    @staticmethod
    def extract_output(output):
        """Robust answer extraction from model output."""
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        answers = []
        for line in answer_blocks[-1].strip().splitlines():
            line = line.strip()
            if line and re.fullmatch(r'-?\d+', line):
                answers.append(int(line))
        return answers if answers else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """Validate extracted answers against correct solutions."""
        expected = [tc['correct_output'] for tc in identity['test_cases']]
        return solution == expected
