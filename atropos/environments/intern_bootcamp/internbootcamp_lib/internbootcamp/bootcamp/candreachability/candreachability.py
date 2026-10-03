"""# 

### 谜题描述
Toad Pimple has an array of integers a_1, a_2, …, a_n.

We say that y is reachable from x if x<y and there exists an integer array p such that x = p_1 < p_2 < … < p_k=y, and a_{p_i}  \&  a_{p_{i+1}} > 0 for all integers i such that 1 ≤ i < k.

Here \& denotes the [bitwise AND operation](https://en.wikipedia.org/wiki/Bitwise_operation#AND).

You are given q pairs of indices, check reachability for each of them.

Input

The first line contains two integers n and q (2 ≤ n ≤ 300 000, 1 ≤ q ≤ 300 000) — the number of integers in the array and the number of queries you need to answer.

The second line contains n space-separated integers a_1, a_2, …, a_n (0 ≤ a_i ≤ 300 000) — the given array.

The next q lines contain two integers each. The i-th of them contains two space-separated integers x_i and y_i (1 ≤ x_i < y_i ≤ n). You need to check if y_i is reachable from x_i. 

Output

Output q lines. In the i-th of them print \"Shi\" if y_i is reachable from x_i, otherwise, print \"Fou\".

Example

Input


5 3
1 3 0 2 1
1 3
2 4
1 4


Output


Fou
Shi
Shi

Note

In the first example, a_3 = 0. You can't reach it, because AND with it is always zero. a_2  \&  a_4 > 0, so 4 is reachable from 2, and to go from 1 to 4 you can use p = [1, 2, 4].

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
const long long mod1 = (long long)1e9 + 7;
const long long mod2 = (long long)1e9 + 9;
const long long BASE = 15879;
const long long inf = (long long)1e18;
const long double e = 2.718281828459;
const long double pi = 3.141592653;
const long double EPS = 1e-9;
using namespace std;
template <class T>
istream &operator>>(istream &in, vector<T> &arr) {
  for (T &cnt : arr) {
    in >> cnt;
  }
  return in;
};
void solve() {
  long long n, q;
  cin >> n >> q;
  vector<long long> arr(n);
  cin >> arr;
  long long B = 20;
  vector<vector<long long>> dp(n, vector<long long>(B, inf));
  vector<long long> nxt(B, -1);
  for (int i = n - 1; i >= 0; i--) {
    for (int j = 0; j < B; j++) {
      if (nxt[j] != -1 and (arr[i] & (1 << j)) != 0) {
        for (int k = 0; k < B; k++) {
          dp[i][k] = min(dp[nxt[j]][k], dp[i][k]);
          if (arr[nxt[j]] & (1 << k)) {
            dp[i][k] = min(dp[i][k], nxt[j]);
          }
        }
      }
      if (arr[i] & (1 << j)) {
        nxt[j] = i;
      }
    }
  }
  while (q--) {
    long long x, y;
    cin >> x >> y;
    x--;
    y--;
    bool ans = false;
    for (int i = 0; i < B; i++) {
      if (arr[y] & (1 << i)) {
        ans |= dp[x][i] <= y;
      }
    }
    if (ans) {
      cout << \"Shi\n\";
    } else {
      cout << \"Fou\n\";
    }
  }
}
int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  cout.precision(30);
  long long seed = time(0);
  srand(seed);
  solve();
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Candreachabilitybootcamp(Basebootcamp):
    def __init__(self, n=5, q=3):
        self.n = n
        self.q = q
    
    def case_generator(self):
        array = [random.randint(0, 300000) for _ in range(self.n)]
        queries = []
        for _ in range(self.q):
            x = random.randint(1, self.n-1)
            y = random.randint(x+1, self.n)
            queries.append((x, y))
        return {
            'n': self.n,
            'array': array,
            'queries': queries
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        array_str = ' '.join(map(str, question_case['array']))
        queries = question_case['queries']
        q = len(queries)
        queries_str = '\n'.join([f"{x} {y}" for x, y in queries])
        return f"""Toad Pimple有一个整数数组，判断每个查询中的y是否可以从x到达。规则如下：
- y可从x到达的条件是存在严格递增序列，起始x，结束y，相邻元素的按位与大于0。
数组（长度{question_case['n']}）：
{array_str}
查询（共{q}个）：
{queries_str}

请输出每个查询的结果（"Shi"或"Fou"），每个结果占一行，并放置在[answer]和[/answer]标签内。"""

    @staticmethod
    def extract_output(output):
        match = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not match:
            return None
        last_answer = match[-1].strip()
        answers = []
        for line in last_answer.split('\n'):
            line = line.strip().lower()
            if line == 'shi' or line == 'fou':
                answers.append('Shi' if line == 'shi' else 'Fou')
            else:
                return None
        return answers if len(answers) == len(match[-1].strip().split('\n')) else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or len(solution) != len(identity['queries']):
            return False
        n = identity['n']
        array = identity['array']
        queries = identity['queries']
        expected = cls._compute_answers(n, array, queries)
        return solution == expected
    
    @staticmethod
    def _compute_answers(n, array, queries):
        B = 20
        INF = float('inf')
        dp = [[INF] * B for _ in range(n)]
        nxt = [-1] * B
        
        for i in range(n-1, -1, -1):
            a_i = array[i]
            old_nxt = nxt.copy()
            # 更新nxt数组
            for j in range(B):
                if a_i & (1 << j):
                    nxt[j] = i
            # 处理当前元素的位关联
            for j in range(B):
                if old_nxt[j] != -1 and (a_i & (1 << j)):
                    for k in range(B):
                        # 合并来自后续节点的可达性
                        dp[i][k] = min(dp[i][k], dp[old_nxt[j]][k])
                        # 处理直接通过该bit可达的情况
                        if (array[old_nxt[j]] & (1 << k)):
                            dp[i][k] = min(dp[i][k], old_nxt[j])
            # 处理当前元素自身的bit
            for k in range(B):
                if (a_i & (1 << k)):
                    dp[i][k] = min(dp[i][k], i)
        
        answers = []
        for x, y in queries:
            x_idx, y_idx = x-1, y-1
            found = False
            for j in range(B):
                if (array[y_idx] & (1 << j)) and (dp[x_idx][j] <= y_idx):
                    found = True
                    break
            answers.append('Shi' if found else 'Fou')
        return answers
