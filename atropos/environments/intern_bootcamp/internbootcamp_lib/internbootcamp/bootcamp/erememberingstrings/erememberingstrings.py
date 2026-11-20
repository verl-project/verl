"""# 

### 谜题描述
You have multiset of n strings of the same length, consisting of lowercase English letters. We will say that those strings are easy to remember if for each string there is some position i and some letter c of the English alphabet, such that this string is the only string in the multiset that has letter c in position i.

For example, a multiset of strings {\"abc\", \"aba\", \"adc\", \"ada\"} are not easy to remember. And multiset {\"abc\", \"ada\", \"ssa\"} is easy to remember because: 

  * the first string is the only string that has character c in position 3; 
  * the second string is the only string that has character d in position 2; 
  * the third string is the only string that has character s in position 2. 



You want to change your multiset a little so that it is easy to remember. For aij coins, you can change character in the j-th position of the i-th string into any other lowercase letter of the English alphabet. Find what is the minimum sum you should pay in order to make the multiset of strings easy to remember.

Input

The first line contains two integers n, m (1 ≤ n, m ≤ 20) — the number of strings in the multiset and the length of the strings respectively. Next n lines contain the strings of the multiset, consisting only of lowercase English letters, each string's length is m.

Next n lines contain m integers each, the i-th of them contains integers ai1, ai2, ..., aim (0 ≤ aij ≤ 106).

Output

Print a single number — the answer to the problem.

Examples

Input

4 5
abcde
abcde
abcde
abcde
1 1 1 1 1
1 1 1 1 1
1 1 1 1 1
1 1 1 1 1


Output

3


Input

4 3
abc
aba
adc
ada
10 10 10
10 1 10
10 10 10
10 1 10


Output

2


Input

3 3
abc
ada
ssa
1 1 1
1 1 1
1 1 1


Output

0

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int INF = 0x3f3f3f3f;
const int maxn = 21;
const int maxs = 1 << 21;
int n, m;
int a[maxn][maxn];
char str[maxn][maxn];
int dp[maxs];
int lowzero(int s) {
  for (int i = 0; i < maxn; ++i) {
    if (!(s & (1 << i))) return i;
  }
  return maxn - 1;
}
int main() {
  while (~scanf(\"%d%d\", &n, &m)) {
    for (int i = 0; i < n; ++i) {
      scanf(\"%s\", str[i]);
    }
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        scanf(\"%d\", &a[i][j]);
      }
    }
    memset(dp, 0xff, sizeof(dp));
    dp[0] = 0;
    int M = 1 << n;
    for (int s = 0; s < M; ++s) {
      if (dp[s] == -1) continue;
      int bit = lowzero(s);
      for (int j = 0; j < m; ++j) {
        if (dp[s | (1 << bit)] == -1 ||
            dp[s | (1 << bit)] > dp[s] + a[bit][j]) {
          dp[s | (1 << bit)] = dp[s] + a[bit][j];
        }
        int sum = 0, bits = 0, mw = 0;
        for (int i = 0; i < n; ++i) {
          if (str[i][j] == str[bit][j]) {
            sum += a[i][j];
            mw = max(mw, a[i][j]);
            bits |= 1 << i;
          }
        }
        if (dp[s | bits] == -1 || dp[s | bits] > dp[s] + sum - mw) {
          dp[s | bits] = dp[s] + sum - mw;
        }
      }
    }
    printf(\"%d\n\", dp[M - 1]);
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Erememberingstringsbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=20, min_m=1, max_m=20, cost_min=0, cost_max=10**6):
        self.min_n = min_n
        self.max_n = max_n
        self.min_m = min_m
        self.max_m = max_m  # 修正拼写错误
        self.cost_min = cost_min
        self.cost_max = cost_max

    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        m = random.randint(self.min_m, self.max_m)
        strings, cost_matrix = self._generate_valid_case(n, m)
        return {
            'n': n,
            'm': m,
            'strings': strings,
            'cost_matrix': cost_matrix
        }

    def _generate_valid_case(self, n, m):
        # 生成目标字符串：每个字符串至少有一个唯一特征位
        target_strings = []
        pos_pool = list(range(m)) * ((n // m) + 1)
        random.shuffle(pos_pool)
        
        for i in range(n):
            s = ['x'] * m
            unique_pos = pos_pool[i]
            # 确保该位置字符唯一
            used_chars = set()
            for ts in target_strings:
                used_chars.add(ts[unique_pos])
            while True:
                c = random.choice('abcdefghijklmnopqrstuvwxyz')
                if c not in used_chars:
                    s[unique_pos] = c
                    break
            # 其他位置随机生成
            for j in range(m):
                if j != unique_pos:
                    s[j] = random.choice('abcdefghijklmnopqrstuvwxyz')
            target_strings.append(''.join(s))
        
        # 构造原始字符串（通过修改目标字符串得到）
        original_strings = []
        cost_matrix = []
        for idx, target in enumerate(target_strings):
            original = list(target)
            modify_pos = random.sample(range(m), k=random.randint(0, m//2))
            costs = []
            for j in range(m):
                if j in modify_pos:
                    # 生成修改成本并改变字符
                    original[j] = random.choice('abcdefghijklmnopqrstuvwxyz'.replace(target[j], ''))
                    costs.append(random.randint(1, 1000))
                else:
                    costs.append(0)
            original_strings.append(''.join(original))
            cost_matrix.append(costs)
        
        return original_strings, cost_matrix

    @staticmethod
    def prompt_func(question_case):
        input_lines = [f"{question_case['n']} {question_case['m']}"] + question_case['strings']
        cost_lines = [' '.join(map(str, row)) for row in question_case['cost_matrix']]
        input_str = '\n'.join(input_lines + cost_lines)
        problem = (
            "You need to make strings easy to remember by minimal cost.\n"
            f"Input:\n{input_str}\n"
            "Output the minimal cost within [answer]...[/answer]."
        )
        return problem

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return int(matches[-1].strip()) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            expected = cls.calculate_min_cost(
                identity['n'], identity['m'],
                identity['strings'], identity['cost_matrix']
            )
            return solution == expected
        except:
            return False

    @staticmethod
    def calculate_min_cost(n, m, strings, cost_matrix):
        INF = float('inf')
        dp = [INF] * (1 << n)
        dp[0] = 0
        
        for state in range(1 << n):
            if dp[state] == INF:
                continue
                
            # Find first unset bit
            bit = None
            for i in range(n):
                if not (state & (1 << i)):
                    bit = i
                    break
            if bit is None:
                continue
                
            # Try all possible positions
            for j in range(m):
                # Option 1: change current string's j-th character
                new_state = state | (1 << bit)
                cost = dp[state] + cost_matrix[bit][j]
                if dp[new_state] > cost:
                    dp[new_state] = cost
                
                # Option 2: group change
                same_chars = [bit]
                for k in range(n):
                    if k != bit and strings[k][j] == strings[bit][j]:
                        same_chars.append(k)
                
                sum_cost = sum(cost_matrix[x][j] for x in same_chars)
                max_cost = max(cost_matrix[x][j] for x in same_chars)
                total_cost = sum_cost - max_cost
                new_state_group = state
                for x in same_chars:
                    new_state_group |= (1 << x)
                
                if dp[new_state_group] > dp[state] + total_cost:
                    dp[new_state_group] = dp[state] + total_cost
        
        return dp[(1 << n) - 1]
