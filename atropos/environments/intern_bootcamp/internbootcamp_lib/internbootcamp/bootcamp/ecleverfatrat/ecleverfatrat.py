"""# 

### 谜题描述
The Fat Rat and his friend Сerealguy have had a bet whether at least a few oats are going to descend to them by some clever construction. The figure below shows the clever construction.

<image>

A more formal description of the clever construction is as follows. The clever construction consists of n rows with scales. The first row has n scales, the second row has (n - 1) scales, the i-th row has (n - i + 1) scales, the last row has exactly one scale. Let's number the scales in each row from the left to the right, starting from 1. Then the value of wi, k in kilograms (1 ≤ i ≤ n; 1 ≤ k ≤ n - i + 1) is the weight capacity parameter of the k-th scale in the i-th row. 

If a body whose mass is not less than wi, k falls on the scale with weight capacity wi, k, then the scale breaks. At that anything that the scale has on it, either falls one level down to the left (if possible) or one level down to the right (if possible). In other words, if the scale wi, k (i < n) breaks, then there are at most two possible variants in which the contents of the scale's pan can fall out: all contents of scale wi, k falls either on scale wi + 1, k - 1 (if it exists), or on scale wi + 1, k (if it exists). If scale wn, 1 breaks, then all its contents falls right in the Fat Rat's claws. Please note that the scales that are the first and the last in a row, have only one variant of dropping the contents.

Initially, oats are simultaneously put on all scales of the first level. The i-th scale has ai kilograms of oats put on it. After that the scales start breaking and the oats start falling down in some way. You can consider everything to happen instantly. That is, the scale breaks instantly and the oats also fall instantly.

The Fat Rat is sure that whatever happens, he will not get the oats from the first level. Cerealguy is sure that there is such a scenario, when the rat gets at least some number of the oats. Help the Fat Rat and the Cerealguy. Determine, which one is right.

Input

The first line contains a single integer n (1 ≤ n ≤ 50) — the number of rows with scales.

The next line contains n space-separated integers ai (1 ≤ ai ≤ 106) — the masses of the oats in kilograms.

The next n lines contain descriptions of the scales: the i-th line contains (n - i + 1) space-separated integers wi, k (1 ≤ wi, k ≤ 106) — the weight capacity parameters for the scales that stand on the i-th row, in kilograms.

Output

Print \"Fat Rat\" if the Fat Rat is right, otherwise print \"Cerealguy\".

Examples

Input

1
1
2


Output

Fat Rat


Input

2
2 2
1 2
4


Output

Cerealguy


Input

2
2 2
1 2
5


Output

Fat Rat

Note

Notes to the examples: 

  * The first example: the scale with weight capacity 2 gets 1. That means that the lower scale don't break. 
  * The second sample: all scales in the top row obviously break. Then the oats fall on the lower row. Their total mass is 4,and that's exactly the weight that the lower scale can \"nearly endure\". So, as 4  ≥  4, the scale breaks.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
const double pi = acos(-1);
const int MOD = 1e9 + 7;
const int INF = 1e9 + 7;
const int MAXN = 5e1 + 5;
const double eps = 1e-9;
using namespace std;
int a[MAXN], w[MAXN][MAXN], dp[MAXN][MAXN][MAXN][MAXN];
int dfs(int i, int j, int l, int r) {
  if (j > r || i + j - 1 < l || l > r) return dp[i][j][l][r] = 0;
  if (i == 1) return dp[i][j][l][r] = (a[j] < w[i][j] ? 0 : a[j]);
  if (dp[i][j][l][r] != -1) return dp[i][j][l][r];
  int sum = 0;
  for (int k = l - 1; k <= r; k++)
    sum = max(sum, dfs(i - 1, j, l, k) + dfs(i - 1, j + 1, k + 1, r));
  return dp[i][j][l][r] = (sum < w[i][j] ? 0 : sum);
}
int main() {
  int n;
  scanf(\"%d\", &(n));
  for (int i = 1; i <= n; i++) scanf(\"%d\", &(a[i]));
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n - i + 1; j++) scanf(\"%d\", &(w[i][j]));
  memset(dp, -1, sizeof dp);
  if (n != 20 && (n != 6 || w[1][2] == 1 && w[1][3] != 2) &&
      dfs(n, 1, 1, n) > 0)
    printf(\"Cerealguy\n\");
  else
    printf(\"Fat Rat\n\");
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from typing import Dict, Any

class Ecleverfatratbootcamp(Basebootcamp):
    def __init__(self, max_n=5, **kwargs):
        super().__init__(**kwargs)
        self.max_n = max_n

    def case_generator(self) -> Dict[str, Any]:
        n = random.randint(1, self.max_n)
        strategy = random.choice([0, 1, 2])  # 0: random, 1: force Cerealguy, 2: force Fat Rat
        
        a = []
        w = []
        
        if strategy == 1:  # Generate cases where Cerealguy is possible
            a = [random.randint(5, 100) for _ in range(n)]
            w = []
            for i in range(n):
                row_size = n - i
                if i == 0:  # First row (i=1 in problem terms)
                    # Ensure a_i >= w_i for all scales in the first row
                    w_row = [random.randint(1, a_val) for a_val in a]
                else:
                    w_row = [random.randint(1, 100) for _ in range(row_size)]
                w.append(w_row)
            # Adjust last row to potentially allow sum >= w
            if n > 1:
                total_upper = sum(a)
                w[-1] = [random.randint(1, total_upper)]
        elif strategy == 2:  # Generate cases where Fat Rat is likely
            a = [random.randint(1, 50) for _ in range(n)]
            w = []
            for i in range(n):
                row_size = n - i
                if i == 0:  # First row
                    # Ensure a_i < w_i for all scales
                    w_row = [random.randint(a_val + 1, 100) for a_val in a]
                else:
                    w_row = [random.randint(1, 100) for _ in range(row_size)]
                w.append(w_row)
        else:  # Random generation
            a = [random.randint(1, 100) for _ in range(n)]
            w = []
            for i in range(n):
                row_size = n - i
                w_row = [random.randint(1, 100) for _ in range(row_size)]
                w.append(w_row)
        
        correct_answer = self.compute_correct_answer(n, a, w)
        return {
            'n': n,
            'a': a,
            'w': w,
            'correct_answer': correct_answer
        }

    @staticmethod
    def compute_correct_answer(n, a_list, w_list):
        memo = {}

        def dfs(i, j, l, r):
            key = (i, j, l, r)
            if key in memo:
                return memo[key]
            if j > r or (i + j - 1) < l or l > r:
                memo[key] = 0
                return 0
            if i == 1:
                a_j = a_list[j-1]
                w_ij = w_list[i-1][j-1]
                result = a_j if a_j >= w_ij else 0
                memo[key] = result
                return result
            max_sum = 0
            for k in range(l-1, r + 1):
                left = dfs(i-1, j, l, k)
                right = dfs(i-1, j+1, k+1, r)
                current_sum = left + right
                if current_sum > max_sum:
                    max_sum = current_sum
            w_ij = w_list[i-1][j-1]
            result = max_sum if max_sum >= w_ij else 0
            memo[key] = result
            return result

        total_sum = dfs(n, 1, 1, n)
        return "Cerealguy" if total_sum > 0 else "Fat Rat"

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        a = ' '.join(map(str, question_case['a']))
        w_rows = []
        for row in question_case['w']:
            w_rows.append(' '.join(map(str, row)))
        w_description = '\n'.join(w_rows)
        prompt = f"""As a programming expert, solve the problem and format your answer within [answer] tags.

Problem:
The Fat Rat and Cerealguy are betting on whether oats will reach the Fat Rat's claws. The structure has {n} rows of scales. Each row's scales have weight capacities. Follow the breaking rules to determine the outcome.

Input:
- Line 1: {n}
- Line 2: {a}
- Next {n} lines:
{w_description}

Rules:
- Scales break if oats ≥ capacity. Contents fall to possible scales below.
- Output "Cerealguy" if any oats reach the Fat Rat, else "Fat Rat".

Provide your answer as [answer]result[/answer], exactly one of the two options."""
        return prompt

    @staticmethod
    def extract_output(output: str):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.IGNORECASE)
        if not matches:
            return None
        last_answer = matches[-1].strip().lower()
        if last_answer == 'fat rat':
            return 'Fat Rat'
        elif last_answer == 'cerealguy':
            return 'Cerealguy'
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
