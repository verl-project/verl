"""# 

### 谜题描述
Kleofáš is participating in an n-thlon - a tournament consisting of n different competitions in n different disciplines (numbered 1 through n). There are m participants in the n-thlon and each of them participates in all competitions.

In each of these n competitions, the participants are given ranks from 1 to m in such a way that no two participants are given the same rank - in other words, the ranks in each competition form a permutation of numbers from 1 to m. The score of a participant in a competition is equal to his/her rank in it.

The overall score of each participant is computed as the sum of that participant's scores in all competitions.

The overall rank of each participant is equal to 1 + k, where k is the number of participants with strictly smaller overall score.

The n-thlon is over now, but the results haven't been published yet. Kleofáš still remembers his ranks in each particular competition; however, he doesn't remember anything about how well the other participants did. Therefore, Kleofáš would like to know his expected overall rank.

All competitors are equally good at each discipline, so all rankings (permutations of ranks of everyone except Kleofáš) in each competition are equiprobable.

Input

The first line of the input contains two space-separated integers n (1 ≤ n ≤ 100) and m (1 ≤ m ≤ 1000) — the number of competitions and the number of participants respectively.

Then, n lines follow. The i-th of them contains one integer xi (1 ≤ xi ≤ m) — the rank of Kleofáš in the i-th competition.

Output

Output a single real number – the expected overall rank of Kleofáš. Your answer will be considered correct if its relative or absolute error doesn't exceed 10 - 9.

Namely: let's assume that your answer is a, and the answer of the jury is b. The checker program will consider your answer correct, if <image>.

Examples

Input

4 10
2
1
2
1


Output

1.0000000000000000


Input

5 5
1
2
3
4
5


Output

2.7500000000000000


Input

3 6
2
4
2


Output

1.6799999999999999

Note

In the first sample, Kleofáš has overall score 6. Nobody else can have overall score less than 6 (but it's possible for one other person to have overall score 6 as well), so his overall rank must be 1.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int inf = 1 << 29;
const int mod = (int)1e9 + 7;
const double eps = 1e-8;
const double pi = acos(-1.0);
int n, m, k;
double dp[110][100010];
double sum[1010];
int a[110];
int main() {
  cin >> n >> m;
  int tot = 0;
  for (int i = 1; i <= n; i++) {
    scanf(\"%d\", &a[i]);
    tot += a[i];
  }
  if (m == 1) {
    printf(\"%.15f\n\", 1.0);
    return 0;
  }
  for (int i = 1; i <= n * m; i++) {
    if (i != a[1] && i <= m) dp[1][i] = 1.0 / (m - 1);
    sum[i] += sum[i - 1] + dp[1][i];
  }
  for (int i = 2; i <= n; i++) {
    for (int j = 1; j <= n * m; j++) {
      dp[i][j] = sum[j - 1] - sum[max(0, j - m - 1)];
      if (j >= a[i]) dp[i][j] -= dp[i - 1][j - a[i]];
      dp[i][j] /= m - 1;
    }
    for (int j = 1; j <= n * m; j++) sum[j] = sum[j - 1] + dp[i][j];
  }
  double ans = sum[tot - 1] * (m - 1) + 1.0;
  printf(\"%.15f\n\", ans);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random

def calculate_expected_rank(n, m, a):
    tot = sum(a)
    if m == 1:
        return 1.0
    max_score = n * m
    dp_prev = [0.0] * (max_score + 2)
    for s in range(1, m + 1):
        if s != a[0]:
            dp_prev[s] = 1.0 / (m - 1)
    sum_prev = [0.0] * (max_score + 2)
    for j in range(1, max_score + 1):
        sum_prev[j] = sum_prev[j - 1] + dp_prev[j]
    for i in range(1, n):
        current_a = a[i]
        dp_curr = [0.0] * (max_score + 2)
        for j in range(1, max_score + 1):
            lower = max(j - m - 1, 0)
            part = sum_prev[j-1] - sum_prev[lower]
            if j >= current_a:
                part -= dp_prev[j - current_a]
            dp_curr[j] = part / (m - 1)
        sum_prev[0] = 0.0
        for j in range(1, max_score + 1):
            sum_prev[j] = sum_prev[j - 1] + dp_curr[j]
        dp_prev = dp_curr.copy()
    total_less = sum_prev[tot -1] if tot >=1 else 0.0
    ans = total_less * (m - 1) + 1.0
    return ans

class Ekleofášandthenthlonbootcamp(Basebootcamp):
    def __init__(self, max_case_n=10, max_case_m=10):
        self.max_case_n = max_case_n
        self.max_case_m = max_case_m
    
    def case_generator(self):
        n = random.randint(1, self.max_case_n)
        m = random.randint(1, self.max_case_m)
        ranks = [random.randint(1, m) for _ in range(n)]
        return {
            'n': n,
            'm': m,
            'ranks': ranks
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        ranks = question_case['ranks']
        input_lines = [f"{n} {m}"] + [str(rank) for rank in ranks]
        input_str = '\n'.join(input_lines)
        prompt = (
            "Kleofáš参加了由n个不同竞赛组成的n-thlon，每个竞赛有m个参与者。每个竞赛中，参与者的排名是1到m的一个排列（无重复）。总得分是每个参与者在所有竞赛中的排名之和。Kleofáš的总体排名等于1加上总得分严格小于他的其他参与者的人数。所有其他参与者在每个竞赛中的排名是等概率的排列（排除Kleofáš的排名）。请计算Kleofáš的期望排名。\n\n"
            "输入数据如下：\n"
            f"{input_str}\n\n"
            "输出一个浮点数，精确到小数点后至少15位。确保你的答案的绝对或相对误差不超过1e-9。将答案放在[answer]和[/answer]之间。\n"
            "例如：\n"
            "[answer]1.0000000000000000[/answer]"
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return float(last_match)
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            solution_float = float(solution)
        except:
            return False
        n = identity['n']
        m = identity['m']
        ranks = identity['ranks']
        correct_ans = calculate_expected_rank(n, m, ranks)
        a = solution_float
        b = correct_ans
        abs_error = abs(a - b)
        if abs_error <= 1e-9:
            return True
        max_val = max(abs(a), abs(b))
        if max_val < 1e-20:
            return True
        rel_error = abs_error / max_val
        return rel_error <= 1e-9
