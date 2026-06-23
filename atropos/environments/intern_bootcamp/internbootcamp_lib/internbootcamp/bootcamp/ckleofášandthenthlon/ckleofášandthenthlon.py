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
int n, m, sum;
int a[101];
long double f[101000], s[101000], ans;
int main() {
  cin >> n >> m;
  if (m == 1) {
    printf(\"1.0000000000000000\n\");
    return 0;
  }
  f[0] = m - 1;
  int now = 0;
  for (int i = 1; i <= n; i++) scanf(\"%d\", &a[i]), sum += a[i];
  for (int i = 1; i <= n; i++) {
    for (int j = 0; j <= m * i; j++) s[j] = 0;
    for (int j = 0; j <= m * i; j++) {
      if (f[j] < 1e-12) continue;
      s[j + 1] += f[j];
      s[j + a[i]] -= f[j];
      s[j + a[i] + 1] += f[j];
      s[j + m + 1] -= f[j];
    }
    for (int j = 1; j <= m * i; j++) s[j] += s[j - 1];
    for (int j = 0; j <= m * i; j++) f[j] = s[j] / (m - 1);
  }
  for (int i = 0; i < sum; i++) ans += f[i];
  printf(\"%.16lf\n\", (double)ans + 1);
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Ckleofášandthenthlonbootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=10, m_min=1, m_max=20):
        self.n_min = n_min
        self.n_max = n_max
        self.m_min = m_min
        self.m_max = m_max

    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        m = random.randint(self.m_min, self.m_max)
        xi = [random.randint(1, m) for _ in range(n)]
        # Ensure xi is 1 if m=1
        if m == 1:
            xi = [1] * n
        return {'n': n, 'm': m, 'xi': xi}

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        xi = question_case['xi']
        xi_str = '\n'.join(map(str, xi))
        problem = (
            "Kleofáš participates in an n-thlon tournament with {n} competitions and {m} participants.\n"
            "Each competition assigns unique ranks from 1 to {m}. Kleofáš's ranks are:\n{xi_str}\n"
            "Calculate his expected overall rank. Format the answer as a float with 12+ decimal places within [answer]...[/answer]."
        ).format(n=n, m=m, xi_str=xi_str)
        return problem

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return float(last_match)
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False
        expected = cls.compute_expected_rank(identity)
        abs_err = abs(solution - expected)
        rel_err = abs_err / max(1e-9, abs(expected))
        return abs_err <= 1e-9 or rel_err <= 1e-9

    @classmethod
    def compute_expected_rank(cls, identity):
        n, m = identity['n'], identity['m']
        xi = identity['xi']
        sum_kl = sum(xi)
        
        if m == 1:
            return 1.0
        
        max_possible = n * m
        dp = [0.0] * (max_possible + 2)
        dp[0] = m - 1.0
        
        for score in xi:
            new_dp = [0.0] * (max_possible + 2)
            for current_sum in range(len(dp)):
                if dp[current_sum] < 1e-12:
                    continue
                # Left segment: 1 to score-1
                start = current_sum + 1
                end = current_sum + score
                if start <= max_possible:
                    new_dp[start] += dp[current_sum]
                    if end <= max_possible:
                        new_dp[end] -= dp[current_sum]
                # Right segment: score+1 to m
                start = current_sum + score + 1
                end = current_sum + m + 1
                if start <= max_possible:
                    new_dp[start] += dp[current_sum]
                    if end <= max_possible:
                        new_dp[end] -= dp[current_sum]
            
            # Compute prefix sums and normalize
            for j in range(1, max_possible + 1):
                new_dp[j] += new_dp[j - 1]
            for j in range(max_possible + 1):
                new_dp[j] /= (m - 1)
            dp = new_dp
        
        total = sum(dp[:sum_kl])
        return total * (m - 1) + 1.0
