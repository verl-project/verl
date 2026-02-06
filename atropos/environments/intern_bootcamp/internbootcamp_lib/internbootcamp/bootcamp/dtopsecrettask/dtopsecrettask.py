"""# 

### 谜题描述
A top-secret military base under the command of Colonel Zuev is expecting an inspection from the Ministry of Defence. According to the charter, each top-secret military base must include a top-secret troop that should... well, we cannot tell you exactly what it should do, it is a top secret troop at the end. The problem is that Zuev's base is missing this top-secret troop for some reasons.

The colonel decided to deal with the problem immediately and ordered to line up in a single line all n soldiers of the base entrusted to him. Zuev knows that the loquacity of the i-th soldier from the left is equal to qi. Zuev wants to form the top-secret troop using k leftmost soldiers in the line, thus he wants their total loquacity to be as small as possible (as the troop should remain top-secret). To achieve this, he is going to choose a pair of consecutive soldiers and swap them. He intends to do so no more than s times. Note that any soldier can be a participant of such swaps for any number of times. The problem turned out to be unusual, and colonel Zuev asked you to help.

Determine, what is the minimum total loquacity of the first k soldiers in the line, that can be achieved by performing no more than s swaps of two consecutive soldiers.

Input

The first line of the input contains three positive integers n, k, s (1 ≤ k ≤ n ≤ 150, 1 ≤ s ≤ 109) — the number of soldiers in the line, the size of the top-secret troop to be formed and the maximum possible number of swap operations of the consecutive pair of soldiers, respectively.

The second line of the input contains n integer qi (1 ≤ qi ≤ 1 000 000) — the values of loquacity of soldiers in order they follow in line from left to right.

Output

Print a single integer — the minimum possible total loquacity of the top-secret troop.

Examples

Input

3 2 2
2 4 1


Output

3


Input

5 4 2
10 1 6 2 5


Output

18


Input

5 2 3
3 1 4 2 5


Output

3

Note

In the first sample Colonel has to swap second and third soldiers, he doesn't really need the remaining swap. The resulting soldiers order is: (2, 1, 4). Minimum possible summary loquacity of the secret troop is 3. In the second sample Colonel will perform swaps in the following order:

  1. (10, 1, 6 — 2, 5) 
  2. (10, 1, 2, 6 — 5) 



The resulting soldiers order is (10, 1, 2, 5, 6). 

Minimum possible summary loquacity is equal to 18.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int n, k, s, ar[1 << 20];
int dp[3][200][200 * 200];
int main() {
  ios_base::sync_with_stdio(0);
  cin >> n >> k >> s;
  for (int i = 1; i <= n; i++) cin >> ar[i];
  if (s > n * n / 2 + 10) s = n * n / 2 + 10;
  for (int done = 0; done <= s; done++)
    for (int pref = 0; pref <= k; pref++) dp[0][pref][done] = 1e9;
  dp[0][0][0] = 0;
  for (int i = 1; i <= n; i++) {
    for (int j = 0; j <= s; j++)
      for (int q = 0; q <= k; q++) dp[i % 2][q][j] = 1e9;
    for (int pref = 0; pref < i && pref <= k; pref++)
      for (int done = 0; done <= s; done++) {
        dp[i % 2][pref][done] =
            min(dp[i % 2][pref][done], dp[1 - i % 2][pref][done]);
        int ndone = done + i - pref - 1;
        if (ndone <= s)
          dp[i % 2][pref + 1][ndone] = min(dp[i % 2][pref + 1][ndone],
                                           dp[1 - i % 2][pref][done] + ar[i]);
      }
  }
  int ans = 1e9;
  for (int i = 0; i <= s; i++) ans = min(ans, dp[n % 2][k][i]);
  cout << ans << endl;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

def calculate_min_loquacity(n, k, s, q):
    adjusted_s = min(s, (n*n)//2 + 10)  # 严格模拟参考代码的调整逻辑
    INF = float('inf')
    
    # 初始化DP数组，使用滚动数组优化
    dp = [[[INF] * (adjusted_s + 1) for _ in range(k+1)] for __ in range(2)]
    dp[0][0][0] = 0  # 初始状态

    for i in range(1, n+1):
        current = i % 2
        prev = 1 - current
        
        # 重置当前层
        for j in range(k+1):
            for t in range(adjusted_s + 1):
                dp[current][j][t] = INF
        
        # 状态转移
        for pref in range(0, min(i-1, k)+1):
            for done in range(adjusted_s + 1):
                if dp[prev][pref][done] == INF:
                    continue
                
                # 情况1：不选当前士兵
                if dp[current][pref][done] > dp[prev][pref][done]:
                    dp[current][pref][done] = dp[prev][pref][done]
                
                # 情况2：选当前士兵
                new_pref = pref + 1
                if new_pref > k:
                    continue
                
                swaps_needed = i - new_pref  # 与参考代码完全一致的计算方式
                new_done = done + swaps_needed
                
                if new_done <= adjusted_s:
                    new_value = dp[prev][pref][done] + q[i-1]
                    if new_value < dp[current][new_pref][new_done]:
                        dp[current][new_pref][new_done] = new_value
        
    # 寻找最终答案
    final_layer = n % 2
    return min(dp[final_layer][k][:adjusted_s+1])

class Dtopsecrettaskbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = params
        # 参数校验确保合法性
        self.params.setdefault('min_n', 3)
        self.params.setdefault('max_n', 150)
        self.params.setdefault('min_k', 1)
        self.params.setdefault('max_k', 150)
        self.params.setdefault('min_s', 1)
        self.params.setdefault('max_s', 10**9)
        self.params.setdefault('q_min', 1)
        self.params.setdefault('q_max', 10**6)
        
        # 确保参数间约束关系
        self.params['min_n'] = max(self.params['min_n'], 1)
        self.params['max_n'] = min(self.params['max_n'], 150)
        self.params['min_k'] = max(self.params['min_k'], 1)
        self.params['max_k'] = min(self.params['max_k'], self.params['max_n'])

    def case_generator(self):
        # 确保k <= n的约束
        n = random.randint(self.params['min_n'], self.params['max_n'])
        k = random.randint(
            self.params['min_k'], 
            min(n, self.params['max_k'])
        )
        s = random.randint(self.params['min_s'], self.params['max_s'])
        
        # 生成loquacity值时保证至少k个非极大值
        q = [
            random.randint(self.params['q_min'], self.params['q_max'])
            for _ in range(n)
        ]
        # 插入k个较小值以确保有解
        for _ in range(k):
            q[random.randint(0, n-1)] = random.randint(
                self.params['q_min'], 
                self.params['q_max']//100
            )
        
        return {
            'n': n,
            'k': k,
            's': s,
            'q': q,
            'expected': calculate_min_loquacity(n, k, s, q)
        }

    @staticmethod
    def prompt_func(question_case):
        return f"""作为军事参谋，你需要解决秘密部队优化问题。请根据以下输入数据计算最小loquacity总和：

输入格式：
第一行：n k s
第二行：q1 q2 ... qn

当前输入：
{question_case['n']} {question_case['k']} {question_case['s']}
{' '.join(map(str, question_case['q']))}

要求：
1. 最多进行s次相邻交换
2. 最终前k个士兵的loquacity总和最小
3. 答案需用[answer]标签包裹，如：[answer]123[/answer]"""

    @staticmethod
    def extract_output(output):
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
        return solution == identity['expected']
