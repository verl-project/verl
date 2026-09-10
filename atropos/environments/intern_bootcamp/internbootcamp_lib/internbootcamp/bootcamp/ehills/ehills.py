"""# 

### 谜题描述
Welcome to Innopolis city. Throughout the whole year, Innopolis citizens suffer from everlasting city construction. 

From the window in your room, you see the sequence of n hills, where i-th of them has height ai. The Innopolis administration wants to build some houses on the hills. However, for the sake of city appearance, a house can be only built on the hill, which is strictly higher than neighbouring hills (if they are present). For example, if the sequence of heights is 5, 4, 6, 2, then houses could be built on hills with heights 5 and 6 only.

The Innopolis administration has an excavator, that can decrease the height of an arbitrary hill by one in one hour. The excavator can only work on one hill at a time. It is allowed to decrease hills up to zero height, or even to negative values. Increasing height of any hill is impossible. The city administration wants to build k houses, so there must be at least k hills that satisfy the condition above. What is the minimum time required to adjust the hills to achieve the administration's plan?

However, the exact value of k is not yet determined, so could you please calculate answers for all k in range <image>? Here <image> denotes n divided by two, rounded up.

Input

The first line of input contains the only integer n (1 ≤ n ≤ 5000)—the number of the hills in the sequence.

Second line contains n integers ai (1 ≤ ai ≤ 100 000)—the heights of the hills in the sequence.

Output

Print exactly <image> numbers separated by spaces. The i-th printed number should be equal to the minimum number of hours required to level hills so it becomes possible to build i houses.

Examples

Input

5
1 1 1 1 1


Output

1 2 2 


Input

3
1 2 3


Output

0 2 


Input

5
1 2 3 2 2


Output

0 1 3 

Note

In the first example, to get at least one hill suitable for construction, one can decrease the second hill by one in one hour, then the sequence of heights becomes 1, 0, 1, 1, 1 and the first hill becomes suitable for construction.

In the first example, to get at least two or at least three suitable hills, one can decrease the second and the fourth hills, then the sequence of heights becomes 1, 0, 1, 0, 1, and hills 1, 3, 5 become suitable for construction.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int maxn = 5005;
const int INF = 0x3f3f3f3f;
int dp[maxn][3];
int high[maxn];
int cal1(int i) {
  int ans = 0;
  if (high[i] <= high[i - 1]) ans += (high[i - 1] - high[i] + 1);
  if (high[i] <= high[i + 1]) ans += (high[i + 1] - high[i] + 1);
  return ans;
}
int cal2(int i) {
  int ans = 0;
  int tmp = high[i - 1];
  if (i >= 2 && high[i - 2] <= high[i - 1]) tmp = high[i - 2] - 1;
  if (high[i] <= tmp) ans += (tmp - high[i] + 1);
  if (high[i] <= high[i + 1]) ans += (high[i + 1] - high[i] + 1);
  return ans;
}
int main() {
  int n;
  scanf(\"%d\", &n);
  for (int i = 1; i <= n; i++) scanf(\"%d\", &high[i]);
  high[0] = -INF;
  high[n + 1] = -INF;
  int m = ceil(n / 2.0);
  int a, b, c;
  memset(dp, INF, sizeof(dp));
  dp[0][0] = 0;
  for (int i = 1; i <= n; i++) {
    for (int j = m; j >= 1; j--) {
      a = dp[j][0];
      b = dp[j][1];
      c = dp[j][2];
      dp[j][0] = min(a, c);
      dp[j][1] = min(dp[j - 1][0] + cal1(i), dp[j - 1][2] + cal2(i));
      dp[j][2] = b;
    }
  }
  for (int i = 1; i <= m; i++)
    cout << min(min(dp[i][0], dp[i][1]), dp[i][2]) << \" \";
  cout << endl;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import random
from typing import List
from bootcamp import Basebootcamp

def compute_min_time(n: int, a_list: List[int]) -> List[int]:
    INF = float('inf')
    high = [-INF] + a_list.copy() + [-INF]
    m = math.ceil(n / 2)
    
    # 初始化DP表，使用二维列表表示当前j和状态0/1/2的最小时间
    dp = [[INF] * 3 for _ in range(m + 1)]
    dp[0][0] = 0  # 初始状态：0个峰，最后状态是0（未选）
    
    for i in range(1, n + 1):
        new_dp = [[INF] * 3 for _ in range(m + 1)]
        for j in range(m + 1):
            for state in range(3):
                if dp[j][state] == INF:
                    continue
                
                if state == 0:
                    # 当前不选i，转移到状态0
                    new_dp[j][0] = min(new_dp[j][0], dp[j][state])
                    # 选择i作为峰，转移到状态1
                    if j < m:
                        cost = 0
                        if high[i] <= high[i - 1]:
                            cost += high[i - 1] - high[i] + 1
                        if high[i] <= high[i + 1]:
                            cost += high[i + 1] - high[i] + 1
                        new_dp[j + 1][1] = min(new_dp[j + 1][1], dp[j][state] + cost)
                
                elif state == 1:
                    # 当前必须不选i（连续不能选），转移到状态2
                    new_dp[j][2] = min(new_dp[j][2], dp[j][state])
                
                elif state == 2:
                    # 当前不选i，转移到状态0
                    new_dp[j][0] = min(new_dp[j][0], dp[j][state])
                    # 选择i作为峰，需考虑前前一个峰的影响
                    if j < m:
                        cost = 0
                        prev_peak_height = high[i - 1]
                        # 考虑i-2的影响
                        if i >= 2 and high[i - 2] <= prev_peak_height:
                            prev_peak_height = high[i - 2] - 1
                        # 计算当前i需要调整的高度
                        if high[i] <= prev_peak_height:
                            cost += prev_peak_height - high[i] + 1
                        if high[i] <= high[i + 1]:
                            cost += high[i + 1] - high[i] + 1
                        new_dp[j + 1][1] = min(new_dp[j + 1][1], dp[j][state] + cost)
        dp = new_dp
    
    # 收集结果
    result = []
    for k in range(1, m + 1):
        min_val = min(dp[k][0], dp[k][1], dp[k][2])
        result.append(min_val if min_val != INF else 0)
    return result

class Ehillsbootcamp(Basebootcamp):
    def __init__(self, max_n: int = 10, max_height: int = 100):
        self.max_n = max_n
        self.max_height = max_height
    
    def case_generator(self):
        n = random.randint(1, self.max_n)
        heights = [random.randint(1, self.max_height) for _ in range(n)]
        expected_output = compute_min_time(n, heights)
        return {
            "n": n,
            "heights": heights,
            "expected_output": expected_output
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        prompt = (
            "Innopolis city needs to adjust hills for building houses. Each hill must be strictly taller than neighbors.\n"
            f"Given {question_case['n']} hills with heights: {', '.join(map(str, question_case['heights']))}.\n"
            f"Calculate the minimum time (hours) needed for each k from 1 to {math.ceil(question_case['n']/2)}. "
            "Output space-separated integers enclosed in [answer]...[/answer].\n"
            "Example Answer Format: [answer]0 1 3[/answer]"
        )
        return prompt
    
    @staticmethod
    def extract_output(output: str) -> List[int]:
        import re
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        last_answer = answer_blocks[-1].strip()
        try:
            return list(map(int, last_answer.split()))
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution: List[int], identity: dict) -> bool:
        return solution == identity['expected_output']
