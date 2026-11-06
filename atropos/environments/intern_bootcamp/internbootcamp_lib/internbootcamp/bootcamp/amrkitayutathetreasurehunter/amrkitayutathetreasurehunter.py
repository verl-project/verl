"""# 

### 谜题描述
The Shuseki Islands are an archipelago of 30001 small islands in the Yutampo Sea. The islands are evenly spaced along a line, numbered from 0 to 30000 from the west to the east. These islands are known to contain many treasures. There are n gems in the Shuseki Islands in total, and the i-th gem is located on island pi.

Mr. Kitayuta has just arrived at island 0. With his great jumping ability, he will repeatedly perform jumps between islands to the east according to the following process: 

  * First, he will jump from island 0 to island d. 
  * After that, he will continue jumping according to the following rule. Let l be the length of the previous jump, that is, if his previous jump was from island prev to island cur, let l = cur - prev. He will perform a jump of length l - 1, l or l + 1 to the east. That is, he will jump to island (cur + l - 1), (cur + l) or (cur + l + 1) (if they exist). The length of a jump must be positive, that is, he cannot perform a jump of length 0 when l = 1. If there is no valid destination, he will stop jumping. 



Mr. Kitayuta will collect the gems on the islands visited during the process. Find the maximum number of gems that he can collect.

Input

The first line of the input contains two space-separated integers n and d (1 ≤ n, d ≤ 30000), denoting the number of the gems in the Shuseki Islands and the length of the Mr. Kitayuta's first jump, respectively.

The next n lines describe the location of the gems. The i-th of them (1 ≤ i ≤ n) contains a integer pi (d ≤ p1 ≤ p2 ≤ ... ≤ pn ≤ 30000), denoting the number of the island that contains the i-th gem.

Output

Print the maximum number of gems that Mr. Kitayuta can collect.

Examples

Input

4 10
10
21
27
27


Output

3


Input

8 8
9
19
28
36
45
55
66
78


Output

6


Input

13 7
8
8
9
16
17
17
18
21
23
24
24
26
30


Output

4

Note

In the first sample, the optimal route is 0  →  10 (+1 gem)  →  19  →  27 (+2 gems)  → ...

In the second sample, the optimal route is 0  →  8  →  15  →  21 →  28 (+1 gem)  →  36 (+1 gem)  →  45 (+1 gem)  →  55 (+1 gem)  →  66 (+1 gem)  →  78 (+1 gem)  → ...

In the third sample, the optimal route is 0  →  7  →  13  →  18 (+1 gem)  →  24 (+2 gems)  →  30 (+1 gem)  → ...

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
template <class T>
void chmin(T& a, const T& b) {
  if (a > b) a = b;
}
template <class T>
void chmax(T& a, const T& b) {
  if (a < b) a = b;
}
int n, d;
const int geta = 260;
const int MX = 30000;
int dp[30010][520];
int cnt[30010];
int main() {
  scanf(\"%d %d\", &n, &d);
  for (int i = 0; i < n; i++) {
    int k;
    scanf(\"%d\", &k);
    cnt[k]++;
  }
  memset(dp, -1, sizeof(dp));
  dp[d][geta] = cnt[d];
  int ans = 0;
  for (int i = d; i <= MX; i++) {
    for (int j = 0; j < 520; j++) {
      if (dp[i][j] == -1) continue;
      chmax(ans, dp[i][j]);
      int len = d + (j - geta);
      if (len > 1 && i + len - 1 <= MX)
        chmax(dp[i + len - 1][j - 1], dp[i][j] + cnt[i + len - 1]);
      if (i + len <= MX) chmax(dp[i + len][j], dp[i][j] + cnt[i + len]);
      if (i + len + 1 <= MX)
        chmax(dp[i + len + 1][j + 1], dp[i][j] + cnt[i + len + 1]);
    }
  }
  printf(\"%d\n\", ans);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import defaultdict
from bootcamp import Basebootcamp

class Amrkitayutathetreasurehunterbootcamp(Basebootcamp):
    def __init__(self, max_n=100, max_d=30000, **params):
        super().__init__(**params)
        self.max_n = min(max_n, 30000)
        self.max_d = min(max_d, 30000)
        self.max_island = 30000
    
    def case_generator(self):
        n = random.randint(1, self.max_n)
        d = random.randint(1, self.max_d)
        gems = []
        # Ensure valid gem distribution with at least one gem >=d
        base = random.randint(d, self.max_island - 50) if d < self.max_island else d
        spreads = sorted(random.sample(range(base, self.max_island + 1), k=min(n, self.max_island - base + 1)))
        gems = [random.choice(spreads) for _ in range(n)]
        gems.sort()
        correct_answer = self.calculate_max_gems(n, d, gems)
        return {
            'n': n,
            'd': d,
            'gems': gems,
            'correct_answer': correct_answer
        }

    @staticmethod
    def calculate_max_gems(n, d, gems):
        cnt = defaultdict(int)
        for p in gems:
            cnt[p] += 1
        
        MX = 30000
        max_offset = 260
        dp = [[-1] * (2*max_offset + 1) for _ in range(MX + 1)]
        initial_pos = d
        
        if initial_pos > MX:
            return 0
        dp[initial_pos][max_offset] = cnt[initial_pos]
        ans = cnt[0] + dp[initial_pos][max_offset]  # Include starting position
        
        for pos in range(initial_pos, MX + 1):
            for offset in range(2*max_offset + 1):
                if dp[pos][offset] == -1:
                    continue
                current_len = d + (offset - max_offset)
                for dl in (-1, 0, 1):
                    new_len = current_len + dl
                    if new_len <= 0:
                        continue
                    next_pos = pos + new_len
                    if next_pos > MX:
                        continue
                    new_offset = offset + dl
                    if 0 <= new_offset <= 2*max_offset:
                        total = dp[pos][offset] + cnt[next_pos]
                        if total > dp[next_pos][new_offset]:
                            dp[next_pos][new_offset] = total
                            ans = max(ans, total)
        return ans

    @staticmethod
    def prompt_func(question_case):
        gems = sorted(question_case['gems'])
        gems_str = '\n'.join(map(str, gems))
        prompt = f"""在Shuseki群岛寻宝问题中，Kitayuta先生从岛0出发，首次跳跃到岛{question_case['d']}。之后每次跳跃长度可为前次±1或相同（必须>0），直到无法继续跳跃。求他能收集的宝石最大数量。

输入参数：
- 宝石总数 n = {question_case['n']}
- 初始跳跃距离 d = {question_case['d']}
- 按非降序排列的宝石位置：
{gems_str}

请计算最大可收集宝石数量，并将最终答案置于[answer]标签内，例如：[answer]5[/answer]。注意岛屿编号上限为30000，跳跃长度必须始终为正。"""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except (ValueError, IndexError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
