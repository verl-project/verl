"""# 

### 谜题描述
You are playing a video game and you have just reached the bonus level, where the only possible goal is to score as many points as possible. Being a perfectionist, you've decided that you won't leave this level until you've gained the maximum possible number of points there.

The bonus level consists of n small platforms placed in a line and numbered from 1 to n from left to right and (n - 1) bridges connecting adjacent platforms. The bridges between the platforms are very fragile, and for each bridge the number of times one can pass this bridge from one of its ends to the other before it collapses forever is known in advance.

The player's actions are as follows. First, he selects one of the platforms to be the starting position for his hero. After that the player can freely move the hero across the platforms moving by the undestroyed bridges. As soon as the hero finds himself on a platform with no undestroyed bridge attached to it, the level is automatically ended. The number of points scored by the player at the end of the level is calculated as the number of transitions made by the hero between the platforms. Note that if the hero started moving by a certain bridge, he has to continue moving in the same direction until he is on a platform.

Find how many points you need to score to be sure that nobody will beat your record, and move to the next level with a quiet heart.

Input

The first line contains a single integer n (2 ≤ n ≤ 105) — the number of platforms on the bonus level. The second line contains (n - 1) integers ai (1 ≤ ai ≤ 109, 1 ≤ i < n) — the number of transitions from one end to the other that the bridge between platforms i and i + 1 can bear.

Output

Print a single integer — the maximum number of points a player can get on the bonus level.

Please, do not use the %lld specifier to read or write 64-bit integers in С++. It is preferred to use the cin, cout streams or the %I64d specifier.

Examples

Input

5
2 1 2 1


Output

5

Note

One possibility of getting 5 points in the sample is starting from platform 3 and consequently moving to platforms 4, 3, 2, 1 and 2. After that the only undestroyed bridge is the bridge between platforms 4 and 5, but this bridge is too far from platform 2 where the hero is located now.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
long long l[100005][2];
long long r[100005][2];
int x[100005];
int n;
int main() {
  scanf(\"%d\", &n);
  n--;
  for (int i = 0; i < n; ++i) {
    scanf(\"%d\", x + i);
  }
  for (int i = n - 1; i >= 0; --i) {
    r[i][1] = (x[i] == 1) ? 0 : r[i + 1][1] + x[i] & (~1);
    if (x[i] % 2)
      r[i][0] = max(r[i][1], x[i] + r[i + 1][0]);
    else
      r[i][0] = max(r[i][1], x[i] - 1 + r[i + 1][0]);
  }
  for (int i = 1; i <= n; ++i) {
    l[i][1] = (x[i - 1] == 1) ? 0 : l[i - 1][1] + x[i - 1] & (~1);
    if (x[i - 1] % 2)
      l[i][0] = max(l[i][1], x[i - 1] + l[i - 1][0]);
    else
      l[i][0] = max(l[i][1], x[i - 1] - 1 + l[i - 1][0]);
  }
  long long q = 0;
  for (int i = 0; i <= n; ++i) {
    q = max(q, r[i][0] + l[i][0]);
  }
  cout << q << endl;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

def solve(n_platforms, a):
    """动态规划解法，包含完整边界校验"""
    if n_platforms < 2:
        return 0
    if len(a) != n_platforms - 1:
        raise ValueError("Bridge count mismatch")
    
    n = n_platforms - 1
    x = a.copy()
    
    # 右侧DP初始化
    r = [[0, 0] for _ in range(n_platforms)]
    for i in range(n-1, -1, -1):
        # 计算r[i][1]
        if x[i] == 1:
            r[i][1] = 0
        else:
            next_i = i + 1
            r_next_1 = r[next_i][1] if next_i < n_platforms else 0
            sum_val = r_next_1 + x[i]
            r[i][1] = sum_val & (~1)
        
        # 计算r[i][0]
        next_i = i + 1
        r_next_0 = r[next_i][0] if next_i < n_platforms else 0
        if x[i] % 2 == 1:
            r[i][0] = max(r[i][1], x[i] + r_next_0)
        else:
            r[i][0] = max(r[i][1], (x[i]-1) + r_next_0)
    
    # 左侧DP初始化
    l = [[0, 0] for _ in range(n_platforms)]
    for i in range(1, n_platforms):
        bridge_idx = i-1
        if bridge_idx < 0:
            continue
            
        x_val = x[bridge_idx]
        # 计算l[i][1]
        if x_val == 1:
            l[i][1] = 0
        else:
            prev_i = i-1
            l_prev_1 = l[prev_i][1] if prev_i >= 0 else 0
            sum_val = l_prev_1 + x_val
            l[i][1] = sum_val & (~1)
        
        # 计算l[i][0]
        prev_i = i-1
        l_prev_0 = l[prev_i][0] if prev_i >= 0 else 0
        if x_val % 2 == 1:
            l[i][0] = max(l[i][1], x_val + l_prev_0)
        else:
            l[i][0] = max(l[i][1], (x_val-1) + l_prev_0)
    
    # 计算最大值
    max_score = 0
    for i in range(n_platforms):
        current = r[i][0] + l[i][0]
        max_score = max(max_score, current)
    return max_score

class Efragilebridgesbootcamp(Basebootcamp):
    def __init__(self, max_platforms=1e5, max_bridge=1e9):
        self.max_platforms = min(int(max_platforms), 100000)
        self.max_bridge = min(int(max_bridge), 10**18)
    
    def case_generator(self):
        # 智能生成测试案例（含边界值）
        platform_choices = [
            2,  # 最小有效值
            3,  # 小奇数平台
            random.randint(4, 100),  # 普通小案例
            self.max_platforms  # 最大规模测试
        ]
        n_platforms = random.choices(
            platform_choices,
            weights=[0.15, 0.15, 0.3, 0.4],
            k=1
        )[0]
        
        # 桥的生成策略
        bridge_patterns = [
            lambda: 1,  # 边界情况
            lambda: 2,  # 偶数基础
            lambda: random.choice([3,5,7]),  # 小奇数
            lambda: random.randint(10, 1000),  # 随机中等数
            lambda: self.max_bridge  # 极值测试
        ]
        a = [random.choice(bridge_patterns)() for _ in range(n_platforms-1)]
        
        return {
            'n': n_platforms,
            'a': a,
            'expected': solve(n_platforms, a)
        }
    
    @staticmethod
    def prompt_func(question_case):
        return f"""游戏奖励关卡计算问题

关卡配置：
- 平台总数：{question_case['n']}
- 桥耐久值：{' '.join(map(str, question_case['a']))}

移动规则：
1. 选择任意起始平台
2. 每次移动消耗桥的耐久值
3. 无法移动时统计总移动次数

计算要求：
1. 找出绝对最大值
2. 考虑所有可能路径
3. 结果需为整数

答案格式：
将最终结果放在[answer]标签内，例如：[answer]42[/answer]

当前测试输入：
{question_case['n']}
{' '.join(map(str, question_case['a']))}"""

    @staticmethod
    def extract_output(output):
        pattern = r'\[answer\](.*?)\[\/answer\]'
        matches = re.findall(pattern, output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip().split()[-1])
        except (ValueError, IndexError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected']
