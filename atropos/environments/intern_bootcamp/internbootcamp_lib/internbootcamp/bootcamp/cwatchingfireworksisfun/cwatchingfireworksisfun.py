"""# 

### 谜题描述
A festival will be held in a town's main street. There are n sections in the main street. The sections are numbered 1 through n from left to right. The distance between each adjacent sections is 1.

In the festival m fireworks will be launched. The i-th (1 ≤ i ≤ m) launching is on time ti at section ai. If you are at section x (1 ≤ x ≤ n) at the time of i-th launching, you'll gain happiness value bi - |ai - x| (note that the happiness value might be a negative value).

You can move up to d length units in a unit time interval, but it's prohibited to go out of the main street. Also you can be in an arbitrary section at initial time moment (time equals to 1), and want to maximize the sum of happiness that can be gained from watching fireworks. Find the maximum total happiness.

Note that two or more fireworks can be launched at the same time.

Input

The first line contains three integers n, m, d (1 ≤ n ≤ 150000; 1 ≤ m ≤ 300; 1 ≤ d ≤ n).

Each of the next m lines contains integers ai, bi, ti (1 ≤ ai ≤ n; 1 ≤ bi ≤ 109; 1 ≤ ti ≤ 109). The i-th line contains description of the i-th launching.

It is guaranteed that the condition ti ≤ ti + 1 (1 ≤ i < m) will be satisfied.

Output

Print a single integer — the maximum sum of happiness that you can gain from watching all the fireworks.

Please, do not write the %lld specifier to read or write 64-bit integers in C++. It is preferred to use the cin, cout streams or the %I64d specifier.

Examples

Input

50 3 1
49 1 1
26 1 4
6 1 10


Output

-31


Input

10 2 1
1 1000 4
9 1000 4


Output

1992

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int n, m, d, a[400], b[400], t[400], now, last, L, R, l, Q[150010];
long long dp[2][150010];
void f1(long long x) {
  while (L <= R && Q[L] < x) L++;
}
void f3(int l) {
  while (L <= R && dp[last][Q[R]] < dp[last][l]) R--;
  Q[++R] = l;
}
void f2(long long y) {
  while (l <= n && l <= y) f3(l), l++;
}
int main() {
  scanf(\"%d%d%d\", &n, &m, &d);
  for (int i = 1; i <= m; i++) scanf(\"%d%d%d\", &a[i], &b[i], &t[i]);
  for (int i = 1; i <= m; i++) {
    now ^= 1;
    last = now ^ 1;
    L = l = 1, R = 0;
    for (int j = 1; j <= n; j++) {
      long long k1 = j - 1ll * (t[i] - t[i - 1]) * d;
      long long k2 = j + 1ll * (t[i] - t[i - 1]) * d;
      f1(k1);
      f2(k2);
      dp[now][j] = dp[last][Q[L]] + b[i] - abs(a[i] - j);
    }
  }
  long long ans = -10101010;
  for (long long i = 1; i <= n; i++) ans = max(ans, dp[now][i]);
  cout << ans << endl;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import deque
from bootcamp import Basebootcamp

class Cwatchingfireworksisfunbootcamp(Basebootcamp):
    def __init__(self, n=50, m=3, d=1):
        # 参数验证增强
        if not (1 <= n <= 150000):
            raise ValueError("n must be between 1 and 150000")
        if not (1 <= m <= 300):
            raise ValueError("m must be between 1 and 300")
        if not (1 <= d <= n):
            raise ValueError("d must be between 1 and n")
        self.n = n
        self.m = m
        self.d = d

    def case_generator(self):
        n, m, d = self.n, self.m, self.d

        # 生成具有多样性的测试案例
        fireworks = []
        t_prev = 0
        
        # 确保合理的时间间隔分布
        time_options = [
            lambda: random.randint(1, 3),  # 密集时间分布
            lambda: random.randint(10, 20) # 稀疏时间分布
        ][random.randint(0, 1)] if m > 1 else lambda: 1

        for i in range(m):
            # 首个烟花时间至少为1
            delta_t = time_options() if i == 0 else max(0, time_options())
            t_prev += delta_t
            
            # 生成特殊案例：同一时间多个烟花
            if i > 0 and random.random() < 0.2:
                t_prev = fireworks[-1]['t']
            
            a = random.randint(1, n)
            # 生成可能的大数值bi
            b = random.choice([random.randint(1, 100), 10**9])
            fireworks.append({'a': a, 'b': b, 't': t_prev})

        # 保证时间序列非递减
        for i in range(1, m):
            if fireworks[i]['t'] < fireworks[i-1]['t']:
                fireworks[i]['t'] = fireworks[i-1]['t']

        try:
            max_happiness = self.compute_max_happiness(n, m, d, fireworks)
            # 处理无解情况
            if max_happiness == -float('inf'):
                raise ValueError("No valid path")
        except:
            return self.case_generator()  # 递归重新生成

        return {
            'n': n,
            'm': m,
            'd': d,
            'fireworks': fireworks,
            'correct_answer': max_happiness
        }

    @staticmethod
    def compute_max_happiness(n, m, d, fireworks_list):
        # 参数预处理
        sorted_fireworks = sorted(fireworks_list, key=lambda x: x['t'])
        a = [0] * (m + 1)
        b = [0] * (m + 1)
        t = [0] * (m + 1)
        for i in range(1, m+1):
            a[i] = sorted_fireworks[i-1]['a']
            b[i] = sorted_fireworks[i-1]['b']
            t[i] = sorted_fireworks[i-1]['t']

        # 优化数据结构
        dp = [[-float('inf')] * (n + 2) for _ in range(2)]
        current = 0
        previous = 1
        
        # 初始状态调整
        dp[previous][1:n+1] = [0]*n  # 初始位置可以是任意位置

        for i in range(1, m+1):
            current ^= 1
            previous = current ^ 1
            window = deque()
            left_ptr = 1
            delta_time = t[i] - t[i-1]
            max_offset = delta_time * d

            for j in range(1, n+1):
                # 滑动窗口维护
                window_left = j - max_offset
                window_right = j + max_offset
                
                # 移除越界元素
                while window and window[0] < window_left:
                    window.popleft()
                
                # 扩展右边界
                while left_ptr <= min(window_right, n):
                    while window and dp[previous][window[-1]] <= dp[previous][left_ptr]:
                        window.pop()
                    window.append(left_ptr)
                    left_ptr += 1

                # 状态转移逻辑修正
                if window:
                    best_prev = dp[previous][window[0]]
                    dp[current][j] = best_prev + b[i] - abs(a[i] - j)
                else:
                    dp[current][j] = -float('inf')

        max_happiness = max(dp[current][1:n+1])
        return max_happiness if max_happiness != -float('inf') else 0

    @staticmethod
    def prompt_func(question_case):
        # 保持原有prompt结构
        input_str = '\n'.join([f"{question_case['n']} {question_case['m']} {question_case['d']}"] +
                              [f"{fw['a']} {fw['b']} {fw['t']}" for fw in question_case['fireworks']])
        
        return f"""【题目背景】
在长度为{question_case['n']}的主街道上，将有{question_case['m']}个烟花按时间顺序燃放。每个烟花i在时间t_i于位置a_i燃放，当你在位置x观看时获得幸福值b_i - |a_i - x|。你每单位时间最多可以移动{question_case['d']}个单位距离。请计算可获得的最大总幸福值。

【输入格式】
{input_str}

【输出格式】
单个整数，表示最大幸福值。

请将最终答案放在[answer]和[/answer]标记之间。"""

    @staticmethod
    def extract_output(output):
        # 强化提取逻辑
        import re
        answer_tag_match = re.search(r'\[answer\]\s*(-?\d+)\s*\[/answer\]', output)
        if answer_tag_match:
            return int(answer_tag_match.group(1))
        
        # 寻找最后出现的整数
        numbers = re.findall(r'-?\d+', output)
        return int(numbers[-1]) if numbers else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return int(solution) == identity['correct_answer']
        except:
            return False
