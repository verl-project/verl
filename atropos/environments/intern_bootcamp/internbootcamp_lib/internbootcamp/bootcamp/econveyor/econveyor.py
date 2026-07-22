"""# 

### 谜题描述
Anton came to a chocolate factory. There he found a working conveyor and decided to run on it from the beginning to the end.

The conveyor is a looped belt with a total length of 2l meters, of which l meters are located on the surface and are arranged in a straight line. The part of the belt which turns at any moment (the part which emerges from under the floor to the surface and returns from the surface under the floor) is assumed to be negligibly short.

The belt is moving uniformly at speed v1 meters per second. Anton will be moving on it in the same direction at the constant speed of v2 meters per second, so his speed relatively to the floor will be v1 + v2 meters per second. Anton will neither stop nor change the speed or the direction of movement.

Here and there there are chocolates stuck to the belt (n chocolates). They move together with the belt, and do not come off it. Anton is keen on the chocolates, but he is more keen to move forward. So he will pick up all the chocolates he will pass by, but nothing more. If a chocolate is at the beginning of the belt at the moment when Anton starts running, he will take it, and if a chocolate is at the end of the belt at the moment when Anton comes off the belt, he will leave it.

<image> The figure shows an example with two chocolates. One is located in the position a1 = l - d, and is now on the top half of the belt, the second one is in the position a2 = 2l - d, and is now on the bottom half of the belt. 

You are given the positions of the chocolates relative to the initial start position of the belt 0 ≤ a1 < a2 < ... < an < 2l. The positions on the belt from 0 to l correspond to the top, and from l to 2l — to the the bottom half of the belt (see example). All coordinates are given in meters.

Anton begins to run along the belt at a random moment of time. This means that all possible positions of the belt at the moment he starts running are equiprobable. For each i from 0 to n calculate the probability that Anton will pick up exactly i chocolates.

Input

The first line contains space-separated integers n, l, v1 and v2 (1 ≤ n ≤ 105, 1 ≤ l, v1, v2 ≤ 109) — the number of the chocolates, the length of the conveyor's visible part, the conveyor's speed and Anton's speed.

The second line contains a sequence of space-separated integers a1, a2, ..., an (0 ≤ a1 < a2 < ... < an < 2l) — the coordinates of the chocolates.

Output

Print n + 1 numbers (one per line): the probabilities that Anton picks up exactly i chocolates, for each i from 0 (the first line) to n (the last line). The answer will be considered correct if each number will have absolute or relative error of at most than 10 - 9.

Examples

Input

1 1 1 1
0


Output

0.75000000000000000000
0.25000000000000000000


Input

2 3 1 2
2 5


Output

0.33333333333333331000
0.66666666666666663000
0.00000000000000000000

Note

In the first sample test Anton can pick up a chocolate if by the moment he starts running its coordinate is less than 0.5; but if by the moment the boy starts running the chocolate's coordinate is greater than or equal to 0.5, then Anton won't be able to pick it up. As all positions of the belt are equiprobable, the probability of picking up the chocolate equals <image>, and the probability of not picking it up equals <image>.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const double eps = 0.0001;
int n, il, ir, m;
double L, v1, v2, l, r, rr;
double a[200001];
double answ[100001] = {0};
void answout() {
  int i;
  for (i = 0; i <= n; ++i) printf(\"%.15lf\n\", answ[i]);
}
int main(void) {
  int i;
  scanf(\"%d%lf%lf%lf\", &n, &L, &v1, &v2);
  for (i = 0; i < n; ++i) scanf(\"%lf\", a + i);
  for (i = 0; i < n; ++i) a[n + i] = a[i] + 2 * L;
  a[2 * n] = a[0] + 4 * L;
  l = 0;
  rr = r = L * v2 / (v1 + v2);
  il = 0;
  ir = -1;
  while (a[ir + 1] < r) ++ir;
  while (true) {
    if (a[il] - l < a[ir + 1] - r) {
      m = ir - il + 1;
      if (a[il] > 2 * L) {
        answ[m] += (2 * L - l) / (2 * L);
        answout();
        return 0;
      }
      answ[m] += (a[il] - l) / (2 * L);
      l = a[il];
      r = a[il] + rr;
      ++il;
    } else {
      m = ir - il + 1;
      if (a[ir + 1] - rr > 2 * L) {
        answ[m] += (2 * L - l) / (2 * L);
        answout();
        return 0;
      }
      answ[m] += (a[ir + 1] - r) / (2 * L);
      r = a[ir + 1];
      l = a[ir + 1] - rr;
      ++ir;
    }
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bisect import bisect_left

class Econveyorbootcamp(Basebootcamp):
    def __init__(self, max_n=100000, l_range=(1, int(1e9)), v1_range=(1, int(1e9)), v2_range=(1, int(1e9))):
        self.max_n = max_n
        self.l_range = l_range
        self.v1_range = v1_range
        self.v2_range = v2_range

    def case_generator(self):
        # 生成随机参数（包含极大值场景）
        n = random.randint(1, min(self.max_n, 1000))  # 测试时限制最大n为1000
        l = random.randint(*self.l_range)
        v1 = random.randint(*self.v1_range)
        v2 = random.randint(*self.v2_range)
        
        # 生成严格递增的坐标（处理极小数和极大值场景）
        a = []
        current = 0.0
        for _ in range(n):
            current += random.uniform(1e-9, (2*l - current)*0.1)
            a.append(current)
        a = [x % (2*l) for x in a]  # 确保在[0, 2l)范围内
        a = sorted(list(set(a)))     # 去重后排序
        while len(a) < n:            # 补充不足的坐标
            a.append(a[-1] + random.uniform(1e-9, 1e-8))
        
        return {
            'n': n,
            'l': float(l),
            'v1': float(v1),
            'v2': float(v2),
            'a': sorted(a[:n])       # 最终确保n个严格递增的坐标
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        l = question_case['l']
        v1 = question_case['v1']
        v2 = question_case['v2']
        a = question_case['a']
        a_str = ' '.join(f"{x:.12f}" for x in a)
        
        return f"""Anton runs on a {2*l:.2f}m conveyor (visible part {l:.2f}m) moving at {v1:.2f}m/s. He runs at {v2:.2f}m/s. Chocolates are at:
{a_str}
Calculate probabilities for picking 0-{n} chocolates. Format answers with 15 decimals in [answer] tags."""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        return [float(line) for line in last_match.splitlines() if line.strip()]

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 精确验证（考虑浮点精度）
        expected = cls.calculate_probabilities(identity)
        return all(abs(s-e) < 1e-9 for s, e in zip(solution, expected))

    @staticmethod
    def calculate_probabilities(identity):
        n = identity['n']
        L = identity['l']
        v1 = identity['v1']
        v2 = identity['v2']
        a = sorted(identity['a'])
        
        # 事件点扩展（处理循环）
        a_ext = a + [x + 2*L for x in a] + [4*L]
        a_ext.sort()
        
        capture_length = L * v2 / (v1 + v2)
        timeline = []
        l_ptr = r_ptr = 0
        current_start = 0.0
        
        # 滑动窗口计算覆盖区域
        while current_start < 2*L:
            # 计算下一个事件点
            next_left = a_ext[l_ptr] if l_ptr < len(a_ext) else float('inf')
            next_right = a_ext[r_ptr] if r_ptr < len(a_ext) else float('inf')
            
            # 确定移动步长
            delta = min(next_left - current_start, next_right - (current_start + capture_length))
            if delta <= 0:  # 处理浮点误差
                delta = 1e-12
            
            # 记录当前区间的覆盖数量
            count = r_ptr - l_ptr
            if 0 <= count <= n:
                timeline.append((delta, count))
            
            # 更新指针
            current_start += delta
            if current_start >= next_left - 1e-12:
                l_ptr += 1
            if current_start + capture_length >= next_right - 1e-12:
                r_ptr += 1
        
        # 统计算法结果
        total = sum(d for d, _ in timeline)
        result = [0.0]*(n+1)
        for duration, cnt in timeline:
            if 0 <= cnt <= n:
                result[cnt] += duration / (2*L)
        return result
