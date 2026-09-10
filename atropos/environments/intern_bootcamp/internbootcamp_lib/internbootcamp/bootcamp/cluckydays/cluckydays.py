"""# 

### 谜题描述
Bob and Alice are often participating in various programming competitions. Like many competitive programmers, Alice and Bob have good and bad days. They noticed, that their lucky and unlucky days are repeating with some period. For example, for Alice days [l_a; r_a] are lucky, then there are some unlucky days: [r_a + 1; l_a + t_a - 1], and then there are lucky days again: [l_a + t_a; r_a + t_a] and so on. In other words, the day is lucky for Alice if it lies in the segment [l_a + k t_a; r_a + k t_a] for some non-negative integer k.

The Bob's lucky day have similar structure, however the parameters of his sequence are different: l_b, r_b, t_b. So a day is a lucky for Bob if it lies in a segment [l_b + k t_b; r_b + k t_b], for some non-negative integer k.

Alice and Bob want to participate in team competitions together and so they want to find out what is the largest possible number of consecutive days, which are lucky for both Alice and Bob.

Input

The first line contains three integers l_a, r_a, t_a (0 ≤ l_a ≤ r_a ≤ t_a - 1, 2 ≤ t_a ≤ 10^9) and describes Alice's lucky days.

The second line contains three integers l_b, r_b, t_b (0 ≤ l_b ≤ r_b ≤ t_b - 1, 2 ≤ t_b ≤ 10^9) and describes Bob's lucky days.

It is guaranteed that both Alice and Bob have some unlucky days.

Output

Print one integer: the maximum number of days in the row that are lucky for both Alice and Bob.

Examples

Input

0 2 5
1 3 5


Output

2


Input

0 1 3
2 3 6


Output

1

Note

The graphs below correspond to the two sample tests and show the lucky and unlucky days of Alice and Bob as well as the possible solutions for these tests.

<image>

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division
from fileinput import input
from fractions import gcd

inp = input()
line = inp.readline().rstrip(\"\n\")
la, ra, ta = map(int, line.split())

line = inp.readline().rstrip(\"\n\")
lb, rb, tb = map(int, line.split())

lengtha = ra - la + 1
lengthb = rb - lb + 1

shift = gcd(ta, tb)
outofsyn = lb - la

abagud = outofsyn % shift
bbagud = -(abagud - shift)



# hvormange = outofsyn // shift
# minouta = outofsyn - hvormange * shift
# minoutb = outofsyn - (hvormange + 1) * shift

no = min(lengtha, lengthb + abagud) - abagud
nt = min(lengtha + bbagud, lengthb) - bbagud

print(max(0, no, nt))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import random
import re
from bootcamp import Basebootcamp

class Cluckydaysbootcamp(Basebootcamp):
    def __init__(self, max_period=1000, **params):
        super().__init__()
        self.max_period = max_period
        self.params = params

    def case_generator(self):
        a_params = self._generate_single_params()
        b_params = self._generate_single_params()
        return {
            'a_params': a_params,
            'b_params': b_params
        }

    def _generate_single_params(self):
        ta = random.randint(2, self.max_period)
        max_s = ta - 1
        s = random.randint(1, max_s)
        la = random.randint(0, ta - s)
        ra = la + s - 1
        return [la, ra, ta]

    @staticmethod
    def prompt_func(question_case) -> str:
        a_la, a_ra, a_ta = question_case['a_params']
        b_lb, b_rb, b_tb = question_case['b_params']
        problem = f"""Alice和Bob在竞赛中发现他们的幸运日存在周期性规律。请你帮助他们找出最长的连续共同幸运天数。

Alice的幸运日周期：
- 周期长度：{a_ta}天
- 每个周期的幸运日为第{a_la}天到第{a_ra}天（包含两端）
- 例如，第0个周期的幸运日是{a_la}~{a_ra}天，第1个周期为{a_ta+a_la}~{a_ta+a_ra}天，依此类推。

Bob的幸运日周期：
- 周期长度：{b_tb}天
- 每个周期的幸运日为第{b_lb}天到第{b_rb}天（包含两端）

任务：
找出两人共同的连续幸运日的最大天数。如果不存在重叠，则答案为0。

输入格式：
第一行三个整数：{a_la} {a_ra} {a_ta}
第二行三个整数：{b_lb} {b_rb} {b_tb}

请将最终答案以整数形式严格放置在[answer]和[/answer]之间，例如：[answer]5[/answer]。"""
        return problem

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(-?\d+)\s*\[/answer\]', output)
        if not matches:
            return None
        try:
            return int(matches[-1])
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        a_la, a_ra, a_ta = identity['a_params']
        b_lb, b_rb, b_tb = identity['b_params']

        length_a = a_ra - a_la + 1
        length_b = b_rb - b_lb + 1
        gcd_val = math.gcd(a_ta, b_tb)
        phase_diff = b_lb - a_la

        ab_shift = phase_diff % gcd_val
        ba_shift = gcd_val - ab_shift

        overlap1 = min(length_a, length_b + ab_shift) - ab_shift
        overlap2 = min(length_a + ba_shift, length_b) - ba_shift

        correct = max(0, overlap1, overlap2)
        return solution == correct
