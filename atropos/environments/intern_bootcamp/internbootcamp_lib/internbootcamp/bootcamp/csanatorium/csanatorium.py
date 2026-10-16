"""# 

### 谜题描述
Vasiliy spent his vacation in a sanatorium, came back and found that he completely forgot details of his vacation! 

Every day there was a breakfast, a dinner and a supper in a dining room of the sanatorium (of course, in this order). The only thing that Vasiliy has now is a card from the dining room contaning notes how many times he had a breakfast, a dinner and a supper (thus, the card contains three integers). Vasiliy could sometimes have missed some meal, for example, he could have had a breakfast and a supper, but a dinner, or, probably, at some days he haven't been at the dining room at all.

Vasiliy doesn't remember what was the time of the day when he arrived to sanatorium (before breakfast, before dinner, before supper or after supper), and the time when he left it (before breakfast, before dinner, before supper or after supper). So he considers any of these options. After Vasiliy arrived to the sanatorium, he was there all the time until he left. Please note, that it's possible that Vasiliy left the sanatorium on the same day he arrived.

According to the notes in the card, help Vasiliy determine the minimum number of meals in the dining room that he could have missed. We shouldn't count as missed meals on the arrival day before Vasiliy's arrival and meals on the departure day after he left.

Input

The only line contains three integers b, d and s (0 ≤ b, d, s ≤ 1018, b + d + s ≥ 1) — the number of breakfasts, dinners and suppers which Vasiliy had during his vacation in the sanatorium. 

Output

Print single integer — the minimum possible number of meals which Vasiliy could have missed during his vacation. 

Examples

Input

3 2 1


Output

1


Input

1 0 0


Output

0


Input

1 1 1


Output

0


Input

1000000000000000000 0 1000000000000000000


Output

999999999999999999

Note

In the first sample, Vasiliy could have missed one supper, for example, in case he have arrived before breakfast, have been in the sanatorium for two days (including the day of arrival) and then have left after breakfast on the third day. 

In the second sample, Vasiliy could have arrived before breakfast, have had it, and immediately have left the sanatorium, not missing any meal.

In the third sample, Vasiliy could have been in the sanatorium for one day, not missing any meal. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
a = map(int, raw_input().split())
ans = max(a) * 3 - sum(a)
if max(a) == 1:
    ans = 0
for start in range(3):
    for end in range(3):
        total = [0 if ind < start and ind > end else (2 if ind >= start and ind <= end else 1) for ind in range(3)]
        k = 0
        for i in range(3):
            k = max(k, a[i] - total[i])
        for i in range(3):
            total[i] += k
        current = 0
        for i in range(3):
            current += total[i] - a[i]
        ans = min(ans, current)
print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Csanatoriumbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = {
            'max_regular': 1000,
            'edge_prob': 0.2,
            **params
        }

    def case_generator(self):
        def gen_val():
            if random.random() < self.params['edge_prob']:
                return random.choice([0, 1, 10**18])
            return random.randint(0, self.params['max_regular'])
        
        while True:
            b, d, s = gen_val(), gen_val(), gen_val()
            if b + d + s == 0:
                choices = [[1,0,0], [0,1,0], [0,0,1]]
                b, d, s = random.choice(choices)
            if b + d + s >= 1:
                return {'b': b, 'd': d, 's': s}

    @staticmethod
    def prompt_func(case):
        b, d, s = case['b'], case['d'], case['s']
        return f"""Vasily的餐卡记录：早餐{b}次，晚餐{d}次，夜宵{s}次。每个完整天必须包含三餐，到达/离开当天可能不完整。需要考虑：
1. 到达可能在任意餐前/后
2. 离开可能在任意餐前/后
3. 相同天数到达离开的情况

求最少可能错过的餐次数（不计算到达前和离开后的）。答案格式：[answer]整数[/answer]。例：[answer]5[/answer]"""

    @staticmethod
    def extract_output(output):
        answers = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(answers[-1]) if answers else None

    @classmethod
    def _verify_correction(cls, solution, case):
        b, d, s = case.values()
        a = [b, d, s]
        
        # 参考代码逻辑实现
        ans = max(a) * 3 - sum(a)
        if max(a) == 1:
            ans = 0
            
        for start in range(3):  # 到达时间点（0=早餐前，1=午餐前...）
            for end in range(3):  # 离开时间点
                # 生成各个餐次应满足的天数模式
                total = []
                for meal in range(3):
                    if start <= end:
                        in_period = start <= meal <= end
                    else:
                        in_period = meal >= start or meal <= end
                        
                    if in_period:
                        # 完整天数中每天的餐次 + 到达/离开日可能包含的
                        base = 2  # 至少包含到达和离开两天的部分
                    else:
                        base = 1  # 仅包含完整天数中的
                        
                    total.append(base)
                
                # 计算需要的最小完整天数
                k = 0
                for i in range(3):
                    if total[i] == 0:
                        continue
                    req = (a[i] - total[i] + 1) // 1
                    k = max(k, (a[i] - total[i]) // 1 if total[i] else 0)
                
                # 验证当前方案
                current = sum((total[i] + k - a[i]) for i in range(3))
                ans = min(ans, current)
                
        return solution == ans
