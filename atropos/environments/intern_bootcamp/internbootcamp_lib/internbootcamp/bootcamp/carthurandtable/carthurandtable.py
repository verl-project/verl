"""# 

### 谜题描述
Arthur has bought a beautiful big table into his new flat. When he came home, Arthur noticed that the new table is unstable.

In total the table Arthur bought has n legs, the length of the i-th leg is li.

Arthur decided to make the table stable and remove some legs. For each of them Arthur determined number di — the amount of energy that he spends to remove the i-th leg.

A table with k legs is assumed to be stable if there are more than half legs of the maximum length. For example, to make a table with 5 legs stable, you need to make sure it has at least three (out of these five) legs of the maximum length. Also, a table with one leg is always stable and a table with two legs is stable if and only if they have the same lengths.

Your task is to help Arthur and count the minimum number of energy units Arthur should spend on making the table stable.

Input

The first line of the input contains integer n (1 ≤ n ≤ 105) — the initial number of legs in the table Arthur bought.

The second line of the input contains a sequence of n integers li (1 ≤ li ≤ 105), where li is equal to the length of the i-th leg of the table.

The third line of the input contains a sequence of n integers di (1 ≤ di ≤ 200), where di is the number of energy units that Arthur spends on removing the i-th leg off the table.

Output

Print a single integer — the minimum number of energy units that Arthur needs to spend in order to make the table stable.

Examples

Input

2
1 5
3 2


Output

2


Input

3
2 4 4
1 1 1


Output

0


Input

6
2 2 1 1 3 3
4 3 5 5 2 1


Output

8

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from collections import defaultdict
n = int(raw_input())
l = map(int, raw_input().split())
d = map(int, raw_input().split())
ld = zip(l, d)
ld.sort(key = lambda x: x[1], reverse = True)

hist = defaultdict(int)
cost = defaultdict(int)
for li, di in zip(l, d):
        hist[li] += 1
        cost[li] += di

ans = 10 ** 9
cost_sum = sum(d)
for li, m in hist.items():
        c = cost_sum
        c -= cost[li]

        rem = m - 1
        for lj, dj in ld:
                if rem == 0: break
                if lj < li:
                        rem -= 1
                        c -= dj
        ans = min(ans, c)

print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from collections import defaultdict
from bootcamp import Basebootcamp
import random
import re

class Carthurandtablebootcamp(Basebootcamp):
    def __init__(self, max_legs=20, max_length=100, max_energy=200):
        self.max_legs = max_legs
        self.max_length = max_length
        self.max_energy = max_energy

    def case_generator(self):
        n = random.randint(1, self.max_legs)
        
        # 生成多维度案例
        max_length = random.randint(1, self.max_length)
        candidates = [
            lambda: random.randint(1, max_length-1) if max_length>1 else 1,
            lambda: max_length
        ]
        
        # 确保存在有效最长腿
        main_count = random.randint(1, n)
        legs = [max_length] * main_count
        
        # 生成其他腿（允许存在与最长腿相同的情况）
        for _ in range(n - main_count):
            legs.append(random.choice(candidates)())
        
        random.shuffle(legs)
        di = [random.randint(1, self.max_energy) for _ in range(n)]
        return {"n": n, "l": legs, "d": di}

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        l = ' '.join(map(str, question_case['l']))
        d = ' '.join(map(str, question_case['d']))
        return f"""Arthur需要稳定桌子。桌子有{n}条腿：
长度：{l}
移除能耗：{d}

稳定条件：
1. 剩余k条腿中，最大长度的腿数量 > k/2
2. 单腿总是稳定，双腿需等长

请计算最小能耗，答案放在[ANSWER]标签内。示例：[ANSWER]42[/ANSWER]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[ANSWER\](\d+)\[\/ANSWER\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return solution == cls.calculate_min_energy(**identity)
        except:
            return False

    @staticmethod
    def calculate_min_energy(n, l, d):
        leg_info = sorted(zip(l, d), key=lambda x: -x[1])
        freq = defaultdict(int)
        cost_map = defaultdict(list)
        
        for li, di in zip(l, d):
            freq[li] += 1
            cost_map[li].append(di)
        
        total_cost = sum(d)
        min_energy = float('inf')
        
        for length in freq:
            current_cost = total_cost - sum(cost_map[length])
            available = freq[length] - 1
            
            # 计算必须移除的长腿
            longer_cost = sum(di for li, di in zip(l, d) if li > length)
            
            # 处理需要移除的短腿
            short_legs = sorted([di for li, di in zip(l, d) if li < length], reverse=True)
            keep = min(available, len(short_legs))
            current_cost -= sum(short_legs[:keep])
            
            # 最终总能耗
            final_cost = longer_cost + (sum(short_legs) - sum(short_legs[:keep]))
            min_energy = min(min_energy, final_cost)
        
        return min_energy
