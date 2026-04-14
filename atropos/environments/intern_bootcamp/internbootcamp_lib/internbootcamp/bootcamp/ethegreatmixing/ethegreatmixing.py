"""# 

### 谜题描述
Sasha and Kolya decided to get drunk with Coke, again. This time they have k types of Coke. i-th type is characterised by its carbon dioxide concentration <image>. Today, on the party in honour of Sergiy of Vancouver they decided to prepare a glass of Coke with carbon dioxide concentration <image>. The drink should also be tasty, so the glass can contain only integer number of liters of each Coke type (some types can be not presented in the glass). Also, they want to minimize the total volume of Coke in the glass.

Carbon dioxide concentration is defined as the volume of carbone dioxide in the Coke divided by the total volume of Coke. When you mix two Cokes, the volume of carbon dioxide sums up, and the total volume of Coke sums up as well.

Help them, find the minimal natural number of liters needed to create a glass with carbon dioxide concentration <image>. Assume that the friends have unlimited amount of each Coke type.

Input

The first line contains two integers n, k (0 ≤ n ≤ 1000, 1 ≤ k ≤ 106) — carbon dioxide concentration the friends want and the number of Coke types.

The second line contains k integers a1, a2, ..., ak (0 ≤ ai ≤ 1000) — carbon dioxide concentration of each type of Coke. Some Coke types can have same concentration.

Output

Print the minimal natural number of liter needed to prepare a glass with carbon dioxide concentration <image>, or -1 if it is impossible.

Examples

Input

400 4
100 300 450 500


Output

2


Input

50 2
100 25


Output

3

Note

In the first sample case, we can achieve concentration <image> using one liter of Coke of types <image> and <image>: <image>.

In the second case, we can achieve concentration <image> using two liters of <image> type and one liter of <image> type: <image>.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, k = map(int, raw_input().split())
c = map(int, raw_input().split())
if n in c:
    ans = 1
else:
    q = set()
    for cc in c:
        q.add(cc)
    q = {c - n for c in q}
    mxq = max(q)
    mnq = min(q)
    if max(q) < 0 or min(q) > 0:
        ans = -1
    else:
        mins = [3000 for i in range(-mnq+1)]
        maxs = [3000 for i in range(mxq+1)]
        for qq in q:
            if qq > 0:
                maxs[qq] = 1
            else:
                mins[-qq] = 1
        ans = 1e10
        mni = len(mins)-1
        mxi = len(maxs) - 1
        while mni > 0 and mxi > 0:
            if mni > mxi:
                mni, mxi = mxi, mni
                mins, maxs = maxs, mins
            for i in range(mni, 0, -1):
                maxs[mxi - i] = min(maxs[mxi - i], maxs[mxi] + mins[i])
            mxi -= 1
            while maxs[mxi] > 2500 and mxi > 0:
                mxi -= 1
        ans = min(maxs[0], mins[0])
print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from itertools import combinations
from bootcamp import Basebootcamp

def solve_case(n, c):
    if n in c:
        return 1
    q = {cc - n for cc in c}
    max_q = max(q)
    min_q = min(q)
    if max_q < 0 or min_q > 0:
        return -1
    max_positive = max_q
    min_negative_abs = -min_q
    maxs = [3000] * (max_positive + 1)
    mins = [3000] * (min_negative_abs + 1)
    for qq in q:
        if qq > 0 and qq <= max_positive:
            maxs[qq] = 1
        elif qq < 0:
            idx = -qq
            if idx <= min_negative_abs:
                mins[idx] = 1
    ans = float('inf')
    mni = len(mins) - 1
    mxi = len(maxs) - 1
    while mni > 0 and mxi > 0:
        if mni > mxi:
            mni, mxi = mxi, mni
            mins, maxs = maxs, mins
        for i in range(mni, 0, -1):
            if mxi - i >= 0:
                maxs[mxi - i] = min(maxs[mxi - i], maxs[mxi] + mins[i])
        mxi -= 1
        while mxi > 0 and maxs[mxi] > 2500:
            mxi -= 1
    final_min = min(maxs[0], mins[0])
    return final_min if final_min <= 2500 else -1

class Ethegreatmixingbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = {
            'min_n': 0,
            'max_n': 1000,
            'min_k': 1,
            'max_k': 1000,  # 允许大k值输入
            'case_type': 'mixed'  # 可配置案例类型: simple/complex/mixed
        }
        self.params.update(params)
    
    def case_generator(self):
        params = self.params
        while True:
            n = random.randint(params['min_n'], params['max_n'])
            k = random.randint(params['min_k'], params['max_k'])
            
            # 处理边界情况
            if n == 0 or n == 1000:
                a = [n] * k  # 边界情况必须包含目标值
            else:
                # 根据配置生成不同类型的案例
                case_strategy = random.choice(['simple', 'complex']) \
                    if params['case_type'] == 'mixed' else params['case_type']
                
                if case_strategy == 'simple':
                    # 生成必含目标值的简单案例
                    a = [n]
                    a += [random.randint(0, 1000) for _ in range(k-1)]
                else:
                    # 生成需要混合的复杂案例
                    a = []
                    attempt = 0
                    valid = False
                    while not valid and attempt < 100:
                        attempt += 1
                        a = []
                        # 添加正负差值元素各至少一个
                        a.append(random.randint(n+1, 1000))
                        a.append(random.randint(0, n-1))
                        # 填充剩余元素
                        a += [random.randint(0, 1000) for _ in range(k-2)]
                        # 确保不包含目标值
                        if n in a: continue
                        # 检查是否存在有效解
                        if solve_case(n, a) != -1:
                            valid = True
                    if not valid:
                        continue
            # 确保长度正确
            a = a[:k]
            # 验证答案有效性
            ans = solve_case(n, a)
            if ans != -1:
                return {'n': n, 'k': k, 'a': a}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = question_case['a']
        k = question_case['k']
        problem_desc = (
            f"调配目标浓度：{n}%，现有{len(a)}种可乐浓度：{', '.join(map(str, a))}\n"
            "规则：混合整数升可乐使浓度恰好等于目标值，求最小总升数\n"
            "注意：若无解需返回-1，答案格式示例：[answer]3[/answer]"
        )
        return problem_desc
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if matches:
            try:
                return int(matches[-1].strip())
            except:
                return None
        return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == solve_case(identity['n'], identity['a'])
