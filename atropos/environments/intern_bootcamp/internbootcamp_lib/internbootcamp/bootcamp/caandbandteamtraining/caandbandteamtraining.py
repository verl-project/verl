"""# 

### 谜题描述
A and B are preparing themselves for programming contests.

An important part of preparing for a competition is sharing programming knowledge from the experienced members to those who are just beginning to deal with the contests. Therefore, during the next team training A decided to make teams so that newbies are solving problems together with experienced participants.

A believes that the optimal team of three people should consist of one experienced participant and two newbies. Thus, each experienced participant can share the experience with a large number of people.

However, B believes that the optimal team should have two experienced members plus one newbie. Thus, each newbie can gain more knowledge and experience.

As a result, A and B have decided that all the teams during the training session should belong to one of the two types described above. Furthermore, they agree that the total number of teams should be as much as possible.

There are n experienced members and m newbies on the training session. Can you calculate what maximum number of teams can be formed?

Input

The first line contains two integers n and m (0 ≤ n, m ≤ 5·105) — the number of experienced participants and newbies that are present at the training session. 

Output

Print the maximum number of teams that can be formed.

Examples

Input

2 6


Output

2


Input

4 5


Output

3

Note

Let's represent the experienced players as XP and newbies as NB.

In the first test the teams look as follows: (XP, NB, NB), (XP, NB, NB).

In the second test sample the teams look as follows: (XP, NB, NB), (XP, NB, NB), (XP, XP, NB).

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
# -*- coding: utf-8 -*-
#ᕕ┌◕ᗜ◕┐ᕗ HELLO HELLO HELLO ᕕ┌◕ᗜ◕┐ᕗ

def teams(n,m):
    if n == 0 or m == 0:
        return 0
    if n > m:
        d = min(n-m,m)
        return d + teams(n-2*d,m-d)
    if n < m:
        d = min(m-n,n)
        return d + teams(n-d,m-2*d)
    if n == m:
        if n >= 3:
            d = n/3
            return 2*d + teams(n-3*d,m-3*d)
        if n == 2:
            return 1
        if n <= 1:
            return 0

n,m = tuple(int(i) for i in raw_input().strip().split(\" \"))
print teams(n,m)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Caandbandteamtrainingbootcamp(Basebootcamp):
    def __init__(self, max_n=500000, max_m=500000):
        self.max_n = max_n
        self.max_m = max_m
    
    def case_generator(self):
        """改进后的案例生成器，确保参数有效性并增加新案例类型"""
        case_type = random.choices(
            population=[0, 1, 2, 3, 4, 5],
            weights=[0.15, 0.15, 0.2, 0.15, 0.15, 0.2],  # 控制分布比例
            k=1
        )[0]
        
        # 类型0: 单边为0
        if case_type == 0:
            if random.random() > 0.5:
                n = 0
                m = random.randint(1, self.max_m)
            else:
                m = 0
                n = random.randint(1, self.max_n)
        
        # 类型1: 极小数值(0-5)
        elif case_type == 1:
            n = random.randint(0, 5)
            m = random.randint(0, 5)
        
        # 类型2: XP主导型(n >= 2m+1)
        elif case_type == 2:
            max_m_for_n = (self.max_n - 1) // 2
            if max_m_for_n >= 0:
                m = random.randint(0, min(max_m_for_n, self.max_m))
                n = random.randint(2*m + 1, self.max_n) if (2*m + 1) <= self.max_n else m
            else:
                n = random.randint(0, self.max_n)
                m = random.randint(0, self.max_m)
        
        # 类型3: NB主导型(m >= 2n+1)
        elif case_type == 3:
            max_n_for_m = (self.max_m - 1) // 2
            if max_n_for_m >= 0:
                n = random.randint(0, min(max_n_for_m, self.max_n))
                m = random.randint(2*n + 1, self.max_m) if (2*n + 1) <= self.max_m else n
            else:
                n = random.randint(0, self.max_n)
                m = random.randint(0, self.max_m)
        
        # 类型4: 完全相等
        elif case_type == 4:
            base = random.randint(0, min(self.max_n, self.max_m))
            n = m = base
        
        # 类型5: 3的倍数关系
        else:
            k = random.randint(0, 166666)  # 保证3k不超过5e5
            n = 3*k + random.choice([0, 1, 2])
            m = 3*k + random.choice([0, 1, 2])
            n = min(n, self.max_n)
            m = min(m, self.max_m)
        
        return {'n': n, 'm': m}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        m = question_case['m']
        return f"""## 组队优化问题
我们需要将 {n} 名经验选手(XP)和 {m} 名新手(NB)组成最多数量的队伍。每个队伍必须符合以下两种类型之一：
1. [类型A] 1XP + 2NB
2. [类型B] 2XP + 1NB

请计算可组成的最大队伍数量，将最终答案用[answer]标签包裹，例如：[answer]3[/answer]"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 数学闭包公式验证
        n, m = identity['n'], identity['m']
        total = solution
        return total == min((n + m) // 3, n + m - 2*abs(n - m))
    
    @staticmethod
    def _calculate_teams(n, m):
        """优化后的数学闭包解法"""
        return min((n + m) // 3, n + m - 2*abs(n - m))
