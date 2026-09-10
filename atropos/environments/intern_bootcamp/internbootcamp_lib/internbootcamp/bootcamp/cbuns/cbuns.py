"""# 

### 谜题描述
Lavrenty, a baker, is going to make several buns with stuffings and sell them.

Lavrenty has n grams of dough as well as m different stuffing types. The stuffing types are numerated from 1 to m. Lavrenty knows that he has ai grams left of the i-th stuffing. It takes exactly bi grams of stuffing i and ci grams of dough to cook a bun with the i-th stuffing. Such bun can be sold for di tugriks.

Also he can make buns without stuffings. Each of such buns requires c0 grams of dough and it can be sold for d0 tugriks. So Lavrenty can cook any number of buns with different stuffings or without it unless he runs out of dough and the stuffings. Lavrenty throws away all excess material left after baking.

Find the maximum number of tugriks Lavrenty can earn.

Input

The first line contains 4 integers n, m, c0 and d0 (1 ≤ n ≤ 1000, 1 ≤ m ≤ 10, 1 ≤ c0, d0 ≤ 100). Each of the following m lines contains 4 integers. The i-th line contains numbers ai, bi, ci and di (1 ≤ ai, bi, ci, di ≤ 100).

Output

Print the only number — the maximum number of tugriks Lavrenty can earn.

Examples

Input

10 2 2 1
7 3 2 100
12 3 1 10


Output

241

Input

100 1 25 50
15 5 20 10


Output

200

Note

To get the maximum number of tugriks in the first sample, you need to cook 2 buns with stuffing 1, 4 buns with stuffing 2 and a bun without any stuffing.

In the second sample Lavrenty should cook 4 buns without stuffings.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n,m,c0,d0 = map(int, raw_input().strip().split())

z = [n] +  [0]*(m+1)
c = [c0] + [0]*(m+1)
d = [d0] + [0]*(m+1)
for i in range(1, m+1):
    a,b,c[i],d[i]=map(int, raw_input().strip().split())
    z[i] = a//b
ans = 0

# dp[j][i] := max gain up to i, using j grams of dough
# = max_{k from 0 up to max(z[i], j // c[i])} {k*d[i] + dp[j - k*c[i]][i-1], dp[j][i-1]}
dp = [[0] * (m+1) for _ in range(n+1)]
for j in range(n+1):
    dp[j][0] = (j//c[0])*d[0]
    for i in range(1, m+1):
        dp[j][i] = dp[j][i-1]
        for k in range(min(z[i], j // c[i])+1):
            g  = k*d[i] + dp[j-k*c[i]][i-1]
            dp[j][i] = max(g, dp[j][i])

print max([dp[j][m] for j in range(n+1)])
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Cbunsbootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=1000, m_min=1, m_max=10, stuffing_min=1, stuffing_max=100):
        self.n_min = n_min
        self.n_max = n_max
        self.m_min = m_min
        self.m_max = m_max
        self.stuffing_min = stuffing_min
        self.stuffing_max = stuffing_max

    def case_generator(self):
        """生成完全符合题目约束的合法案例"""
        m = random.randint(self.m_min, self.m_max)
        n = random.randint(max(self.n_min, 1), self.n_max)  # 确保n≥1
        
        # 保证c0/d0的取值满足1 ≤ c0,d0 ≤ 100
        c0 = random.randint(max(self.stuffing_min, 1), min(self.stuffing_max, 100))
        d0 = random.randint(max(self.stuffing_min, 1), min(self.stuffing_max, 100))
        
        stuffings = []
        for _ in range(m):
            # 保证所有参数满足1 ≤ ai,bi,ci,di ≤100
            ai = random.randint(1, 100)
            bi = random.randint(1, 100)
            ci = random.randint(1, 100)
            di = random.randint(1, 100)
            # 确保bi≥1避免除零错误
            bi = max(1, bi)
            stuffings.append((ai, bi, ci, di))
        
        return {
            'n': n,
            'm': m,
            'c0': c0,
            'd0': d0,
            'stuffings': stuffings
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        input_lines = [
            f"{question_case['n']} {question_case['m']} {question_case['c0']} {question_case['d0']}"
        ]
        for stuffing in question_case['stuffings']:
            input_lines.append(" ".join(map(str, stuffing)))
        
        return (
            "Lavrenty需要制作包子获取最大利润，规则如下：\n"
            "1. 总共有n克面团和m种馅料\n"
            "2. 无馅包子消耗c0克面团，利润d0\n"
            "3. 第i种馅料的参数：可用ai克，每个包子需要bi克馅料和ci克面团，利润di\n"
            "4. 输出最大利润\n\n"
            "输入数据：\n" + "\n".join(input_lines) +
            "\n\n将最终答案放在[answer]和[/answer]之间"
        )

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return int(matches[-1].strip()) if matches else None

    @classmethod
    def _calculate_correct_answer(cls, identity):
        """优化后的答案计算逻辑"""
        n = identity['n']
        m = identity['m']
        c0 = identity['c0']
        d0 = identity['d0']
        stuffings = identity['stuffings']

        # 合并所有包子类型（0为无馅，1~m为有馅）
        items = [(c0, d0, float('inf'))]  # (ci, di, 最大数量)
        for ai, bi, ci, di in stuffings:
            max_count = ai // bi
            items.append((ci, di, max_count))

        # 背包DP优化实现
        dp = [0] * (n + 1)
        for ci, di, max_count in items:
            for j in range(n, ci-1, -1):
                max_k = min(max_count, j // ci)
                dp[j] = max(dp[j - k*ci] + k*di for k in range(0, max_k+1))
        
        return max(dp)

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return int(solution) == cls._calculate_correct_answer(identity)
        except:
            return False
