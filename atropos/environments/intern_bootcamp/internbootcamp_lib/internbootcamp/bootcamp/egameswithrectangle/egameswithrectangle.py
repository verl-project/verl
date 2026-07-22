"""# 

### 谜题描述
In this task Anna and Maria play the following game. Initially they have a checkered piece of paper with a painted n × m rectangle (only the border, no filling). Anna and Maria move in turns and Anna starts. During each move one should paint inside the last-painted rectangle a new lesser rectangle (along the grid lines). The new rectangle should have no common points with the previous one. Note that when we paint a rectangle, we always paint only the border, the rectangles aren't filled.

Nobody wins the game — Anna and Maria simply play until they have done k moves in total. Count the number of different ways to play this game.

Input

The first and only line contains three integers: n, m, k (1 ≤ n, m, k ≤ 1000).

Output

Print the single number — the number of the ways to play the game. As this number can be very big, print the value modulo 1000000007 (109 + 7).

Examples

Input

3 3 1


Output

1


Input

4 4 1


Output

9


Input

6 7 2


Output

75

Note

Two ways to play the game are considered different if the final pictures are different. In other words, if one way contains a rectangle that is not contained in the other way.

In the first sample Anna, who performs her first and only move, has only one possible action plan — insert a 1 × 1 square inside the given 3 × 3 square.

In the second sample Anna has as much as 9 variants: 4 ways to paint a 1 × 1 square, 2 ways to insert a 1 × 2 rectangle vertically, 2 more ways to insert it horizontally and one more way is to insert a 2 × 2 square.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, m, k = map(int, raw_input().strip().split())
def fac(x):
    ret = 1
    for i in xrange(1, x + 1):
        ret *= i
    return ret
def c(n, r):
    return fac(n) / (fac(n - r) * fac(r)) if r <= n else 0
print c(n - 1, k * 2) * c(m - 1, k * 2) % (10 ** 9 + 7)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

MOD = 10**9 + 7

class Egameswithrectanglebootcamp(Basebootcamp):
    def __init__(self, n_min=3, n_max=20, m_min=3, m_max=20, k_min=1, k_max=5):
        # 确保参数的合理性
        self.n_min = max(n_min, 3)
        self.n_max = max(n_max, self.n_min)
        self.m_min = max(m_min, 3)
        self.m_max = max(m_max, self.m_min)
        
        # 动态调整k的取值范围
        max_possible_k = min((self.n_max-1)//2, (self.m_max-1)//2)
        self.k_min = max(k_min, 1)
        self.k_max = min(k_max, max_possible_k)
        if self.k_max < self.k_min:
            raise ValueError("Invalid parameter range: Unable to generate valid k")

    def case_generator(self):
        """生成保证有解的合法测试用例"""
        while True:
            # 计算当前参数允许的k范围
            max_k = min((self.n_max-1)//2, (self.m_max-1)//2)
            k = random.randint(self.k_min, min(self.k_max, max_k))
            
            # 动态调整维数生成范围
            min_n = 2*k + 1
            min_m = 2*k + 1
            valid_n_range = (max(self.n_min, min_n), self.n_max)
            valid_m_range = (max(self.m_min, min_m), self.m_max)
            
            if valid_n_range[0] > valid_n_range[1] or valid_m_range[0] > valid_m_range[1]:
                continue  # 跳过无效范围
                
            n = random.randint(*valid_n_range)
            m = random.randint(*valid_m_range)
            return {'n': n, 'm': m, 'k': k}

    @staticmethod
    def prompt_func(question_case):
        n, m, k = question_case['n'], question_case['m'], question_case['k']
        return f"""Anna和Maria正在进行矩形嵌套游戏。初始有一个{n}x{m}的矩形边框，需进行正好{k}步操作。计算不同的游戏方式总数（模10^9+7）。

游戏规则：
1. 每个新矩形必须完全在前一个矩形内部且不共享任何边界点
2. 每个矩形必须在两个维度上都至少缩小2个单位
3. 不同顺序的矩形添加视为不同方案
4. 最终答案需模10^9+7

请将最终答案包裹在[answer]标签内，例如：[answer]42[/answer]。"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(
            r'\[answer\]\s*(\d+)\s*\[/answer\]', 
            output, 
            re.IGNORECASE | re.DOTALL
        )
        return matches[-1] if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        n, m, k = identity['n'], identity['m'], identity['k']
        t = 2 * k

        def comb_mod(n_, r):
            """使用动态规划计算组合数模MOD"""
            if r < 0 or r > n_:
                return 0
            C = [0] * (r + 1)
            C[0] = 1
            for i in range(1, n_ + 1):
                for j in range(min(i, r), 0, -1):
                    C[j] = (C[j] + C[j-1]) % MOD
            return C[r]

        try:
            expected = (comb_mod(n-1, t) * comb_mod(m-1, t)) % MOD
            return int(solution.strip()) == expected
        except (ValueError, TypeError):
            return False
