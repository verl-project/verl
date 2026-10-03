"""# 

### 谜题描述
Nastya received a gift on New Year — a magic wardrobe. It is magic because in the end of each month the number of dresses in it doubles (i.e. the number of dresses becomes twice as large as it is in the beginning of the month).

Unfortunately, right after the doubling the wardrobe eats one of the dresses (if any) with the 50% probability. It happens every month except the last one in the year. 

Nastya owns x dresses now, so she became interested in the [expected number](https://en.wikipedia.org/wiki/Expected_value) of dresses she will have in one year. Nastya lives in Byteland, so the year lasts for k + 1 months.

Nastya is really busy, so she wants you to solve this problem. You are the programmer, after all. Also, you should find the answer modulo 109 + 7, because it is easy to see that it is always integer.

Input

The only line contains two integers x and k (0 ≤ x, k ≤ 1018), where x is the initial number of dresses and k + 1 is the number of months in a year in Byteland.

Output

In the only line print a single integer — the expected number of dresses Nastya will own one year later modulo 109 + 7.

Examples

Input

2 0


Output

4


Input

2 1


Output

7


Input

3 2


Output

21

Note

In the first example a year consists on only one month, so the wardrobe does not eat dresses at all.

In the second example after the first month there are 3 dresses with 50% probability and 4 dresses with 50% probability. Thus, in the end of the year there are 6 dresses with 50% probability and 8 dresses with 50% probability. This way the answer for this test is (6 + 8) / 2 = 7.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
x, k = map(lambda x: int(x), raw_input().split(\" \"))
def p(x, k):
    return (p(x, k/2)**2 * (x ** (k%2))) % (10**9+7) if k else 1

print ((x*2-1)*p(2,k) + 1) % (10**9+7) if x else 0
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cnastyaandawardrobebootcamp(Basebootcamp):
    MOD = 10**9 + 7
    
    def __init__(self, x_min=0, x_max=10**18, k_min=0, k_max=10**18):
        # 参数有效性校验
        if x_min < 0 or x_max > 10**18 or x_min > x_max:
            raise ValueError("Invalid x range: must satisfy 0 ≤ x_min ≤ x_max ≤ 1e18")
        if k_min < 0 or k_max > 10**18 or k_min > k_max:
            raise ValueError("Invalid k range: must satisfy 0 ≤ k_min ≤ k_max ≤ 1e18")
        
        self.x_min = x_min
        self.x_max = x_max
        self.k_min = k_min
        self.k_max = k_max
    
    def case_generator(self):
        def generate_value(v_min, v_max, prefer_edge_prob=0.3):
            """增强版数值生成，确保：'''
            1. 正确生成极大数（兼容1e18）
            2. 优先生成边界值的概率
            3. 包含0的特殊处理"""
            edge_candidates = []
            if v_min == v_max:
                return v_min
            
            # 添加合法边界候选
            if v_min <= 0 <= v_max:
                edge_candidates.append(0)
            if v_max != 0 and v_max > 0:
                edge_candidates.append(v_max)
            
            # 添加典型候选值（如1等）
            if 1 > v_min and 1 < v_max:
                edge_candidates.append(1)
            
            # 概率选择边界值
            if edge_candidates and random.random() < prefer_edge_prob:
                return random.choice(edge_candidates)
            
            # 大数安全生成（Python的randint支持大整数）
            return random.randint(v_min, v_max)
        
        return {
            'x': generate_value(self.x_min, self.x_max),
            'k': generate_value(self.k_min, self.k_max)
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        x = question_case['x']
        k_val = question_case['k']
        return f"""
Nastya的魔法衣柜问题解析

背景描述：
每个月初衣柜中的裙子数量会翻倍。在翻倍后（除最后一个月份外），衣柜有50%的概率吃掉1件裙子（如果当前有至少1件）。需要计算经过k+1个月后的期望裙子数量。

输入参数：
x = {x} (初始裙子数)
k = {k_val} (决定总月份数为k+1)

计算规则：
1. 当x=0时，结果直接为0
2. 否则使用公式：((2x-1) * 2^k + 1) mod (10^9+7)
3. 注意k=0时表示只经过1个月（不执行吃裙子操作）

示例验证：
输入 2 0 → 输出：4
输入 2 1 → 输出：7
输入 3 2 → 输出：21

请将最终答案放在[answer]标签内，例如：[answer]42[/answer]
当前题目参数：x={x}, k={k_val}
"""

    @staticmethod
    def extract_output(output):
        # 多模式匹配（应对不同排版）
        patterns = [
            r'\[answer\](.*?)\[/answer\]',  # 标准格式
            r'answer:\s*(\d+)',             # 兼容无标签格式
            r'最终答案\s*[:：]\s*(\d+)'     # 中文格式
        ]
        for pattern in patterns:
            matches = re.findall(pattern, output, re.DOTALL)
            if matches:
                return matches[-1].strip()
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            x = identity['x']
            k = identity['k']
            
            if x == 0:
                return int(solution) == 0
            
            # 大数安全计算
            mod = cls.MOD
            term = pow(2, k, mod)
            expected = ((2*x - 1) * term + 1) % mod
            return int(solution) == expected
        except:
            return False
