"""# 

### 谜题描述
Dreamoon loves summing up something for no reason. One day he obtains two integers a and b occasionally. He wants to calculate the sum of all nice integers. Positive integer x is called nice if <image> and <image>, where k is some integer number in range [1, a].

By <image> we denote the quotient of integer division of x and y. By <image> we denote the remainder of integer division of x and y. You can read more about these operations here: http://goo.gl/AcsXhT.

The answer may be large, so please print its remainder modulo 1 000 000 007 (109 + 7). Can you compute it faster than Dreamoon?

Input

The single line of the input contains two integers a, b (1 ≤ a, b ≤ 107).

Output

Print a single integer representing the answer modulo 1 000 000 007 (109 + 7).

Examples

Input

1 1


Output

0


Input

2 2


Output

8

Note

For the first sample, there are no nice integers because <image> is always zero.

For the second sample, the set of nice integers is {3, 5}.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/usr/bin/python

a, b = map(int, raw_input().split())

M = 10**9+7

answer = ((b*(b-1)//2) % M) * ((a*(a*b+b+2)//2) % M)
answer %= M

print answer
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Cdreamoonandsumsbootcamp(Basebootcamp):
    def __init__(self, a_min=1, a_max=10**7, b_min=1, b_max=10**7):
        """
        参数范围调整为与题目一致，默认支持1到1e7范围
        """
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
    
    def case_generator(self):
        """
        确保覆盖边界值和大数情况，增加10%的边界采样
        """
        # 10%概率生成边界值
        if random.random() < 0.1:
            a = random.choice([self.a_min, self.a_max])
            b = random.choice([self.b_min, self.b_max])
        else:
            a = random.randint(self.a_min, self.a_max)
            b = random.randint(self.b_min, self.b_max)
        return {"a": a, "b": b}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        """
        添加数学公式的LaTeX表达和更严谨的条件说明
        """
        a = question_case['a']
        b = question_case['b']
        prompt = (
            "## Dreamoon's Summation Problem\n\n"
            "Given two integers **a = {a}** and **b = {b}**, compute the sum of all positive integers x satisfying:\n\n"
            "1. ∃k ∈ [1, a] such that \n"
            "   - Quotient equals remainder: \n"
            "     $$\left\lfloor\\frac{{x}}{{k}}\\right\\rfloor = x \\% k$$\n"
            "   - Remainder limit: \n"
            "     $$x \\% k < {b}$$\n\n"
            "**Output Requirements:**\n"
            "- Sum modulo 1,000,000,007\n"
            "- Put final answer within [answer][/answer] tags\n\n"
            "## Examples\n"
            "Input: 1 1 → Output: [answer]0[/answer]\n"
            "Input: 2 2 → Valid x: 3(3//1=3%1=0<2),5(5//2=5%2=1<2) → Sum: 8 → [answer]8[/answer]"
        ).format(a=a, b=b)
        return prompt
    
    @staticmethod
    def extract_output(output):
        """
        增强模式匹配，处理多标签和格式错误
        """
        import re
        # 匹配最后一个answer标签，允许前后有空格
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL | re.IGNORECASE)
        if not matches:
            return None
        last_answer = matches[-1].strip()
        
        # 清理非法字符并尝试转换整数
        cleaned = re.sub(r'[^\d]', '', last_answer)
        try:
            return int(cleaned) if cleaned else None
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        """
        添加公式推导验证，而不仅依赖参考代码
        """
        a = identity['a']
        b = identity['b']
        MOD = 10**9 + 7
        
        # 原始验证逻辑
        ref_answer = ((b*(b-1)//2) % MOD) * ((a*(a*b + b + 2)//2) % MOD) % MOD
        
        # 新增数学验证逻辑（推导过程验证）
        # 根据x = k*q + r，且q=r，得x = k*r + r = r(k+1)
        # 约束条件：0 ≤ r < k，且 r < b → r ∈ [0, min(b-1, k-1)]
        valid = True
        if b == 1:
            derived_answer = 0
        else:
            sigma_r = (b-1)*b//2  # sum(r) from r=0 to b-1
            sigma_k = a*(a+1)//2 + a*b*(a+1)//2
            derived_answer = (sigma_r * sigma_k) % MOD
        
        # 交叉验证两种计算方式
        return int(solution) in {ref_answer, derived_answer} and ref_answer == derived_answer
