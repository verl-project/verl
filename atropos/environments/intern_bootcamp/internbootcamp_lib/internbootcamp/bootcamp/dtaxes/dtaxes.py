"""# 

### 谜题描述
Mr. Funt now lives in a country with a very specific tax laws. The total income of mr. Funt during this year is equal to n (n ≥ 2) burles and the amount of tax he has to pay is calculated as the maximum divisor of n (not equal to n, of course). For example, if n = 6 then Funt has to pay 3 burles, while for n = 25 he needs to pay 5 and if n = 2 he pays only 1 burle.

As mr. Funt is a very opportunistic person he wants to cheat a bit. In particular, he wants to split the initial n in several parts n1 + n2 + ... + nk = n (here k is arbitrary, even k = 1 is allowed) and pay the taxes for each part separately. He can't make some part equal to 1 because it will reveal him. So, the condition ni ≥ 2 should hold for all i from 1 to k.

Ostap Bender wonders, how many money Funt has to pay (i.e. minimal) if he chooses and optimal way to split n in parts.

Input

The first line of the input contains a single integer n (2 ≤ n ≤ 2·109) — the total year income of mr. Funt.

Output

Print one integer — minimum possible number of burles that mr. Funt has to pay as a tax.

Examples

Input

4


Output

2


Input

27


Output

3

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import math

flag= True

inpt= int(raw_input(\"\"))

if inpt==2 or inpt==3 or inpt==5 or inpt==7:
    print \"1\"
    flag= False
    


if flag==True and inpt%2== 0 and flag== True:
    print \"2\"
    flag= False

def isprime(n):
    for x in range(2, int(math.floor(math.sqrt(n)+2))):
        
        if n%x==0:
            return False
    return True

if flag==True and isprime(inpt) and flag == True:
    print \"1\"
    flag= False

if flag==True and isprime(inpt-2) and flag== True:
    print \"2\"
    flag= False

if flag== True:
    print \"3\"
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import random
from bootcamp import Basebootcamp

class Dtaxesbootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=2_000_000_000):
        """
        修正默认参数范围，允许生成大数
        增加类型标记字段用于验证生成多样性
        """
        self.min_n = min_n
        self.max_n = max_n
    
    def case_generator(self):
        """
        改进案例生成策略，确保生成多种类型：
        1. 随机生成基准数后调整确保覆盖所有情况
        2. 包含质数、偶数和需要组合拆分的复杂情况
        """
        # 生成基准数并强制类型分布
        base_n = random.randint(self.min_n, self.max_n)
        
        # 控制类型分布：30%偶数，30%质数，40%复杂奇数
        if random.random() < 0.3:
            n = base_n if base_n % 2 == 0 and base_n >= 4 else base_n + (base_n % 2)
        elif random.random() < 0.5:
            # 生成质数（简化实现，实际应用需优化大质数生成）
            n = self._nearest_prime(base_n)
        else:
            # 生成复杂奇数（需要拆分成2+3的情况）
            n = base_n | 1  # 强制为奇数
            while n < 5 or self._is_prime(n) or self._is_prime(n-2):
                n += 2
        return {'n': max(2, min(n, self.max_n))}
    
    def _nearest_prime(self, n):
        """简单质数查找实现（演示用，实际应优化）"""
        n = max(2, n)
        while True:
            if self._is_prime(n):
                return n
            n += 1
    
    @staticmethod
    def _is_prime(num):
        """优化后的质数判断"""
        if num < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:  # 预判小因子
            if num % p == 0:
                return num == p
        i = 37
        w = 2
        while i * i <= num:
            if num % i == 0:
                return False
            i += w
            w = 6 - w
        return True
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        return f"""Mr. Funt needs to split his income of {n} burles into parts (each ≥2) to minimize tax. 
Tax for each part is its largest divisor other than itself. 
For example:
- 6 → tax 3 (split as [6] or [3,3])
- 27 → tax 3 (split as [25,2] → 5+1=6 is wrong; optimal is [2,2,23] → 1+1+1=3)

Calculate the minimal total tax. Put your final answer within [answer][/answer]."""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return int(matches[-1].strip()) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        
        # 优化验证逻辑
        if n == 2 or n == 3: return solution == 1
        if n % 2 == 0: return solution == 2
        if cls._is_prime(n): return solution == 1
        return solution == (2 if cls._is_prime(n-2) else 3)
