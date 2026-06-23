"""# 

### 谜题描述
Ivan has number b. He is sorting through the numbers a from 1 to 10^{18}, and for every a writes ([a,    b])/(a) on blackboard. Here [a,    b] stands for least common multiple of a and b. Ivan is very lazy, that's why this task bored him soon. But he is interested in how many different numbers he would write on the board if he would finish the task. Help him to find the quantity of different numbers he would write on the board.

Input

The only line contains one integer — b (1 ≤ b ≤ 10^{10}).

Output

Print one number — answer for the problem.

Examples

Input

1


Output

1

Input

2


Output

2

Note

In the first example [a,    1] = a, therefore ([a,    b])/(a) is always equal to 1.

In the second example [a,    2] can be equal to a or 2 ⋅ a depending on parity of a. ([a,    b])/(a) can be equal to 1 and 2.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
m = int(raw_input())

ct=0
i=1
while i*i<m:
    if m%i==0:
        ct+=2
    i+=1
if i*i==m:
    ct+=1
print ct
exit()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
import math
from bootcamp import Basebootcamp

class Blcmbootcamp(Basebootcamp):
    def __init__(self, min_b=1, max_b=10**6):
        """
        参数范围优化：默认上限设为1e6保证效率，同时仍覆盖所有可能情况
        确保min_b >=1 符合题目约束条件
        """
        self.min_b = max(1, min_b)  # 强制下限保护
        self.max_b = min(10**10, max_b)  # 强制上限保护
    
    def case_generator(self):
        """
        增强型case生成逻辑，确保生成案例的多样性：
        - 50%概率生成随机数
        - 30%概率生成质数
        - 20%概率生成完全平方数
        """
        choice = random.random()
        
        if choice < 0.5:
            # 生成普通随机数
            b = random.randint(self.min_b, self.max_b)
        elif choice < 0.8:
            # 生成质数（使用随机质数生成逻辑）
            primes = [x for x in self._primes_up_to(10**4) if x >= self.min_b and x <= self.max_b]
            if not primes:
                primes = [2,3,5,7,11,13,17,19,23,29]
            b = random.choice(primes)
        else:
            # 生成完全平方数（确保平方根为整数）
            sqrt_min = math.isqrt(self.min_b)
            sqrt_max = math.isqrt(self.max_b)
            sqrt_val = random.randint(sqrt_min, sqrt_max)
            b = sqrt_val * sqrt_val
        
        return {'b': b}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        b = question_case['b']
        prompt = f"""
你是一位数学专家，请解决以下问题：

给定正整数b={b}，计算当a遍历1到10^18的所有整数时，表达式[a,b]/a可能得到的不同结果的数量。其中[a,b]表示a和b的最小公倍数。

请遵循以下步骤：
1. 分析表达式[a,b]/a的数学性质
2. 推导该表达式可能值的个数与b的关系
3. 最终答案应为b的正因数个数

将答案用[answer]标签包裹，例如[answer]答案[/answer]
"""
        return prompt.strip()
    
    @staticmethod
    def extract_output(output):
        # 增强匹配模式，允许数值前后的空格
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output, re.IGNORECASE)
        return int(matches[-1]) if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        def count_factors(n):
            """优化后的因数计数器"""
            if n == 1:
                return 1
            total = 1
            # 预生成质数列表加速分解
            for p in cls._primes_up_to(math.isqrt(n) + 1):
                if p*p > n:
                    break
                exponent = 0
                while n % p == 0:
                    exponent += 1
                    n = n // p
                if exponent > 0:
                    total *= (exponent + 1)
            if n > 1:  # 剩余的大质数
                total *= 2
            return total
        
        return solution == count_factors(identity['b'])
    
    # 新增工具方法
    @staticmethod
    def _primes_up_to(n):
        """使用Sieve算法生成质数列表"""
        sieve = [True] * (n+1)
        sieve[0:2] = [False, False]
        for i in range(2, int(math.sqrt(n)) +1):
            if sieve[i]:
                sieve[i*i : n+1 : i] = [False]*len(sieve[i*i : n+1 : i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]
