"""# 

### 谜题描述
Today, Osama gave Fadi an integer X, and Fadi was wondering about the minimum possible value of max(a, b) such that LCM(a, b) equals X. Both a and b should be positive integers.

LCM(a, b) is the smallest positive integer that is divisible by both a and b. For example, LCM(6, 8) = 24, LCM(4, 12) = 12, LCM(2, 3) = 6.

Of course, Fadi immediately knew the answer. Can you be just like Fadi and find any such pair?

Input

The first and only line contains an integer X (1 ≤ X ≤ 10^{12}).

Output

Print two positive integers, a and b, such that the value of max(a, b) is minimum possible and LCM(a, b) equals X. If there are several possible such pairs, you can print any.

Examples

Input


2


Output


1 2


Input


6


Output


2 3


Input


4


Output


1 4


Input


1


Output


1 1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def gcd(a,b):
    while b:
        a,b=b,a%b
    return a
n=input()
maxx=n
for i in range(1,int(n**(0.5)+1)):
    if n%i==0:
        s=n/i
        x=gcd(i,s)
        m=max(i,s)
        m*=x
        maxx=min(maxx,m)
a=maxx
b=n/maxx
b*=gcd(a,b)
print min(a,b),max(a,b)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import math
from math import gcd
from bootcamp import Basebootcamp

class Cfadiandlcmbootcamp(Basebootcamp):
    def __init__(self, min_X=1, max_X=10**6):
        self.min_X = min_X
        self.max_X = max_X
    
    def case_generator(self):
        # 增加特殊案例生成逻辑
        if random.random() < 0.3:  # 30%概率生成边界案例
            candidates = [
                1,  # 最小边界
                random.choice([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]),  # 小质数
                int(math.sqrt(self.max_X))**2,  # 最大平方数
                2**random.randint(1, 20),  # 2的幂次方
                2 * 3  # 两个质数的乘积
            ]
            X = random.choice(candidates)
            X = max(self.min_X, min(X, self.max_X))
        else:
            X = random.randint(self.min_X, self.max_X)
        return {'X': X}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        X = question_case['X']
        prompt = f"""Given X = {X}, find two positive integers a and b such that:
1. LCM(a, b) = X
2. max(a, b) is minimized

Output format: [answer]a b[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\]\s*(\d+\s+\d+)\s*\[/answer\]', output)
        return matches[-1] if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        X = identity['X']
        
        # 解析答案
        try:
            a, b = map(int, solution.split())
            if a <= 0 or b <= 0:
                return False
        except:
            return False
        
        # 验证LCM正确性
        current_gcd = gcd(a, b)
        if a * b // current_gcd != X:
            return False
        
        # 计算最优解
        min_max = math.inf
        sqrt_x = int(math.sqrt(X))
        for i in range(1, sqrt_x + 1):
            if X % i == 0:
                j = X // i
                pair_gcd = gcd(i, j)
                candidate_max = max(i, j) * pair_gcd  # 关键优化点
                min_max = min(min_max, candidate_max)
        
        return max(a, b) == min_max
