"""# 

### 谜题描述
Let f(i) denote the minimum positive integer x such that x is not a divisor of i.

Compute ∑_{i=1}^n f(i) modulo 10^9+7. In other words, compute f(1)+f(2)+...+f(n) modulo 10^9+7.

Input

The first line contains a single integer t (1≤ t≤ 10^4), the number of test cases. Then t cases follow.

The only line of each test case contains a single integer n (1≤ n≤ 10^{16}).

Output

For each test case, output a single integer ans, where ans=∑_{i=1}^n f(i) modulo 10^9+7.

Example

Input


6
1
2
3
4
10
10000000000000000


Output


2
5
7
10
26
366580019

Note

In the fourth test case n=4, so ans=f(1)+f(2)+f(3)+f(4).

  * 1 is a divisor of 1 but 2 isn't, so 2 is the minimum positive integer that isn't a divisor of 1. Thus, f(1)=2. 
  * 1 and 2 are divisors of 2 but 3 isn't, so 3 is the minimum positive integer that isn't a divisor of 2. Thus, f(2)=3. 
  * 1 is a divisor of 3 but 2 isn't, so 2 is the minimum positive integer that isn't a divisor of 3. Thus, f(3)=2. 
  * 1 and 2 are divisors of 4 but 3 isn't, so 3 is the minimum positive integer that isn't a divisor of 4. Thus, f(4)=3. 



Therefore, ans=f(1)+f(2)+f(3)+f(4)=2+3+2+3=10.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
import os
from io import BytesIO

sys.stdin = BytesIO(os.read(0, os.fstat(0).st_size))

f = sys.stdin

line = lambda: f.readline().strip('\r\n').split()

def solve():

    idx = 0
    off = 1
    s = N
    mul = 1
     
    while mul <= N:
        mul = 1
        for i in range(idx,-1,-1):
            f = FACT[i]
            if mul % f != 0:
                mul *= f
         
        start = mul
        step = mul
        
        d = (N-start)//step + 1
        off = 1 if idx == 0 else FACT[idx+1] - FACT[idx]
        s += d * off
        s %= M
         
        idx += 1
        
    return str(s)

FACT = [1,2,3,4,5,7,8,9,11,13,16,17,19,23,25,27,29,31,32,37,41,43]
M = 10**9 + 7

T = int(line()[0])
for test in range(1,T+1):
    
    N = int(line()[0])
    
    sys.stdout.write(solve())
    sys.stdout.write(\"\n\")
    
f.close()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cstrangefunctionbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10**16, include_boundary=True):
        if min_n < 1:
            raise ValueError("min_n must be at least 1")
        if max_n < min_n:
            raise ValueError("max_n must be greater than or equal to min_n")
        self.min_n = min_n
        self.max_n = max_n
        self.include_boundary = include_boundary
    
    def case_generator(self):
        # 生成具有全面覆盖的测试用例
        if self.include_boundary and random.random() < 0.15:
            boundary_n = [1, 2, 3, 4, 10, 10000000000000000]
            n = random.choice(boundary_n)
        else:
            r = random.random()
            if r < 0.3:  # 30%小值
                n = random.randint(1, 100)
            elif r < 0.6:  # 30%中等值
                n = random.randint(10**3, 10**6)
            elif r < 0.85:  # 25%大值
                n = random.randint(10**9, 10**12)
            else:  # 15%极值
                n = random.randint(10**15, self.max_n)
            n = max(self.min_n, min(n, self.max_n))
        return {'n': n}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        return f"""请计算∑f(i) mod 1e9+7（1≤i≤{n}），其中f(i)是i的最小非因数。
规则说明：
1. f(i)定义为i的最小非除数正整数
2. 例如f(4)=3，因为1、2是4的因数但3不是
3. 总和需要对1e9+7取模

输出要求：
将最终答案放在[answer]标签内，如[answer]123[/answer]"""

    @staticmethod
    def extract_output(output):
        # 增强型数字提取逻辑
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        number_str = re.sub(r'[^0-9]', '', matches[-1].strip())
        try:
            return int(number_str) % (10**9+7) if number_str else None
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return solution == cls.compute_answer(identity['n'])
        except:
            return False

    @staticmethod
    def compute_answer(n):
        # 优化后的参考实现
        FACT = [1,2,3,4,5,7,8,9,11,13,16,17,19,23,25,27,29,31,32,37,41,43]
        MOD = 10**9+7
        res = n % MOD
        idx = 0
        
        while idx < len(FACT):
            # 计算当前阶乘基
            product = 1
            for f in reversed(FACT[:idx+1]):
                if product % f != 0:
                    product *= f
                    if product > n:
                        break
            
            if product > n:
                break
            
            # 计算贡献值
            offset = FACT[idx+1] - FACT[idx] if idx > 0 else 1
            count = (n - product) // product + 1
            res = (res + count * offset) % MOD
            idx += 1
        
        return res
