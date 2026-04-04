"""# 

### 谜题描述
On the math lesson a teacher asked each pupil to come up with his own lucky numbers. As a fan of number theory Peter chose prime numbers. Bob was more original. He said that number t is his lucky number, if it can be represented as: 

t = a2 + b2,  where a, b are arbitrary positive integers.

Now, the boys decided to find out how many days of the interval [l, r] (l ≤ r) are suitable for pair programming. They decided that the day i (l ≤ i ≤ r) is suitable for pair programming if and only if the number i is lucky for Peter and lucky for Bob at the same time. Help the boys to find the number of such days.

Input

The first line of the input contains integer numbers l, r (1 ≤ l, r ≤ 3·108).

Output

In the only line print the number of days on the segment [l, r], which are lucky for Peter and Bob at the same time.

Examples

Input

3 5


Output

1


Input

6 66


Output

7

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
f = lambda x: int(x**.5)+1
h,z = map(bytearray,'1\0')

L,R = map(int,raw_input().split())

n = f(R); b = n/2*h; b[0] = 0
for k in xrange(1,f(n)/2):
    if b[k]: p = 2*k+1; s = k*(p+1); b[s::p] = z*len(b[s::p])

g = ((i*(i+1),2*i+1) for i,v in enumerate(b) if v)

r = (R+3)/4*h; r[0] = 0
for s,p in g:
    r[s::p] = z*len(r[s::p])

print r.count(h,(L+2)/4)+(L<=2<=R)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from math import isqrt
from bootcamp import Basebootcamp

class Cdoublehappinessbootcamp(Basebootcamp):
    def __init__(self, max_r=10**5):
        """
        初始化训练场参数，默认最大右边界为1e5以保证生成效率
        :param max_r: 生成区间右端点最大值，默认为100000
        """
        self.max_r = max_r
    
    def _sieve(self, l, r):
        """高效生成区间[l, r]内的质数列表"""
        if r < 2:
            return []
        
        # 生成基础质数用于筛法
        limit = isqrt(r) + 1
        sieve = [True] * (limit + 1)
        sieve[0:2] = [False, False]
        for i in range(2, isqrt(limit) + 1):
            if sieve[i]:
                sieve[i*i : limit+1 : i] = [False] * len(sieve[i*i : limit+1 : i])
        base_primes = [i for i, prime in enumerate(sieve) if prime]
        
        # 区间筛法
        segment_size = r - l + 1
        sieve = [True] * segment_size
        for p in base_primes:
            start = max(p * p, ((l + p - 1) // p) * p)
            for i in range(start, r+1, p):
                sieve[i - l] = False
        
        # 处理小质数的平方
        for i in range(max(2, l), isqrt(r) + 1):
            if sieve[i - l]:
                for j in range(i*i, r+1, i):
                    sieve[j - l] = False
        
        return [i + l for i in range(segment_size) if sieve[i]]
    
    def case_generator(self):
        """生成随机区间案例并准确计算答案"""
        while True:
            # 生成有效区间
            l = random.randint(1, self.max_r - 1)
            r = random.randint(l, self.max_r)
            
            # 获取区间质数
            primes = self._sieve(l, r)
            
            # 计算有效质数数量
            valid_primes = [p for p in primes if p == 2 or p % 4 == 1]
            
            # 确保生成有效案例（可包含0个解）
            return {
                'l': l,
                'r': r,
                'answer': len(valid_primes)
            }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        l = question_case['l']
        r = question_case['r']
        return f"""你是数学课代表，需要帮助Peter和Bob计算在区间[{l}, {r}]内同时满足两个条件的数字：
1. Peter条件：必须是质数（只能被1和自身整除）
2. Bob条件：可以表示为两个正整数的平方和（即存在a,b>0使得t=a²+b²）

重要数学规则：
- 根据费马定理，质数p可表示为两个平方数之和当且仅当p=2或p≡1(mod4)

请逐步分析后输出准确答案，并将最终答案放在[answer]和[/answer]标签之间。

例如：
当输入为6 66时，正确质数是13,17,29,37,41,53,61，因此输出为7

当前需要解决的输入：
l = {l}, r = {r}

请按照以下格式输出：
[answer]答案数字[/answer]"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output, re.IGNORECASE)
        if matches:
            try:
                return int(matches[-1])
            except ValueError:
                return None
        return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['answer']
