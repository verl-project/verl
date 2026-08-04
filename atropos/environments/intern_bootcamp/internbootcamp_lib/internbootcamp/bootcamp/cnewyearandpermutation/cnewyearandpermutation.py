"""# 

### 谜题描述
Recall that the permutation is an array consisting of n distinct integers from 1 to n in arbitrary order. For example, [2,3,1,5,4] is a permutation, but [1,2,2] is not a permutation (2 appears twice in the array) and [1,3,4] is also not a permutation (n=3 but there is 4 in the array).

A sequence a is a subsegment of a sequence b if a can be obtained from b by deletion of several (possibly, zero or all) elements from the beginning and several (possibly, zero or all) elements from the end. We will denote the subsegments as [l, r], where l, r are two integers with 1 ≤ l ≤ r ≤ n. This indicates the subsegment where l-1 elements from the beginning and n-r elements from the end are deleted from the sequence.

For a permutation p_1, p_2, …, p_n, we define a framed segment as a subsegment [l,r] where max\\{p_l, p_{l+1}, ..., p_r\} - min\\{p_l, p_{l+1}, ..., p_r\} = r - l. For example, for the permutation (6, 7, 1, 8, 5, 3, 2, 4) some of its framed segments are: [1, 2], [5, 8], [6, 7], [3, 3], [8, 8]. In particular, a subsegment [i,i] is always a framed segments for any i between 1 and n, inclusive.

We define the happiness of a permutation p as the number of pairs (l, r) such that 1 ≤ l ≤ r ≤ n, and [l, r] is a framed segment. For example, the permutation [3, 1, 2] has happiness 5: all segments except [1, 2] are framed segments.

Given integers n and m, Jongwon wants to compute the sum of happiness for all permutations of length n, modulo the prime number m. Note that there exist n! (factorial of n) different permutations of length n.

Input

The only line contains two integers n and m (1 ≤ n ≤ 250 000, 10^8 ≤ m ≤ 10^9, m is prime).

Output

Print r (0 ≤ r < m), the sum of happiness for all permutations of length n, modulo a prime number m.

Examples

Input


1 993244853


Output


1


Input


2 993244853


Output


6


Input


3 993244853


Output


32


Input


2019 993244853


Output


923958830


Input


2020 437122297


Output


265955509

Note

For sample input n=3, let's consider all permutations of length 3:

  * [1, 2, 3], all subsegments are framed segment. Happiness is 6. 
  * [1, 3, 2], all subsegments except [1, 2] are framed segment. Happiness is 5. 
  * [2, 1, 3], all subsegments except [2, 3] are framed segment. Happiness is 5. 
  * [2, 3, 1], all subsegments except [2, 3] are framed segment. Happiness is 5. 
  * [3, 1, 2], all subsegments except [1, 2] are framed segment. Happiness is 5. 
  * [3, 2, 1], all subsegments are framed segment. Happiness is 6. 



Thus, the sum of happiness is 6+5+5+5+5+6 = 32.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, m =  map(int, raw_input().split())
f = [0] * (n + 1)
f[0] = 1
for i in range(1, n + 1):
    f[i] = f[i - 1]  * i % m
ans = 0
for i in range(n):
    ans += (n - i) * (n - i) * f[i + 1] * f[n - i - 1] % m
print ans % m
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cnewyearandpermutationbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10, possible_ms=None):
        """
        参数优化说明：
        - max_n默认提升至10以覆盖更多测试案例
        - 增加素数验证逻辑
        """
        super().__init__()
        self.min_n = max(1, min_n)
        self.max_n = max(self.min_n, max_n)
        
        # 设置素数列表并验证
        if possible_ms is None:
            possible_ms = [993244853, 437122297, 100000007]  # 增加一个常用测试素数
        self.possible_ms = [p for p in possible_ms if self.is_prime(p)]  # 素数过滤

    @staticmethod
    def is_prime(n):
        """简单素数验证"""
        if n <= 1:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        sqrt_n = int(n**0.5) + 1
        for i in range(3, sqrt_n, 2):
            if n % i == 0:
                return False
        return True

    def case_generator(self):
        """生成有效案例的增强版"""
        # 确保至少有一个可用素数
        if not self.possible_ms:
            raise ValueError("No valid prime numbers available")
        
        n = random.randint(self.min_n, self.max_n)
        m = random.choice(self.possible_ms)
        return {
            'n': n,
            'm': m,
            'expected': self.compute_expected(n, m)
        }

    @staticmethod
    def compute_expected(n, m):
        """优化计算过程，添加模运算校验"""
        if m <= 1:
            return 0
        fact = [1]*(n+1)
        for i in range(1, n+1):
            fact[i] = fact[i-1] * i % m
        total = 0
        for i in range(n):
            k = n - i
            term = k * k % m
            term = term * fact[i+1] % m
            term = term * fact[n - (i+1)] % m
            total = (total + term) % m
        return total

    @staticmethod
    def prompt_func(question_case) -> str:
        """添加中文规则描述"""
        n = question_case['n']
        m = question_case['m']
        return f"""请解决以下数学问题：

[问题描述]
给定两个整数n和m，其中m是素数。需要计算所有长度为n的排列的happiness总和，对m取模后的结果。

[happiness定义]
- 排列：包含1到n每个数字恰好一次的序列
- framed segment：区间[l, r]满足max(p_l,...,p_r) - min(p_l,...,p_r) = r - l
- 排列的happiness是其所有framed segment的数量

[输入要求]
- 1 ≤ n ≤ 250,000
- 1e8 ≤ m ≤ 1e9 且m为素数

[示例]
例如当n=3时，总共有6种排列：
- [1,2,3] 的happiness是6  
- [1,3,2] 的happiness是5  
- ...其他排列类似...
总和为32（模m后）

[当前测试案例]
n = {n}
m = {m}

请将最终答案用[answer]标签包裹，例如：[answer]123[/answer]"""

    @staticmethod
    def extract_output(output: str):
        """增强提取逻辑：处理负数、科学计数法等情况"""
        # 匹配最后一个answer标签
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        
        raw_answer = matches[-1].strip()
        
        # 提取所有数字字符（包括负号）
        number_str = re.sub(r'[^-+0-9]', '', raw_answer)
        
        try:
            return int(number_str) if number_str else None
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """精确验证，处理负余数情况"""
        expected = identity['expected']
        m = identity['m']
        if solution is None:
            return False
        return (solution % m) == expected
