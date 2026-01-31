"""# 

### 谜题描述
There are less than 60 years left till the 900-th birthday anniversary of a famous Italian mathematician Leonardo Fibonacci. Of course, such important anniversary needs much preparations.

Dima is sure that it'll be great to learn to solve the following problem by the Big Day: You're given a set A, consisting of numbers l, l + 1, l + 2, ..., r; let's consider all its k-element subsets; for each such subset let's find the largest common divisor of Fibonacci numbers with indexes, determined by the subset elements. Among all found common divisors, Dima is interested in the largest one.

Dima asked to remind you that Fibonacci numbers are elements of a numeric sequence, where F1 = 1, F2 = 1, Fn = Fn - 1 + Fn - 2 for n ≥ 3.

Dima has more than half a century ahead to solve the given task, but you only have two hours. Count the residue from dividing the sought largest common divisor by m.

Input

The first line contains four space-separated integers m, l, r and k (1 ≤ m ≤ 109; 1 ≤ l < r ≤ 1012; 2 ≤ k ≤ r - l + 1).

Please, do not use the %lld specifier to read or write 64-bit integers in С++. It is preferred to use cin, cout streams or the %I64d specifier.

Output

Print a single integer — the residue from dividing the sought greatest common divisor by m.

Examples

Input

10 1 8 2


Output

3


Input

10 1 8 3


Output

1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import os
import sys
import traceback


def recfib(n,m):
  if n == 0: return (0, 1)
  a, b = recfib(n / 2,m)
  c = a * ( ((2 * b) - a) % m )
  d = ((b * b)%m) + ((a * a)%m)
  c %= m
  d %= m
  if n % 2 == 0: return (c, d)
  return (d, (c + d)%m,)

def fibn(n,m):
    return recfib(n,m)[0]

def d_down( r,l,k ):
  d = (r-l)/(k-1)
  while d > 1:
    if (1+(r/d)-((l+d-1)/d))>=k: return d
    N = 1 + (r/d)
    d -= ( ((N*d)-r) + (N-1) ) / N
  return 1

def d_up( d,r,l,k ):
  while (1+(r/d)-((l+d-1)/d))>=k: d+=1
  return d-1

def ddu( r,l,k):
  d = d_down(r,l,k )
  if not (\"EPYNO_D_UP\" in os.environ): d = d_up( d,r,l,k )
  return d

def solve():
  m,l,r,k = map( int, sys.stdin.readline().strip().split()[:4] )
  d = ddu(r,l,k)
  if \"EPYDEBUG\" in os.environ: print( (d,fibn(d,m),) )
  print( fibn( d,m ) )

if __name__==\"__main__\":
  solve()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Canniversarybootcamp(Basebootcamp):
    def __init__(self, max_k=1000, **params):
        super().__init__(**params)
        self.max_k = max_k  # 控制k的生成范围

    def case_generator(self):
        for _ in range(100):
            try:
                # 改进的参数生成逻辑
                base = 10**random.randint(0, 12)
                l = random.randint(1, base * 100)
                r = l + random.randint(2, 1000)
                
                # 确保满足题目约束条件
                r = min(r, 10**12)
                if r <= l:
                    continue
                
                max_possible_k = r - l + 1
                if max_possible_k < 2:
                    continue
                
                # 生成k的分布：50%小值，20%极值，30%随机值
                if random.random() < 0.5:
                    k = random.randint(2, min(10, max_possible_k))
                else:
                    k_options = [
                        2, 
                        max_possible_k,
                        random.randint(3, min(max_possible_k, self.max_k))
                    ]
                    k = random.choice(k_options)

                m = random.randint(1, 10**9)
                
                # 计算正确答案
                d = self.calculate_d(r, l, k)
                fib_result = self.fibn(d, m)
                
                return {
                    'm': m,
                    'l': l,
                    'r': r,
                    'k': k,
                    'correct_answer': fib_result % m
                }
            except Exception as e:
                continue
        raise RuntimeError("无法生成有效测试用例")

    @staticmethod
    def prompt_func(question_case) -> str:
        params = question_case
        return f"""根据以下参数计算斐波那契最大公约数问题：
- 模数 m = {params['m']}
- 区间 l = {params['l']} 到 r = {params['r']}
- 子集大小 k = {params['k']}

输出最终答案到[answer]标签内"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\s*\]\s*(-?\d+)\s*\[/answer\s*\]', output, re.IGNORECASE)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity['correct_answer']
        return solution % identity['m'] == expected

    # 核心算法实现
    @classmethod
    def calculate_d(cls, r, l, k):
        """改进的二分查找算法"""
        def count(d_val):
            return (r // d_val) - ((l - 1) // d_val)
        
        left, right = 1, r
        best = 1
        while left <= right:
            mid = (left + right) // 2
            if count(mid) >= k:
                best = mid
                left = mid + 1
            else:
                right = mid - 1
        return best

    @staticmethod
    def fibn(n, m):
        """优化的斐波那契计算（修正索引偏移）"""
        if n == 0: return 0
        a, b = 0, 1
        for _ in range(n-1):
            a, b = b, (a + b) % m
        return b % m
