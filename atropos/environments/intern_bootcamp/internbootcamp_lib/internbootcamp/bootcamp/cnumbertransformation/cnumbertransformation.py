"""# 

### 谜题描述
Little Petya likes positive integers a lot. Recently his mom has presented him a positive integer a. There's only one thing Petya likes more than numbers: playing with little Masha. It turned out that Masha already has a positive integer b. Petya decided to turn his number a into the number b consecutively performing the operations of the following two types:

  1. Subtract 1 from his number. 
  2. Choose any integer x from 2 to k, inclusive. Then subtract number (a mod x) from his number a. Operation a mod x means taking the remainder from division of number a by number x. 



Petya performs one operation per second. Each time he chooses an operation to perform during the current move, no matter what kind of operations he has performed by that moment. In particular, this implies that he can perform the same operation any number of times in a row.

Now he wonders in what minimum number of seconds he could transform his number a into number b. Please note that numbers x in the operations of the second type are selected anew each time, independently of each other.

Input

The only line contains three integers a, b (1 ≤ b ≤ a ≤ 1018) and k (2 ≤ k ≤ 15).

Please do not use the %lld specifier to read or write 64-bit integers in С++. It is preferred to use the cin, cout streams or the %I64d specifier.

Output

Print a single integer — the required minimum number of seconds needed to transform number a into number b.

Examples

Input

10 1 4


Output

6


Input

6 3 10


Output

2


Input

1000000000000000000 1 3


Output

666666666666666667

Note

In the first sample the sequence of numbers that Petya gets as he tries to obtain number b is as follows: 10  →  8  →  6  →  4  →  3  →  2  →  1.

In the second sample one of the possible sequences is as follows: 6  →  4  →  3.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
long long a, b, k;
long long d[360365];
const long long lcm = 360360;
long long giam(long long a, long long b) {
  d[0] = 0;
  for (int i = 1; i <= a - b; i++) {
    d[i] = d[i - 1] + 1;
    for (int j = 2; j <= k; j++)
      if ((b + i) % j <= i) d[i] = min(d[i], d[i - (b + i) % j] + 1);
  }
  return d[a - b];
}
int main() {
  scanf(\"%lld %lld %lld\", &a, &b, &k);
  if ((a - b) < lcm)
    printf(\"%lld\n\", giam(a, b));
  else
    printf(\"%lld\n\", giam(a % lcm, 0) + giam(lcm, b % lcm) +
                         giam(lcm, 0) * (a / lcm - (b + lcm - 1) / lcm));
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from math import gcd
from bootcamp import Basebootcamp

class Cnumbertransformationbootcamp(Basebootcamp):
    _lcm_cache = {}  # Class-level cache for LCM values
    
    def __init__(self, max_a=10**18, max_k=15, **kwargs):
        super().__init__(**kwargs)
        self.max_a = max_a
        self.max_k = max_k
    
    def case_generator(self):
        while True:
            k = random.randint(2, self.max_k)
            b = random.randint(1, 100)  # 提高小数值案例比例
            a = random.randint(b, min(b + 10**6, self.max_a))
            
            # 确保至少25%案例有a-b ≥ LCM(k)
            if random.random() < 0.25:
                lcm = self.compute_lcm(k)
                a = max(b + lcm + random.randint(1, 100), a)
            
            # 验证案例有效性
            if a >= b and 2 <= k <= 15:
                return {"a": a, "b": b, "k": k}

    @staticmethod
    def prompt_func(question_case):
        case = question_case
        return f"""给定初始值a={case['a']}，目标值b={case['b']}，k={case['k']}。每次操作可选：
1. 减1
2. 选x∈[2,k]，减去a mod x
求最短时间。答案放入[answer][/answer]。正确格式示例：[answer]5[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        a, b, k = identity['a'], identity['b'], identity['k']
        try:
            return solution == cls.min_steps(a, b, k)
        except:
            return False

    @classmethod
    def compute_lcm(cls, k):
        if k in cls._lcm_cache:
            return cls._lcm_cache[k]
        
        lcm = 1
        for i in range(2, k+1):
            lcm = lcm * i // gcd(lcm, i)
        cls._lcm_cache[k] = lcm
        return lcm

    @classmethod
    def giam(cls, s, t, k):
        """优化后的动态规划实现"""
        delta = s - t
        if delta <= 0:
            return 0
        
        d = list(range(delta + 1))  # 初始化为纯减法方案
        
        for i in range(1, delta + 1):
            current = t + i
            min_ops = d[i-1] + 1  # 初始为减1方案
            
            for j in range(2, k+1):
                mod = current % j
                # 关键修复：只有当mod>0时才有效
                if mod > 0 and mod <= i:
                    min_ops = min(min_ops, d[i - mod] + 1)
            
            d[i] = min_ops
        return d[delta]

    @classmethod
    def min_steps(cls, a, b, k):
        if a == b:
            return 0
        if a < b:
            return -1  # 无效输入
        
        lcm = cls.compute_lcm(k)
        base_cycles = (a - b) // lcm
        
        # 分三个阶段计算
        if (a - b) < lcm:
            return cls.giam(a, b, k)
        
        phase1 = cls.giam(a % lcm, 0, k)
        phase2 = cls.giam(lcm, b % lcm, k)
        phase3 = base_cycles * cls.giam(lcm, 0, k)
        
        return phase1 + phase2 + phase3
