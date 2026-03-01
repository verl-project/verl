"""# 

### 谜题描述
You've got an array a, consisting of n integers. The array elements are indexed from 1 to n. Let's determine a two step operation like that:

  1. First we build by the array a an array s of partial sums, consisting of n elements. Element number i (1 ≤ i ≤ n) of array s equals <image>. The operation x mod y means that we take the remainder of the division of number x by number y. 
  2. Then we write the contents of the array s to the array a. Element number i (1 ≤ i ≤ n) of the array s becomes the i-th element of the array a (ai = si). 



You task is to find array a after exactly k described operations are applied.

Input

The first line contains two space-separated integers n and k (1 ≤ n ≤ 2000, 0 ≤ k ≤ 109). The next line contains n space-separated integers a1, a2, ..., an — elements of the array a (0 ≤ ai ≤ 109).

Output

Print n integers — elements of the array a after the operations are applied to it. Print the elements in the order of increasing of their indexes in the array a. Separate the printed numbers by spaces.

Examples

Input

3 1
1 2 3


Output

1 3 6


Input

5 0
3 14 15 92 6


Output

3 14 15 92 6

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
M=10**9+7
I=lambda:map(int,raw_input().split())
n,k=I()
a=I()
l=[1]
for i in range(n):print sum(i*j for i,j in zip(l[::-1],a))%M,;l.append(l[-1]*(i+k)*pow(i+1,M-2,M)%M);
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

MOD = 10**9 + 7

class Cpartialsumsbootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=2000, k_max=10**9, a_max=10**9):
        self.n_min = n_min
        self.n_max = n_max
        self.k_max = k_max
        self.a_max = a_max
    
    def case_generator(self):
        n = random.randint(self.n_min, min(10, self.n_max))  # 测试时限制n范围
        k = random.choices([0, 1, self.k_max], weights=[0.3,0.4,0.3], k=1)[0]
        a = [random.randint(0, self.a_max) for _ in range(n)]
        return {
            'n': n,
            'k': k,
            'input_array': a,
            'expected_output': self._generate_expected_output(n, k, a)
        }
    
    @staticmethod
    def _generate_expected_output(n, k, a):
        l = [1]
        res = []
        for i in range(n):
            res.append(sum(l[j] * a[i-j] % MOD for j in range(i+1)) % MOD)
            next_l = l[-1] * (i + k) % MOD
            inv_denominator = pow(i+1, MOD-2, MOD)
            l.append(next_l * inv_denominator % MOD)
        return res
    
    @staticmethod
    def prompt_func(case):
        return (
            f"给定数组a，执行k次特定变换操作后的结果。每次操作分为两步：\n"
            f"1. 生成前缀和数组s（每个元素模{MOD}）\n"
            f"2. 用s替换原数组a\n"
            f"输入：n={case['n']}, k={case['k']}\n初始数组：{' '.join(map(str, case['input_array']))}\n"
            f"请输出最终数组，答案格式：[answer]结果[/answer]（示例：[answer]1 2 3[/answer]）"
        )
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last = matches[-1].strip().replace(',', ' ')
        try:
            return list(map(int, last.split()))
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected_output']
