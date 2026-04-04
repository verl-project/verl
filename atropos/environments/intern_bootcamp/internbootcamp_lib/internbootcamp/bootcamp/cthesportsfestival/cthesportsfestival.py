"""# 

### 谜题描述
The student council is preparing for the relay race at the sports festival.

The council consists of n members. They will run one after the other in the race, the speed of member i is s_i. The discrepancy d_i of the i-th stage is the difference between the maximum and the minimum running speed among the first i members who ran. Formally, if a_i denotes the speed of the i-th member who participated in the race, then d_i = max(a_1, a_2, ..., a_i) - min(a_1, a_2, ..., a_i).

You want to minimize the sum of the discrepancies d_1 + d_2 + ... + d_n. To do this, you are allowed to change the order in which the members run. What is the minimum possible sum that can be achieved?

Input

The first line contains a single integer n (1 ≤ n ≤ 2000) — the number of members of the student council.

The second line contains n integers s_1, s_2, ..., s_n (1 ≤ s_i ≤ 10^9) – the running speeds of the members.

Output

Print a single integer — the minimum possible value of d_1 + d_2 + ... + d_n after choosing the order of the members.

Examples

Input


3
3 1 2


Output


3


Input


1
5


Output


0


Input


6
1 6 3 3 6 3


Output


11


Input


6
104 943872923 6589 889921234 1000000000 69


Output


2833800505

Note

In the first test case, we may choose to make the third member run first, followed by the first member, and finally the second. Thus a_1 = 2, a_2 = 3, and a_3 = 1. We have:

  * d_1 = max(2) - min(2) = 2 - 2 = 0. 
  * d_2 = max(2, 3) - min(2, 3) = 3 - 2 = 1. 
  * d_3 = max(2, 3, 1) - min(2, 3, 1) = 3 - 1 = 2. 



The resulting sum is d_1 + d_2 + d_3 = 0 + 1 + 2 = 3. It can be shown that it is impossible to achieve a smaller value.

In the second test case, the only possible rearrangement gives d_1 = 0, so the minimum possible result is 0.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
n = int(input())
a=[0 for _ in range(n)]
for i in range(n):
   while(True):
       ch = sys.stdin.read(1)
       if(ch >='0' and ch <= '9'):
            a[i]*=10
            a[i]+=ord(ch)-ord('0')
       else:
            break
a.sort()
def myMin(a,b):
    if(a<b): return a
    return b
 
dp = [-1 for r in range(n) for l in range(n)]
for i in range(n):
    dp[i*n+i] = 0
 
for d in range(1, n):
    for l in range(n - d):
        r = l + d
        dp[l*n+r] = a[r] - a[l] + myMin(dp[(l+1)*n+r], dp[l*n+r-1])
 
print(dp[n-1])
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cthesportsfestivalbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=100, s_min=1, s_max=10**9):
        self.min_n = min_n
        self.max_n = max(max_n, min_n)
        self.s_min = s_min
        self.s_max = s_max
    
    def case_generator(self):
        # 多样化生成策略
        case_type = random.random()
        n = random.randint(self.min_n, self.max_n)
        
        if case_type < 0.2:  # 全同元素
            s = [random.randint(self.s_min, self.s_max)] * n
        elif case_type < 0.4:  # 递增序列
            base = random.randint(self.s_min, self.s_max//2)
            s = sorted([base + i*10 for i in range(n)])
        elif case_type < 0.6:  # 递减序列
            base = random.randint(self.s_min + n*10, self.s_max)
            s = sorted([base - i*10 for i in range(n)], reverse=True)
        else:  # 随机序列
            s = [random.randint(self.s_min, self.s_max) for _ in range(n)]
        
        sorted_s = sorted(s)
        
        # DP优化：滚动数组
        dp = [0] * n
        for r in range(n):
            new_dp = [0] * n
            for l in range(r, -1, -1):
                if l == r:
                    new_dp[l] = 0
                else:
                    new_dp[l] = sorted_s[r] - sorted_s[l] + min(dp[l+1], new_dp[l+1])
            dp = new_dp
        
        return {
            'n': n,
            's': s,
            'correct_sum': dp[0] if n > 0 else 0
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        return (
            "作为学生会成员，你需要安排接力赛顺序使差异总和最小。差异d_i为前i人速度极差。\n\n"
            "输入格式：\n"
            f"{question_case['n']}\n{' '.join(map(str, question_case['s']))}\n\n"
            "将答案放入[answer]标签，如：[answer]123[/answer]。仅接受整数。"
        )
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output, re.IGNORECASE|re.MULTILINE)
        return int(matches[-1]) if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_sum']
