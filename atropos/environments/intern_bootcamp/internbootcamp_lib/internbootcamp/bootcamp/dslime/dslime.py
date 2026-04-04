"""# 

### 谜题描述
There are n slimes in a row. Each slime has an integer value (possibly negative or zero) associated with it.

Any slime can eat its adjacent slime (the closest slime to its left or to its right, assuming that this slime exists). 

When a slime with a value x eats a slime with a value y, the eaten slime disappears, and the value of the remaining slime changes to x - y.

The slimes will eat each other until there is only one slime left. 

Find the maximum possible value of the last slime.

Input

The first line of the input contains an integer n (1 ≤ n ≤ 500 000) denoting the number of slimes.

The next line contains n integers a_i (-10^9 ≤ a_i ≤ 10^9), where a_i is the value of i-th slime.

Output

Print an only integer — the maximum possible value of the last slime.

Examples

Input

4
2 1 2 1


Output

4

Input

5
0 -1 -1 -1 -1


Output

4

Note

In the first example, a possible way of getting the last slime with value 4 is:

  * Second slime eats the third slime, the row now contains slimes 2, -1, 1
  * Second slime eats the third slime, the row now contains slimes 2, -2
  * First slime eats the second slime, the row now contains 4 



In the second example, the first slime can keep eating slimes to its right to end up with a value of 4.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = input()
a = map(int, raw_input().strip().split())
if n == 1:
    print a[0]
    exit()
m = min(a)
s = sum(a)
X = s - 2 * m
isn = 0
isp = 0
for num in a:
    if num < 0:
        isn = 1
    if num >= 0:
        isp = 1
    if isn and isp: break

ev = isn and isp
if ev:
    ans = 0
    for num in a:
        ans += abs(num)
    print ans
else:
    if isp: print X
    else:
        print 2 * max(a) - s
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Dslimebootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10, allow_positive=True, allow_negative=True, min_val=-100, max_val=100):
        # 参数有效性验证
        if min_n < 1:
            raise ValueError("min_n must be at least 1")
        if max_n < min_n:
            raise ValueError("max_n must be >= min_n")
        if min_val > max_val:
            raise ValueError("Invalid value range: min_val > max_val")
        if not allow_positive and not allow_negative:
            if not (min_val <= 0 and max_val >= 0):
                raise ValueError("When both signs disabled, value range must include 0")

        self.min_n = min_n
        self.max_n = max_n
        self.allow_positive = allow_positive
        self.allow_negative = allow_negative
        self.min_val = min_val
        self.max_val = max_val

    def case_generator(self):
        # 调整有效取值范围
        effective_min = max(0, self.min_val) if not self.allow_negative else self.min_val
        effective_max = min(0, self.max_val) if not self.allow_positive else self.max_val

        n = random.randint(self.min_n, self.max_n)
        if n == 1:
            return {'n': 1, 'a': [random.randint(effective_min, effective_max)]}

        # 确定可能生成的案例类型
        case_types = []
        
        # 混合类型需要同时支持正负数
        if self.allow_positive and self.allow_negative:
            if effective_min <= -1 and effective_max >= 1:
                case_types.append('mixed')
        
        # 全正数类型要求至少能生成正数
        if self.allow_positive and effective_max >= 1:
            case_types.append('positive')
        
        # 全负数类型要求至少能生成负数
        if self.allow_negative and effective_min <= -1:
            case_types.append('negative')

        # 处理无法生成有效类型的情况
        if not case_types:
            return {'n': n, 'a': [0]*n}

        case_type = random.choice(case_types)
        a = []

        if case_type == 'mixed':
            # 强制生成至少一个正数和负数
            positive_pos = random.randint(0, n-1)
            negative_pos = random.choice([i for i in range(n) if i != positive_pos])
            
            a = [random.randint(effective_min, effective_max) for _ in range(n)]
            a[positive_pos] = abs(a[positive_pos]) + 1  # 确保正数
            a[negative_pos] = -abs(a[negative_pos]) - 1  # 确保负数

        elif case_type == 'positive':
            # 保证至少一个正数
            a = [random.randint(0, effective_max) for _ in range(n)]
            a[random.randint(0, n-1)] = random.randint(1, effective_max)

        elif case_type == 'negative':
            # 全负数
            a = [random.randint(effective_min, -1) for _ in range(n)]

        return {'n': n, 'a': a}

    @staticmethod
    def prompt_func(question_case) -> str:
        n, a = question_case['n'], question_case['a']
        return f"""Solve this slime puzzle where {n} slimes with values {a} merge until one remains. 
The rules are: Each slime can eat adjacent neighbors, merging x and y becomes x-y. 
Find the maximum possible final value. Put your answer in [answer]...[/answer]."""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return int(matches[-1].strip()) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 保持原有验证逻辑不变
        a = identity['a']
        if len(a) == 1:
            return solution == a[0]
        
        total = sum(a)
        if any(x < 0 for x in a) ^ any(x >= 0 for x in a):
            min_val, max_val = min(a), max(a)
            return solution == ((total - 2*min_val) if any(x >=0 for x in a) else (2*max_val - total))
        return solution == sum(abs(x) for x in a)
