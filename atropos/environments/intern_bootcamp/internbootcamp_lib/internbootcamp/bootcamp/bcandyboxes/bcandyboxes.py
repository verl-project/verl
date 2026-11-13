"""# 

### 谜题描述
There is an old tradition of keeping 4 boxes of candies in the house in Cyberland. The numbers of candies are special if their arithmetic mean, their median and their range are all equal. By definition, for a set {x1, x2, x3, x4} (x1 ≤ x2 ≤ x3 ≤ x4) arithmetic mean is <image>, median is <image> and range is x4 - x1. The arithmetic mean and median are not necessary integer. It is well-known that if those three numbers are same, boxes will create a \"debugging field\" and codes in the field will have no bugs.

For example, 1, 1, 3, 3 is the example of 4 numbers meeting the condition because their mean, median and range are all equal to 2.

Jeff has 4 special boxes of candies. However, something bad has happened! Some of the boxes could have been lost and now there are only n (0 ≤ n ≤ 4) boxes remaining. The i-th remaining box contains ai candies.

Now Jeff wants to know: is there a possible way to find the number of candies of the 4 - n missing boxes, meeting the condition above (the mean, median and range are equal)?

Input

The first line of input contains an only integer n (0 ≤ n ≤ 4).

The next n lines contain integers ai, denoting the number of candies in the i-th box (1 ≤ ai ≤ 500).

Output

In the first output line, print \"YES\" if a solution exists, or print \"NO\" if there is no solution.

If a solution exists, you should output 4 - n more lines, each line containing an integer b, denoting the number of candies in a missing box.

All your numbers b must satisfy inequality 1 ≤ b ≤ 106. It is guaranteed that if there exists a positive integer solution, you can always find such b's meeting the condition. If there are multiple answers, you are allowed to print any of them.

Given numbers ai may follow in any order in the input, not necessary in non-decreasing.

ai may have stood at any positions in the original set, not necessary on lowest n first positions.

Examples

Input

2
1
1


Output

YES
3
3


Input

3
1
1
1


Output

NO


Input

4
1
2
2
3


Output

YES

Note

For the first sample, the numbers of candies in 4 boxes can be 1, 1, 3, 3. The arithmetic mean, the median and the range of them are all 2.

For the second sample, it's impossible to find the missing number of candies.

In the third example no box has been lost and numbers satisfy the condition.

You may output b in any order.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/usr/bin/env python
#-*- coding:utf-8 -*-

import sys
import math
import random
import operator
from fractions import Fraction, gcd
from decimal import Decimal, getcontext
from itertools import product, permutations, combinations
getcontext().prec = 100

MOD = 10**9 + 7
INF = float(\"+inf\")

n = int(raw_input())

arr = [int(raw_input()) for _ in range(n)]


def check(arr):
    x = arr
    v1 = sum(x) / 4.0
    v2 = (x[1] + x[2]) / 2.0
    v3 = x[3] - x[0]
    return v1 == v2 == v3

if len(arr) == 4:
    print \"YES\" if check(sorted(arr)) else \"NO\"
    quit()

for x1 in xrange(1, 501):
    for x2 in xrange(1, 501):
        val = 2 * x1
        x4 = 3 * x1
        x3 = 2 * val - x2
        res = [x1, x2, x3, x4]
        if not (x1 <= x2 <= x3 <= x4):
            continue
        if not check(res):
            continue
        for v in arr:
            if v not in res:
                break
            res.remove(v)
        else:
            print \"YES\"
            # print res
            for c in res:
                print c
            quit()

print \"NO\"
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Bcandyboxesbootcamp(Basebootcamp):
    def __init__(self, solution_prob=0.5, max_x1=166, max_retry=100):
        self.solution_prob = solution_prob
        self.max_x1 = max_x1
        self.max_retry = max_retry
    
    def case_generator(self):
        if random.random() < self.solution_prob:
            # 生成可解案例：四数满足特定结构
            x1 = random.randint(1, self.max_x1)
            x2 = random.randint(x1, 2 * x1)
            x3 = 4 * x1 - x2
            x4 = 3 * x1
            quadruple = sorted([x1, x2, x3, x4])  # 保证有序结构
            n = random.randint(0, 4)
            ai = random.sample(quadruple, n) if n else []
            return {'n': n, 'ai': ai}
        else:
            # 生成不可解案例，保证n与ai长度严格匹配
            n = random.choice([2, 3, 4])
            for _ in range(self.max_retry):
                ai = [random.randint(1, 500) for _ in range(n)]
                if not self._has_solution(n, ai):
                    return {'n': n, 'ai': ai}
            
            # 回退策略：根据n生成已验证的不可解案例
            if n == 2:
                ai = [1, 5]   # 无法形成有效四元组
            elif n == 3:
                ai = [1, 1, 1]  # 三个相同数不可解
            else:
                ai = [1, 2, 3, 4]  # 四数不符合条件
            return {'n': n, 'ai': ai}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        ai = question_case['ai']
        problem = f"""Jeff has 4 special boxes of candies where the arithmetic mean, median, and range must all be equal. Currently, {n} box(es) remain with these candy counts: {', '.join(map(str, ai)) if n else 'None'}.

Determine if the missing boxes can be filled (1-1,000,000 candies each) to satisfy the condition. If possible, write "YES" followed by the missing numbers; otherwise, write "NO".

Format your answer as:
[answer]
YES|NO
[missing numbers (if YES)]
[/answer]"""
        return problem
    
    @staticmethod
    def extract_output(output):
        # 提取最后一个answer块，确保捕获最终答案
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        content = matches[-1].strip()
        
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines or lines[0].upper() not in ('YES', 'NO'):
            return None
        
        result = {'answer': lines[0].upper(), 'missing': []}
        if result['answer'] == 'YES' and len(lines) >= 2:
            for line in lines[1:]:
                if not line.isdigit():
                    return None
                num = int(line)
                if not (1 <= num <= 10**6):
                    return None
                result['missing'].append(num)
        return result
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        answer = solution.get('answer')
        missing = solution.get('missing', [])
        n = identity['n']
        ai = identity['ai']
        
        if answer not in ('YES', 'NO'):
            return False
        
        if answer == 'YES':
            # 验证数量与数值范围
            if len(missing) != 4 - n or any(not (1 <= num <= 10**6) for num in missing):
                return False
            combined = sorted(ai + missing)
            # 检查四数条件
            mean = sum(combined) / 4
            median = (combined[1] + combined[2]) / 2
            range_val = combined[-1] - combined[0]
            return abs(mean - median) < 1e-9 and abs(median - range_val) < 1e-9
        else:
            # 检查确实无解
            return not cls._has_solution(n, ai)
    
    @classmethod
    def _has_solution(cls, n, ai):
        if n == 4:
            sorted_ai = sorted(ai)
            mean = sum(sorted_ai) / 4
            median = (sorted_ai[1] + sorted_ai[2]) / 2
            range_val = sorted_ai[-1] - sorted_ai[0]
            return abs(mean - median) < 1e-9 and abs(median - range_val) < 1e-9
        
        # 遍历所有可能四元组结构
        for x1 in range(1, 167):
            for x2 in range(x1, 2 * x1 + 1):
                x3 = 4 * x1 - x2
                x4 = 3 * x1
                quad = sorted([x1, x2, x3, x4])
                temp = quad.copy()
                try:
                    for num in ai:
                        temp.remove(num)
                except ValueError:
                    continue  # ai包含不在四元组中的数
                if len(temp) == 4 - n:
                    return True
        return False
