"""# 

### 谜题描述
You invited n guests to dinner! You plan to arrange one or more circles of chairs. Each chair is going to be either occupied by one guest, or be empty. You can make any number of circles. 

Your guests happen to be a little bit shy, so the i-th guest wants to have a least l_i free chairs to the left of his chair, and at least r_i free chairs to the right. The \"left\" and \"right\" directions are chosen assuming all guests are going to be seated towards the center of the circle. Note that when a guest is the only one in his circle, the l_i chairs to his left and r_i chairs to his right may overlap.

What is smallest total number of chairs you have to use?

Input

First line contains one integer n — number of guests, (1 ⩽ n ⩽ 10^5). 

Next n lines contain n pairs of space-separated integers l_i and r_i (0 ⩽ l_i, r_i ⩽ 10^9).

Output

Output a single integer — the smallest number of chairs you have to use.

Examples

Input

3
1 1
1 1
1 1


Output

6


Input

4
1 2
2 1
3 5
5 3


Output

15


Input

1
5 6


Output

7

Note

In the second sample the only optimal answer is to use two circles: a circle with 5 chairs accomodating guests 1 and 2, and another one with 10 chairs accomodationg guests 3 and 4.

In the third sample, you have only one circle with one person. The guest should have at least five free chairs to his left, and at least six free chairs to his right to the next person, which is in this case the guest herself. So, overall number of chairs should be at least 6+1=7.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
\"\"\"
What I cannot create, I do not understand.

https://github.com/Cheran-Senthil/PyRival
Copyright (c) 2018 Cheran Senthilkumar
\"\"\"
# IMPORTS---------------------------------------------------------------------#
from __future__ import division, print_function

import itertools
import sys
from atexit import register
from io import BytesIO

# import cmath
# import math
# import operator as op
# import random
# from bisect import bisect, bisect_left
# from collections import Counter, defaultdict, deque
# from copy import deepcopy
# from cPickle import dumps
# from decimal import Decimal, getcontext
# from difflib import SequenceMatcher
# from fractions import Fraction, gcd
# from heapq import heappop, heappush
# from Queue import PriorityQueue, Queue


# PYTHON3---------------------------------------------------------------------#
class dict(dict):
    def items(self):
        return dict.iteritems(self)

    def keys(self):
        return dict.iterkeys(self)

    def values(self):
        return dict.itervalues(self)


filter = itertools.ifilter
map = itertools.imap
zip = itertools.izip

input = raw_input
range = xrange


# FASTIO----------------------------------------------------------------------#
# sys.stdout = BytesIO()
# register(lambda: sys.__stdout__.write(sys.stdout.getvalue()))
# sys.stdin = BytesIO(sys.stdin.read())

# input = lambda: sys.stdin.readline().rstrip()
# print = lambda *args: sys.stdout.write(' '.join(str(x) for x in args) + '\n')
# flush = sys.stdout.flush


# SETTINGS--------------------------------------------------------------------#
# getcontext().prec = 100
# sys.setrecursionlimit(100000)


# CONSTANTS-------------------------------------------------------------------#
MOD = 1000000007
INF = float('+inf')

ASCII_LOWERCASE = 'abcdefghijklmnopqrstuvwxyz'
ASCII_UPPERCASE = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


# MAIN------------------------------------------------------------------------#
def main():
    n = int(input())
    l, r = [0] * n, [0] * n
    for i in range(n):
        l[i], r[i] = map(int, input().split(' '))
    l.sort(), r.sort()
    print(sum(max(li, ri) + 1 for li, ri in zip(l, r)))

if __name__ == '__main__':
    main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Dsocialcirclesbootcamp(Basebootcamp):
    def __init__(self, max_guests=10000, min_lr=0, max_lr=10**9):
        """
        参数优化说明:
            max_guests: 默认提升到10^4量级
            min_lr/max_lr: 严格遵循题目约束
        """
        self.max_guests = max_guests
        self.min_lr = min_lr
        self.max_lr = max_lr
    
    def case_generator(self):
        # 提高单人案例概率到30%
        if random.random() < 0.3:
            n = 1
        else:
            n = random.randint(1, self.max_guests)
        
        guests = []
        for _ in range(n):
            # 加强边界条件覆盖
            rand_type = random.random()
            if rand_type < 0.3:  # 完全对称型
                base = random.randint(self.min_lr, self.max_lr)
                l = r = base
            elif rand_type < 0.6:  # 单边极大
                l = random.choice([self.min_lr, self.max_lr])
                r = random.randint(self.min_lr, self.max_lr)
            else:  # 完全随机
                l = random.randint(self.min_lr, self.max_lr)
                r = random.randint(self.min_lr, self.max_lr)
            guests.append([l, r])
        return {"n": n, "guests": guests}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case["n"]
        guests = question_case["guests"]
        input_lines = [f"{n}"] + [f"{li} {ri}" for li, ri in guests]
        input_str = '\n'.join(input_lines)
        
        problem = (
            "## 晚宴椅子安排谜题\n"
            "需要为客人安排环形座位，每个客人要求：\n"
            "- 左侧至少有l_i把空椅子（朝向圆心方向）\n"
            "- 右侧至少有r_i把空椅子（朝向圆心方向）\n\n"
            "**关键规则**：\n"
            "1. 每个环形区域至少有1个客人\n"
            "2. 不同环形区域的椅子不共享\n"
            "3. 单独客人时：左右要求可以重叠\n"
            "   - 例：客人(5,6)需要7把椅子：max(5,6)+1=7\n\n"
            "**输入格式**：\n"
            f"- 第1行：n (1 ≤ n ≤ 1e5)\n"
            f"- 后接{n}行：每行两个整数l_i r_i\n\n"
            "**解答要求**：\n"
            "输出最小总椅子数，将答案放在[answer]标签内\n\n"
            "**当前题目**：\n"
            f"{input_str}\n"
            "[answer]在此填写答案[/answer]"
        )
        return problem
    
    @staticmethod
    def extract_output(output):
        # 支持多种数字格式（包含逗号、空格、汉字数字）
        pattern = r'\[answer\][\s*]*([+-]?[\d, ]+)(?:\[/answer\]|$)'
        matches = re.findall(pattern, output, re.IGNORECASE)
        if not matches:
            return None
        
        try:
            last_match = matches[-1].replace(',', '').replace(' ', '')
            # 处理中文数字
            if '万' in last_match:
                return int(float(last_match.replace('万', '')) * 10000)
            return int(last_match)
        except (ValueError, TypeError):
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            if not isinstance(solution, int):
                sol = int(str(solution).strip())
            else:
                sol = solution
            
            guests = identity['guests']
            l_sorted = sorted(li for li, ri in guests)
            r_sorted = sorted(ri for li, ri in guests)
            correct = sum(max(l, r) + 1 for l, r in zip(l_sorted, r_sorted))
            return sol == correct
        except:
            return False
