"""# 

### 谜题描述
One very well-known internet resource site (let's call it X) has come up with a New Year adventure. Specifically, they decided to give ratings to all visitors.

There are n users on the site, for each user we know the rating value he wants to get as a New Year Present. We know that user i wants to get at least ai rating units as a present.

The X site is administered by very creative and thrifty people. On the one hand, they want to give distinct ratings and on the other hand, the total sum of the ratings in the present must be as small as possible.

Help site X cope with the challenging task of rating distribution. Find the optimal distribution.

Input

The first line contains integer n (1 ≤ n ≤ 3·105) — the number of users on the site. The next line contains integer sequence a1, a2, ..., an (1 ≤ ai ≤ 109).

Output

Print a sequence of integers b1, b2, ..., bn. Number bi means that user i gets bi of rating as a present. The printed sequence must meet the problem conditions. 

If there are multiple optimal solutions, print any of them.

Examples

Input

3
5 1 1


Output

5 1 2


Input

1
1000000000


Output

1000000000

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
# -*- coding: utf-8 -*-

import sys

n = int(raw_input())
a = map(int, raw_input().split())
new_a = []
b = [0] * n
for i in xrange(len(a)):
    new_a.append([a[i],i])
sorted_new_a = sorted(new_a, key=lambda x:x[0])
now = 1
for l in sorted_new_a:
    now = max(l[0], now)
    b[l[1]] = now
    now += 1

print ' '.join(map(str, b))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cnewyearratingschangebootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n_min = params.get('n_min', 1)
        self.n_max = params.get('n_max', 100)
        self.a_min = params.get('a_min', 1)
        self.a_max = params.get('a_max', 10**5)
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        a = [random.randint(self.a_min, self.a_max) for _ in range(n)]
        return {'n': n, 'a': a}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = question_case['a']
        problem_text = (
            "你是一个网站的管理员，需要为每个用户分配唯一的评分作为新年礼物。每个用户希望获得至少a_i的评分。所有评分必须唯一且总和最小。\n\n"
            "输入格式：\n"
            "- 第一行是整数n，表示用户数。\n"
            "- 第二行是n个整数a_1 a_2 ... a_n，表示每个用户的最小需求。\n\n"
            "输出格式：\n"
            "- 输出n个整数b_1 b_2 ... b_n，满足b_i ≥ a_i、所有b_i唯一且总和最小。\n\n"
            "示例输入：\n3\n5 1 1\n示例输出：\n5 1 2\n\n"
            f"当前输入：\n{n}\n{' '.join(map(str, a))}\n"
            "请将答案放在[answer]和[/answer]之间，例如：[answer]1 2 3[/answer]"
        )
        return problem_text
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            solution = list(map(int, last_match.split()))
        except ValueError:
            return None
        return solution
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not isinstance(solution, list):
            return False
        n = identity['n']
        a_list = identity['a']
        if len(solution) != n:
            return False
        
        sorted_a = sorted([(a, idx) for idx, a in enumerate(a_list)], key=lambda x: x[0])
        b = [0] * n
        current = 1
        for val, idx in sorted_a:
            current = max(val, current)
            b[idx] = current
            current += 1
        
        return solution == b
