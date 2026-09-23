"""# 

### 谜题描述
A new agent called Killjoy invented a virus COVID-2069 that infects accounts on Codeforces. Each account has a rating, described by an integer (it can possibly be negative or very large).

Killjoy's account is already infected and has a rating equal to x. Its rating is constant. There are n accounts except hers, numbered from 1 to n. The i-th account's initial rating is a_i. Any infected account (initially the only infected account is Killjoy's) instantly infects any uninfected account if their ratings are equal. This can happen at the beginning (before any rating changes) and after each contest. If an account is infected, it can not be healed.

Contests are regularly held on Codeforces. In each contest, any of these n accounts (including infected ones) can participate. Killjoy can't participate. After each contest ratings are changed this way: each participant's rating is changed by an integer, but the sum of all changes must be equal to zero. New ratings can be any integer.

Find out the minimal number of contests needed to infect all accounts. You can choose which accounts will participate in each contest and how the ratings will change.

It can be proven that all accounts can be infected in some finite number of contests.

Input

The first line contains a single integer t (1 ≤ t ≤ 100) — the number of test cases. The next 2t lines contain the descriptions of all test cases.

The first line of each test case contains two integers n and x (2 ≤ n ≤ 10^3, -4000 ≤ x ≤ 4000) — the number of accounts on Codeforces and the rating of Killjoy's account.

The second line of each test case contains n integers a_1, a_2, ..., a_n (-4000 ≤ a_i ≤ 4000) — the ratings of other accounts.

Output

For each test case output the minimal number of contests needed to infect all accounts.

Example

Input


3
2 69
68 70
6 4
4 4 4 4 4 4
9 38
-21 83 50 -59 -77 15 -71 -78 20


Output


1
0
2

Note

In the first test case it's possible to make all ratings equal to 69. First account's rating will increase by 1, and second account's rating will decrease by 1, so the sum of all changes will be equal to zero.

In the second test case all accounts will be instantly infected, because all ratings (including Killjoy's account's rating) are equal to 4.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/usr/bin/env python
from __future__ import division, print_function

import os
import sys
from io import BytesIO, IOBase

if sys.version_info[0] < 3:
    from __builtin__ import xrange as range
    from future_builtins import ascii, filter, hex, map, oct, zip


def main():
    t = int(input())

    for _ in range(t):
        n, x = map(int, input().split())
        a = [int(ai) for ai in input().split()]

        diff = 0
        flag = True
        one = False

        for i in a:
            diff += x - i
            flag = flag and (x == i)
            one = one or (x == i)

        if flag:
            print(0)
        elif diff == 0 or one:
            print(1)
        else:
            print(2)


# region fastio

BUFSIZE = 8192


class FastIO(IOBase):
    newlines = 0

    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = \"x\" in file.mode or \"r\" not in file.mode
        self.write = self.buffer.write if self.writable else None

    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()

    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b\"\n\") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()

    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)


class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode(\"ascii\"))
        self.read = lambda: self.buffer.read().decode(\"ascii\")
        self.readline = lambda: self.buffer.readline().decode(\"ascii\")


def print(*args, **kwargs):
    \"\"\"Prints the values to a stream, or to sys.stdout by default.\"\"\"
    sep, file = kwargs.pop(\"sep\", \" \"), kwargs.pop(\"file\", sys.stdout)
    at_start = True
    for x in args:
        if not at_start:
            file.write(sep)
        file.write(str(x))
        at_start = False
    file.write(kwargs.pop(\"end\", \"\n\"))
    if kwargs.pop(\"flush\", False):
        file.flush()


if sys.version_info[0] < 3:
    sys.stdin, sys.stdout = FastIO(sys.stdin), FastIO(sys.stdout)
else:
    sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)

input = lambda: sys.stdin.readline().rstrip(\"\r\n\")

# endregion

if __name__ == \"__main__\":
    main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Ckilljoybootcamp(Basebootcamp):
    def __init__(self, n_min=2, n_max=10, a_range=(-4000, 4000), x_range=(-4000, 4000)):
        self.n_min = n_min
        self.n_max = n_max
        self.a_range = a_range
        self.x_range = x_range
    
    def case_generator(self):
        ans_type = random.choices([0, 1, 2], weights=[1, 2, 3])[0]
        if ans_type == 0:
            return self._generate_case_0()
        elif ans_type == 1:
            return random.choice([self._generate_case_1_initial_infect, self._generate_case_1_balance_sum])()
        else:
            return self._generate_case_2()
    
    def _generate_case_0(self):
        n = random.randint(self.n_min, self.n_max)
        x = random.randint(*self.x_range)
        return {'n': n, 'x': x, 'a': [x]*n}
    
    def _generate_case_1_initial_infect(self):
        n = random.randint(self.n_min, self.n_max)
        x = random.randint(*self.x_range)
        num_infect = random.randint(1, n-1)
        a = [x]*num_infect
        remaining = n - num_infect
        for _ in range(remaining):
            while True:
                ai = random.randint(*self.a_range)
                if ai != x:
                    a.append(ai)
                    break
        random.shuffle(a)
        return {'n': n, 'x': x, 'a': a}
    
    def _generate_case_1_balance_sum(self):
        for _ in range(100):
            n = random.randint(self.n_min, self.n_max)
            x = random.randint(*self.x_range)
            sum_total = n * x
            a = []
            for _ in range(n-1):
                ai = random.randint(*self.a_range)
                while ai == x:
                    ai = random.randint(*self.a_range)
                a.append(ai)
            last = sum_total - sum(a)
            if last != x and self.a_range[0] <= last <= self.a_range[1]:
                a.append(last)
                return {'n': n, 'x': x, 'a': a}
        return {'n': 2, 'x': 0, 'a': [1, -1]}
    
    def _generate_case_2(self):
        while True:
            n = random.randint(self.n_min, self.n_max)
            x = random.randint(*self.x_range)
            a = [random.randint(*self.a_range) for _ in range(n)]
            sum_total = sum(a)
            has_x = x in a
            if not has_x and sum_total != n * x:
                return {'n': n, 'x': x, 'a': a}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        problem = f"""Killjoy的账户（评分固定为{question_case['x']}）已感染COVID-2069病毒。现有{question_case['n']}个其他账户，初始评分为：{question_case['a']}。

感染规则：
1. 初始时只有Killjoy的账户被感染
2. 评分相同的账户会立即相互感染
3. 已感染账户不可恢复

比赛规则：
- 每场比赛中可以选择任意账户参赛
- 参赛账户评分变化总和必须为0
- 评分可变为任意整数

请计算感染所有账户所需的最少比赛次数，并将答案放入[answer]标签内，例如：[answer]1[/answer]。"""
        return problem
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output, re.IGNORECASE)
        return matches[-1] if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            user_ans = int(solution)
        except ValueError:
            return False
        
        n, x, a = identity['n'], identity['x'], identity['a']
        
        if all(num == x for num in a):
            return user_ans == 0
        
        sum_diff = sum(x - num for num in a)
        has_infected = x in a
        
        if has_infected or sum_diff == 0:
            return user_ans == 1
        
        return user_ans == 2
