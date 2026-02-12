"""# 

### 谜题描述
There are n students numerated from 1 to n. The level of the i-th student is a_i. You need to split the students into stable groups. A group of students is called stable, if in the sorted array of their levels no two neighboring elements differ by more than x.

For example, if x = 4, then the group with levels [1, 10, 8, 4, 4] is stable (because 4 - 1 ≤ x, 4 - 4 ≤ x, 8 - 4 ≤ x, 10 - 8 ≤ x), while the group with levels [2, 10, 10, 7] is not stable (7 - 2 = 5 > x).

Apart from the n given students, teachers can invite at most k additional students with arbitrary levels (at teachers' choice). Find the minimum number of stable groups teachers can form from all students (including the newly invited).

For example, if there are two students with levels 1 and 5; x = 2; and k ≥ 1, then you can invite a new student with level 3 and put all the students in one stable group.

Input

The first line contains three integers n, k, x (1 ≤ n ≤ 200 000, 0 ≤ k ≤ 10^{18}, 1 ≤ x ≤ 10^{18}) — the initial number of students, the number of students you can additionally invite, and the maximum allowed level difference.

The second line contains n integers a_1, a_2, ..., a_n (1 ≤ a_i ≤ 10^{18}) — the students levels.

Output

In the only line print a single integer: the minimum number of stable groups you can split the students into.

Examples

Input


8 2 3
1 1 5 8 12 13 20 22


Output


2

Input


13 0 37
20 20 80 70 70 70 420 5 1 5 1 60 90


Output


3

Note

In the first example you can invite two students with levels 2 and 11. Then you can split the students into two stable groups: 

  1. [1, 1, 2, 5, 8, 11, 12, 13], 
  2. [20, 22]. 



In the second example you are not allowed to invite new students, so you need 3 groups: 

  1. [1, 1, 5, 5, 20, 20] 
  2. [60, 70, 70, 70, 80, 90] 
  3. [420] 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division, print_function

import bisect
import os
import sys
from io import BytesIO, IOBase



def main():
    pass


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


# For getting input from input.txt file
#sys.stdin = open('input.txt', 'r')

# Printing the Output to output.txt file
#sys.stdout = open('output.txt', 'w')



n, k, x = list(map(int, input().rstrip().split()))
list1 = list(map(int, input().rstrip().split()))
list1.sort()
count1 = 0
list2 = []
for i in range(1, n):
    if list1[i]-list1[i-1] > x:
        d = list1[i]-list1[i-1]
        if d%x == 0:
            d = (d//x)-1
        else:
            d = d//x
        list2.append(d)
list2.sort()
count1 = 0
for i in list2:
    if k-i >= 0:
        k = k-i
    else:
        count1 += 1
print(count1+1)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Cstablegroupsbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = {
            'n_min': params.get('n_min', 1),   # 允许生成n=0/1的边界情况
            'n_max': params.get('n_max', 10),
            'x_min': params.get('x_min', 1),
            'x_max': params.get('x_max', 10),
            'k_max': params.get('k_max', 10**18),
            'a_max': params.get('a_max', 10**18),
        }

    def case_generator(self):
        params = self.params
        n = random.randint(params['n_min'], params['n_max'])
        x = random.randint(params['x_min'], params['x_max'])
        
        # 生成混合包含稳定间隔和非稳定间隔的测试用例
        a_sorted = []
        if n > 0:
            a_sorted = [random.randint(1, 100)]
            for _ in range(n-1):
                # 50%概率生成稳定间隔（包括重复值）
                if random.random() < 0.5:
                    delta = random.randint(0, x)
                else:
                    delta = x + random.randint(1, 10)
                a_sorted.append(a_sorted[-1] + delta)
            a_sorted.sort()

        # 计算所有需要填补的间隙
        gaps = []
        for i in range(1, len(a_sorted)):
            d = a_sorted[i] - a_sorted[i-1]
            if d > x:
                req = (d-1) // x  # 等效于 d//x 的向上取整减一
                gaps.append(req)
        gaps.sort()

        # 生成合理的k值（允许k为0或覆盖部分间隙）
        k_val = 0
        if len(gaps) > 0:
            cover = random.randint(0, len(gaps))
            required = sum(gaps[:cover])
            k_val = min(required, params['k_max'])
        else:
            # 无间隙时k仍然可以随机设置（不影响结果）
            k_val = random.randint(0, params['k_max'])

        return {
            'n': n,
            'k': k_val,
            'x': x,
            'a': a_sorted
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        return f"""You need to split students into stable groups. Two rules:
1. A group is stable if adjacent student level differences ≤ {question_case['x']}
2. You can add up to {question_case['k']} students with any levels

Students (sorted): {question_case['a']}
Output the minimal number of groups. Put answer in [answer][/answer]."""

    @staticmethod
    def extract_output(output):
        import re
        answers = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(answers[-1]) if answers else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 处理边界情况（n=0/1时直接验证）
        n = identity['n']
        if n <= 1:
            return solution == 1
        
        # 标准验证逻辑
        gaps = []
        a_sorted = identity['a']
        x = identity['x']
        for i in range(1, len(a_sorted)):
            d = a_sorted[i] - a_sorted[i-1]
            if d > x:
                gaps.append((d-1)//x)  # 等效d//x的向上取整减一
        gaps.sort()
        
        covered = 0
        k = identity['k']
        for g in gaps:
            if k >= g:
                k -= g
                covered += 1
            else:
                break
        
        expected = len(gaps) + 1 - covered
        return solution == expected
