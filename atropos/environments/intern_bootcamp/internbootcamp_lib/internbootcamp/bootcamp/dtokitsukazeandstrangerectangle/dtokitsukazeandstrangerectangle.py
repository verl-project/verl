"""# 

### 谜题描述
There are n points on the plane, the i-th of which is at (x_i, y_i). Tokitsukaze wants to draw a strange rectangular area and pick all the points in the area.

The strange area is enclosed by three lines, x = l, y = a and x = r, as its left side, its bottom side and its right side respectively, where l, r and a can be any real numbers satisfying that l < r. The upper side of the area is boundless, which you can regard as a line parallel to the x-axis at infinity. The following figure shows a strange rectangular area.

<image>

A point (x_i, y_i) is in the strange rectangular area if and only if l < x_i < r and y_i > a. For example, in the above figure, p_1 is in the area while p_2 is not.

Tokitsukaze wants to know how many different non-empty sets she can obtain by picking all the points in a strange rectangular area, where we think two sets are different if there exists at least one point in one set of them but not in the other.

Input

The first line contains a single integer n (1 ≤ n ≤ 2 × 10^5) — the number of points on the plane.

The i-th of the next n lines contains two integers x_i, y_i (1 ≤ x_i, y_i ≤ 10^9) — the coordinates of the i-th point.

All points are distinct.

Output

Print a single integer — the number of different non-empty sets of points she can obtain.

Examples

Input

3
1 1
1 2
1 3


Output

3


Input

3
1 1
2 1
3 1


Output

6


Input

4
2 1
2 2
3 1
3 2


Output

6

Note

For the first example, there is exactly one set having k points for k = 1, 2, 3, so the total number is 3.

For the second example, the numbers of sets having k points for k = 1, 2, 3 are 3, 2, 1 respectively, and their sum is 6.

For the third example, as the following figure shows, there are

  * 2 sets having one point; 
  * 3 sets having two points; 
  * 1 set having four points. 



Therefore, the number of different non-empty sets in this example is 2 + 3 + 0 + 1 = 6.

<image>

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

import random
import collections
import math
import itertools
import bisect

class FenwickTree:
    def __init__(self, x):
        \"\"\"transform list into BIT\"\"\"
        self.bit = x
        for i in range(len(x)):
            j = i | (i + 1)
            if j < len(x):
                x[j] += x[i]

    def update(self, idx, x):
        \"\"\"updates bit[idx] += x\"\"\"
        while idx < len(self.bit):
            self.bit[idx] += x
            idx |= idx + 1

    def query(self, end):
        \"\"\"calc sum(bit[:end])\"\"\"
        x = 0
        while end:
            x += self.bit[end - 1]
            end &= end - 1
        return x

    def findkth(self, k):
        \"\"\"Find largest idx such that sum(bit[:idx]) <= k\"\"\"
        idx = -1
        for d in reversed(range(len(self.bit).bit_length())):
            right_idx = idx + (1 << d)
            if right_idx < len(self.bit) and k >= self.bit[right_idx]:
                idx = right_idx
                k -= self.bit[idx]
        return idx + 1


def real_main():
    n = read_int()
    points = []
    X = set()
    for _ in range(n):
        x, y = read_int_array()
        X.add(x)
        points.append((-y, x, y))

    index = {x: i for i, x in enumerate(sorted(X))}
    ft = FenwickTree([0]*len(index))

    order = mergesort_simple(points)
    ans = 0
    seen = set()
    for idx, i in enumerate(order):
        _, x, y = points[i]
        max_xi = len(index) if (idx + 1 == len(order) or points[order[idx+1]][2] != y) else index[points[order[idx+1]][1]]
        xi = index[x]
        less = ft.query(xi)
        more = ft.query(max_xi) - ft.query(xi+1)
        ans += (less + 1) * (more + 1)
        if xi not in seen:
            ft.update(xi, 1)
            seen.add(xi)
    print(ans)


def solve():
    pass


def main():
    if False and 'PYCHARM_HOSTED' in os.environ:
        main_pycharm()
    else:
        real_main()


from copy import deepcopy
def main_pycharm():
    solution = solve

    test_inputs = None
    test_outputs = None
    judge = None
    slow_solution = None
    if solution is not None:
        if test_outputs is not None:
            test_with_answers(solution, test_inputs, test_outputs)
        if judge is not None:
            test_with_judge(solution, test_inputs, judge)
        if slow_solution is not None:
            test_with_slow_solution(solution, test_inputs, slow_solution)


def test_with_answers(solution, inputs, answers):
    total, wrong = 0, 0
    for args, test_ans in zip(inputs, answers):
        ans = solution(*deepcopy(args))
        if ans != test_ans:
            print('WRONG! ans=%s, test_ans=%s, args=%s' % (ans, test_ans, args))
            wrong += 1
        else:
            print('GOOD')
        total += 1
    print('ALL %d TESTS PASSED' % total if not wrong else '%d out of %d tests are WRONG' % (wrong, total))


def test_with_judge(solution, inputs_gen, judge):
    total, wrong = 0, 0
    for args in inputs_gen:
        ans = solution(*deepcopy(args))
        if not judge(deepcopy(ans), *deepcopy(args)):
            print('WRONG! ans=%s, args=%s' % (ans, args))
            wrong += 1
        total += 1
    print('ALL %d TESTS PASSED' % total if not wrong else '%d out of %d tests are WRONG' % (wrong, total))


def test_with_slow_solution(solution, inputs_gen, solution_slow):
    total, wrong = 0, 0
    for args in inputs_gen:
        ans = solution(*deepcopy(args))
        slow = solution_slow(*deepcopy(args))
        if ans != slow:
            print('WRONG! ans=%s, slow=%s, args=%s' % (ans, slow, args))
            wrong += 1
        total += 1
    print('ALL %d TESTS PASSED' % total if not wrong else '%d out of %d tests are WRONG' % (wrong, total))

def generate_nums(n, min, max, check_if_good=None):
    while True:
        nums = [random.randint(min, max) for _ in range(n)]
        if check_if_good is None or check_if_good(nums):
            return nums

# This mergesort can be like 7 times faster than build in sort
# (for stupid reasons)
def mergesort(A, key=lambda x: x, reverse=False):
    C = A
    A = list(range(len(A)))
    B = list(A)

    n = len(A)
    for i in range(0, n - 1, 2):
        if key(C[A[i]]) > key(C[A[i ^ 1]]):
            A[i], A[i ^ 1] = A[i ^ 1], A[i]

    width = 2
    while width < n:
        for i in range(0, n, 2 * width):
            R1, R2 = min(i + width, n), min(i + 2 * width, n)
            j, k = R1, i
            while i < R1 and j < R2:
                if key(C[A[i]]) > key(C[A[j]]):
                    B[k] = A[j]
                    j += 1
                else:
                    B[k] = A[i]
                    i += 1
                k += 1
            while i < R1:
                B[k] = A[i]
                k += 1
                i += 1
            while k < R2:
                B[k] = A[k]
                k += 1
        A, B = B, A
        width *= 2

    if reverse:
        A.reverse()
    return A

def mergesort_simple(A, reverse=False):
    C = A
    A = list(range(len(A)))
    B = list(A)

    n = len(A)
    for i in range(0, n - 1, 2):
        if C[A[i]] > C[A[i ^ 1]]:
            A[i], A[i ^ 1] = A[i ^ 1], A[i]

    width = 2
    while width < n:
        for i in range(0, n, 2 * width):
            R1, R2 = min(i + width, n), min(i + 2 * width, n)
            j, k = R1, i
            while i < R1 and j < R2:
                if C[A[i]] > C[A[j]]:
                    B[k] = A[j]
                    j += 1
                else:
                    B[k] = A[i]
                    i += 1
                k += 1
            while i < R1:
                B[k] = A[i]
                k += 1
                i += 1
            while k < R2:
                B[k] = A[k]
                k += 1
        A, B = B, A
        width *= 2

    if reverse:
        A.reverse()
    return A


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

def read():
    return input()

def read_int():
    return int(input())

def read_array(sep=None, maxsplit=-1):
    return input().split(sep, maxsplit)

def read_int_array():
    return [int(a) for a in read_array()]

# endregion

if __name__ == \"__main__\":
    main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re
from copy import deepcopy

class Dtokitsukazeandstrangerectanglebootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=5, **kwargs):
        super().__init__(**kwargs)
        self.min_n = min_n
        self.max_n = max_n

    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        points = []
        existing = set()
        for _ in range(n):
            while True:
                x = random.randint(1, 100)
                y = random.randint(1, 100)
                if (x, y) not in existing:
                    existing.add((x, y))
                    points.append([x, y])
                    break
        answer = self.calculate_answer(n, deepcopy(points))
        return {
            'n': n,
            'points': points,
            'answer': answer
        }
    
    @staticmethod
    def prompt_func(question_case):
        case = question_case
        points = case['points']
        points_str = '\n'.join([f"{p[0]} {p[1]}" for p in points])
        prompt = (
            "在平面直角坐标系中有n个互不相同的点。Tokitsukaze想要绘制一个特殊的矩形区域，该区域由三条直线x=l、x=r（满足l < r）和y=a围成。"
            "区域的上方是无界的，包含所有满足l < x_i < r且y_i > a的点。不同的参数选择会产生不同的点集合。请计算可以形成的不同的非空点集合的总数。\n\n"
            "输入格式：\n"
            "第一行包含一个整数n（1 ≤ n ≤ 2×10^5），接下来的n行每行包含两个整数x_i和y_i，表示点的坐标。\n\n"
            "当前测试输入为：\n"
            f"{case['n']}\n"
            f"{points_str}\n\n"
            "请输出一个整数作为答案，并将其放置在[answer]和[/answer]标签之间。例如：[answer]42[/answer]。\n"
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['answer']
    
    class FenwickTree:
        def __init__(self, size):
            self.n = size
            self.tree = [0] * (self.n + 2)  # +2防止越界
        
        def update(self, idx):
            idx += 1  # 转换为1-based索引
            while idx <= self.n:
                self.tree[idx] += 1
                idx += idx & -idx
        
        def query(self, idx):
            idx += 1  # 转换为1-based查询
            res = 0
            while idx > 0:
                res += self.tree[idx]
                idx -= idx & -idx
            return res
    
    @classmethod
    def calculate_answer(cls, n, input_points):
        # 按y升序处理（通过-y降序实现）
        sorted_points = sorted([(x, -y) for x, y in input_points], key=lambda x: (x[1], x[0]))
        
        # 离散化x坐标
        x_coords = sorted({x for x, _ in sorted_points})
        x_mapping = {x: i for i, x in enumerate(x_coords)}
        max_x_index = len(x_coords) - 1
        
        ft = cls.FenwickTree(len(x_coords))
        ans = 0
        prev_y = None
        buffer = []
        
        for x, neg_y in sorted_points:
            current_y = neg_y
            if current_y != prev_y:
                # 处理缓冲区的点
                for bx in buffer:
                    ft.update(x_mapping[bx])
                buffer = []
                prev_y = current_y
            
            xi = x_mapping[x]
            left = ft.query(xi - 1)
            # 寻找右边第一个存在的x坐标
            right_count = ft.query(max_x_index) - ft.query(xi)
            ans += (left + 1) * (right_count + 1)
            buffer.append(x)
        
        # 处理最后一批点
        for bx in buffer:
            ft.update(x_mapping[bx])
        
        return ans
