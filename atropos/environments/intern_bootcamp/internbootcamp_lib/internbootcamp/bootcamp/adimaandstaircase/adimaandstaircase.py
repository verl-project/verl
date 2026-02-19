"""# 

### 谜题描述
Dima's got a staircase that consists of n stairs. The first stair is at height a1, the second one is at a2, the last one is at an (1 ≤ a1 ≤ a2 ≤ ... ≤ an). 

Dima decided to play with the staircase, so he is throwing rectangular boxes at the staircase from above. The i-th box has width wi and height hi. Dima throws each box vertically down on the first wi stairs of the staircase, that is, the box covers stairs with numbers 1, 2, ..., wi. Each thrown box flies vertically down until at least one of the two following events happen:

  * the bottom of the box touches the top of a stair; 
  * the bottom of the box touches the top of a box, thrown earlier. 



We only consider touching of the horizontal sides of stairs and boxes, at that touching with the corners isn't taken into consideration. Specifically, that implies that a box with width wi cannot touch the stair number wi + 1.

You are given the description of the staircase and the sequence in which Dima threw the boxes at it. For each box, determine how high the bottom of the box after landing will be. Consider a box to fall after the previous one lands.

Input

The first line contains integer n (1 ≤ n ≤ 105) — the number of stairs in the staircase. The second line contains a non-decreasing sequence, consisting of n integers, a1, a2, ..., an (1 ≤ ai ≤ 109; ai ≤ ai + 1).

The next line contains integer m (1 ≤ m ≤ 105) — the number of boxes. Each of the following m lines contains a pair of integers wi, hi (1 ≤ wi ≤ n; 1 ≤ hi ≤ 109) — the size of the i-th thrown box.

The numbers in the lines are separated by spaces.

Output

Print m integers — for each box the height, where the bottom of the box will be after landing. Print the answers for the boxes in the order, in which the boxes are given in the input.

Please, do not use the %lld specifier to read or write 64-bit integers in C++. It is preferred to use the cin, cout streams or the %I64d specifier.

Examples

Input

5
1 2 3 6 6
4
1 1
3 1
1 1
4 3


Output

1
3
4
6


Input

3
1 2 3
2
1 1
3 1


Output

1
3


Input

1
1
5
1 2
1 10
1 10
1 10
1 10


Output

1
3
13
23
33

Note

The first sample are shown on the picture.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
\"\"\"Template for Python Competitive Programmers prepared by Mayank Chaudhary aka chaudhary_19\"\"\"

# to use the print and division function of Python3
from __future__ import division, print_function

\"\"\"value of mod\"\"\"
MOD = 998244353

\"\"\"use resource\"\"\"
# import resource
# resource.setrlimit(resource.RLIMIT_STACK, [0x100000000, resource.RLIM_INFINITY])

\"\"\"for factorial\"\"\"
# fact = [1]
# for i in range(1, 100005):
#     fact.append((fact[-1] * i) % MOD)
# ifact = [0] * 100005
# ifact[100004] = pow(fact[100004], MOD - 2, MOD)
# for i in range(100004, 0, -1):
#     ifact[i - 1] = (i * ifact[i]) % MOD

\"\"\"uncomment next 4 lines while doing recursion based question\"\"\"
# import threading
# threading.stack_size(2**27)
import sys

# sys.setrecursionlimit(10**6)



\"\"\"uncomment modules according to your need\"\"\"
# from bisect import bisect_left, bisect_right, insort
# import itertools
# from math import floor, ceil, sqrt
# import heapq
# from random import randint as rn
# from Queue import Queue as Q
# from collections import Counter, defaultdict

'''
def modinv(n, p):
    return pow(n, p - 2, p)
'''

\"\"\"
def ncr(n, r, p):  # for using this uncomment the lines calculating fact and ifact
    t = ((fact[n]) * ((ifact[r] * ifact[n - r]) % p)) % p
    return t
\"\"\"


def get_ints(): return map(int, sys.stdin.readline().strip().split())
def get_array(): return list(map(int, sys.stdin.readline().strip().split()))
def input(): return sys.stdin.readline().strip()


# def GCD(x, y):
#     while (y):
#         x, y = y, x % y
#     return x
#
# def lcm(x, y):
#     return (x*y)//(GCD(x, y))
#
# def get_xor(n):
#     return [n,1,n+1,0][n%4]


\"\"\"*******************************************************\"\"\"


def main():

    n = int(input())
    Arr = get_array()
    m = int(input())
    l = w = h = 0
    for i in range(m):
        W,H = get_ints()
        l = max(l+h, Arr[W-1])
        w,h = W,H
        print(l)


\"\"\" -------- Python 2 and 3 footer by Pajenegod and c1729 ---------\"\"\"

py2 = round(0.5)
if py2:
    from future_builtins import ascii, filter, hex, map, oct, zip

    range = xrange

import os, sys
from io import IOBase, BytesIO

BUFSIZE = 8192


class FastIO(BytesIO):
    newlines = 0

    def __init__(self, file):
        self._file = file
        self._fd = file.fileno()
        self.writable = \"x\" in file.mode or \"w\" in file.mode
        self.write = super(FastIO, self).write if self.writable else None

    def _fill(self):
        s = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
        self.seek((self.tell(), self.seek(0, 2), super(FastIO, self).write(s))[0])
        return s

    def read(self):
        while self._fill(): pass
        return super(FastIO, self).read()

    def readline(self):
        while self.newlines == 0:
            s = self._fill();
            self.newlines = s.count(b\"\n\") + (not s)
        self.newlines -= 1
        return super(FastIO, self).readline()

    def flush(self):
        if self.writable:
            os.write(self._fd, self.getvalue())
            self.truncate(0), self.seek(0)


class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        if py2:
            self.write = self.buffer.write
            self.read = self.buffer.read
            self.readline = self.buffer.readline
        else:
            self.write = lambda s: self.buffer.write(s.encode('ascii'))
            self.read = lambda: self.buffer.read().decode('ascii')
            self.readline = lambda: self.buffer.readline().decode('ascii')


sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
input = lambda: sys.stdin.readline().rstrip('\r\n')

\"\"\" main function\"\"\"

if __name__ == '__main__':
    main()
# threading.Thread(target=main).start()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Adimaandstaircasebootcamp(Basebootcamp):
    def __init__(self, max_stairs=10, max_boxes=10, max_height=10**9):
        self.max_stairs = max_stairs
        self.max_boxes = max_boxes
        self.max_height = max_height
    
    def case_generator(self):
        n = random.randint(1, self.max_stairs)
        a = [random.randint(1, self.max_height) for _ in range(n)]
        a.sort()
        m = random.randint(1, self.max_boxes)
        boxes = []
        for _ in range(m):
            wi = random.randint(1, n)
            hi = random.randint(1, self.max_height)
            boxes.append([wi, hi])
        return {
            'n': n,
            'a': a,
            'm': m,
            'boxes': boxes
        }
    
    @staticmethod
    def prompt_func(question_case):
        a_str = ' '.join(map(str, question_case['a']))
        boxes = question_case['boxes']
        boxes_str = '\n'.join(f"{w} {h}" for w, h in boxes)
        prompt = f"""Dima有一个由{question_case['n']}个台阶组成的楼梯，各台阶初始高度依次为：{a_str}。接下来抛下{question_case['m']}个盒子，每个盒子的宽度和高度如下：
{boxes_str}

每个盒子的底部会落在前w个台阶上，下落后将停留在当前最高的平台（可能是台阶顶或之前的盒子顶）。请按顺序输出每个盒子落地后的底部高度。

答案应为一个包含{question_case['m']}个整数的列表，每个整数对应一个盒子的结果，按输入顺序排列，每个数占一行，并放置在[answer]和[/answer]标签之间。

例如，输出格式应为：
[answer]
答案1
答案2
...
[/answer]
请确保答案准确无误，并严格遵循上述格式。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        numbers = []
        for line in last_match.split('\n'):
            stripped = line.strip()
            if stripped:
                try:
                    numbers.append(int(stripped))
                except ValueError:
                    continue
        return numbers if numbers else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        a = identity['a']
        boxes = identity['boxes']
        correct = []
        current_l = 0
        previous_h = 0
        for w, h in boxes:
            current_l = max(current_l + previous_h, a[w-1])
            correct.append(current_l)
            previous_h = h
        return solution == correct
