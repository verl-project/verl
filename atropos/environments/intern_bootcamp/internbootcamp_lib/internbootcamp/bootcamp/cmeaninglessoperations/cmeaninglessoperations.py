"""# 

### 谜题描述
Can the greatest common divisor and bitwise operations have anything in common? It is time to answer this question.

Suppose you are given a positive integer a. You want to choose some integer b from 1 to a - 1 inclusive in such a way that the [greatest common divisor (GCD)](https://en.wikipedia.org/wiki/Greatest_common_divisor) of integers a ⊕ b and a \> \& \> b is as large as possible. In other words, you'd like to compute the following function:

$$$f(a) = max_{0 < b < a}{gcd(a ⊕ b, a \> \& \> b)}.$$$

Here ⊕ denotes the [bitwise XOR operation](https://en.wikipedia.org/wiki/Bitwise_operation#XOR), and \& denotes the [bitwise AND operation](https://en.wikipedia.org/wiki/Bitwise_operation#AND).

The greatest common divisor of two integers x and y is the largest integer g such that both x and y are divided by g without remainder.

You are given q integers a_1, a_2, …, a_q. For each of these integers compute the largest possible value of the greatest common divisor (when b is chosen optimally). 

Input

The first line contains an integer q (1 ≤ q ≤ 10^3) — the number of integers you need to compute the answer for.

After that q integers are given, one per line: a_1, a_2, …, a_q (2 ≤ a_i ≤ 2^{25} - 1) — the integers you need to compute the answer for. 

Output

For each integer, print the answer in the same order as the integers are given in input.

Example

Input


3
2
3
5


Output


3
1
7

Note

For the first integer the optimal choice is b = 1, then a ⊕ b = 3, a \> \& \> b = 0, and the greatest common divisor of 3 and 0 is 3.

For the second integer one optimal choice is b = 2, then a ⊕ b = 1, a \> \& \> b = 2, and the greatest common divisor of 1 and 2 is 1.

For the third integer the optimal choice is b = 2, then a ⊕ b = 7, a \> \& \> b = 0, and the greatest common divisor of 7 and 0 is 7.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/usr/bin/env python
\"\"\"
This file is part of https://github.com/Cheran-Senthil/PyRival.
Copyright 2019 Cheran Senthilkumar <hello@cheran.io>

\"\"\"
from __future__ import division, print_function

import cmath
import itertools
import math
import operator as op
# import random
import sys
from atexit import register
from bisect import bisect_left, bisect_right
# from collections import Counter, defaultdict, deque
# from copy import deepcopy
# from decimal import Decimal
# from difflib import SequenceMatcher
# from functools import reduce
# from heapq import heappop, heappush
from io import BytesIO, FileIO, StringIO


if sys.version_info[0] < 3:
    class dict(dict):
        \"\"\"dict() -> new empty dictionary\"\"\"
        def items(self):
            \"\"\"D.items() -> a set-like object providing a view on D's items\"\"\"
            return dict.iteritems(self)

        def keys(self):
            \"\"\"D.keys() -> a set-like object providing a view on D's keys\"\"\"
            return dict.iterkeys(self)

        def values(self):
            \"\"\"D.values() -> an object providing a view on D's values\"\"\"
            return dict.itervalues(self)

    input = raw_input
    range = xrange

    filter = itertools.ifilter
    map = itertools.imap
    zip = itertools.izip


INP_FILE = 0
OUT_FILE = 1

if sys.version_info[0] < 3:
    sys.stdin = BytesIO(FileIO(INP_FILE).read())
    sys.stdout = BytesIO()
    register(lambda: FileIO(OUT_FILE, 'w').write(sys.stdout.getvalue()))
else:
    sys.stdin = StringIO(FileIO(INP_FILE).read().decode())
    sys.stdout = StringIO()
    register(lambda: FileIO(OUT_FILE, 'w').write(sys.stdout.getvalue().encode()))

input = lambda: sys.stdin.readline().rstrip('\r\n')


def main():
    q = int(input())

    n = [1, 1, 5, 1, 21, 1, 85, 73, 341, 89, 1365, 1, 5461, 4681, 21845, 1, 87381, 1, 349525, 299593, 1398101, 178481, 5592405, 1082401, 22369621]

    for _ in range(q):
        a = int(input())
        m = ((1 << a.bit_length()) - 1)

        if m == a:
            print(n[a.bit_length() - 2])
        else:
            print(m)


if __name__ == '__main__':
    main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Cmeaninglessoperationsbootcamp(Basebootcamp):
    n = [
        1, 1, 5, 1, 21, 1, 85, 73, 341, 89, 1365, 1, 5461, 4681, 21845,
        1, 87381, 1, 349525, 299593, 1398101, 178481, 5592405, 1082401, 22369621
    ]

    def __init__(self, all_ones_prob=0.5, min_k=2, max_k=25):
        """
        Parameters:
            all_ones_prob: 生成全1二进制数的概率 (0-1)
            min_k: 最小二进制位数范围 (2-25)
            max_k: 最大二进制位数范围 (2-25)
        """
        self.all_ones_prob = min(max(all_ones_prob, 0.0), 1.0)
        self.min_k = max(2, min_k)
        self.max_k = min(25, max_k)

    def case_generator(self):
        """生成保证有效的测试用例"""
        if random.random() < self.all_ones_prob:
            # 生成全1二进制数 (格式: 2^k - 1)
            k = random.randint(self.min_k, self.max_k)
            return {"a": (1 << k) - 1}
        else:
            # 生成非全1数，确保至少有一个0位
            min_valid_k = max(self.min_k, 3)  # 保证k>=3才有非全1解
            k = random.randint(min_valid_k, self.max_k)
            
            # 生成有效范围内数字
            a_min = (1 << (k-1)) + 1
            a_max = (1 << k) - 2
            a = random.randint(a_min, a_max)
            
            # 二次验证数字有效性
            while bin(a).count('0') == 0 or a.bit_length() != k:
                a = random.randint(a_min, a_max)
            
            return {"a": a}

    @staticmethod
    def prompt_func(question_case) -> str:
        a = question_case["a"]
        return f"""你需要解决一个基于位运算和最大公约数（GCD）的数学谜题。

给定正整数 a={a}，请找到一个整数 b（1 ≤ b < a），使得以下两个值的GCD最大：
1. a XOR b（按位异或）
2. a AND b（按位与）

请通过以下步骤解决：
1. 分析不同b值对应的计算结果
2. 找出使GCD最大的最优b值
3. 计算并返回最大GCD值

示例：
当a=5时，选择b=2：
- XOR: 5 ^ 2 = 7
- AND: 5 & 2 = 0
- GCD(7, 0) = 7

请将最终答案放在[answer]和[/answer]标签之间，例如：
[answer]7[/answer]"""

    @staticmethod
    def extract_output(output):
        # 提取最后一个符合格式的答案
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        a = identity["a"]
        bit_len = a.bit_length()
        
        # 全1二进制数验证逻辑
        if (a & (a + 1)) == 0:
            # 参考数组索引计算
            correct_index = bit_len - 2
            if 0 <= correct_index < len(cls.n):
                correct = cls.n[correct_index]
            else:
                # 超出数组范围时使用数学公式计算
                correct = (1 << (bit_len - 1)) - 1
        else:
            # 非全1数逻辑
            correct = (1 << bit_len) - 1
        
        try:
            return int(solution) == correct
        except ValueError:
            return False
