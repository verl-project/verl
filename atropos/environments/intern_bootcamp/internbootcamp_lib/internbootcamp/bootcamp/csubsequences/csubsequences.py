"""# 

### 谜题描述
For the given sequence with n different elements find the number of increasing subsequences with k + 1 elements. It is guaranteed that the answer is not greater than 8·1018.

Input

First line contain two integer values n and k (1 ≤ n ≤ 105, 0 ≤ k ≤ 10) — the length of sequence and the number of elements in increasing subsequences.

Next n lines contains one integer ai (1 ≤ ai ≤ n) each — elements of sequence. All values ai are different.

Output

Print one integer — the answer to the problem.

Examples

Input

5 2
1
2
3
5
4


Output

7

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/usr/bin/env python2
\"\"\"
This file is part of https://github.com/cheran-senthil/PyRival
Copyright 2019 Cheran Senthilkumar <hello@cheran.io>

\"\"\"
from __future__ import division, print_function

import math
import itertools
import operator as op
import os
import sys
from atexit import register
from io import BytesIO

range = xrange

filter = itertools.ifilter
map = itertools.imap
zip = itertools.izip

sys.stdin = BytesIO(os.read(0, os.fstat(0).st_size))
sys.stdout = BytesIO()
register(lambda: os.write(1, sys.stdout.getvalue()))

input = lambda: sys.stdin.readline().rstrip('\r\n')


class FenwickTree:
    def __init__(self, n):
        self.s = [0] * n

    def update(self, pos, dif):
        while pos < len(self.s):
            self.s[pos] += dif
            pos |= pos + 1

    def query(self, pos):
        res = 0
        while pos > 0:
            res += self.s[pos - 1]
            pos &= pos - 1
        return res

    def lower_bound(self, val):
        if val <= 0:
            return -1

        pos = 0

        pw = 1 << len(self.s).bit_length()
        while pw != 0:
            if pw + pos <= len(self.s):
                if self.s[pos + pw - 1] < val:
                    pos += pw
                    val -= self.s[pos - 1]
            pw >>= 1

        return pos


def main():
    n, k = map(int, input().split())
    a = [int(input()) for _ in range(n)]

    pref_trees = [FenwickTree(n + 1) for _ in range(k + 1)]

    res = [1] * n
    for i, ai in enumerate(a):
        pref_trees[0].update(ai, 1)
        for j in range(1, k + 1):
            res[i] = pref_trees[j - 1].query(ai)
            pref_trees[j].update(ai, res[i])

    print(sum(res))


if __name__ == '__main__':
    main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Csubsequencesbootcamp(Basebootcamp):
    class FenwickTree:
        def __init__(self, size):
            self.size = size
            self.tree = [0] * (size + 1)  # 索引范围1~size
        
        def update(self, index, value):
            while index <= self.size:
                self.tree[index] += value
                index += index & -index
        
        def query(self, index):
            res = 0
            while index > 0:
                res += self.tree[index]
                index -= index & -index
            return res

    def __init__(self, **params):
        self.n = params.get('n', 5)
        self.k = params.get('k', 2)
        super().__init__(**params)

    @classmethod
    def calculate_solution(cls, n, k, sequence):
        if k == 0:
            return n
        
        ft_list = [cls.FenwickTree(n) for _ in range(k+1)]
        result = 0
        
        for num in sequence:
            for j in range(1, k+1):
                if j == 1:
                    prev = ft_list[j-1].query(num-1)
                else:
                    prev = ft_list[j-1].query(num-1)
                
                if j == k:
                    result += prev
                ft_list[j].update(num, prev)
            ft_list[0].update(num, 1)
        
        return result

    def case_generator(self):
        sequence = random.sample(range(1, self.n+1), self.n)
        return {
            'n': self.n,
            'k': self.k,
            'sequence': sequence,
            'correct_answer': self.calculate_solution(self.n, self.k, sequence)
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k = question_case['k']
        seq = '\n'.join(map(str, question_case['sequence']))
        return f"""Given a sequence of {n} distinct integers, find the number of strictly increasing subsequences with exactly {k+1} elements. 

Input format:
First line contains n and k: {n} {k}
Following {n} lines contain the sequence: 
{seq}

Rules:
1. A subsequence must maintain original element order
2. Elements must be strictly increasing
3. Count all possible valid subsequences

Provide your final answer as an integer within [answer] and [/answer] tags."""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity.get('correct_answer', None)
