"""# 

### 谜题描述
You are given an integer array a_1, a_2, …, a_n.

The array b is called to be a subsequence of a if it is possible to remove some elements from a to get b.

Array b_1, b_2, …, b_k is called to be good if it is not empty and for every i (1 ≤ i ≤ k) b_i is divisible by i.

Find the number of good subsequences in a modulo 10^9 + 7. 

Two subsequences are considered different if index sets of numbers included in them are different. That is, the values ​of the elements ​do not matter in the comparison of subsequences. In particular, the array a has exactly 2^n - 1 different subsequences (excluding an empty subsequence).

Input

The first line contains an integer n (1 ≤ n ≤ 100 000) — the length of the array a.

The next line contains integers a_1, a_2, …, a_n (1 ≤ a_i ≤ 10^6).

Output

Print exactly one integer — the number of good subsequences taken modulo 10^9 + 7.

Examples

Input

2
1 2


Output

3

Input

5
2 2 1 22 14


Output

13

Note

In the first example, all three non-empty possible subsequences are good: \{1\}, \{1, 2\}, \{2\}

In the second example, the possible good subsequences are: \{2\}, \{2, 2\}, \{2, 22\}, \{2, 14\}, \{2\}, \{2, 22\}, \{2, 14\}, \{1\}, \{1, 22\}, \{1, 14\}, \{22\}, \{22, 14\}, \{14\}.

Note, that some subsequences are listed more than once, since they occur in the original array multiple times.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/usr/bin/env python
\"\"\"
This file is part of https://github.com/Cheran-Senthil/PyRival.

Copyright 2018 Cheran Senthilkumar all rights reserved,
Cheran Senthilkumar <hello@cheran.io>
Permission to use, modify, and distribute this software is given under the
terms of the MIT License.

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
# from collections import Counter, MutableSequence, defaultdict, deque
# from copy import deepcopy
# from decimal import Decimal
# from difflib import SequenceMatcher
# from fractions import Fraction
# from heapq import heappop, heappush

if sys.version_info[0] < 3:
    # from cPickle import dumps
    from io import BytesIO as stream
    # from Queue import PriorityQueue, Queue
else:
    # from functools import reduce
    from io import StringIO as stream
    from math import gcd
    # from pickle import dumps
    # from queue import PriorityQueue, Queue


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

    def gcd(x, y):
        \"\"\"gcd(x, y) -> int
        greatest common divisor of x and y
        \"\"\"
        while y:
            x, y = y, x % y
        return x

    input = raw_input
    range = xrange

    filter = itertools.ifilter
    map = itertools.imap
    zip = itertools.izip


def sync_with_stdio(sync=True):
    \"\"\"Set whether the standard Python streams are allowed to buffer their I/O.

    Args:
        sync (bool, optional): The new synchronization setting.

    \"\"\"
    global input, flush

    if sync:
        flush = sys.stdout.flush
    else:
        sys.stdin = stream(sys.stdin.read())
        input = lambda: sys.stdin.readline().rstrip('\r\n')

        sys.stdout = stream()
        register(lambda: sys.__stdout__.write(sys.stdout.getvalue()))


def memodict(f):
    \"\"\" Memoization decorator for a function taking a single argument. \"\"\"
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret
    return memodict().__getitem__


@memodict
def all_factors(n):
    return set(reduce(list.__add__,
                      ([i, n//i] for i in range(1, int(n**0.5) + 1, 2 if n % 2 else 1) if n % i == 0)))


def main():
    n = int(input())
    a = list(map(int, input().split(' ')))

    div_cnt = [0] * (n + 1)
    div_cnt[0] = 1

    res = 0

    for i in a:
        for j in sorted(all_factors(i), reverse=True):
            try:
                res += div_cnt[j - 1]
                if res > 1000000007:
                    res -= 1000000007

                div_cnt[j] += div_cnt[j - 1]
                if div_cnt[j] > 1000000007:
                    div_cnt[j] -= 1000000007
            except:
                pass

    print(res)


if __name__ == '__main__':
    sync_with_stdio(False)
    main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
from collections import defaultdict
from bootcamp import Basebootcamp

MOD = 10**9 + 7

class FactorCache:
    """优化后的因数缓存机制"""
    def __init__(self):
        self.cache = defaultdict(set)
    
    def get_factors(self, n):
        if n not in self.cache:
            factors = set()
            if n > 0:
                max_factor = int(math.isqrt(n))
                step = 2 if n % 2 else 1
                for i in range(1, max_factor + 1, step):
                    if n % i == 0:
                        factors.add(i)
                        factors.add(n//i)
            self.cache[n] = factors
        return self.cache[n]

factor_cache = FactorCache()

def compute_answer(n, a):
    """完全重构的动态规划解法"""
    dp = defaultdict(int)
    dp[0] = 1  # 初始状态：空序列
    total = 0
    
    for num in a:
        factors = sorted(factor_cache.get_factors(num), reverse=True)
        for f in factors:
            if f == 0:
                continue
            prev = f - 1
            if prev in dp:
                contribution = dp[prev]
                total = (total + contribution) % MOD
                dp[f] = (dp[f] + contribution) % MOD
                
    return total

class Cmultiplicitybootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_a=100):
        self.max_n = max_n    # 限制最大数组长度
        self.max_a = max_a    # 限制元素最大值
    
    def case_generator(self):
        import random
        n = random.randint(1, self.max_n)
        a = [random.randint(1, self.max_a) for _ in range(n)]
        return {
            "n": n,
            "a": a,
            "correct_answer": compute_answer(n, a)
        }
    
    @staticmethod
    def prompt_func(question_case):
        a_str = ' '.join(map(str, question_case['a']))
        return f"""请计算以下数组中满足条件的所有非空子序列数量：

条件：子序列的第i个元素（从1开始计数）必须能被i整除

输入数组：
{question_case['n']}
{a_str}

答案请用[answer]标签包裹，例如：[answer]123[/answer]"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return matches[-1] if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return int(solution) == identity['correct_answer']
        except (ValueError, KeyError):
            return False
