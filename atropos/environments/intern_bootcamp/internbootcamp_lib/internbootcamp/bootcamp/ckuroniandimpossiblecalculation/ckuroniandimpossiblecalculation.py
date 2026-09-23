"""# 

### 谜题描述
To become the king of Codeforces, Kuroni has to solve the following problem.

He is given n numbers a_1, a_2, ..., a_n. Help Kuroni to calculate ∏_{1≤ i<j≤ n} |a_i - a_j|. As result can be very big, output it modulo m.

If you are not familiar with short notation, ∏_{1≤ i<j≤ n} |a_i - a_j| is equal to |a_1 - a_2|⋅|a_1 - a_3|⋅ ... ⋅|a_1 - a_n|⋅|a_2 - a_3|⋅|a_2 - a_4|⋅ ... ⋅|a_2 - a_n| ⋅ ... ⋅ |a_{n-1} - a_n|. In other words, this is the product of |a_i - a_j| for all 1≤ i < j ≤ n.

Input

The first line contains two integers n, m (2≤ n ≤ 2⋅ 10^5, 1≤ m ≤ 1000) — number of numbers and modulo.

The second line contains n integers a_1, a_2, ..., a_n (0 ≤ a_i ≤ 10^9).

Output

Output the single number — ∏_{1≤ i<j≤ n} |a_i - a_j| mod m.

Examples

Input


2 10
8 5


Output


3

Input


3 12
1 4 5


Output


0

Input


3 7
1 4 9


Output


1

Note

In the first sample, |8 - 5| = 3 ≡ 3 mod 10.

In the second sample, |1 - 4|⋅|1 - 5|⋅|4 - 5| = 3⋅ 4 ⋅ 1 = 12 ≡ 0 mod 12.

In the third sample, |1 - 4|⋅|1 - 9|⋅|4 - 9| = 3 ⋅ 8 ⋅ 5 = 120 ≡ 1 mod 7.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from collections import defaultdict as dd, Counter, OrderedDict as od, deque
from itertools import combinations as comb, permutations as perm
from functools import reduce
from math import log, ceil, floor
from fractions import gcd

import os, sys

#sys.stdin = open(\"pb.txt\", \"r\")
raw_input = sys.stdin.readline

int_input = lambda : int(raw_input())
list_input = lambda : list(raw_input().strip())
explode_input = lambda sep :raw_input().split(sep)
format_input = lambda object : map(object, raw_input().split())


n, m = format_input(int)
a = format_input(int)

if n > m:
	print 0
else:
	cum = 1
	n_itr1 = 0
	while cum > 0 and n_itr1 < n:
		for n_itr2 in xrange(n_itr1 + 1, n):
			cum = (cum * abs(a[n_itr1] - a[n_itr2])) % m
		n_itr1 += 1
	print cum
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import defaultdict
from bootcamp import Basebootcamp

class Ckuroniandimpossiblecalculationbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.min_n = params.get('min_n', 2)
        self.max_n = params.get('max_n', 20)
        self.min_m = params.get('min_m', 1)
        self.max_m = params.get('max_m', 1000)
        self.a_min = params.get('a_min', 0)
        self.a_max = params.get('a_max', 10**9)
        self.n_gt_m_prob = params.get('n_gt_m_prob', 0.5)

    def case_generator(self):
        if random.random() < self.n_gt_m_prob:
            # 确保m的取值范围适配当前max_n
            m_upper = min(self.max_m, self.max_n-1)
            if m_upper < self.min_m:
                m = self.max_n  # 当无法满足则转为生成n <= m模式
                n = random.randint(self.min_n, min(self.max_n, m))
            else:
                m = random.randint(self.min_m, m_upper)
                n = random.randint(m+1, self.max_n)
            a = [random.randint(self.a_min, self.a_max) for _ in range(n)]
            return {
                'n': n,
                'm': m,
                'a': a,
                'expected': 0
            }
        else:
            # 生成n <= m的案例
            m = random.randint(self.min_m, self.max_m)
            n = random.randint(self.min_n, min(self.max_n, m))
            mods = defaultdict(int)
            duplicates = False
            a = []
            for _ in range(n):
                val = random.randint(self.a_min, self.a_max)
                mod = val % m
                if mods[mod] >= 1:
                    duplicates = True
                mods[mod] += 1
                a.append(val)
            
            expected = 0 if duplicates else self._safe_compute(n, m, a)
            return {
                'n': n,
                'm': m,
                'a': a,
                'expected': expected
            }

    def _safe_compute(self, n, m, a):
        """安全计算模式（限制n不超过100）"""
        if n > 100:
            return 0
        product = 1
        for i in range(n):
            for j in range(i+1, n):
                product = (product * abs(a[i]-a[j])) % m
        return product

    @staticmethod
    def prompt_func(question_case) -> str:
        input_case = f"{question_case['n']} {question_case['m']}\n{' '.join(map(str, question_case['a']))}"
        return f"""编程竞赛题目：
计算所有两两绝对差值的乘积模m

输入格式：
第一行：n m（2≤n≤2e5，1≤m≤1000）
第二行：a_1 a_2 ... a_n（0≤a_i≤1e9）

示例说明：
当n>m时结果为0，否则计算各对差值的乘积取模

当前题目：
输入：
{input_case}

请将最终答案用[answer]标签包裹，例如：[answer]0[/answer]"""

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
        return solution == identity['expected']
