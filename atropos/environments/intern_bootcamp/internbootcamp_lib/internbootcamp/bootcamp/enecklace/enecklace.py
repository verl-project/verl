"""# 

### 谜题描述
Ivan wants to make a necklace as a present to his beloved girl. A necklace is a cyclic sequence of beads of different colors. Ivan says that necklace is beautiful relative to the cut point between two adjacent beads, if the chain of beads remaining after this cut is a palindrome (reads the same forward and backward).

<image>

Ivan has beads of n colors. He wants to make a necklace, such that it's beautiful relative to as many cuts as possible. He certainly wants to use all the beads. Help him to make the most beautiful necklace.

Input

The first line of the input contains a single number n (1 ≤ n ≤ 26) — the number of colors of beads. The second line contains after n positive integers ai — the quantity of beads of i-th color. It is guaranteed that the sum of ai is at least 2 and does not exceed 100 000.

Output

In the first line print a single number — the maximum number of beautiful cuts that a necklace composed from given beads may have. In the second line print any example of such necklace.

Each color of the beads should be represented by the corresponding lowercase English letter (starting with a). As the necklace is cyclic, print it starting from any point.

Examples

Input

3
4 2 1


Output

1
abacaba

Input

1
4


Output

4
aaaa


Input

2
1 1


Output

0
ab

Note

In the first sample a necklace can have at most one beautiful cut. The example of such a necklace is shown on the picture.

In the second sample there is only one way to compose a necklace.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/usr/bin/python

import os
import sys
import itertools

def solve(f):
    s = 'abcdefghijklmnopqrstuvwxyz'

    n = f.read_int()
    a = f.read_int_list()

    e = 0

    if n == 1:
        return str(a[0]) + \"\n\" + 'a'*a[0]

    odd = len([ai for ai in a if ai%2 == 1])
    if odd > 1:
        return \"0\n\" + ''.join([s[i]*ai for i, ai in enumerate(a)])

    g = reduce(lambda x,y: gcd(x,y), a)

    if g % 2 == 0:
        elem = ''.join([s[i]*(ai/g) for i, ai in enumerate(a)])
        return str(g) + \"\n\" + (elem + elem[::-1]) * (g/2)
    else:
        other = ''
        center = ''

        for i, ai in enumerate(a):
            if ai % 2 == 1:
                center = s[i]*(ai/g)
            else:
                other += s[i]*(ai/g/2)

        return str(g) + \"\n\" + (other + center + other[::-1]) * g



def gcd(a, b):
    if b == 0:
        return a
    else:
        return gcd(b, a % b)


class Reader(object):
    def __init__(self, filename=None):
        if filename: self.f = open(filename)
        else: self.f = sys.stdin

    def read_int(self):
        return int(self.f.readline().strip())
    def read_float(self):
        return float(self.f.readline().strip())
    def read_long(self):
        return long(self.f.readline().strip())
    def read_str(self):
        return self.f.readline().strip()

    def read_int_list(self):
        return [int(item) for item in self.f.readline().split()]
    def read_float_list(self):
        return [float(item) for item in self.f.readline().split()]
    def read_long_list(self):
        return [long(item) for item in self.f.readline().split()]
    def read_str_list(self):
        return self.f.readline().split()

if __name__ == '__main__':
    filename = sys.argv[1] if len(sys.argv)>1 else None
    f = Reader(filename)
    print solve(f)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from functools import reduce
from collections import Counter
from bootcamp import Basebootcamp

class Enecklacebootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = {
            'min_n': 1,
            'max_n': 26,
            'min_ai': 1,
            'max_ai': 10,
            'max_total': 100000  # Increased default max_total
        }
        self.params.update(params)
    
    def case_generator(self):
        params = self.params
        min_n, max_n = params['min_n'], params['max_n']
        min_ai, max_ai = params['min_ai'], params['max_ai']
        max_total = params['max_total']

        n = random.randint(min_n, max_n)
        
        # Special case handling for single color
        if n == 1:
            count = random.randint(max(2, min_ai), min(max_ai, max_total))
            return {'n': 1, 'a': [count]}
        
        # Generate valid bead counts
        a = [min_ai] * n
        total = n * min_ai
        remaining = min(max_total - total, max_ai*n - total)
        
        # Distribute remaining beads
        while remaining > 0:
            idx = random.randint(0, n-1)
            available = min(max_ai - a[idx], remaining)
            if available <= 0:
                continue
            add = random.randint(0, available)
            a[idx] += add
            remaining -= add
            total += add
        
        # Ensure minimum total of 2
        if sum(a) < 2:
            a[-1] += 2 - sum(a)
        
        return {'n': n, 'a': a}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = question_case['a']
        return f"""Ivan wants to create the most beautiful necklace possible. The necklace is circular, and a cut is beautiful if the remaining chain is a palindrome. Using {n} colors (a-{chr(ord('a')+n-1)}) with counts {a}, find:

1. Maximum number of beautiful cuts
2. A valid necklace arrangement

Format your answer as:
[answer]
{{max_cuts}}
{{necklace}}
[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            lines = solution.split('\n')
            if len(lines) < 2: return False
            k_user = int(lines[0].strip())
            necklace = lines[1].strip().lower()
        except:
            return False

        # Validate bead counts
        if not cls._check_bead_counts(necklace, identity['n'], identity['a']):
            return False

        # Validate palindrome cuts
        valid_k = cls._calculate_max_cuts(identity['n'], identity['a'])
        return k_user == valid_k and valid_k == cls._count_beautiful_cuts(necklace)

    @classmethod
    def _calculate_max_cuts(cls, n, a):
        # Implementation from reference solution
        if n == 1:
            return a[0]
        
        odd_count = sum(1 for x in a if x % 2)
        if odd_count > 1:
            return 0
        
        g = reduce(lambda x,y: cls._gcd(x,y), a)
        return g

    @staticmethod
    def _gcd(a, b):
        return a if b == 0 else Enecklacebootcamp._gcd(b, a % b)

    @classmethod
    def _check_bead_counts(cls, necklace, n, a):
        counts = Counter(necklace)
        expected = {chr(97+i): cnt for i, cnt in enumerate(a)}
        return counts == expected

    @staticmethod
    def _count_beautiful_cuts(necklace):
        return sum(1 for i in range(len(necklace)) 
                   if (necklace[i+1:]+necklace[:i+1]) == (necklace[i+1:]+necklace[:i+1])[::-1])
