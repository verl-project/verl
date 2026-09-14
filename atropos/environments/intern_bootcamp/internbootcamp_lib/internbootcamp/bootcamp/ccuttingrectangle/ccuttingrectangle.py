"""# 

### 谜题描述
A rectangle with sides A and B is cut into rectangles with cuts parallel to its sides. For example, if p horizontal and q vertical cuts were made, (p + 1) ⋅ (q + 1) rectangles were left after the cutting. After the cutting, rectangles were of n different types. Two rectangles are different if at least one side of one rectangle isn't equal to the corresponding side of the other. Note that the rectangle can't be rotated, this means that rectangles a × b and b × a are considered different if a ≠ b.

For each type of rectangles, lengths of the sides of rectangles are given along with the amount of the rectangles of this type that were left after cutting the initial rectangle.

Calculate the amount of pairs (A; B) such as the given rectangles could be created by cutting the rectangle with sides of lengths A and B. Note that pairs (A; B) and (B; A) are considered different when A ≠ B.

Input

The first line consists of a single integer n (1 ≤ n ≤ 2 ⋅ 10^{5}) — amount of different types of rectangles left after cutting the initial rectangle.

The next n lines each consist of three integers w_{i}, h_{i}, c_{i} (1 ≤ w_{i}, h_{i}, c_{i} ≤ 10^{12}) — the lengths of the sides of the rectangles of this type and the amount of the rectangles of this type.

It is guaranteed that the rectangles of the different types are different.

Output

Output one integer — the answer to the problem.

Examples

Input

1
1 1 9


Output

3


Input

2
2 3 20
2 4 40


Output

6


Input

2
1 2 5
2 3 5


Output

0

Note

In the first sample there are three suitable pairs: (1; 9), (3; 3) and (9; 1).

In the second sample case there are 6 suitable pairs: (2; 220), (4; 110), (8; 55), (10; 44), (20; 22) and (40; 11).

Here the sample of cut for (20; 22).

<image>

The third sample has no suitable pairs.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
# Red Riot Unbreakable!!!!!!!!!!!!!!
from sys import stdin
from fractions import gcd

n = input()
inp = stdin.readlines()

freq = dict()

for line in inp:
    r, c, cnt = map(int, line.strip().split())
    try:
        freq[c].append((r, cnt))
    except:
        freq[c] = [(r, cnt)]

r = []
g = []
cG = 0
size = -1
for ci in freq:
    llist = freq[ci]
    
    if size != -1:
        if len(llist) != size:
            print 0
            exit()
        else:
            llist.sort()
            frac = -1
            for i in xrange(size):
                if r[i] != llist[i][0]:
                    print 0
                    exit()
                else:
                    gx = gcd(g[i], llist[i][1])
                    if frac == -1:
                        frac = (g[i] / gx, llist[i][1] / gx)
                    else:
                        cur = (g[i] / gx, llist[i][1] / gx)
                        if cur != frac:
                            print 0
                            exit()

            for num in llist: cG = gcd(cG, num[1])
    else:
        size = len(llist)
        llist.sort()
        r = [num[0] for num in llist]
        g = [num[1] for num in llist]

    for num in g: cG = gcd(cG, num)

sqcG = int(cG ** 0.5) + 1
ans = dsq = 0
for d in xrange(1, sqcG):
    if cG % d == 0: ans += 2
    dsq += (2 * d - 1)
    if dsq == cG: ans -= 1

print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from math import gcd
from functools import reduce
from collections import defaultdict
from bootcamp import Basebootcamp

class Ccuttingrectanglebootcamp(Basebootcamp):
    def __init__(self, max_row_types=3, max_col_types=3, invalid_case_ratio=0.3):
        self.max_row_types = max_row_types
        self.max_col_types = max_col_types
        self.invalid_case_ratio = invalid_case_ratio
    
    def case_generator(self):
        # Generate valid base parameters
        is_valid = random.random() > self.invalid_case_ratio
        
        # Generate row parameters with GCD
        m = random.randint(1, self.max_row_types)
        row_gcd = random.randint(1, 5)
        row_factors = [random.randint(1, 3) for _ in range(m)]
        row_base = [row_gcd * f for f in row_factors]
        
        # Generate column parameters with GCD
        n = random.randint(1, self.max_col_types)
        col_gcd = random.randint(1, 5) if is_valid else random.randint(2, 6)
        col_factors = self._generate_coprimes(n)  # coprime factors
        col_base = [col_gcd * f for f in col_factors]
        
        # Build rectangles data
        rectangles = []
        for w in row_base:
            for h in col_base:
                rectangles.append({
                    'w': w,
                    'h': h,
                    'c': (sum(row_factors) * sum(col_factors))  # Valid baseline
                })
        
        # Introduce errors for invalid cases
        if not is_valid:
            # Corrupt either row or column base
            if random.choice([True, False]):
                row_base[0] += 1  # Break row consistency
            else:
                col_base[0] += 1  # Break column consistency
        
        # Shuffle and format output
        random.shuffle(rectangles)
        total_gcd = row_gcd * col_gcd if is_valid else 0
        
        return {
            'n': len(rectangles),
            'rectangles': [dict(r) for r in rectangles],  # Ensure serialization
            'correct_answer': self._count_ordered_factor_pairs(total_gcd) if is_valid else 0
        }

    def _generate_coprimes(self, size):
        """Generate list of coprimes with guaranteed success"""
        coprimes = []
        candidates = list(range(1, 10))
        random.shuffle(candidates)
        
        for _ in range(size):
            for num in candidates:
                if all(gcd(num, e) == 1 for e in coprimes):
                    coprimes.append(num)
                    break
        return coprimes

    def _count_ordered_factor_pairs(self, num):
        """Accurate ordered pair counter matching problem requirements"""
        if num == 0:
            return 0
            
        pairs = set()
        for i in range(1, int(num**0.5)+1):
            if num % i == 0:
                pairs.add((i, num//i))
                pairs.add((num//i, i))
        return len(pairs)

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        rects = question_case['rectangles']
        example = "\n".join([f"{r['w']} {r['h']} {r['c']}" for r in rects])
        return f"""Calculate valid (A,B) pairs for rectangle cutting. Enclose your answer in [answer] tags.

Input:
{n}
{example}

[answer]...[/answer]"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity.get('correct_answer', -1)
