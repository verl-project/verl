"""# 

### 谜题描述
Ivan has got an array of n non-negative integers a1, a2, ..., an. Ivan knows that the array is sorted in the non-decreasing order. 

Ivan wrote out integers 2a1, 2a2, ..., 2an on a piece of paper. Now he wonders, what minimum number of integers of form 2b (b ≥ 0) need to be added to the piece of paper so that the sum of all integers written on the paper equalled 2v - 1 for some integer v (v ≥ 0). 

Help Ivan, find the required quantity of numbers.

Input

The first line contains integer n (1 ≤ n ≤ 105). The second input line contains n space-separated integers a1, a2, ..., an (0 ≤ ai ≤ 2·109). It is guaranteed that a1 ≤ a2 ≤ ... ≤ an.

Output

Print a single integer — the answer to the problem.

Examples

Input

4
0 1 1 1


Output

0


Input

1
3


Output

3

Note

In the first sample you do not need to add anything, the sum of numbers already equals 23 - 1 = 7.

In the second sample you need to add numbers 20, 21, 22.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from math import sqrt as s
from collections import *
from fractions import gcd
n=input()
arr=map(int,raw_input().split())
di={}
for i in arr:
    j=i
    if j in di:
        while j in di:
            del di[j]
            j=j+1
    di[j]=1

#print di
print max(di)-len(di)+1
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import re
import random

class Civanandpowersoftwobootcamp(Basebootcamp):
    def __init__(self, **params):
        self.max_n = params.get('max_n', 1000)
        self.min_n = params.get('min_n', 1)
        self.max_a = params.get('max_a', 2 * 10**9)
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        # Generate problem instance with realistic distributions
        a = [random.randint(0, self.max_a) for _ in range(n)]
        a.sort()
        
        # Simulate optimal merging logic
        di = {}
        for num in a:
            j = num
            while j in di:
                del di[j]
                j += 1
            di[j] = 1
        
        # Calculate expected answer
        max_j = max(di.keys()) if di else 0
        len_j = len(di)
        expected = max_j - len_j + 1
        
        return {
            'n': n,
            'a': a,
            'expected': expected
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        a_str = ' '.join(map(str, question_case['a']))
        return f"""Ivan's array of exponents (sorted): {a_str}

Task: Determine the minimum number of distinct 2^b terms to add 
such that total sum equals 2^v -1 for some integer v.

Rules:
1. Added terms must be distinct (b values are unique)
2. Original array is sorted non-decreasing
3. Final sum should be exactly one less than a power of two

Output format: Single integer enclosed in [answer] tags.

Example: For input '3', valid response: [answer]3[/answer]"""

    @staticmethod
    def extract_output(output):
        # Robust extraction with error tolerance
        matches = re.findall(
            r'\[answer\s*\]\s*(\d+)\s*\[\s*/answer\s*\]', 
            output, 
            re.IGNORECASE
        )
        if matches:
            try:
                return int(matches[-1].strip())
            except ValueError:
                return None
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity.get('expected', -1)
