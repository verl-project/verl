"""# 

### 谜题描述
You've got array a[1], a[2], ..., a[n], consisting of n integers. Count the number of ways to split all the elements of the array into three contiguous parts so that the sum of elements in each part is the same. 

More formally, you need to find the number of such pairs of indices i, j (2 ≤ i ≤ j ≤ n - 1), that <image>.

Input

The first line contains integer n (1 ≤ n ≤ 5·105), showing how many numbers are in the array. The second line contains n integers a[1], a[2], ..., a[n] (|a[i]| ≤ 109) — the elements of array a.

Output

Print a single integer — the number of ways to split the array into three parts with the same sum.

Examples

Input

5
1 2 3 0 3


Output

2


Input

4
0 1 -1 0


Output

1


Input

2
4 1


Output

0

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def solve(n, a):
    res = 0
    if n < 3: return res

    d = [0]*(n+1)
    for i in range(1, n+1):
        d[i] = d[i-1]+a[i-1]

    if d[n] % 3 != 0: return res
    t = d[n]/3

    rc = [0]*(n+1)
    for i in range(1, n+1):
        if d[n]-d[i-1] == t:
            rc[i] = 1
    for i in range(n-1, 0, -1):
        rc[i] += rc[i+1]

    for i in range(1, n-1):
        if d[i] != t: continue
        res += rc[i+2]

    return res

n = int(raw_input())
a = map(int, raw_input().split())
print solve(n, a)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cnumberofwaysbootcamp(Basebootcamp):
    def __init__(self, max_n=10, element_range=100):
        self.max_n = max_n
        self.element_range = element_range
    
    def case_generator(self):
        while True:
            n = random.randint(1, self.max_n)
            # Generate array with exactly n elements
            a = [random.randint(-self.element_range, self.element_range) for _ in range(n)]
            total = sum(a)
            remainder = total % 3
            
            # Adjust last element to make total divisible by 3
            adjust_range = [
                x for x in range(-self.element_range, self.element_range + 1)
                if (total - a[-1] + x) % 3 == 0
            ]
            
            if adjust_range:
                a[-1] = random.choice(adjust_range)
                return {'n': n, 'a': a}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = ' '.join(map(str, question_case['a']))
        prompt = (
            "You are given an array of integers and need to split it into three contiguous parts with equal sums. "
            "Find the number of valid ways to make such splits.\n\n"
            "**Rules:**\n"
            "1. The array must be split into three contiguous non-empty parts by choosing indices i and j where 2 ≤ i ≤ j ≤ n-1.\n"
            "2. All three parts must have the same sum.\n\n"
            "**Input Format:**\n"
            "- The first line contains an integer n (1 ≤ n ≤ 5*10^5), the array length.\n"
            "- The second line contains n integers a_1 to a_n (|a_i| ≤ 1e9).\n\n"
            "**Output:**\n"
            "A single integer representing the number of valid splits.\n\n"
            "**Example:**\n"
            "Input:\n5\n1 2 3 0 3\nOutput:\n2\n\n"
            "**Current Problem:**\n"
            f"n = {n}\n"
            f"Array: {a}\n\n"
            "Calculate the answer and put your final answer within [answer] and [/answer]."
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
        n = identity['n']
        a = identity['a']
        return solution == cls.solve(n, a)
    
    @staticmethod
    def solve(n, a):
        if n < 3:
            return 0
        total = sum(a)
        if total % 3 != 0:
            return 0
        target = total // 3
        
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + a[i]
        
        suffix_count = [0] * (n + 2)
        for i in range(n, 0, -1):
            suffix_count[i] = suffix_count[i + 1]
            if prefix[n] - prefix[i - 1] == target:
                suffix_count[i] += 1
        
        result = 0
        for i in range(1, n - 1):
            if prefix[i] == target:
                result += suffix_count[i + 2]
        
        return result
