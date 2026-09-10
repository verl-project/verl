"""# 

### 谜题描述
Thanos wants to destroy the avengers base, but he needs to destroy the avengers along with their base.

Let we represent their base with an array, where each position can be occupied by many avengers, but one avenger can occupy only one position. Length of their base is a perfect power of 2. Thanos wants to destroy the base using minimum power. He starts with the whole base and in one step he can do either of following: 

  * if the current length is at least 2, divide the base into 2 equal halves and destroy them separately, or 
  * burn the current base. If it contains no avenger in it, it takes A amount of power, otherwise it takes his B ⋅ n_a ⋅ l amount of power, where n_a is the number of avengers and l is the length of the current base.

Output the minimum power needed by Thanos to destroy the avengers' base.

Input

The first line contains four integers n, k, A and B (1 ≤ n ≤ 30, 1 ≤ k ≤ 10^5, 1 ≤ A,B ≤ 10^4), where 2^n is the length of the base, k is the number of avengers and A and B are the constants explained in the question.

The second line contains k integers a_{1}, a_{2}, a_{3}, …, a_{k} (1 ≤ a_{i} ≤ 2^n), where a_{i} represents the position of avenger in the base.

Output

Output one integer — the minimum power needed to destroy the avengers base.

Examples

Input


2 2 1 2
1 3


Output


6


Input


3 2 1 2
1 7


Output


8

Note

Consider the first example.

One option for Thanos is to burn the whole base 1-4 with power 2 ⋅ 2 ⋅ 4 = 16.

Otherwise he can divide the base into two parts 1-2 and 3-4.

For base 1-2, he can either burn it with power 2 ⋅ 1 ⋅ 2 = 4 or divide it into 2 parts 1-1 and 2-2.

For base 1-1, he can burn it with power 2 ⋅ 1 ⋅ 1 = 2. For 2-2, he can destroy it with power 1, as there are no avengers. So, the total power for destroying 1-2 is 2 + 1 = 3, which is less than 4. 

Similarly, he needs 3 power to destroy 3-4. The total minimum power needed is 6.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin
from bisect import *


def solve(be, en):
    num = bisect_right(a, en) - bisect_left(a, be)

    if num <= 0:
        return A
    elif be == en:
        return B * num

    md = (be + en) >> 1
    be1, be2, en1, en2 = be, md + 1, md, en
    return min(solve(be1, en1) + solve(be2, en2), B * num * (en - be + 1))


rints = lambda: [int(x) for x in stdin.readline().split()]
n, k, A, B = rints()
a = sorted(rints())
print(solve(1, 1 << n))
# print(bisect_left([1, 1, 2, 5], 3), bisect_right([1, 1, 2, 5], 1))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import bisect
from bootcamp import Basebootcamp

class Ccreativesnapbootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=20, k_min=1, k_max=10000, A_min=1, A_max=10000, B_min=1, B_max=10000):
        self.n_min = n_min
        self.n_max = n_max
        self.k_min = k_min
        self.k_max = k_max
        self.A_min = A_min
        self.A_max = A_max
        self.B_min = B_min
        self.B_max = B_max
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        k = random.randint(self.k_min, self.k_max)
        A = random.randint(self.A_min, self.A_max)
        B = random.randint(self.B_min, self.B_max)
        base_length = 1 << n  # Equivalent to 2^n
        positions = [random.randint(1, base_length) for _ in range(k)]
        positions.sort()
        return {
            'n': n,
            'k': k,
            'A': A,
            'B': B,
            'positions': positions
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k = question_case['k']
        A = question_case['A']
        B = question_case['B']
        positions = question_case['positions']
        pos_str = ', '.join(map(str, positions))
        prompt = f"""Ccreativesnap wishes to destroy the Avengers' base, which is structured as an array of length 2^{n}. Each position in the array can hold multiple Avengers, and each Avenger occupies exactly one position. Ccreativesnap aims to accomplish this with minimum power expenditure. He can either split the current segment into two halves (if its length is at least 2) or burn the segment, incurring power costs based on the presence of Avengers.

**Rules for Power Calculation:**
- Burning an empty segment costs {A} units of power.
- Burning a segment with Avengers costs {B} multiplied by the number of Avengers in the segment and the segment's length.

**Input Parameters:**
- n = {n} (the base length is 2^{n})
- k = {k} (number of Avengers)
- A = {A} (empty segment cost)
- B = {B} (cost multiplier for non-empty segments)
- Positions of the Avengers: {pos_str}

**Task:**
Calculate the minimum power required for Ccreativesnap to destroy the entire base. Present your answer as a single integer enclosed within [answer] and [/answer] tags.

For example, if your calculated answer is 42, format it as:
[answer]42[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_answer = matches[-1].strip()
        try:
            return int(last_answer)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        A = identity['A']
        B = identity['B']
        positions = identity['positions']
        a_sorted = sorted(positions)
        total_length = 1 << n  # 2^n
        
        def solve(be, en):
            left = bisect.bisect_left(a_sorted, be)
            right = bisect.bisect_right(a_sorted, en)
            num = right - left
            
            if num == 0:
                return A
            if be == en:
                return B * num  # segment length is 1
            
            mid = (be + en) // 2
            cost_split = solve(be, mid) + solve(mid + 1, en)
            cost_burn = B * num * (en - be + 1)
            return min(cost_split, cost_burn)
        
        correct_answer = solve(1, total_length)
        return solution == correct_answer
