"""# 

### 谜题描述
You are given an array of positive integers a1, a2, ..., an × T of length n × T. We know that for any i > n it is true that ai = ai - n. Find the length of the longest non-decreasing sequence of the given array.

Input

The first line contains two space-separated integers: n, T (1 ≤ n ≤ 100, 1 ≤ T ≤ 107). The second line contains n space-separated integers a1, a2, ..., an (1 ≤ ai ≤ 300).

Output

Print a single number — the length of a sought sequence.

Examples

Input

4 3
3 1 4 2


Output

5

Note

The array given in the sample looks like that: 3, 1, 4, 2, 3, 1, 4, 2, 3, 1, 4, 2. The elements in bold form the largest non-decreasing subsequence. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
input = sys.stdin.readline 

n, t = map(int, input().split())
a = list(map(int, input().split()))

dp = [0 for _ in range(301)]

for i in a * min(t, 2 * n): 
  dp[i] = max(dp[:i + 1]) + 1

max_n = max(dp)


count = [0 for _ in range(301)]
for x in a: 
  count[x] += 1

print(max_n + max((t - n * 2) * max(count), 0))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import Counter
from bootcamp import Basebootcamp

class Bonceagainbootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=100, T_min=1, T_max=10**7, a_min=1, a_max=300):
        self.n_min = max(n_min, 1)  # Ensure n >= 1
        self.n_max = n_max
        self.T_min = max(T_min, 1)  # Ensure T >= 1
        self.T_max = T_max
        self.a_min = a_min
        self.a_max = a_max
    
    def case_generator(self):
        # Generate base array
        n = random.randint(self.n_min, self.n_max)
        T = random.choice([
            1,  # Edge case: minimal T
            2 * n,  # Boundary case
            random.randint(2 * n + 1, self.T_max),  # Large T case
            random.randint(self.T_min, self.T_max)  # General case
        ])
        
        # Generate array with element distribution
        if random.random() < 0.3:
            # Generate uniform array
            val = random.randint(self.a_min, self.a_max)
            a = [val] * n
        else:
            # Generate normal array ensuring max frequency cases
            a = [random.randint(self.a_min, self.a_max) for _ in range(n)]
            if random.random() < 0.2:
                # Create a dominant element
                dominant = random.choice(a)
                a = [dominant if random.random() < 0.7 else x for x in a]
        
        return {'n': n, 'T': T, 'array': a}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        T = question_case['T']
        a = question_case['array']
        return f"""Find the length of the longest non-decreasing subsequence in a concatenated array.

Problem Statement:
- The base array [{', '.join(map(str, a))}] is repeated {T} times
- The subsequence can select elements from any position in the concatenated array
- Subsequence must maintain original order and be non-decreasing

Output Format:
Your answer should be a single integer enclosed in [answer][/answer] tags.

Example:
For input (n=4, T=3) with array [3, 1, 4, 2], the correct answer is:
[answer]5[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        try:
            return int(matches[-1]) if matches else None
        except (IndexError, ValueError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        T = identity['T']
        a = identity['array']
        
        # Phase 1: Calculate first segment
        k = min(T, 2 * n)
        dp = [0] * 301
        for num in a * k:
            dp[num] = max(dp[:num+1]) + 1
        max_segment = max(dp)
        
        # Phase 2: Handle large T cases
        if T > 2 * n:
            counter = Counter(a)
            max_frequency = max(counter.values())
            return solution == max_segment + (T - 2 * n) * max_frequency
        
        return solution == max_segment
