"""# 

### 谜题描述
You are given a sequence of numbers a1, a2, ..., an, and a number m.

Check if it is possible to choose a non-empty subsequence aij such that the sum of numbers in this subsequence is divisible by m.

Input

The first line contains two numbers, n and m (1 ≤ n ≤ 106, 2 ≤ m ≤ 103) — the size of the original sequence and the number such that sum should be divisible by it.

The second line contains n integers a1, a2, ..., an (0 ≤ ai ≤ 109).

Output

In the single line print either \"YES\" (without the quotes) if there exists the sought subsequence, or \"NO\" (without the quotes), if such subsequence doesn't exist.

Examples

Input

3 5
1 2 3


Output

YES


Input

1 6
5


Output

NO


Input

4 6
3 1 1 3


Output

YES


Input

6 6
5 5 5 5 5 5


Output

YES

Note

In the first sample test you can choose numbers 2 and 3, the sum of which is divisible by 5.

In the second sample test the single non-empty subsequence of numbers is a single number 5. Number 5 is not divisible by 6, that is, the sought subsequence doesn't exist.

In the third sample test you need to choose two numbers 3 on the ends.

In the fourth sample test you can take the whole subsequence.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, m = map(int, raw_input().split())
if n > m:
    print \"YES\"
    raise SystemExit()
a = map(int, raw_input().split())
prev = set()
for x in a:
    new = set([x % m])
    for i in prev:
        new.add((i + x) % m)
        new.add(i)
    if 0 in new:
        print \"YES\"
        raise SystemExit()
    prev = new
print \"NO\"
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Bmodulosumbootcamp(Basebootcamp):
    def __init__(self, max_n=100, max_m=1000, max_a=10**9):
        self.max_n = max_n
        self.max_m = max_m
        self.max_a = max_a
    
    def case_generator(self):
        # Determine trivial case (n ≥ m) possibility
        can_generate_trivial = self.max_n >= 2 and self.max_m >= 2
        
        if can_generate_trivial and random.random() < 0.5:
            # Generate trivial case (n ≥ m)
            m = random.randint(2, min(self.max_m, self.max_n))
            n = random.randint(m, self.max_n)
        else:
            # Generate non-trivial case (n < m)
            m = random.randint(2, self.max_m)
            max_n = min(self.max_n, m-1)
            if max_n < 1: max_n = 1
            n = random.randint(1, max_n)
        
        a = [random.randint(0, self.max_a) for _ in range(n)]
        return {'n': n, 'm': m, 'a': a}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        m = question_case['m']
        a = question_case['a']
        a_str = ' '.join(map(str, a))
        return f"""Determine if there exists a non-empty subsequence of the array divisible by {m}. 
The array has {n} elements: {a_str}. 
Answer with [answer]YES[/answer] or [answer]NO[/answer]."""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](YES|NO)\[/answer\]', output, re.IGNORECASE)
        return matches[-1].upper() if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n, m, a = identity['n'], identity['m'], identity['a']
        
        # Apply Pigeonhole Principle correction (n ≥ m)
        if n >= m:
            return solution == "YES"
        
        # Verify using reference algorithm
        prev = set()
        for x in a:
            mod = x % m
            new = {mod}
            for i in prev:
                new.add((i + mod) % m)
            new.update(prev)
            if 0 in new:
                return solution == "YES"
            prev = new
        return solution == "NO"

# Key modifications:
# 1. Corrected trivial case generation to include n ≥ m
# 2. Fixed verification logic to use n ≥ m check
# 3. Improved regex pattern in extract_output
# 4. Added proper set update in verification algorithm
