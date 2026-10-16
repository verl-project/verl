"""# 

### 谜题描述
You're given an integer n. For every integer i from 2 to n, assign a positive integer a_i such that the following conditions hold:

  * For any pair of integers (i,j), if i and j are coprime, a_i ≠ a_j. 
  * The maximal value of all a_i should be minimized (that is, as small as possible). 



A pair of integers is called [coprime](https://en.wikipedia.org/wiki/Coprime_integers) if their [greatest common divisor](https://en.wikipedia.org/wiki/Greatest_common_divisor) is 1.

Input

The only line contains the integer n (2 ≤ n ≤ 10^5).

Output

Print n-1 integers, a_2, a_3, …, a_n (1 ≤ a_i ≤ n). 

If there are multiple solutions, print any of them.

Examples

Input


4


Output


1 2 1 

Input


3


Output


2 1

Note

In the first example, notice that 3 and 4 are coprime, so a_3 ≠ a_4. Also, notice that a=[1,2,3] satisfies the first condition, but it's not a correct answer because its maximal value is 3.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
resp = [0]*100001

n = int(raw_input())
count = 1
for i in range(2, n+1):
	if resp[i] == 0:
		for j in range(i, n+1, i):
			resp[j] = count
		count += 1

for i in range(2, n+1):
	print resp[i],
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
from collections import defaultdict
from itertools import combinations
from math import gcd
import re

class Cehabandaspecialcoloringproblembootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=100):
        self.min_n = min_n
        self.max_n = max_n
    
    def case_generator(self):
        import random
        n = random.randint(self.min_n, self.max_n)
        return {'n': n}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        prompt = f"""You are given an integer n = {n}. Your task is to assign a positive integer a_i to each integer i from 2 to n such that the following conditions are met:

1. For every pair of coprime integers (i, j), a_i ≠ a_j.
2. The maximum value among all a_i is as small as possible.

Two integers are coprime if their greatest common divisor (GCD) is 1. Your solution should output the values a_2, a_3, ..., a_n separated by spaces.

Examples:

Input:
4
Output:
1 2 1

Input:
3
Output:
2 1

Your answer should include the sequence of numbers for a_2 to a_n enclosed within [answer] and [/answer] tags. For example, if your solution is "1 2 1" for n=4, write it as [answer]1 2 1[/answer]."""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            solution = list(map(int, last_match.split()))
            return solution
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        if len(solution) != n - 1:
            return False
        
        def count_primes(k):
            if k < 2:
                return 0
            sieve = [True] * (k + 1)
            sieve[0] = sieve[1] = False
            for i in range(2, int(math.sqrt(k)) + 1):
                if sieve[i]:
                    sieve[i*i : k+1 : i] = [False] * len(sieve[i*i : k+1 : i])
            return sum(sieve)
        primes_count = count_primes(n)
        
        max_color = max(solution) if solution else 0
        if max_color != primes_count:
            return False
        
        color_groups = defaultdict(list)
        for idx, color in enumerate(solution):
            i = idx + 2  # i is from 2 to n
            color_groups[color].append(i)
        
        for group in color_groups.values():
            for a, b in combinations(group, 2):
                if gcd(a, b) == 1:
                    return False
        return True
