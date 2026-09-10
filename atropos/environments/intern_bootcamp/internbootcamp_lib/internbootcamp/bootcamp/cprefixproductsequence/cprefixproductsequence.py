"""# 

### 谜题描述
Consider a sequence [a1, a2, ... , an]. Define its prefix product sequence <image>.

Now given n, find a permutation of [1, 2, ..., n], such that its prefix product sequence is a permutation of [0, 1, ..., n - 1].

Input

The only input line contains an integer n (1 ≤ n ≤ 105).

Output

In the first output line, print \"YES\" if such sequence exists, or print \"NO\" if no such sequence exists.

If any solution exists, you should output n more lines. i-th line contains only an integer ai. The elements of the sequence should be different positive integers no larger than n.

If there are multiple solutions, you are allowed to print any of them.

Examples

Input

7


Output

YES
1
4
3
6
5
2
7


Input

6


Output

NO

Note

For the second sample, there are no valid sequences.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from math import sqrt
n=input()
if n==4 :
    print \"YES\"
    print 1
    print 3
    print 2
    print 4
elif n==1:
    print \"YES\"
    print 1
else :
    prime=[1]*(n+1)
    prime[0]=0
    prime[1]=0
    i=2
    while i<= int(sqrt(n))+1:
        if prime[i]==1:
            for s in range(i*i,n+1,i):
                prime[s]=0
        if prime[n]==0:
            break
        i=i+1
    if prime[n]:
        print \"YES\"
        print 1
        i=2
        while i<n:
            print ((i*pow(i-1,n-2,n))%n)
            i=i+1
        print n
    else:
        print \"NO\"
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from math import isqrt
from bootcamp import Basebootcamp

class Cprefixproductsequencebootcamp(Basebootcamp):
    def __init__(self, max_n=1e5, **kwargs):
        super().__init__(**kwargs)
        self.max_n = int(max_n)
    
    def sieve(self):
        """Optimized sieve with bytearray storage"""
        if self.max_n < 2:
            return []
        sieve = bytearray([1])*(self.max_n+1)
        sieve[0] = sieve[1] = 0
        for i in range(2, isqrt(self.max_n)+1):
            if sieve[i]:
                sieve[i*i : self.max_n+1 : i] = b'\x00'*len(sieve[i*i : self.max_n+1 : i])
        return [i for i, v in enumerate(sieve) if v]

    def case_generator(self):
        # Valid case candidates
        valid = {1, 4}
        primes = set(self.sieve())
        valid.update(primes)
        
        # Generate case type
        candidates = list(range(1, self.max_n+1))
        exist_cases = [n for n in candidates if n in valid]
        non_exist_cases = [n for n in candidates if n not in valid]
        
        # Ensure balanced case generation
        if exist_cases and (random.random() < 0.5 or not non_exist_cases):
            n = random.choice(exist_cases)
            return {'n': n, 'exists': True}
        elif non_exist_cases:
            return {'n': random.choice(non_exist_cases), 'exists': False}
        else:  # Fallback when all cases are valid
            return {'n': 1, 'exists': True}
    
    @staticmethod
    def prompt_func(case):
        n = case['n']
        return f"""Given n={n}, determine if there exists a permutation of 1-{n} where: 
1. All prefix products modulo {n} 
2. Form a permutation of 0-{n-1}

Output format (inside [answer] tags):
[answer]
YES
<p1>
<p2>
...
<p{n}>
[/answer]
OR
[answer]
NO
[/answer]"""

    @staticmethod
    def extract_output(text):
        # Robust extraction with negative lookahead
        pattern = r'\[answer\][\s]*((?!\[answer\]).*?)[\s]*\[/answer\]'
        matches = re.findall(pattern, text, re.DOTALL|re.IGNORECASE)
        if not matches:
            return None
        
        content = matches[-1].strip().upper()
        lines = [l.strip() for l in content.split('\n')]
        
        if lines[0].startswith('NO'):
            return {'answer': 'NO'} if len(lines) == 1 else None
        
        if lines[0].startswith('YES') and len(lines) == int(lines[0][3:].strip() or 0)+1:
            try:
                nums = list(map(int, lines[1:]))
                return {'answer': 'YES', 'sequence': nums}
            except ValueError:
                pass
        return None

    @classmethod
    def _verify_correction(cls, solution, case):
        # Structural validation
        if not solution or case['exists'] != (solution.get('answer') == 'YES'):
            return False
        
        if solution['answer'] == 'NO':
            return True
        
        # Numerical validation
        n = case['n']
        seq = solution.get('sequence', [])
        if sorted(seq) != list(range(1, n+1)):
            return False
        
        # Prefix product verification
        seen = set()
        product = 1
        for num in seq:
            product = (product * num) % n
            if product in seen:
                return False
            seen.add(product)
        return seen == set(range(n))
