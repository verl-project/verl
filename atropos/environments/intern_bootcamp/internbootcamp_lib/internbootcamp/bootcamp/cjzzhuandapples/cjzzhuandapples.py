"""# 

### 谜题描述
Jzzhu has picked n apples from his big apple tree. All the apples are numbered from 1 to n. Now he wants to sell them to an apple store. 

Jzzhu will pack his apples into groups and then sell them. Each group must contain two apples, and the greatest common divisor of numbers of the apples in each group must be greater than 1. Of course, each apple can be part of at most one group.

Jzzhu wonders how to get the maximum possible number of groups. Can you help him?

Input

A single integer n (1 ≤ n ≤ 105), the number of the apples.

Output

The first line must contain a single integer m, representing the maximum number of groups he can get. Each of the next m lines must contain two integers — the numbers of apples in the current group.

If there are several optimal answers you can print any of them.

Examples

Input

6


Output

2
6 3
2 4


Input

9


Output

3
9 3
2 4
6 8


Input

2


Output

0

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n=input()
ans=[]
used={i:0 for i in range(1,n+1)}
an=0
for i in range(3,n/2+1,2)+[2]:
    prime=True
    for j in range(3,int(i**0.5)+1,2):
        if i%j==0:
            prime=False
            break
    if prime:
        cur=[i]
        l=1
        for j in range(3,n/i+1):
            if used[j*i]==0:
                cur.append(j*i)
                used[j*i]=1
                l+=1
        if l%2==1 and 2*i<=n:
            cur.append(2*i)
            used[2*i]=1
            l+=1
        for j in range(l/2):
            an+=1
            ans.append((cur[j*2],cur[2*j+1]))
print an
for i in ans:
    print i[0],i[1]
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import random
from bootcamp import Basebootcamp

class Cjzzhuandapplesbootcamp(Basebootcamp):
    def __init__(self, n_min=2, n_max=100):
        self.n_min = max(1, n_min)
        self.n_max = n_max
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        m_correct = self.compute_max_groups(n)
        return {'n': n, 'm_correct': m_correct}
    
    @staticmethod
    def compute_max_groups(n):
        if n < 2:
            return 0
        used = [False] * (n + 1)
        primes = []
        
        # Efficient sieve to find primes up to n//2
        sieve_size = (n // 2) + 1
        sieve = [True] * sieve_size
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.isqrt(sieve_size)) + 1):
            if sieve[i]:
                sieve[i*i::i] = [False] * len(sieve[i*i::i])
        
        # Collect primes in the order: odd primes first, then 2
        primes = [i for i in range(3, sieve_size, 2) if sieve[i]]
        if 2 <= sieve_size:
            primes.append(2)
        
        total_groups = 0
        for prime in primes:
            if prime > n // 2:
                continue
            
            # Collect multiples of prime
            multiples = []
            if not used[prime]:
                multiples.append(prime)
                used[prime] = True
            
            max_multiple = n // prime
            for multiplier in range(3, max_multiple + 1):
                num = prime * multiplier
                if not used[num]:
                    multiples.append(num)
                    used[num] = True
            
            # Handle odd count
            if len(multiples) % 2 != 0:
                candidate = prime * 2
                if candidate <= n and not used[candidate]:
                    multiples.append(candidate)
                    used[candidate] = True
            
            total_groups += len(multiples) // 2
        
        return total_groups
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        return f"""Jzzhu has {n} apples numbered 1-{n}. Group them into pairs where each pair's GCD >1. Find the maximum groups.

Output format:
m
a1 b1
...
am bm

Put your answer between [answer] and [/answer]. Example:

[answer]
2
6 3
2 4
[/answer]"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        
        answer = matches[-1].strip().split('\n')
        try:
            m = int(answer[0])
            pairs = [tuple(map(int, line.split())) for line in answer[1:m+1]]
            if len(pairs) != m:
                return None
            return pairs
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return identity['m_correct'] == 0
        
        n = identity['n']
        expected_m = identity['m_correct']
        
        if len(solution) != expected_m:
            return False
        
        used = set()
        for a, b in solution:
            if a < 1 or b < 1 or a > n or b > n:
                return False
            if math.gcd(a, b) == 1:
                return False
            if a in used or b in used:
                return False
            used.update({a, b})
        
        return True
