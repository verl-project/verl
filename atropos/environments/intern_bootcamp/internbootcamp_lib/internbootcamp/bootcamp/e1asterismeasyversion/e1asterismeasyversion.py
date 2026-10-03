"""# 

### 谜题描述
This is the easy version of the problem. The difference between versions is the constraints on n and a_i. You can make hacks only if all versions of the problem are solved.

First, Aoi came up with the following idea for the competitive programming problem:

Yuzu is a girl who collecting candies. Originally, she has x candies. There are also n enemies numbered with integers from 1 to n. Enemy i has a_i candies.

Yuzu is going to determine a permutation P. A permutation is an array consisting of n distinct integers from 1 to n in arbitrary order. For example, \{2,3,1,5,4\} is a permutation, but \{1,2,2\} is not a permutation (2 appears twice in the array) and \{1,3,4\} is also not a permutation (because n=3 but there is the number 4 in the array).

After that, she will do n duels with the enemies with the following rules:

  * If Yuzu has equal or more number of candies than enemy P_i, she wins the duel and gets 1 candy. Otherwise, she loses the duel and gets nothing. 
  * The candy which Yuzu gets will be used in the next duels. 



Yuzu wants to win all duels. How many valid permutations P exist?

This problem was easy and wasn't interesting for Akari, who is a friend of Aoi. And Akari made the following problem from the above idea:

Let's define f(x) as the number of valid permutations for the integer x.

You are given n, a and a prime number p ≤ n. Let's call a positive integer x good, if the value f(x) is not divisible by p. Find all good integers x.

Your task is to solve this problem made by Akari.

Input

The first line contains two integers n, p (2 ≤ p ≤ n ≤ 2000). It is guaranteed, that the number p is prime (it has exactly two divisors 1 and p).

The second line contains n integers a_1, a_2, …, a_n (1 ≤ a_i ≤ 2000).

Output

In the first line, print the number of good integers x.

In the second line, output all good integers x in the ascending order.

It is guaranteed that the number of good integers x does not exceed 10^5.

Examples

Input


3 2
3 4 5


Output


1
3


Input


4 3
2 3 5 6


Output


2
3 4


Input


4 3
9 1 1 1


Output


0

Note

In the first test, p=2.

  * If x ≤ 2, there are no valid permutations for Yuzu. So f(x)=0 for all x ≤ 2. The number 0 is divisible by 2, so all integers x ≤ 2 are not good. 
  * If x = 3, \{1,2,3\} is the only valid permutation for Yuzu. So f(3)=1, so the number 3 is good. 
  * If x = 4, \{1,2,3\} , \{1,3,2\} , \{2,1,3\} , \{2,3,1\} are all valid permutations for Yuzu. So f(4)=4, so the number 4 is not good. 
  * If x ≥ 5, all 6 permutations are valid for Yuzu. So f(x)=6 for all x ≥ 5, so all integers x ≥ 5 are not good. 



So, the only good number is 3.

In the third test, for all positive integers x the value f(x) is divisible by p = 3.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
if sys.subversion[0] == \"PyPy\":
    import io, atexit
    sys.stdout = io.BytesIO()
    atexit.register(lambda: sys.__stdout__.write(sys.stdout.getvalue()))
    
    sys.stdin = io.BytesIO(sys.stdin.read())
    input = lambda: sys.stdin.readline().rstrip()

RS = raw_input
RI = lambda x=int: map(x,RS().split())
RN = lambda x=int: x(RS())
''' ...................................................................... '''

def ok(x):
    j=0
    
    for i in xrange(n):
        while j<n and arr[j]<=(x+i):
            j+=1
        cnt = j -i
        if cnt%p==0: return 0
    return 1
        

n,p = RI()
arr = RI()
arr.sort()
m = arr[n-1]
ans = []
for x in xrange(max(0,m-n)+1,m+1):
    if ok(x): ans.append(x)

print len(ans)
print ' '.join(map(str,ans))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class E1asterismeasyversionbootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=10, max_a=2000, **kwargs):
        super().__init__(**kwargs)
        self.min_n = max(2, min_n)  # Ensure minimum n is 2
        self.max_n = max(self.min_n, max_n)
        self.max_a = max_a

    def case_generator(self):
        # Generate n within the specified range
        n = random.randint(self.min_n, self.max_n)
        
        # Function to generate primes up to a given number
        def get_primes_up_to(num):
            sieve = [True] * (num + 1)
            sieve[0] = sieve[1] = False
            for i in range(2, int(num**0.5) + 1):
                if sieve[i]:
                    sieve[i*i : num+1 : i] = [False] * len(sieve[i*i : num+1 : i])
            return [i for i, is_prime in enumerate(sieve) if is_prime]
        
        possible_p = get_primes_up_to(n)
        if not possible_p:
            possible_p = [2]  # Fallback, though this should not happen for n >= 2
        p = random.choice(possible_p)
        
        # Generate and sort the array a
        a = [random.randint(1, self.max_a) for _ in range(n)]
        a.sort()
        
        return {
            'n': n,
            'p': p,
            'a': a
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        p = question_case['p']
        a = question_case['a']
        a_str = ' '.join(map(str, a))
        prompt = f"""Akari's programming problem involves finding 'good' integers x based on Yuzu's candy duel scenario. Yuzu must defeat {n} enemies, each with a certain number of candies. The goal is to determine all x where the number of valid permutations for Yuzu is not divisible by the prime p={p}.

**Problem Details:**
- Yuzu starts with x candies.
- Each enemy i has {a_str} candies (sorted).
- Yuzu wins all duels in a permutation if she has enough candies each time, gaining 1 candy per win.
- Find all x such that the number of valid permutations is not divisible by p.

**Input Constraints:**
- n = {n}, prime p = {p} (p ≤ n)
- Enemy candies: {a_str}

**Output Format:**
1. First line: Number of good integers x.
2. Second line: List of good x values in ascending order.

Example:
If x values are 3 and 4, output:
2
3 4

Place your answer between [answer] and [/answer], following the format exactly."""
        return prompt

    @staticmethod
    def extract_output(output):
        import re
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        last_block = answer_blocks[-1].strip()
        lines = [line.strip() for line in last_block.split('\n') if line.strip()]
        if len(lines) < 2:
            return None
        try:
            count = int(lines[0])
            x_list = list(map(int, lines[1].split()))
            if len(x_list) != count or x_list != sorted(x_list):
                return None
            return x_list
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False
        n = identity['n']
        p = identity['p']
        a = sorted(identity['a'])
        m = a[-1] if n > 0 else 0
        start_x = max(0, m - n) + 1
        end_x = m
        
        correct_x = []
        for x in range(start_x, end_x + 1):
            j = 0
            valid = True
            for i in range(n):
                while j < n and a[j] <= x + i:
                    j += 1
                cnt = j - i
                if cnt % p == 0:
                    valid = False
                    break
            if valid:
                correct_x.append(x)
        
        return solution == correct_x
