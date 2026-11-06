"""# 

### 谜题描述
Limak is a little polar bear. In the snow he found a scroll with the ancient prophecy. Limak doesn't know any ancient languages and thus is unable to understand the prophecy. But he knows digits!

One fragment of the prophecy is a sequence of n digits. The first digit isn't zero. Limak thinks that it's a list of some special years. It's hard to see any commas or spaces, so maybe ancient people didn't use them. Now Limak wonders what years are listed there.

Limak assumes three things:

  * Years are listed in the strictly increasing order; 
  * Every year is a positive integer number; 
  * There are no leading zeros. 



Limak is going to consider all possible ways to split a sequence into numbers (years), satisfying the conditions above. He will do it without any help. However, he asked you to tell him the number of ways to do so. Since this number may be very large, you are only asked to calculate it modulo 109 + 7.

Input

The first line of the input contains a single integer n (1 ≤ n ≤ 5000) — the number of digits.

The second line contains a string of digits and has length equal to n. It's guaranteed that the first digit is not '0'.

Output

Print the number of ways to correctly split the given sequence modulo 109 + 7.

Examples

Input

6
123434


Output

8


Input

8
20152016


Output

4

Note

In the first sample there are 8 ways to split the sequence:

  * \"123434\" = \"123434\" (maybe the given sequence is just one big number) 
  * \"123434\" = \"1\" + \"23434\" 
  * \"123434\" = \"12\" + \"3434\" 
  * \"123434\" = \"123\" + \"434\" 
  * \"123434\" = \"1\" + \"23\" + \"434\" 
  * \"123434\" = \"1\" + \"2\" + \"3434\" 
  * \"123434\" = \"1\" + \"2\" + \"3\" + \"434\" 
  * \"123434\" = \"1\" + \"2\" + \"3\" + \"4\" + \"34\" 



Note that we don't count a split \"123434\" = \"12\" + \"34\" + \"34\" because numbers have to be strictly increasing.

In the second sample there are 4 ways:

  * \"20152016\" = \"20152016\" 
  * \"20152016\" = \"20\" + \"152016\" 
  * \"20152016\" = \"201\" + \"52016\" 
  * \"20152016\" = \"2015\" + \"2016\" 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import itertools
from pprint import pprint

M = 10**9+7
n = int(raw_input())

d = [int(c) for c in raw_input()]

comp = [[0]*(n+1) for i in range(n-1)]

for l in range(1,n):
    num_eq = 0
    for i in range(n-l):
        j = i+l
        if d[i] == d[j]:
            num_eq += 1
            if num_eq >= l:
                #comp[i-l+1][j-l+1] = 0
                num_eq -= 1
            continue
        elif d[i] < d[j]:
            for (i1, j1) in zip(range(i-num_eq,i+1), range(j-num_eq,j+1)):
                comp[i1][j1] = 1
            num_eq = 0
        else:
            num_eq = 0

N = [[0]*(n+1) for _ in range(n+1)]
for j in range(1,n+1):
    N[j][j] = 1

for i in range(1,n):
    if d[i] == 0: continue
    s = 0
    for l in range(1,n+1-i):
        s = (s + N[i][l-1])% M
        #N[i+l][l] = sum(N[i][k] for k in range(min(l,i)))
        if l <= i and comp[i-l][i]:
            N[i+l][l] = (s + N[i][l])% M
        else:
            N[i+l][l] = s

ans = 0
for l in range(1,n+1): ans = (ans + N[n][l])%M
print(ans)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

MOD = 10**9 + 7

def compute_answer(n, digits_str):
    if n == 0:
        return 0
    d = [int(c) for c in digits_str]
    if n == 1:
        return 1
    
    # Initialize comparison matrix
    comp = [[0]*(n+1) for _ in range(n)]
    
    for l in range(1, n):
        equal_count = 0
        for i in range(n - l):
            j = i + l
            if d[i] == d[j]:
                equal_count += 1
                if equal_count >= l:
                    equal_count = l - 1
            else:
                if d[i] < d[j]:
                    # Mark all positions in the equal prefix
                    start = i - equal_count
                    end = i + 1
                    for k in range(start, end):
                        if k >= 0 and j - equal_count + (k - start) < n:
                            comp[k][j - equal_count + (k - start) + 1] = 1
                equal_count = 0
    
    # Dynamic programming table
    dp = [[0]*(n+1) for _ in range(n+1)]
    for j in range(1, n+1):
        dp[j][j] = 1
    
    # Fill DP table
    for i in range(1, n):
        if d[i] == 0:
            continue
        prefix_sum = 0
        for l in range(1, n - i + 1):
            prefix_sum = (prefix_sum + dp[i][l-1]) % MOD
            if l <= i:
                compare_pos = i - l
                if compare_pos >= 0 and comp[compare_pos][i]:
                    dp[i+l][l] = (prefix_sum + dp[i][l]) % MOD
                else:
                    dp[i+l][l] = prefix_sum
            else:
                dp[i+l][l] = prefix_sum
    
    # Calculate final answer
    total = 0
    for l in range(1, n+1):
        total = (total + dp[n][l]) % MOD
    return total

class Dnewyearandancientprophecybootcamp(Basebootcamp):
    def __init__(self, min_length=1, max_length=20):
        self.min_length = min_length
        self.max_length = max_length

    def case_generator(self):
        # Ensure reasonable case generation for computation efficiency
        n = random.randint(self.min_length, min(self.max_length, 100))
        first_digit = random.randint(1, 9)
        rest = ''.join(str(random.randint(0, 9)) for _ in range(n-1))
        digits = str(first_digit) + rest
        
        return {
            'n': n,
            'digits': digits,
            'correct_answer': compute_answer(n, digits)
        }

    @staticmethod
    def prompt_func(question_case):
        return f"""Given a digit sequence of length {question_case['n']}: {question_case['digits']}
        
Task requirements:
1. Split the sequence into strictly increasing integers
2. No leading zeros in any number
3. Calculate the number of valid splits modulo 10^9+7

Examples:
Input: 6\n123434 → Output:8
Input:8\n20152016 → Output:4

Format your final answer within [answer] tags like: [answer]123[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](\d+)\[/answer\]', output, re.I)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
