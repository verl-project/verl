"""# 

### 谜题描述
You're given an array a of length n. You can perform the following operations on it:

  * choose an index i (1 ≤ i ≤ n), an integer x (0 ≤ x ≤ 10^6), and replace a_j with a_j+x for all (1 ≤ j ≤ i), which means add x to all the elements in the prefix ending at i. 
  * choose an index i (1 ≤ i ≤ n), an integer x (1 ≤ x ≤ 10^6), and replace a_j with a_j \% x for all (1 ≤ j ≤ i), which means replace every element in the prefix ending at i with the remainder after dividing it by x. 



Can you make the array strictly increasing in no more than n+1 operations?

Input

The first line contains an integer n (1 ≤ n ≤ 2000), the number of elements in the array a.

The second line contains n space-separated integers a_1, a_2, ..., a_n (0 ≤ a_i ≤ 10^5), the elements of the array a.

Output

On the first line, print the number of operations you wish to perform. On the next lines, you should print the operations.

To print an adding operation, use the format \"1 i x\"; to print a modding operation, use the format \"2 i x\". If i or x don't satisfy the limitations above, or you use more than n+1 operations, you'll get wrong answer verdict.

Examples

Input

3
1 2 3


Output

0

Input

3
7 6 3


Output

2
1 1 1
2 2 4

Note

In the first sample, the array is already increasing so we don't need any operations.

In the second sample:

In the first step: the array becomes [8,6,3].

In the second step: the array becomes [0,2,3].

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n=input()
l=map(int,raw_input().split())
print n+1 
print 1,n,900000
#arr=[l[i]+900000 for i in xrange(n)]
for i in xrange(n):
    print 2,i+1,l[i]-i-1+900000
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Cehabanda2operationtaskbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.max_n = params.get('max_n', 2000)
        self.a_min = params.get('a_min', 0)
        self.a_max = params.get('a_max', 10**5)
    
    def case_generator(self):
        n = random.randint(1, min(20, self.max_n))
        if random.random() < 0.5:
            # Generate strictly increasing array
            max_initial = self.a_max - (n-1)
            if max_initial < self.a_min:
                max_initial = self.a_min
            current = random.randint(self.a_min, max_initial)
            a = [current]
            for _ in range(n-1):
                current += 1
                a.append(current)
        else:
            # Generate random array with validation
            a = []
            for _ in range(n):
                a.append(random.randint(self.a_min, self.a_max))
            
            # Check for strict increasing
            is_increasing = True
            for i in range(n-1):
                if a[i] >= a[i+1]:
                    is_increasing = False
                    break
            
            # Handle edge case when n=1 or array accidentally becomes increasing
            if is_increasing and n >= 2:
                # Break the increasing property
                a[-1] = a[-2] - 1 if (a[-2] > 0) else a[-2] + 1
        
        return {'n': n, 'a': a}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = question_case['a']
        return f"""You are given an array of {n} integers. Your task is to make it strictly increasing in no more than {n+1} operations. 

Original array: {' '.join(map(str, a))}

Allowed operations:
1. "1 i x" - Add x to the first i elements (0 ≤ x ≤ 1e6, 1 ≤ i ≤ {n})
2. "2 i x" - Take modulo x of the first i elements (1 ≤ x ≤ 1e6, 1 ≤ i ≤ {n})

Output format:
- First line: number of operations m
- Next m lines: operations in the specified format

If the array is already strictly increasing, output 0.

Put your answer within [answer] and [/answer] tags. Example:
[answer]
2
1 2 5
2 3 10
[/answer]
"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            a = identity['a'].copy()
            n = identity['n']
            lines = solution.strip().split('\n')
            if not lines:
                return False
            
            m = int(lines[0])
            if m != len(lines)-1 or m < 0 or m > n+1:
                return False
            
            for line in lines[1:]:
                parts = line.strip().split()
                if len(parts) != 3:
                    return False
                
                try:
                    op = int(parts[0])
                    i = int(parts[1])
                    x = int(parts[2])
                except ValueError:
                    return False
                
                if i < 1 or i > n:
                    return False
                
                if op == 1:
                    if not (0 <= x <= 1e6):
                        return False
                    for j in range(i):
                        a[j] += x
                elif op == 2:
                    if not (1 <= x <= 1e6):
                        return False
                    for j in range(i):
                        a[j] %= x
                else:
                    return False
            
            # Verify strict increasing
            for i in range(n-1):
                if a[i] >= a[i+1]:
                    return False
            return True
        
        except Exception as e:
            return False
