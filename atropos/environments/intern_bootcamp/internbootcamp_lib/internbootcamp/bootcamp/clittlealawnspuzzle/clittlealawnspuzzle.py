"""# 

### 谜题描述
When he's not training for IOI, Little Alawn enjoys playing with puzzles of various types to stimulate his brain. Today, he's playing with a puzzle that consists of a 2 × n grid where each row is a permutation of the numbers 1,2,3,…,n.

The goal of Little Alawn's puzzle is to make sure no numbers on the same column or row are the same (we'll call this state of the puzzle as solved), and to achieve this he is able to swap the numbers in any column. However, after solving the puzzle many times, Little Alawn got bored and began wondering about the number of possible solved configurations of the puzzle he could achieve from an initial solved configuration only by swapping numbers in a column.

Unfortunately, Little Alawn got stuck while trying to solve this harder problem, so he was wondering if you could help him with it. Find the answer modulo 10^9+7.

Input

Each test contains multiple test cases. The first line contains the number of test cases t (1 ≤ t ≤ 10^4). Description of the test cases follows.

The first line of each test case contains a single integer n (2 ≤ n ≤ 4 ⋅ 10^5).

The next two lines of each test case describe the initial state of the puzzle grid. Each line will be a permutation of the numbers 1,2,3,…,n and the numbers in each column and row will be pairwise distinct.

It is guaranteed that the sum of n over all test cases does not exceed 4 ⋅ 10^5.

Output

For each test case output a single integer, the number of possible solved configurations of the puzzle Little Alawn can achieve from an initial solved configuration only by swapping numbers in a column. As the answer can be very large, please output it modulo 10^9+7.

The answer for each test case should be on a separate line.

Example

Input


2
4
1 4 2 3
3 2 1 4
8
2 6 5 1 4 3 7 8
3 8 7 5 1 2 4 6


Output


2
8

Note

The two possible puzzle configurations for example 1 are:

  * [1,4,2,3] in the first row and [3,2,1,4] in the second; 
  * [3,2,1,4] in the first row and [1,4,2,3] in the second. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division,print_function
from heapq import*
import sys
le = sys.__stdin__.read().split(\"\n\")[::-1]
af=[]
mo=10**9+7
for zorg in range(int(le.pop())):
    n = int(le.pop())
    p1 = list(map(int,le.pop().split()))
    p2 = list(map(int,le.pop().split()))
    d = {p1[k]-1:p2[k]-1 for k in range(n)}
    vu=[False]*n
    log=0
    for k in range(n):
        if vu[k]==False:
            log+=1
            i=d[k]
            vu[k]=True
            while i!=k:
                vu[i]=True
                i=d[i]
    af.append(pow(2,log,mo))
print(\"\n\".join(map(str,af)))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random

class Clittlealawnspuzzlebootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = params.copy()
        self.params.setdefault('min_n', 2)
        self.params.setdefault('max_n', 10)
    
    def case_generator(self):
        min_n = self.params['min_n']
        max_n = self.params['max_n']
        n = random.randint(min_n, max_n)
        
        row1 = list(range(1, n+1))
        while True:
            row2 = random.sample(row1, n)
            if all(a != b for a, b in zip(row1, row2)):
                break
        
        return {
            'n': n,
            'row1': row1,
            'row2': row2
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        row1 = ' '.join(map(str, question_case['row1']))
        row2 = ' '.join(map(str, question_case['row2']))
        return f"""You are solving Little Alawn's puzzle on a 2×{n} grid. Both rows are permutations of 1-{n}, and initially no column has duplicates.

Current grid:
Row 1: {row1}
Row 2: {row2}

Calculate the number of distinct valid configurations achievable by vertical column swaps. Follow these steps:
1. Analyze the permutation relationships between columns
2. Identify independent cycles in the permutation graph
3. Compute 2^cycles mod 10^9+7

Format your answer as [answer]N[/answer] where N is the final number."""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](\d+)\[/answer\]', output)
        return matches[-1] if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            n = identity['n']
            row1 = identity['row1']
            row2 = identity['row2']
            
            # Build permutation mapping (0-based)
            perm = {row1[i]-1: row2[i]-1 for i in range(n)}
            
            # Find cycle count
            visited = [False] * n
            cycles = 0
            for node in range(n):
                if not visited[node]:
                    cycles += 1
                    current = node
                    while not visited[current]:
                        visited[current] = True
                        current = perm[current]
            
            correct = pow(2, cycles, 10**9+7)
            return int(solution) == correct
        except:
            return False
