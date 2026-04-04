"""# 

### 谜题描述
Petya and Vasya are competing with each other in a new interesting game as they always do.

At the beginning of the game Petya has to come up with an array of N positive integers. Sum of all elements in his array should be equal to S. Then Petya has to select an integer K such that 0 ≤ K ≤ S.

In order to win, Vasya has to find a non-empty subarray in Petya's array such that the sum of all selected elements equals to either K or S - K. Otherwise Vasya loses.

You are given integers N and S. You should determine if Petya can win, considering Vasya plays optimally. If Petya can win, help him to do that.

Input

The first line contains two integers N and S (1 ≤ N ≤ S ≤ 10^{6}) — the required length of the array and the required sum of its elements.

Output

If Petya can win, print \"YES\" (without quotes) in the first line. Then print Petya's array in the second line. The array should contain N positive integers with sum equal to S. In the third line print K. If there are many correct answers, you can print any of them.

If Petya can't win, print \"NO\" (without quotes).

You can print each letter in any register (lowercase or uppercase).

Examples

Input


1 4


Output


YES
4
2

Input


3 4


Output


NO

Input


3 8


Output


YES
2 1 5
4

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys

__author__ = 'ratmir'

from collections import deque

alphabet = \"abcdefghijklmnopqrstuvwxyz\"

def solve(n, a, graph):
    return 1

def execute():
    [n, s] = [int(x1) for x1 in sys.stdin.readline().split()]
    if n==1 and s==1:
        print \"NO\"
        return
    if n%2==0 and s%2==0 and s>=2*n:
        print \"YES\"
        a = [2]*(n-1)
        a.append(s-2*n+2)
        print(''.join('{} '.format(k) for k in a))
        print 1
    else:
        if n-1<s-n:
            print \"YES\"
            a = [1]*(n-1)
            a.append(s-n+1)
            print(''.join('{} '.format(k) for k in a))
            print n
        else:
            print \"NO\"



execute()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Dgamewitharraybootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = {
            'max_n': params.get('max_n', 1000),
            'max_s': params.get('max_s', 10**6),
            's_margin': params.get('s_margin', 1000)  # Control s generation range
        }
    
    def case_generator(self):
        """Generate valid puzzle instances covering edge cases"""
        n = random.randint(1, self.params['max_n'])
        s_max = min(self.params['max_s'], n + self.params['s_margin'])
        s = random.randint(n, s_max)
        
        # Special case handling
        if n == 1:
            if s == 1:
                return {'n': 1, 's': 1, 'has_solution': False}
            else:
                return {
                    'n': 1, 's': s, 
                    'has_solution': True,
                    'array': [s],
                    'k': s//2  # Guaranteed valid for s >1
                }

        # Main logic
        valid_case = None
        # Case 1: Even structure
        if n%2 ==0 and s%2 ==0 and s >= 2*n:
            last = s - 2*(n-1)
            if last > 0:
                valid_case = ([2]*(n-1) + [last], 1)
        
        # Case 2: 1...1 + big number
        if not valid_case and (n-1 < s -n):
            last = s - (n-1)
            if last > 0:
                valid_case = ([1]*(n-1) + [last], n)
        
        if valid_case:
            array, k = valid_case
            return {
                'n': n,
                's': s,
                'has_solution': True,
                'array': array,
                'k': k
            }
        else:
            # Recheck impossibility
            if n ==1 and s ==1:
                return {'n':1, 's':1, 'has_solution': False}
            # For other cases, return no solution
            return {'n':n, 's':s, 'has_solution': False}

    @staticmethod
    def prompt_func(question_case):
        """Enhanced problem description with explicit constraints"""
        n, s = question_case['n'], question_case['s']
        return f"""## Programming Challenge: Petya's Array Game

**Problem Statement:**
Petya needs to construct an array of exactly {n} positive integers that sum to {s}. He then selects an integer K (0 ≤ K ≤ {s}). Vasya wins if he finds ANY contiguous subarray whose sum equals either K or {s}-K.

**Your Task:**
Determine if Petya can choose an array and K to guarantee victory. If possible:
1. Output YES
2. Provide the array
3. Specify K

**Constraints:**
- Array elements must be positive integers
- Total sum must exactly equal {s}
- Subarray must be non-empty and contiguous
- K must be in [0, {s}]

**Output Format:**
[answer]
YES
a1 a2 ... an
K
[/answer]
OR
[answer]
NO
[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """Robust validation with subarray sum checking"""
        if not solution:
            return False

        lines = [l.strip() for l in solution.split('\n') if l.strip()]
        required_no = not identity['has_solution']
        n, s = identity['n'], identity['s']

        # Handle NO cases
        if len(lines) == 1 and lines[0].upper() == 'NO':
            return required_no

        # Validate YES structure
        if required_no or len(lines)<3 or lines[0].upper()!='YES':
            return False

        try:
            array = list(map(int, lines[1].split()))
            k = int(lines[2])
        except:
            return False

        # Basic validation
        if (
            len(array) != n or
            sum(array) != s or
            any(x<=0 for x in array) or
            not (0 <= k <= s)
        ):
            return False

        # Subarray sum verification
        prefix = [0]
        for num in array:
            prefix.append(prefix[-1] + num)
        
        seen = set()
        for i in range(len(prefix)):
            for j in range(i+1, len(prefix)):
                current = prefix[j] - prefix[i]
                if current == k or current == (s -k):
                    return False
                seen.add(current)
        return True
