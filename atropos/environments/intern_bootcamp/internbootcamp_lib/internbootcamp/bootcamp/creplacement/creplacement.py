"""# 

### 谜题描述
Daniel has a string s, consisting of lowercase English letters and period signs (characters '.'). Let's define the operation of replacement as the following sequence of steps: find a substring \"..\" (two consecutive periods) in string s, of all occurrences of the substring let's choose the first one, and replace this substring with string \".\". In other words, during the replacement operation, the first two consecutive periods are replaced by one. If string s contains no two consecutive periods, then nothing happens.

Let's define f(s) as the minimum number of operations of replacement to perform, so that the string does not have any two consecutive periods left.

You need to process m queries, the i-th results in that the character at position xi (1 ≤ xi ≤ n) of string s is assigned value ci. After each operation you have to calculate and output the value of f(s).

Help Daniel to process all queries.

Input

The first line contains two integers n and m (1 ≤ n, m ≤ 300 000) the length of the string and the number of queries.

The second line contains string s, consisting of n lowercase English letters and period signs.

The following m lines contain the descriptions of queries. The i-th line contains integer xi and ci (1 ≤ xi ≤ n, ci — a lowercas English letter or a period sign), describing the query of assigning symbol ci to position xi.

Output

Print m numbers, one per line, the i-th of these numbers must be equal to the value of f(s) after performing the i-th assignment.

Examples

Input

10 3
.b..bz....
1 h
3 c
9 f


Output

4
3
1


Input

4 4
.cc.
2 .
3 .
2 a
1 a


Output

1
3
1
1

Note

Note to the first sample test (replaced periods are enclosed in square brackets).

The original string is \".b..bz....\".

  * after the first query f(hb..bz....) = 4 (\"hb[..]bz....\"  →  \"hb.bz[..]..\"  →  \"hb.bz[..].\"  →  \"hb.bz[..]\"  →  \"hb.bz.\")
  * after the second query f(hbс.bz....) = 3 (\"hbс.bz[..]..\"  →  \"hbс.bz[..].\"  →  \"hbс.bz[..]\"  →  \"hbс.bz.\")
  * after the third query f(hbс.bz..f.) = 1 (\"hbс.bz[..]f.\"  →  \"hbс.bz.f.\")



Note to the second sample test.

The original string is \".cc.\".

  * after the first query: f(..c.) = 1 (\"[..]c.\"  →  \".c.\")
  * after the second query: f(....) = 3 (\"[..]..\"  →  \"[..].\"  →  \"[..]\"  →  \".\")
  * after the third query: f(.a..) = 1 (\".a[..]\"  →  \".a.\")
  * after the fourth query: f(aa..) = 1 (\"aa[..]\"  →  \"aa.\")

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/usr/bin/env python3
from __future__ import print_function

import sys
if sys.version_info[0] < 3:
    input = raw_input

n, m = input().split(' ')
n = int(n)
m = int(m)
s = input()

last_char = None
start = None

score = 0

for i, c in enumerate(s):
    if last_char == \".\" and c != \".\":
        score += (i - start - 1)
        start = None
    if last_char != \".\" and c == \".\": start = i
    last_char = c

if start is not None:
    score += (n - start - 1)

s = list(s)

so = []

for _ in range(m):
    i, c = input().split()
    i = int(i) - 1

    if c == \".\" and s[i] != \".\":
        if i-1 >= 0 and s[i-1] == \".\": score += 1
        if i+1 < n and s[i+1] == \".\": score += 1
    if c != \".\" and s[i] == \".\":
        if i-1 >= 0 and s[i-1] == \".\": score -= 1
        if i+1 < n and s[i+1] == \".\": score -= 1

    s[i] = c
    so += [str(score)]

print('\n'.join(so))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Creplacementbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', random.randint(5, 10))
        self.m = params.get('m', random.randint(3, 5))
    
    def case_generator(self):
        n, m = self.n, self.m
        s = [
            '.' if random.random() < 0.3 else random.choice('abcdefghijklmnopqrstuvwxyz')
            for _ in range(n)
        ]
        initial_str = ''.join(s)
        current_s = list(s)
        
        # Calculate initial score
        score = 0
        start = None
        last_char = None
        for i, c in enumerate(current_s):
            if last_char == '.' and c != '.':
                score += (i - start - 1)
                start = None
            if last_char != '.' and c == '.':
                start = i
            last_char = c
        if start is not None:
            score += (n - start - 1)
        
        # Generate queries and compute expected outputs
        queries = []
        expected_output = []
        for _ in range(m):
            xi = random.randint(1, n)
            ci = random.choice(['.', random.choice('abcdefghijklmnopqrstuvwxyz')])
            queries.append((xi, ci))
            
            # Update score based on query
            i = xi - 1
            original = current_s[i]
            new_char = ci
            if original == '.' and new_char != '.':
                if i > 0 and current_s[i-1] == '.':
                    score -= 1
                if i < n-1 and current_s[i+1] == '.':
                    score -= 1
            elif original != '.' and new_char == '.':
                if i > 0 and current_s[i-1] == '.':
                    score += 1
                if i < n-1 and current_s[i+1] == '.':
                    score += 1
            current_s[i] = new_char
            expected_output.append(score)
        
        return {
            'n': n,
            'm': m,
            'initial_string': initial_str,
            'queries': queries,
            'expected_output': expected_output
        }
    
    @staticmethod
    def prompt_func(case):
        prompt = [
            "Daniel has a string composed of lowercase letters and periods ('.').",
            "The minimum number of replacements needed to eliminate all consecutive '..' is denoted as f(s).",
            "Each query changes a character in the string, and you must compute f(s) after each change.",
            "\nProblem Instance:",
            f"Initial string: {case['initial_string']}",
            f"String length (n): {case['n']}, Number of queries (m): {case['m']}",
            "Queries (position and new character):"
        ]
        for idx, (pos, char) in enumerate(case['queries'], 1):
            prompt.append(f"{pos} {char}")
        prompt.append(
            "\nOutput the m results as space-separated integers. Place your answer within [answer] and [/answer], e.g., [answer]1 2 3[/answer]."
        )
        return '\n'.join(prompt)
    
    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        last_answer = answer_blocks[-1].strip()
        try:
            return list(map(int, last_answer.split()))
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, case):
        return solution == case['expected_output']
