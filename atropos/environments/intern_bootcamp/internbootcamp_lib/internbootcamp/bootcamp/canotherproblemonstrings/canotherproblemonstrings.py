"""# 

### 谜题描述
A string is binary, if it consists only of characters \"0\" and \"1\".

String v is a substring of string w if it has a non-zero length and can be read starting from some position in string w. For example, string \"010\" has six substrings: \"0\", \"1\", \"0\", \"01\", \"10\", \"010\". Two substrings are considered different if their positions of occurrence are different. So, if some string occurs multiple times, we should consider it the number of times it occurs.

You are given a binary string s. Your task is to find the number of its substrings, containing exactly k characters \"1\".

Input

The first line contains the single integer k (0 ≤ k ≤ 106). The second line contains a non-empty binary string s. The length of s does not exceed 106 characters.

Output

Print the single number — the number of substrings of the given string, containing exactly k characters \"1\".

Please do not use the %lld specifier to read or write 64-bit integers in С++. It is preferred to use the cin, cout streams or the %I64d specifier.

Examples

Input

1
1010


Output

6


Input

2
01010


Output

4


Input

100
01010


Output

0

Note

In the first sample the sought substrings are: \"1\", \"1\", \"10\", \"01\", \"10\", \"010\".

In the second sample the sought substrings are: \"101\", \"0101\", \"1010\", \"01010\".

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division

def read(func=None):
    return func(raw_input()) if func else raw_input()

################################

k = read(int)
s = read()

a = [1]
for c in s:
    if c == '0':
        a[-1] += 1
    else:
        a.append(1)

if k == 0:
    print sum(n * (n - 1) // 2 for n in a)
else:
    print sum(a * b for (a, b) in zip(a, a[k:]))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Canotherproblemonstringsbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.min_s_length = params.get('min_s_length', 1)
        self.max_s_length = params.get('max_s_length', 20)
        self.min_k = params.get('min_k', 0)
        self.max_k = params.get('max_k', 20)
    
    def case_generator(self):
        s_length = random.randint(self.min_s_length, self.max_s_length)
        s = ''.join(random.choice('01') for _ in range(s_length))
        k = random.randint(self.min_k, self.max_k)
        return {"k": k, "s": s}
    
    @staticmethod
    def prompt_func(question_case):
        k = question_case['k']
        s = question_case['s']
        problem_desc = f"""
You are a programming competition contestant. Your task is to solve the following problem:

Given an integer k and a binary string s, find the number of substrings of s that contain exactly k '1' characters. A substring is a contiguous non-empty sequence of characters. Two substrings are considered different if they start or end at different positions, even if their contents are identical.

Input Format:
- The first line contains the integer k.
- The second line contains the binary string s.

Output Format:
- A single integer representing the number of valid substrings.

Example:
Input:
1
1010
Output:
6

Your Task:
The current problem instance has the following inputs:
k = {k}
s = "{s}"

Please compute the correct answer and enclose it within [answer] and [/answer] tags. For example, if the answer is 5, write it as [answer]5[/answer].
"""
        return problem_desc.strip()
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        k = identity['k']
        s = identity['s']
        def compute_answer(k_val, s_str):
            a = [1]
            for c in s_str:
                if c == '0':
                    a[-1] += 1
                else:
                    a.append(1)
            if k_val == 0:
                return sum(n * (n - 1) // 2 for n in a)
            else:
                return sum(a * b for a, b in zip(a, a[k_val:])) if len(a) >= k_val else 0
        correct_answer = compute_answer(k, s)
        return solution == correct_answer
