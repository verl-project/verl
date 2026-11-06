"""# 

### 谜题描述
Mahmoud and Ehab are on the third stage of their adventures now. As you know, Dr. Evil likes sets. This time he won't show them any set from his large collection, but will ask them to create a new set to replenish his beautiful collection of sets.

Dr. Evil has his favorite evil integer x. He asks Mahmoud and Ehab to find a set of n distinct non-negative integers such the bitwise-xor sum of the integers in it is exactly x. Dr. Evil doesn't like big numbers, so any number in the set shouldn't be greater than 106.

Input

The only line contains two integers n and x (1 ≤ n ≤ 105, 0 ≤ x ≤ 105) — the number of elements in the set and the desired bitwise-xor, respectively.

Output

If there is no such set, print \"NO\" (without quotes).

Otherwise, on the first line print \"YES\" (without quotes) and on the second line print n distinct integers, denoting the elements in the set is any order. If there are multiple solutions you can print any of them.

Examples

Input

5 5


Output

YES
1 2 4 5 7

Input

3 6


Output

YES
1 2 5

Note

You can read more about the bitwise-xor operation here: <https://en.wikipedia.org/wiki/Bitwise_operation#XOR>

For the first sample <image>.

For the second sample <image>.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/usr/bin/env python
#-*- coding: utf-8 -*-

from collections import defaultdict
from math import factorial as f
from fractions import gcd as g
import operator

n, x = [int (i) for i in raw_input ().split ()]
if n == 2 and x == 0:
    print \"NO\"
else:
    l = range (1, n - 2)
    y = reduce (operator.xor, l, 0)
    p = 1 << 18
    if n == 1:
        l = [x]
    elif n == 2:
        l = [0, x]
    elif x != y:
        l.append (0)
        l.append (p)
        l.append (p ^ x ^ y)
    else:
        l.append (p)
        l.append (2 * p)
        l.append (p ^ (2 * p))
    print \"YES\"
    print \" \".join (str (i) for i in l)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from functools import reduce
from operator import xor

class Cmahmoudandehabandthexorbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.min_n = params.get('min_n', 1)
        self.max_n = params.get('max_n', 1000)
        self.min_x = params.get('min_x', 0)
        self.max_x = params.get('max_x', 10**5)
        self.allow_unsolvable = params.get('allow_unsolvable', True)

    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        if n == 2 and self.allow_unsolvable:
            if random.choice([True, False]):
                x = 0
            else:
                x = random.randint(1, self.max_x)
        else:
            x = random.randint(self.min_x, self.max_x)
        return {'n': n, 'x': x}

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        x = question_case['x']
        prompt = f"""You are participating in a puzzle-solving challenge. Your task is to solve the following problem based on the given rules.

Problem Description:

Dr. Evil has asked you to find a set of {n} distinct non-negative integers such that the bitwise XOR of all the numbers in the set equals {x}. Each number in the set must not exceed 1,000,000.

Rules:

1. The set must contain exactly {n} distinct non-negative integers.
2. Each integer in the set must be between 0 and 1,000,000, inclusive.
3. The bitwise XOR of all the integers in the set must be exactly {x}.
4. If it is impossible to create such a set, you must respond with "NO".

Your task is to determine if such a set exists. If it exists, provide the set as specified; otherwise, respond with "NO".

Input:

The input consists of two integers: n = {n} and x = {x}.

Output Format:

- If a solution exists:
  On the first line, print "YES".
  On the second line, print the {n} distinct integers separated by spaces.
- If no solution exists:
  Print "NO" on a single line.

Examples:

Example 1:
Input:
n = 5, x = 5

Output:
YES
1 2 4 5 7

Example 2:
Input:
n = 3, x = 6

Output:
YES
1 2 5

Example 3:
Input:
n = 2, x = 0

Output:
NO

Important Notes:
- The order of the integers in the output does not matter.
- All integers must be distinct.
- Ensure that each number is within the valid range (0 to 1,000,000).

Please provide your answer within the following tags:
[answer]
your_answer_here
[/answer]

Replace 'your_answer_here' with either "YES" followed by the numbers on the next line or "NO" if no solution exists.

For example, a valid answer format is:
[answer]
YES
1 2 3 4 5
[/answer]

Or for "NO":
[answer]
NO
[/answer]

Ensure that your answer is enclosed within the [answer] and [/answer] tags."""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        answer_content = matches[-1].strip()
        lines = [line.strip() for line in answer_content.split('\n') if line.strip()]
        if not lines:
            return None
        first_line = lines[0].upper()
        if first_line == 'YES':
            if len(lines) < 2:
                return None
            numbers_line = lines[1]
            try:
                nums = list(map(int, numbers_line.split()))
                return nums
            except ValueError:
                return None
        elif first_line == 'NO':
            return 'NO'
        else:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        x = identity['x']

        def is_possible(_n, _x):
            if _n == 1:
                return True
            elif _n == 2:
                return _x != 0
            else:
                return True

        if solution == 'NO':
            return not is_possible(n, x)
        elif isinstance(solution, list):
            if len(solution) != n:
                return False
            seen = set()
            for num in solution:
                if not isinstance(num, int) or num < 0 or num > 10**6:
                    return False
                if num in seen:
                    return False
                seen.add(num)
            xor_sum = reduce(xor, solution, 0)
            return xor_sum == x
        else:
            return False
