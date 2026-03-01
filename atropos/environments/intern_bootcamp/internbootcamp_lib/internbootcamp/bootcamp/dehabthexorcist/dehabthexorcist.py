"""# 

### 谜题描述
Given 2 integers u and v, find the shortest array such that [bitwise-xor](https://en.wikipedia.org/wiki/Bitwise_operation#XOR) of its elements is u, and the sum of its elements is v.

Input

The only line contains 2 integers u and v (0 ≤ u,v ≤ 10^{18}).

Output

If there's no array that satisfies the condition, print \"-1\". Otherwise:

The first line should contain one integer, n, representing the length of the desired array. The next line should contain n positive integers, the array itself. If there are multiple possible answers, print any.

Examples

Input


2 4


Output


2
3 1

Input


1 3


Output


3
1 1 1

Input


8 5


Output


-1

Input


0 0


Output


0

Note

In the first sample, 3⊕ 1 = 2 and 3 + 1 = 4. There is no valid array of smaller length.

Notice that in the fourth sample the array is empty.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
u,v=map(int, raw_input().split())
if u>v or u%2!=v%2:
    print -1
    exit(0)
if u==v:
    if u==0:
        print 0
    else:
        print 1
        print u
    exit(0)
x=(v-u)/2
if x&u:
    print 3
    print u, x, x
else:
    print 2
    print u+x, x
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re

from bootcamp import Basebootcamp

class Dehabthexorcistbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__(**params)
    
    def case_generator(self):
        case_type = random.choice([0, 1, 2, 3, 4, 5])
        u = 0
        v = 0

        if case_type == 0:
            u = 0
            v = 0
        elif case_type == 1:
            u = random.randint(1, 10**18)
            v = u
        elif case_type == 2:
            while True:
                x = random.randint(1, (10**18 - 1) // 2)
                max_u = 10**18 - 2 * x
                if max_u < 0:
                    continue
                k = x.bit_length()
                upper_limit = max_u >> k
                if upper_limit == 0:
                    u = 0
                else:
                    upper = random.randint(0, upper_limit)
                    u = upper << k
                if u <= max_u:
                    break
            v = u + 2 * x
        elif case_type == 3:
            while True:
                x = random.randint(1, (10**18 - 1) // 2)
                max_u = 10**18 - 2 * x
                if max_u < 1:
                    continue
                u = random.randint(1, max_u)
                if (x & u) != 0:
                    break
            v = u + 2 * x
        elif case_type == 4:
            v = random.randint(0, 10**18 - 1)
            delta = random.randint(1, 10**18 - v)
            u = v + delta
        elif case_type == 5:
            u = random.randint(0, 10**18)
            v = random.randint(0, 10**18)
            if (u % 2) == (v % 2):
                v += 1
                if v > 10**18:
                    v = u - 1 if u > 0 else 1

        return {'u': u, 'v': v}

    @staticmethod
    def prompt_func(question_case) -> str:
        u = question_case['u']
        v = question_case['v']
        prompt = f"""You are given two integers u and v. Your task is to find the shortest possible array of positive integers such that the bitwise XOR of all its elements equals {u}, and the sum of all elements equals {v}. If no such array exists, output -1.

Input:
u = {u}, v = {v}

Output Format:
If no solution exists, output -1. Otherwise, output the length of the array followed by the array elements. If there are multiple valid answers, output any.

Examples:

Example 1:
Input: u=2, v=4
Output:
2
3 1

Example 2:
Input: u=1, v=3
Output:
3
1 1 1

Please provide your answer within [answer] and [/answer]. For example:

[answer]
2
3 1
[/answer]"""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        lines = [line.strip() for line in last_match.split('\n') if line.strip()]
        if not lines:
            return None
        if lines[0] == '-1':
            return -1 if len(lines) == 1 else None
        try:
            n = int(lines[0])
            if n == 0:
                return [] if len(lines) == 1 else None
            if len(lines) < 2:
                return None
            elements = list(map(int, lines[1].split()))
            if len(elements) != n or any(e <= 0 for e in elements):
                return None
            return elements
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        u = identity['u']
        v = identity['v']
        
        if solution == -1:
            return u > v or (u % 2) != (v % 2)
        
        if isinstance(solution, list):
            if not solution:
                return u == 0 and v == 0
            if any(not isinstance(e, int) or e <= 0 for e in solution):
                return False
        else:
            return False
        
        xor = 0
        total = 0
        for num in solution:
            xor ^= num
            total += num
        
        if xor != u or total != v:
            return False
        
        if u > v or (u % 2) != (v % 2):
            return False
        
        if u == v:
            correct_len = 0 if u == 0 else 1
        else:
            x = (v - u) // 2
            correct_len = 3 if (x & u) else 2
        
        return len(solution) == correct_len
