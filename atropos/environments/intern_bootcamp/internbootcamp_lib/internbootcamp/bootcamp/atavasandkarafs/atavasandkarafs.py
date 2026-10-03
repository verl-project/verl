"""# 

### 谜题描述
Karafs is some kind of vegetable in shape of an 1 × h rectangle. Tavaspolis people love Karafs and they use Karafs in almost any kind of food. Tavas, himself, is crazy about Karafs.

<image>

Each Karafs has a positive integer height. Tavas has an infinite 1-based sequence of Karafses. The height of the i-th Karafs is si = A + (i - 1) × B.

For a given m, let's define an m-bite operation as decreasing the height of at most m distinct not eaten Karafses by 1. Karafs is considered as eaten when its height becomes zero.

Now SaDDas asks you n queries. In each query he gives you numbers l, t and m and you should find the largest number r such that l ≤ r and sequence sl, sl + 1, ..., sr can be eaten by performing m-bite no more than t times or print -1 if there is no such number r.

Input

The first line of input contains three integers A, B and n (1 ≤ A, B ≤ 106, 1 ≤ n ≤ 105).

Next n lines contain information about queries. i-th line contains integers l, t, m (1 ≤ l, t, m ≤ 106) for i-th query.

Output

For each query, print its answer in a single line.

Examples

Input

2 1 4
1 5 3
3 3 10
7 10 2
6 4 8


Output

4
-1
8
-1


Input

1 5 2
1 5 10
2 7 4


Output

1
2

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
a, b, n = map(int, raw_input().split())
res = [0] * n

def parseints(s, n):
    r = [0] * n
    i = 0
    x = 0
    for c in s:
        if c == ' ':
            r[i] = x
            i += 1
            x = 0
        else:
            x = x * 10 + ord(c) - 48
    return r
# z = map(int, sys.stdin.read().split())
z = parseints(sys.stdin.read().replace('\n', ' '), 3 * n)

for i in xrange(n):
    l0, t, m = z[3 * i: 3 * i + 3]
    r = (t - a) / b + 2
    l = l0 - 1
    g = t * m + a * l + b * l * (l - 1) / 2
    while l + 1 < r:
        c = (l + r) / 2
        s = a * c + b * c * (c - 1) / 2
        if s <= g:
            l = c
        else:
            r = c

    res[i] = -1 if l < l0 else l

sys.stdout.write(repr(res).replace(', ', '\n')[1:-1])
# print '\n'.join(map(str, res))
# print
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
from bootcamp import Basebootcamp

def compute_r(A, B, l0, t, m):
    if B == 0:
        B = 1
    if A > t:
        return -1
    numerator = t - A
    if numerator < 0:
        return -1
    r_max1 = numerator // B + 1
    if r_max1 < l0:
        return -1
    low = l0
    high = r_max1
    best = -1
    while low <= high:
        mid = (low + high) // 2
        terms = mid - l0 + 1
        first = A + (l0 - 1) * B
        last = A + (mid - 1) * B
        current_sum = terms * (first + last) // 2
        if current_sum <= t * m:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best if best != -1 else -1

class Atavasandkarafsbootcamp(Basebootcamp):
    def __init__(self, min_A=1, max_A=100, min_B=1, max_B=100, n_queries=3):
        self.min_A = min_A
        self.max_A = max_A
        self.min_B = min_B
        self.max_B = max_B
        self.n_queries = n_queries
    
    def case_generator(self):
        import random
        A = random.randint(self.min_A, self.max_A)
        B = random.randint(self.min_B, self.max_B)
        queries = []
        expected_outputs = []
        for _ in range(self.n_queries):
            l = random.randint(1, 10**6)
            t = random.randint(1, 10**6)
            m = random.randint(1, 10**6)
            expected = compute_r(A, B, l, t, m)
            queries.append({'l': l, 't': t, 'm': m})
            expected_outputs.append(expected)
        return {
            'A': A,
            'B': B,
            'queries': queries,
            'expected_outputs': expected_outputs
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        prompt = (
            "Tavas is crazy about Atavasandkarafs, a vegetable arranged in an infinite sequence. "
            "The height of the i-th Atavasandkarafs is s_i = A + (i-1)*B. Each m-bite operation allows decreasing "
            "the height of up to m distinct Atavasandkarafses by 1. A Atavasandkarafs is eaten when its height becomes zero.\n\n"
            f"You are given A = {question_case['A']}, B = {question_case['B']}, and the following {len(question_case['queries'])} queries:\n"
        )
        for idx, q in enumerate(question_case['queries'], 1):
            prompt += f"Query {idx}: l = {q['l']}, t = {q['t']}, m = {q['m']}\n"
        prompt += (
            "\nFor each query, determine the largest r such that all Atavasandkarafses from s_l to s_r can be eaten "
            "using at most t m-bite operations. If impossible, output -1.\n\n"
            "Format your answer with each result on a new line enclosed within [answer] and [/answer]. Example:\n"
            "[answer]\n"
            "3\n"
            "-1\n"
            "[/answer]"
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        last_block = answer_blocks[-1].strip()
        answers = []
        for line in last_block.split('\n'):
            line = line.strip()
            if line:
                try:
                    answers.append(int(line))
                except ValueError:
                    return None
        return answers
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity.get('expected_outputs', [])
        return solution == expected
