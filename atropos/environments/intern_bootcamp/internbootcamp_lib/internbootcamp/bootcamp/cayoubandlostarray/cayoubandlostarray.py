"""# 

### 谜题描述
Ayoub had an array a of integers of size n and this array had two interesting properties: 

  * All the integers in the array were between l and r (inclusive). 
  * The sum of all the elements was divisible by 3. 



Unfortunately, Ayoub has lost his array, but he remembers the size of the array n and the numbers l and r, so he asked you to find the number of ways to restore the array. 

Since the answer could be very large, print it modulo 10^9 + 7 (i.e. the remainder when dividing by 10^9 + 7). In case there are no satisfying arrays (Ayoub has a wrong memory), print 0.

Input

The first and only line contains three integers n, l and r (1 ≤ n ≤ 2 ⋅ 10^5 , 1 ≤ l ≤ r ≤ 10^9) — the size of the lost array and the range of numbers in the array.

Output

Print the remainder when dividing by 10^9 + 7 the number of ways to restore the array.

Examples

Input


2 1 3


Output


3


Input


3 2 2


Output


1


Input


9 9 99


Output


711426616

Note

In the first example, the possible arrays are : [1,2], [2,1], [3, 3].

In the second example, the only possible array is [2, 2, 2].

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin
from copy import deepcopy

n, l, r = map(int, stdin.readline().split())
l -= 1
div1, div2, mod = l // 3, r // 3, 1000000007
all = [div2, div2 + 1 if r % 3 else div2, div2 + (min(r % 3, 2) // 2)]
minus = [div1, div1 + 1 if l % 3 else div1, div1 + (min(l % 3, 2) // 2)]
all = [all[i] - minus[i] for i in range(3)]

mem, p = [deepcopy(all), [0, 0, 0]], 0

for i in range(1, n):
    p ^= 1
    for j in range(1, 4):
        for k in range(1, 4):
            tem = (j + k) % 3
            mem[p][tem] = mem[p][tem] % mod + (mem[p ^ 1][j % 3] * all[k % 3]) % mod

    for j in range(3):
        mem[p ^ 1][j] = 0

print(mem[p][0] % mod)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def compute_answer(n, l, r):
    mod = 10**9 + 7

    def count_mod(low, high, m):
        remainder = low % 3
        if remainder <= m:
            first = low + (m - remainder)
        else:
            first = low + (3 - remainder + m)
        if first > high:
            return 0
        last = high - ((high - m) % 3)
        return ((last - first) // 3) + 1

    count0 = count_mod(l, r, 0)
    count1 = count_mod(l, r, 1)
    count2 = count_mod(l, r, 2)
    counts = [count0, count1, count2]

    # Dynamic programming approach
    dp_prev = counts.copy()
    for _ in range(n - 1):
        dp_next = [0] * 3
        for prev_mod in range(3):
            for curr_mod in range(3):
                new_mod = (prev_mod + curr_mod) % 3
                dp_next[new_mod] = (dp_next[new_mod] + dp_prev[prev_mod] * counts[curr_mod]) % mod
        dp_prev = dp_next

    return dp_prev[0] % mod

class Cayoubandlostarraybootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=20, l_min=1, r_max=10**9):
        self.n_min = n_min
        self.n_max = n_max
        self.l_min = l_min
        self.r_max = r_max

    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        l = random.randint(self.l_min, self.r_max)
        r = random.randint(l, self.r_max)
        correct_answer = compute_answer(n, l, r)
        return {
            'n': n,
            'l': l,
            'r': r,
            'correct_answer': correct_answer
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        l = question_case['l']
        r = question_case['r']
        problem = f"""Given three integers n, l, and r, calculate the number of arrays of length n where each element is between l and r (inclusive) and the total sum is divisible by 3. Return the result modulo 10^9+7.

Input:
n = {n}, l = {l}, r = {r}

Put your final answer within [answer] tags like [answer]123[/answer]."""
        return problem

    @staticmethod
    def extract_output(output):
        pattern = r'\[answer\](.*?)\[/answer\]'
        matches = re.findall(pattern, output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
