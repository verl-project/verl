"""# 

### 谜题描述
Vladimir would like to prepare a present for his wife: they have an anniversary! He decided to buy her exactly n flowers.

Vladimir went to a flower shop, and he was amazed to see that there are m types of flowers being sold there, and there is unlimited supply of flowers of each type. Vladimir wants to choose flowers to maximize the happiness of his wife. He knows that after receiving the first flower of the i-th type happiness of his wife increases by a_i and after receiving each consecutive flower of this type her happiness increases by b_i. That is, if among the chosen flowers there are x_i > 0 flowers of type i, his wife gets a_i + (x_i - 1) ⋅ b_i additional happiness (and if there are no flowers of type i, she gets nothing for this particular type).

Please help Vladimir to choose exactly n flowers to maximize the total happiness of his wife.

Input

The first line contains the only integer t (1 ≤ t ≤ 10 000), the number of test cases. It is followed by t descriptions of the test cases.

Each test case description starts with two integers n and m (1 ≤ n ≤ 10^9, 1 ≤ m ≤ 100 000), the number of flowers Vladimir needs to choose and the number of types of available flowers.

The following m lines describe the types of flowers: each line contains integers a_i and b_i (0 ≤ a_i, b_i ≤ 10^9) for i-th available type of flowers.

The test cases are separated by a blank line. It is guaranteed that the sum of values m among all test cases does not exceed 100 000.

Output

For each test case output a single integer: the maximum total happiness of Vladimir's wife after choosing exactly n flowers optimally.

Example

Input


2
4 3
5 0
1 4
2 2

5 3
5 2
4 2
3 1


Output


14
16

Note

In the first example case Vladimir can pick 1 flower of the first type and 3 flowers of the second type, in this case the total happiness equals 5 + (1 + 2 ⋅ 4) = 14.

In the second example Vladimir can pick 2 flowers of the first type, 2 flowers of the second type, and 1 flower of the third type, in this case the total happiness equals (5 + 1 ⋅ 2) + (4 + 1 ⋅ 2) + 3 = 16.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from bisect import bisect_left
t=input()
for _ in range(t):
    n,m=[int(i) for i in raw_input().split()]
    a=[[int(i) for i in raw_input().split()] for j in range(m)]
    a.sort()
    arr=[a[i][0] for i in range(m)]
    ans=0
    p=[0 for i in range(m+1)]
    for i in range(1,m+1):
        p[i]=p[i-1]+arr[i-1]
    for i in range(m):
        pos=bisect_left(arr,a[i][1])
        if(m-pos>=n):
            ans=max(ans,p[m]-p[m-n])
            continue
        val=p[m]-p[pos]
        val+=(n-(m+1-pos))*a[i][1]
        if pos>i:
            val+=a[i][0]
        else:
            val+=a[i][1]
        ans=max(ans,val)
    print ans
    if(_!=t-1):
        x=raw_input()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bisect import bisect_right
import re
from bootcamp import Basebootcamp

def solve_max_happiness(n, m, flowers):
    # Sort flowers by b_i in descending order, then a_i in descending order
    flowers.sort(key=lambda x: (-x[1], -x[0]))
    max_b = [f[1] for f in flowers]
    a_list = [f[0] for f in flowers]
    prefix_a = [0] * (m + 1)
    for i in range(m):
        prefix_a[i+1] = prefix_a[i] + a_list[i]
    
    max_total = 0
    for k in range(1, min(m, n) + 1):
        if k > n:
            continue
        rem = n - k
        current_total = prefix_a[k] + rem * max_b[0]
        max_total = max(max_total, current_total)
    return max_total

class Cchoosingflowersbootcamp(Basebootcamp):
    def __init__(self, max_n=1000, max_m=100):
        self.max_n = max_n
        self.max_m = max_m

    def case_generator(self):
        import random
        n = random.randint(1, self.max_n)
        m = random.randint(1, self.max_m)
        flowers = []
        for _ in range(m):
            a_i = random.randint(0, 10**9)
            b_i = random.randint(0, 10**9)
            flowers.append((a_i, b_i))
        correct_answer = solve_max_happiness(n, m, flowers)
        return {
            'n': n,
            'm': m,
            'flowers': flowers,
            'correct_answer': correct_answer
        }

    @staticmethod
    def prompt_func(question_case):
        input_lines = ["1"]  # t=1 for one test case
        input_lines.append(f"{question_case['n']} {question_case['m']}")
        for a, b in question_case['flowers']:
            input_lines.append(f"{a} {b}")
        input_str = '\n'.join(input_lines)
        prompt = f"""Vladimir wants to choose exactly n flowers to maximize his wife's happiness. There are m types of flowers available. Each type i has parameters a_i and b_i: the first flower of type i contributes a_i happiness, and each subsequent flower of the same type adds b_i happiness. For example, selecting x_i flowers of type i (where x_i ≥ 1) contributes a_i + (x_i - 1) * b_i happiness from that type. The goal is to select exactly n flowers to maximize the total happiness.

Input:
{input_str}

Please compute the maximum possible total happiness. Format your answer as a single integer enclosed within [answer] and [/answer] tags. For example:

[answer]42[/answer]

Ensure the answer is the only content within the tags."""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_answer = matches[-1].strip()
        numbers = re.findall(r'-?\d+', last_answer)
        if not numbers:
            return None
        try:
            return int(numbers[-1])
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
