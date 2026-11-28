"""# 

### 谜题描述
This is the first subtask of problem F. The only differences between this and the second subtask are the constraints on the value of m and the time limit. You need to solve both subtasks in order to hack this one.

There are n+1 distinct colours in the universe, numbered 0 through n. There is a strip of paper m centimetres long initially painted with colour 0. 

Alice took a brush and painted the strip using the following process. For each i from 1 to n, in this order, she picks two integers 0 ≤ a_i < b_i ≤ m, such that the segment [a_i, b_i] is currently painted with a single colour, and repaints it with colour i. 

Alice chose the segments in such a way that each centimetre is now painted in some colour other than 0. Formally, the segment [i-1, i] is painted with colour c_i (c_i ≠ 0). Every colour other than 0 is visible on the strip.

Count the number of different pairs of sequences \\{a_i\}_{i=1}^n, \\{b_i\}_{i=1}^n that result in this configuration. 

Since this number may be large, output it modulo 998244353.

Input

The first line contains a two integers n, m (1 ≤ n ≤ 500, n = m) — the number of colours excluding the colour 0 and the length of the paper, respectively.

The second line contains m space separated integers c_1, c_2, …, c_m (1 ≤ c_i ≤ n) — the colour visible on the segment [i-1, i] after the process ends. It is guaranteed that for all j between 1 and n there is an index k such that c_k = j.

Note that since in this subtask n = m, this means that c is a permutation of integers 1 through n.

Output

Output a single integer — the number of ways Alice can perform the painting, modulo 998244353.

Examples

Input


3 3
1 2 3


Output


5


Input


7 7
4 5 1 6 2 3 7


Output


165

Note

In the first example, there are 5 ways, all depicted in the figure below. Here, 0 is white, 1 is red, 2 is green and 3 is blue.

<image>

Below is an example of a painting process that is not valid, as in the second step the segment 1 3 is not single colour, and thus may not be repainted with colour 2.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
range = xrange
input = raw_input

MOD = 998244353 

n,n = [int(x) for x in input().split()]
C = [int(x) - 1 for x in input().split()]

pos = [0]*n
for i in range(n):
    pos[C[i]] = i

DP = [[1]*(n + 1) for _ in range(n + 1)]
for le in range(1, n + 1):
    for i in range(n - le + 1):
        j = i + le
        k = min(range(i,j), key = C.__getitem__)
        
        ans1 = 0
        for split in range(i,k + 1):
            ans1 += DP[i][split] * DP[split][k] % MOD   
        
        ans2 = 0
        for split in range(k+1, j + 1):
            ans2 += DP[k + 1][split] * DP[split][j] % MOD

        DP[i][j] = int((ans1 % MOD) * (ans2 % MOD) % MOD)

print DP[0][n]
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

MOD = 998244353

def solve(n, c_list):
    C = [x - 1 for x in c_list]
    DP = [[1] * (n + 1) for _ in range(n + 1)]
    for le in range(1, n + 1):
        for i in range(n - le + 1):
            j = i + le
            k = min(range(i, j), key=lambda x: C[x])
            ans1 = 0
            for split in range(i, k + 1):
                ans1 = (ans1 + DP[i][split] * DP[split][k]) % MOD
            ans2 = 0
            for split in range(k + 1, j + 1):
                ans2 = (ans2 + DP[k + 1][split] * DP[split][j]) % MOD
            DP[i][j] = (ans1 * ans2) % MOD
    return DP[0][n] % MOD

class F1shortcolorfulstripbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=7):
        self.min_n = min_n
        self.max_n = max_n
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        c = list(range(1, n + 1))
        random.shuffle(c)
        correct_answer = solve(n, c)
        return {
            'n': n,
            'm': n,
            'c': c,
            'correct_answer': correct_answer
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        c = question_case['c']
        c_str = ' '.join(map(str, c))
        prompt = f"""You are tasked with solving a programming puzzle. Please find the number of valid ways Alice could have painted a strip of paper according to the rules described, and provide the answer modulo 998244353.

Problem Description:
- The strip is initially painted with color 0.
- Alice repaints segments in n steps, each time using a new color from 1 to n.
- Each segment repainted must be a single continuous block of the same color before repainting.
- The final color of each 1 cm segment [i-1, i] is given as a permutation of 1 through n.

Input:
- The first line contains two integers n and m (n = m = {n} in this case).
- The second line contains the colors of each segment as a permutation: {c_str}

Output:
- A single integer representing the number of valid ways modulo 998244353.

Ensure your final answer is enclosed within [answer] and [/answer] tags. Example: [answer]123[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        # 移除可能的逗号等非数字字符
        cleaned = last_match.replace(',', '').replace(' ', '')
        try:
            return int(cleaned)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        correct = identity['correct_answer']
        return solution == correct
