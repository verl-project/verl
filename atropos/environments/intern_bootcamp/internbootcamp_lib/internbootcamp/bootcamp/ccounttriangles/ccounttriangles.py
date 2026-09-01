"""# 

### 谜题描述
Like any unknown mathematician, Yuri has favourite numbers: A, B, C, and D, where A ≤ B ≤ C ≤ D. Yuri also likes triangles and once he thought: how many non-degenerate triangles with integer sides x, y, and z exist, such that A ≤ x ≤ B ≤ y ≤ C ≤ z ≤ D holds?

Yuri is preparing problems for a new contest now, so he is very busy. That's why he asked you to calculate the number of triangles with described property.

The triangle is called non-degenerate if and only if its vertices are not collinear.

Input

The first line contains four integers: A, B, C and D (1 ≤ A ≤ B ≤ C ≤ D ≤ 5 ⋅ 10^5) — Yuri's favourite numbers.

Output

Print the number of non-degenerate triangles with integer sides x, y, and z such that the inequality A ≤ x ≤ B ≤ y ≤ C ≤ z ≤ D holds.

Examples

Input


1 2 3 4


Output


4


Input


1 2 2 5


Output


3


Input


500000 500000 500000 500000


Output


1

Note

In the first example Yuri can make up triangles with sides (1, 3, 3), (2, 2, 3), (2, 3, 3) and (2, 3, 4).

In the second example Yuri can make up triangles with sides (1, 2, 2), (2, 2, 2) and (2, 2, 3).

In the third example Yuri can make up only one equilateral triangle with sides equal to 5 ⋅ 10^5.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
# difference array ftw

A, B, C, D = map(int, raw_input().strip().split())

# x < y + z always true
# x > z - y = d
# 0 <= d <= B - 1
# ans = sum {f(d) * (B - d)} over all d
# f(d) = number of y, z pairs s.t. z - y = d
# calculate f() using difference array ;)

f = [0 for i in xrange(D - B + 2)]

for y in xrange(B, C + 1):
    f[C - y] += 1
    f[D - y + 1] -= 1

ans = f[0] * (B - A + 1)
for d in xrange(1, B):
    # exceeded size of f array
    if d >= D - B + 2: break
    
    f[d] += f[d - 1]
    
    vals = min(B - A + 1, B - d)
    ans += (f[d] * vals)

print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def calculate_answer(A, B, C, D):
    len_f = D - B + 2
    f = [0] * len_f

    for y in range(B, C + 1):
        start = C - y
        end = D - y + 1
        if start < len_f:
            f[start] += 1
        if end < len_f and end > 0:
            f[end] -= 1

    ans = f[0] * (B - A + 1)
    for d in range(1, B):
        if d >= len_f:
            break
        f[d] += f[d-1]
        current_min = min(B - A + 1, B - d)
        ans += f[d] * current_min

    return ans

class Ccounttrianglesbootcamp(Basebootcamp):
    def __init__(self, max_A=1000, max_step=1000):
        self.max_A = min(max_A, 500000)
        self.max_step = min(max_step, 500000)
    
    def case_generator(self):
        MAX_LIMIT = 500000
        
        A = random.randint(1, self.max_A)
        B = random.randint(A, min(A + self.max_step, MAX_LIMIT))
        C = random.randint(B, min(B + self.max_step, MAX_LIMIT))
        D = random.randint(C, min(C + self.max_step, MAX_LIMIT))
        
        return {
            'A': A,
            'B': B,
            'C': C,
            'D': D,
            'correct_answer': calculate_answer(A, B, C, D)
        }
    
    @staticmethod
    def prompt_func(question_case):
        params = question_case
        return f"""Yuri最喜欢的四个数字满足A ≤ B ≤ C ≤ D。请计算满足以下条件的三元组(x, y, z)数量：
        
- {params['A']} ≤ x ≤ {params['B']}
- {params['B']} ≤ y ≤ {params['C']}
- {params['C']} ≤ z ≤ {params['D']}
- 构成非退化三角形（x + y > z）

答案请用[answer]标签包裹，例如：[answer]42[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
