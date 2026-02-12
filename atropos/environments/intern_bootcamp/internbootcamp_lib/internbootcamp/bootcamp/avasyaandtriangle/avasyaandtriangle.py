"""# 

### 谜题描述
Vasya has got three integers n, m and k. He'd like to find three integer points (x_1, y_1), (x_2, y_2), (x_3, y_3), such that 0 ≤ x_1, x_2, x_3 ≤ n, 0 ≤ y_1, y_2, y_3 ≤ m and the area of the triangle formed by these points is equal to nm/k.

Help Vasya! Find such points (if it's possible). If there are multiple solutions, print any of them.

Input

The single line contains three integers n, m, k (1≤ n, m ≤ 10^9, 2 ≤ k ≤ 10^9).

Output

If there are no such points, print \"NO\".

Otherwise print \"YES\" in the first line. The next three lines should contain integers x_i, y_i — coordinates of the points, one point per line. If there are multiple solutions, print any of them.

You can print each letter in any case (upper or lower).

Examples

Input

4 3 3


Output

YES
1 0
2 3
4 1


Input

4 4 7


Output

NO

Note

In the first example area of the triangle should be equal to nm/k = 4. The triangle mentioned in the output is pictured below: 

<image>

In the second example there is no triangle with area nm/k = 16/7.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def gcd(a, b):
    if b == 0:
       return a
    else:
       return gcd(b, a % b)




n,m,k = map(int,raw_input().split())

was_even = False

if (2*n*m)%k == 0:

    if k%2 == 0:
        k =k/2
        was_even = True

    g = gcd(k,n)
    k_j = k/g
    a = n/g
    b = m/k_j

    if was_even == False:
        if a*2<n:
            a = a*2
        else:
            b = b*2

    print 'YES'
    print 0,0
    print a,0
    print 0,b

else:
    print 'NO'
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from math import gcd
from bootcamp import Basebootcamp

class Avasyaandtrianglebootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10**3, min_m=1, max_m=10**3, min_k=2, max_k=10**6, ensure_solvable=None):
        self.min_n = min_n
        self.max_n = max_n
        self.min_m = min_m
        self.max_m = max_m
        self.min_k = min_k
        self.max_k = max_k
        self.ensure_solvable = ensure_solvable

    def case_generator(self):
        max_attempts = 1000
        for _ in range(max_attempts):
            # 生成参数时根据 ensure_solvable 调整策略
            if self.ensure_solvable:
                n = random.randint(self.min_n, self.max_n)
                m = random.randint(self.min_m, self.max_m)
                a = random.randint(1, n)
                b = random.randint(1, m)
                numerator = 2 * n * m
                denominator = a * b
                if denominator == 0:
                    continue
                if numerator % denominator != 0:
                    continue
                k = numerator // denominator
                if k < self.min_k or k > self.max_k or k < 2:
                    continue
                points = [(0, 0), (a, 0), (0, b)]
                valid = all(0 <= x <= n and 0 <= y <= m for x, y in points)
                if valid:
                    return {
                        'n': n,
                        'm': m,
                        'k': k,
                        'solvable': True,
                        'points': points
                    }
            else:
                n = random.randint(self.min_n, self.max_n)
                m = random.randint(self.min_m, self.max_m)
                k = random.randint(self.min_k, self.max_k)
                solvable = (2 * n * m) % k == 0
                if self.ensure_solvable is not None and solvable != self.ensure_solvable:
                    continue

                if not solvable:
                    return {
                        'n': n,
                        'm': m,
                        'k': k,
                        'solvable': False,
                        'points': None
                    }

                was_even = False
                current_k = k
                if current_k % 2 == 0:
                    current_k //= 2
                    was_even = True

                g = gcd(current_k, n)
                k_j = current_k // g
                a = n // g
                b = m // k_j

                if not was_even:
                    if 2 * a <= n:
                        a *= 2
                    else:
                        b *= 2
                        if b > m:
                            continue

                points = [(0, 0), (a, 0), (0, b)]
                valid = all(0 <= x <= n and 0 <= y <= m for x, y in points)
                if valid:
                    return {
                        'n': n,
                        'm': m,
                        'k': k,
                        'solvable': True,
                        'points': points
                    }

        # Fallback for ensure_solvable=True
        if self.ensure_solvable:
            n, m = self.min_n, self.min_m
            a, b = n, m
            k = (2 * n * m) // (a * b)
            while k < 2 or (2 * n * m) % (a * b) != 0 or k < self.min_k or k > self.max_k:
                a = random.randint(1, n)
                b = random.randint(1, m)
                k = (2 * n * m) // (a * b)
            points = [(0, 0), (a, 0), (0, b)]
            return {
                'n': n,
                'm': m,
                'k': k,
                'solvable': True,
                'points': points
            }

        # Fallback for other cases
        n, m, k = self.max_n, self.max_m, self.max_k
        while (2 * n * m) % k == 0:
            k = random.randint(self.min_k, self.max_k)
        return {
            'n': n,
            'm': m,
            'k': k,
            'solvable': False,
            'points': None
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        m = question_case['m']
        k = question_case['k']
        target_expr = f"{n}×{m}/{k} = {n*m}/{k}"
        prompt = f"""Vasya has three integers n={n}, m={m}, and k={k}. He wants to find three integer points (x1, y1), (x2, y2), (x3, y3) such that:
- All coordinates satisfy 0 ≤ xi ≤ {n} and 0 ≤ yi ≤ {m} for i = 1, 2, 3.
- The area of the triangle formed by these points is exactly (n×m)/k = {target_expr}.

Determine if such points exist. If yes, output "YES" followed by the coordinates. Otherwise, output "NO".

Format your answer as:
[answer]
YES
x1 y1
x2 y2
x3 y3
[/answer]
or
[answer]
NO
[/answer]

Place your final answer within [answer] tags."""
        return prompt

    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, flags=re.DOTALL | re.IGNORECASE)
        if not answer_blocks:
            return None
        last_answer = answer_blocks[-1].strip()
        lines = [line.strip() for line in last_answer.split('\n') if line.strip()]
        if not lines:
            return None
        if lines[0].upper() == 'NO':
            return 'NO'
        elif lines[0].upper() == 'YES' and len(lines) == 4:
            points = []
            for line in lines[1:4]:
                parts = line.split()
                if len(parts) != 2:
                    return None
                try:
                    x, y = int(parts[0]), int(parts[1])
                    points.append((x, y))
                except ValueError:
                    return None
            return points
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        m = identity['m']
        k = identity['k']
        solvable = identity['solvable']

        if not solvable:
            return isinstance(solution, str) and solution.upper() == 'NO'
        if solution == 'NO':
            return False
        if not isinstance(solution, list) or len(solution) != 3:
            return False

        for x, y in solution:
            if not (0 <= x <= n and 0 <= y <= m):
                return False

        x1, y1 = solution[0]
        x2, y2 = solution[1]
        x3, y3 = solution[2]
        area_twice = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        expected_twice_area = (2 * n * m) // k
        return area_twice == expected_twice_area
