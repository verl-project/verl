"""# 

### 谜题描述
Memory is now interested in the de-evolution of objects, specifically triangles. He starts with an equilateral triangle of side length x, and he wishes to perform operations to obtain an equilateral triangle of side length y.

In a single second, he can modify the length of a single side of the current triangle such that it remains a non-degenerate triangle (triangle of positive area). At any moment of time, the length of each side should be integer.

What is the minimum number of seconds required for Memory to obtain the equilateral triangle of side length y?

Input

The first and only line contains two integers x and y (3 ≤ y < x ≤ 100 000) — the starting and ending equilateral triangle side lengths respectively.

Output

Print a single integer — the minimum number of seconds required for Memory to obtain the equilateral triangle of side length y if he starts with the equilateral triangle of side length x.

Examples

Input

6 3


Output

4


Input

8 5


Output

3


Input

22 4


Output

6

Note

In the first sample test, Memory starts with an equilateral triangle of side length 6 and wants one of side length 3. Denote a triangle with sides a, b, and c as (a, b, c). Then, Memory can do <image>.

In the second sample test, Memory can do <image>.

In the third sample test, Memory can do: <image>

<image>.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys

y, x = map(int, raw_input().split())
current = [ x, x, x ]
goal = [ y, y, y ]
steps = 0
while current != goal:
    best_index, best_value = -1, -1
    for i, a in enumerate(current):
        b, c = current[(i + 1) % 3], current[(i + 2) % 3]
        x = min(y, b + c - 1)
        if x > a and x > best_value:
            best_index, best_value = i, x
    current[best_index] = best_value
    steps += 1
print(steps)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cmemoryanddeevolutionbootcamp(Basebootcamp):
    def __init__(self, min_y=3, max_x=100000):
        self.min_y = min_y
        self.max_x = max_x

    def case_generator(self):
        y = random.randint(self.min_y, self.max_x - 1)
        x = random.randint(y + 1, self.max_x)
        return {'x': x, 'y': y}

    @staticmethod
    def prompt_func(question_case):
        x = question_case['x']
        y = question_case['y']
        prompt = f"""Memory starts with an equilateral triangle with side length {x} and wants to transform it into one with side length {y}. Each second, he can modify one side's length to any positive integer, provided the new triangle remains non-degenerate (sum of any two sides exceeds the third). All side lengths must be integers. What is the minimum number of seconds required?

Output the answer as an integer enclosed within [answer] and [/answer]. Example: [answer]4[/answer]"""
        return prompt

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
        # 交换x和y以适配参考代码逻辑
        original_x = identity['x']
        original_y = identity['y']
        x = original_y  # 参考代码中的输入xy是颠倒的
        y = original_x
        
        current = [x, x, x]
        goal = [y, y, y]
        steps = 0
        
        while current != goal:
            best_index = -1
            best_value = -1
            for i in range(3):
                a = current[i]
                b = current[(i+1) % 3]
                c = current[(i+2) % 3]
                new_val = min(y, b + c - 1)
                if new_val > a and new_val > best_value:
                    best_index = i
                    best_value = new_val
            
            if best_index == -1:  # 无解
                return False
            
            current[best_index] = best_value
            steps += 1
            
            # 安全阀防止无限循环
            if steps > 1000:
                return False
        
        return solution == steps
