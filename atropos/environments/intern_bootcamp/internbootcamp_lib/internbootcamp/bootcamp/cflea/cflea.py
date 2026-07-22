"""# 

### 谜题描述
It is known that fleas in Berland can jump only vertically and horizontally, and the length of the jump is always equal to s centimeters. A flea has found herself at the center of some cell of the checked board of the size n × m centimeters (each cell is 1 × 1 centimeters). She can jump as she wishes for an arbitrary number of times, she can even visit a cell more than once. The only restriction is that she cannot jump out of the board.

The flea can count the amount of cells that she can reach from the starting position (x, y). Let's denote this amount by dx, y. Your task is to find the number of such starting positions (x, y), which have the maximum possible value of dx, y.

Input

The first line contains three integers n, m, s (1 ≤ n, m, s ≤ 106) — length of the board, width of the board and length of the flea's jump.

Output

Output the only integer — the number of the required starting positions of the flea.

Examples

Input

2 3 1000000


Output

6


Input

3 3 2


Output

4

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, m, s = map(int, raw_input().split())
print ((n - 1) % s + 1) * ((n - 1) / s + 1) * ((m - 1) % s + 1) * ((m - 1) / s + 1)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Cfleabootcamp(Basebootcamp):
    def __init__(self, max_dim=10**6):
        self.max_dim = max_dim
    
    def case_generator(self):
        s = random.choice([
            random.randint(1, self.max_dim),
            random.randint(self.max_dim//2, self.max_dim),
            1
        ])
        n = random.choice([
            random.randint(1, 10),
            random.randint(1, self.max_dim),
            self.max_dim
        ])
        m = random.choice([
            random.randint(1, 10),
            random.randint(1, self.max_dim),
            self.max_dim
        ])
        return {'n': n, 'm': m, 's': s}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        s = question_case['s']
        problem_text = f"""你是Berland棋盘上的一只跳蚤，棋盘的大小是{n}厘米（长）x {m}厘米（宽），每个单元格是1x1厘米。你只能在水平或垂直方向上跳跃，每次跳跃的长度固定为s厘米。你希望找出有多少个起始位置(x,y)使得可达的单元格数目是所有起始位置中的最大值。

规则说明：
1. 棋盘共有n×m个单元格，坐标范围为(1 ≤ x ≤ n, 1 ≤ y ≤ m)
2. 每次跳跃必须为s厘米，且在水平或垂直方向，可跳跃任意次数（包括0次）
3. 不能跳出棋盘边界，允许多次访问同一单元格
4. dx,y表示从(x,y)出发可达的单元格总数
5. 你需要计算具有最大dx,y值的起始位置数量

输入参数：
n = {n}
m = {m}
s = {s}

请将最终答案放在[answer]标签内，例如：[answer]42[/answer]。"""
        return problem_text
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        m = identity['m']
        s = identity['s']
        
        a = (n-1) % s + 1 if s != 0 else 1
        b = (n-1) // s + 1 if s != 0 else 1
        c = (m-1) % s + 1 if s != 0 else 1
        d = (m-1) // s + 1 if s != 0 else 1
        
        correct = a * b * c * d
        return solution == correct
