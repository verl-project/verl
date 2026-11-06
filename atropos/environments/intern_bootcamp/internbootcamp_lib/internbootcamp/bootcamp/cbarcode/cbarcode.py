"""# 

### 谜题描述
You've got an n × m pixel picture. Each pixel can be white or black. Your task is to change the colors of as few pixels as possible to obtain a barcode picture.

A picture is a barcode if the following conditions are fulfilled: 

  * All pixels in each column are of the same color. 
  * The width of each monochrome vertical line is at least x and at most y pixels. In other words, if we group all neighbouring columns of the pixels with equal color, the size of each group can not be less than x or greater than y. 

Input

The first line contains four space-separated integers n, m, x and y (1 ≤ n, m, x, y ≤ 1000; x ≤ y).

Then follow n lines, describing the original image. Each of these lines contains exactly m characters. Character \".\" represents a white pixel and \"#\" represents a black pixel. The picture description doesn't have any other characters besides \".\" and \"#\".

Output

In the first line print the minimum number of pixels to repaint. It is guaranteed that the answer exists. 

Examples

Input

6 5 1 2
##.#.
.###.
###..
#...#
.##.#
###..


Output

11


Input

2 5 1 1
#####
.....


Output

5

Note

In the first test sample the picture after changing some colors can looks as follows: 
    
    
      
    .##..  
    .##..  
    .##..  
    .##..  
    .##..  
    .##..  
    

In the second test sample the picture after changing some colors can looks as follows: 
    
    
      
    .#.#.  
    .#.#.  
    

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, m, x, y = map(int, raw_input().split())
u = [i.count('.') for i in list(zip(*[raw_input() for i in range(n)]))]
v = [n - i for i in u]
a, b, s = [u[0]], [v[0]], x - 1
for i in range(1, x):
    a, b = [1000001] + [j + u[i] for j in a], [1000001] + [j + v[i] for j in b]
for i in range(x, min(y, m)):
    a, b = [min(b[s: ]) + u[i]] + [j + u[i] for j in a], [min(a[s: ]) + v[i]] + [j + v[i] for j in b]
for i in range(min(y, m), m):
    a, b = [min(b[s: ]) + u[i]] + [j + u[i] for j in a[: -1]], [min(a[s: ]) + v[i]] + [j + v[i] for j in b[: -1]]
print(min(min(a[s: ]), min(b[s: ])))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def calculate_min_pixels(n, m, x, y, rows):
    # 转置行以获取各列
    cols = list(zip(*rows))
    u = [col.count('.') for col in cols]  # 每列改为白色所需的修改次数（即列中黑色像素数）
    v = [n - count for count in u]        # 每列改为黑色所需的修改次数（即列中白色像素数）
    
    a = [u[0]]  # 以白色结尾的段的最小修改次数
    b = [v[0]]  # 以黑色结尾的段的最小修改次数
    s = x - 1   # 段长度至少为x，因此需要保留前s个状态
    
    # 处理前x-1列
    for i in range(1, x):
        # 由于段长度必须>=x，此时只能继续延长当前颜色段
        a = [float('inf')] + [prev + u[i] for prev in a]
        b = [float('inf')] + [prev + v[i] for prev in b]
    
    # 处理x到min(y, m)-1列
    for i in range(x, min(y, m)):
        # 可以开始新的颜色段，此时需要取另一种颜色的最小值
        min_b = min(b[s:]) if b[s:] else float('inf')
        new_a = [min_b + u[i]] + [prev + u[i] for prev in a]
        min_a = min(a[s:]) if a[s:] else float('inf')
        new_b = [min_a + v[i]] + [prev + v[i] for prev in b]
        a, b = new_a, new_b
    
    # 处理剩下的列（当m > y时）
    for i in range(min(y, m), m):
        # 需要确保段长度不超过y，因此保留前y个状态
        min_b = min(b[s:]) if b[s:] else float('inf')
        new_a = [min_b + u[i]] + [prev + u[i] for prev in a[:-1]]  # 保留前y-1个状态
        min_a = min(a[s:]) if a[s:] else float('inf')
        new_b = [min_a + v[i]] + [prev + v[i] for prev in b[:-1]]
        a, b = new_a, new_b
    
    # 最后，取所有可能状态中的最小值
    valid_a = a[s:] if a[s:] else [float('inf')]
    valid_b = b[s:] if b[s:] else [float('inf')]
    return min(min(valid_a), min(valid_b))

class Cbarcodebootcamp(Basebootcamp):
    def __init__(self, n=6, m=5, x=1, y=2):
        self.n = n
        self.m = m
        self.x = min(x, y)
        self.y = max(x, y)
        
    def case_generator(self):
        # 生成随机图像
        rows = []
        for _ in range(self.n):
            row = ''.join(random.choice(['#', '.']) for _ in range(self.m))
            rows.append(row)
        
        # 计算正确答案
        correct_answer = calculate_min_pixels(self.n, self.m, self.x, self.y, rows)
        
        return {
            'n': self.n,
            'm': self.m,
            'x': self.x,
            'y': self.y,
            'rows': rows,
            'correct_answer': correct_answer
        }
    
    @staticmethod
    def prompt_func(question_case):
        prompt = [
            "You are a barcode conversion expert. Your task is to transform the given image into a valid barcode by repainting as few pixels as possible.",
            "",
            "A valid barcode must satisfy:",
            "1. Each column consists of entirely black (#) or white (.) pixels.",
            "2. The width of every consecutive vertical line (group of same-color columns) must be between x and y, inclusive.",
            "",
            f"Problem parameters:",
            f"- Rows (n): {question_case['n']}",
            f"- Columns (m): {question_case['m']}",
            f"- Minimum width (x): {question_case['x']}",
            f"- Maximum width (y): {question_case['y']}",
            "",
            "Original image (each line represents a row):"
        ]
        prompt.extend(question_case['rows'])
        prompt.append(
            "\nCalculate the minimal number of pixels to repaint. Put your final answer within [answer] and [/answer]."
        )
        return '\n'.join(prompt)
    
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
        return solution == identity.get('correct_answer', None)
