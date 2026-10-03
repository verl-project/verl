"""# 

### 谜题描述
During the quarantine, Sicromoft has more free time to create the new functions in \"Celex-2021\". The developers made a new function GAZ-GIZ, which infinitely fills an infinite table to the right and down from the upper left corner as follows:

<image> The cell with coordinates (x, y) is at the intersection of x-th row and y-th column. Upper left cell (1,1) contains an integer 1.

The developers of the SUM function don't sleep either. Because of the boredom, they teamed up with the developers of the RAND function, so they added the ability to calculate the sum on an arbitrary path from one cell to another, moving down or right. Formally, from the cell (x,y) in one step you can move to the cell (x+1, y) or (x, y+1). 

After another Dinwows update, Levian started to study \"Celex-2021\" (because he wants to be an accountant!). After filling in the table with the GAZ-GIZ function, he asked you to calculate the quantity of possible different amounts on the path from a given cell (x_1, y_1) to another given cell (x_2, y_2), if you can only move one cell down or right.

Formally, consider all the paths from the cell (x_1, y_1) to cell (x_2, y_2) such that each next cell in the path is located either to the down or to the right of the previous one. Calculate the number of different sums of elements for all such paths.

Input

The first line contains one integer t (1 ≤ t ≤ 57179) — the number of test cases.

Each of the following t lines contains four natural numbers x_1, y_1, x_2, y_2 (1 ≤ x_1 ≤ x_2 ≤ 10^9, 1 ≤ y_1 ≤ y_2 ≤ 10^9) — coordinates of the start and the end cells. 

Output

For each test case, in a separate line, print the number of possible different sums on the way from the start cell to the end cell.

Example

Input


4
1 1 2 2
1 2 2 4
179 1 179 100000
5 7 5 7


Output


2
3
1
1

Note

In the first test case there are two possible sums: 1+2+5=8 and 1+3+5=9. <image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin, stdout
from collections import Counter, defaultdict
from itertools import permutations, combinations
raw_input = stdin.readline
pr = stdout.write
mod=10**9+7

def ni():
    return int(raw_input())


def li():
    return map(int,raw_input().split())


def pn(n):
    stdout.write(str(n)+'\n')


def pa(arr):
    pr(' '.join(map(str,arr))+'\n')

# fast read function for total integer input

def inp():
    # this function returns whole input of
    # space/line seperated integers
    # Use Ctrl+D to flush stdin.
    return map(int,stdin.read().split())

range = xrange # not for python 3.0+

# main code

for t in range(ni()):
    x1,y1,x2,y2=li()
    m=x2-x1
    n=y2-y1
    pn((n*m)+1)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Ccelexupdatebootcamp(Basebootcamp):
    def __init__(self, max_x=10**9, max_y=10**9, max_dx=100, max_dy=100):
        self.max_x = max_x
        self.max_y = max_y
        self.max_dx = max_dx
        self.max_dy = max_dy
    
    def case_generator(self):
        dx = random.randint(0, self.max_dx)
        x1 = random.randint(1, self.max_x - dx) if dx < self.max_x else 1
        x2 = x1 + dx
        
        dy = random.randint(0, self.max_dy)
        y1 = random.randint(1, self.max_y - dy) if dy < self.max_y else 1
        y2 = y1 + dy
        
        return {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        x1 = question_case['x1']
        y1 = question_case['y1']
        x2 = question_case['x2']
        y2 = question_case['y2']
        return f"""在Ccelexupdate-2021的GAZ-GIZ函数生成的无限表格中，每个单元格的数值有特定的填充规则。Levian需要你解决一个路径总和计数问题：

任务描述：
从起点({x1},{y1})出发，只能向右或向下移动，到达终点({x2},{y2})。计算所有可能路径中不同元素总和的个数。

规则说明：
1. 路径必须严格向右或向下移动，即每一步只能增加x或y坐标。
2. 多个不同路径可能产生相同的总和，需要统计所有唯一的总和值数量。

示例提示：
当起点为(1,1)，终点为(2,2)时，正确答案是2种不同总和。

请将你的最终答案放置在[answer]和[/answer]标签之间，例如：[answer]5[/answer]。"""
    
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
        m = identity['x2'] - identity['x1']
        n = identity['y2'] - identity['y1']
        return solution == m * n + 1
