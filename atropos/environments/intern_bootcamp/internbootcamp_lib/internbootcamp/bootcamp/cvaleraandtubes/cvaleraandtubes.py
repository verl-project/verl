"""# 

### 谜题描述
Valera has got a rectangle table consisting of n rows and m columns. Valera numbered the table rows starting from one, from top to bottom and the columns – starting from one, from left to right. We will represent cell that is on the intersection of row x and column y by a pair of integers (x, y).

Valera wants to place exactly k tubes on his rectangle table. A tube is such sequence of table cells (x1, y1), (x2, y2), ..., (xr, yr), that: 

  * r ≥ 2; 
  * for any integer i (1 ≤ i ≤ r - 1) the following equation |xi - xi + 1| + |yi - yi + 1| = 1 holds; 
  * each table cell, which belongs to the tube, must occur exactly once in the sequence. 



Valera thinks that the tubes are arranged in a fancy manner if the following conditions are fulfilled: 

  * no pair of tubes has common cells; 
  * each cell of the table belongs to some tube. 



Help Valera to arrange k tubes on his rectangle table in a fancy manner.

Input

The first line contains three space-separated integers n, m, k (2 ≤ n, m ≤ 300; 2 ≤ 2k ≤ n·m) — the number of rows, the number of columns and the number of tubes, correspondingly. 

Output

Print k lines. In the i-th line print the description of the i-th tube: first print integer ri (the number of tube cells), then print 2ri integers xi1, yi1, xi2, yi2, ..., xiri, yiri (the sequence of table cells).

If there are multiple solutions, you can print any of them. It is guaranteed that at least one solution exists. 

Examples

Input

3 3 3


Output

3 1 1 1 2 1 3
3 2 1 2 2 2 3
3 3 1 3 2 3 3


Input

2 3 1


Output

6 1 1 1 2 1 3 2 3 2 2 2 1

Note

Picture for the first sample: 

<image>

Picture for the second sample: 

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, m, k = map(int, raw_input().split())
path = []
for x in xrange(1, n+1):
  if x % 2 == 1:
    for y in xrange(1, m+1):
      path.append((x, y))
  else:
    for y in xrange(m, 0, -1):
      path.append((x, y))
for i in xrange(k-1):
  print 2, path[2*i][0], path[2*i][1], path[2*i+1][0], path[2*i+1][1]
print n*m - 2*(k-1),
for i in xrange(2*k-2, n*m):
  print path[i][0], path[i][1],
print
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random

class Cvaleraandtubesbootcamp(Basebootcamp):

    def __init__(self, min_n=2, max_n=10, min_m=2, max_m=10):
        self.min_n = min_n
        self.max_n = max_n
        self.min_m = min_m
        self.max_m = max_m

    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        m = random.randint(self.min_m, self.max_m)
        max_k = (n * m) // 2
        k = random.randint(1, max_k)
        return {
            'n': n,
            'm': m,
            'k': k
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        m = question_case['m']
        k = question_case['k']
        prompt = f"""你是一名网格谜题解决专家。Valera有一个{n}行{m}列的矩形网格，他需要在这个网格上放置{k}个管子。每个管子由一系列相邻的网格单元格组成，具体要求如下：

1. 每个管子必须是一个连续的单元格序列，相邻单元格之间上下或左右相邻（即曼哈顿距离为1）。
2. 每个管子至少包含2个单元格，且每个单元格在管子中只能出现一次。
3. 所有{k}个管子必须完全覆盖网格的所有单元格，且管子之间不能有任何重叠。

请编写一个程序，按照以下格式输出解决方案：

- 输出{k}行，每行描述一个管子。
- 每行的格式为：首先是一个整数r_i表示该管子的单元格数量，接着是2*r_i个整数，依次表示每个单元格的x坐标和y坐标（按照管子从起点到终点的顺序排列）。

示例输入（n=3, m=3, k=3）的输出可能为：

[answer]
3 1 1 1 2 1 3
3 2 1 2 2 2 3
3 3 1 3 2 3 3
[/answer]

注意：
- x的范围是1到{n}，y的范围是1到{m}。
- 确保每个单元格被恰好一个管子使用，且所有管子遵守相邻规则。
- 将你的答案放置在[answer]和[/answer]标签之间。

请解决以下具体问题：
n = {n}, m = {m}, k = {k}
"""
        return prompt

    @staticmethod
    def extract_output(output):
        lines = output.split('\n')
        answer_blocks = []
        current_block = []
        in_answer = False

        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith('[answer]'):
                in_answer = True
                current_block = []
            elif stripped_line.startswith('[/answer]'):
                in_answer = False
                if current_block:
                    answer_blocks.append(current_block)
                    current_block = []
            elif in_answer:
                current_block.append(stripped_line)

        if not answer_blocks:
            return None

        last_block = answer_blocks[-1]
        tubes = []

        for line in last_block:
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 3:
                continue

            try:
                r_i = int(parts[0])
            except ValueError:
                continue

            if len(parts) != 1 + 2 * r_i:
                continue

            if r_i < 2:
                continue

            try:
                coords = []
                for i in range(r_i):
                    x = int(parts[1 + 2 * i])
                    y = int(parts[2 + 2 * i])
                    coords.append((x, y))
                tubes.append(coords)
            except (ValueError, IndexError):
                continue

        return tubes or None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False

        n = identity['n']
        m = identity['m']
        k = identity['k']

        if len(solution) != k:
            return False

        for tube in solution:
            if len(tube) < 2:
                return False

        all_cells = set()
        for tube in solution:
            for (x, y) in tube:
                if x < 1 or x > n or y < 1 or y > m:
                    return False
                if (x, y) in all_cells:
                    return False
                all_cells.add((x, y))

        if len(all_cells) != n * m:
            return False

        for tube in solution:
            prev = tube[0]
            visited = {prev}
            for current in tube[1:]:
                dx = abs(current[0] - prev[0])
                dy = abs(current[1] - prev[1])
                if dx + dy != 1:
                    return False
                if current in visited:
                    return False
                visited.add(current)
                prev = current

        return True
