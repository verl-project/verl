"""# 

### 谜题描述
As the boat drifts down the river, a wood full of blossoms shows up on the riverfront.

\"I've been here once,\" Mino exclaims with delight, \"it's breathtakingly amazing.\"

\"What is it like?\"

\"Look, Kanno, you've got your paintbrush, and I've got my words. Have a try, shall we?\" 

There are four kinds of flowers in the wood, Amaranths, Begonias, Centaureas and Dianthuses.

The wood can be represented by a rectangular grid of n rows and m columns. In each cell of the grid, there is exactly one type of flowers.

According to Mino, the numbers of connected components formed by each kind of flowers are a, b, c and d respectively. Two cells are considered in the same connected component if and only if a path exists between them that moves between cells sharing common edges and passes only through cells containing the same flowers.

You are to help Kanno depict such a grid of flowers, with n and m arbitrarily chosen under the constraints given below. It can be shown that at least one solution exists under the constraints of this problem.

Note that you can choose arbitrary n and m under the constraints below, they are not given in the input.

Input

The first and only line of input contains four space-separated integers a, b, c and d (1 ≤ a, b, c, d ≤ 100) — the required number of connected components of Amaranths, Begonias, Centaureas and Dianthuses, respectively.

Output

In the first line, output two space-separated integers n and m (1 ≤ n, m ≤ 50) — the number of rows and the number of columns in the grid respectively.

Then output n lines each consisting of m consecutive English letters, representing one row of the grid. Each letter should be among 'A', 'B', 'C' and 'D', representing Amaranths, Begonias, Centaureas and Dianthuses, respectively.

In case there are multiple solutions, print any. You can output each letter in either case (upper or lower).

Examples

Input

5 3 2 1


Output

4 7
DDDDDDD
DABACAD
DBABACD
DDDDDDD

Input

50 50 1 1


Output

4 50
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
ABABABABABABABABABABABABABABABABABABABABABABABABAB
BABABABABABABABABABABABABABABABABABABABABABABABABA
DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD

Input

1 6 4 5


Output

7 7
DDDDDDD
DDDBDBD
DDCDCDD
DBDADBD
DDCDCDD
DBDBDDD
DDDDDDD

Note

In the first example, each cell of Amaranths, Begonias and Centaureas forms a connected component, while all the Dianthuses form one.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from fractions import gcd
from math import factorial, ceil, sqrt, atan2, log, pi, e, asin,acos, cos, sin, floor
from itertools import *
from fractions import Fraction
import string
import copy
import random
import bisect
from decimal import *
def id_generator(size=20, chars=string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
 
def mp():
    return map(int,str(raw_input()).split())

a,b,c,d=mp()
l=[['A' for i in range(50)] for j in range(50)]
for i in range(50):
    for j in range(25,50):
        l[i][j]='B'
a-=1
b-=1
for i in range(1,50,2):
    for j in range(1,25,2):
        if b:
            l[i][j]='B'
            b-=1
        else:
            break
            
for i in range(1,50,2):
    for j in range(1,25,2):
        if c:
            if l[i][j]=='A':
                l[i][j]='C'
                c-=1
        else:
            break
        
for i in range(1,50,2):
    for j in range(1,25,2):
        if d:
            if l[i][j]=='A':
                l[i][j]='D'
                d-=1
        else:
            break
            
for i in range(1,50,2):
    for j in range(26,50,2):
        if a:
            l[i][j]='A'
            a-=1
        else:
            break
            
for i in range(1,50,2):
    for j in range(26,50,2):
        if c:
            if l[i][j]=='B':
                l[i][j]='C'
                c-=1
        else:
            break
        
for i in range(1,50,2):
    for j in range(26,50,2):
        if d:
            if l[i][j]=='B':
                l[i][j]='D'
                d-=1
        else:
            break

print 50,50
for i in l:
    s=''
    for j in i:
        s+=j
    print s
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random

class Camistofflorescencebootcamp(Basebootcamp):
    def __init__(self):
        pass

    def case_generator(self):
        # 随机选择n和m的值，范围在1到50之间
        n = random.randint(1, 50)
        m = random.randint(1, 50)
        # 生成一个n行m列的网格
        grid = self.generate_grid(n, m)
        # 计算各花种的连通块数目
        a = self.count_components(grid, 'A')
        b = self.count_components(grid, 'B')
        c = self.count_components(grid, 'C')
        d = self.count_components(grid, 'D')
        # 返回问题实例
        return {
            'n': n,
            'm': m,
            'grid': [ ''.join(row) for row in grid ],
            'a': a,
            'b': b,
            'c': c,
            'd': d
        }

    @staticmethod
    def prompt_func(question_case):
        a = question_case['a']
        b = question_case['b']
        c = question_case['c']
        d = question_case['d']
        prompt = f"给定四个整数a={a}, b={b}, c={c}, d={d}，请构造一个矩形网格，使得网格中A、B、C、D的连通块数目分别为a、b、c、d。连通块的定义是上下左右相邻且花相同的区域。输出格式为：第一行是网格的行数n和列数m，接下来n行是每行的字符串。请将答案放在[answer]和[/answer]标签之间。"
        return prompt

    @staticmethod
    def extract_output(output):
        # 查找最后的[answer]和[/answer]标签
        start = output.rfind('[answer]')
        end = output.rfind('[/answer]')
        if start == -1 or end == -1 or start >= end:
            return None
        content = output[start + len('[answer]'):end].strip()
        # 分割内容为行
        lines = content.split('\n')
        if len(lines) < 2:
            return None
        # 提取n和m
        first_line = lines[0].strip()
        try:
            n, m = map(int, first_line.split())
        except:
            return None
        if len(lines) - 1 < n:
            return None
        grid = []
        for i in range(n):
            line = lines[i + 1].strip()
            if len(line) != m:
                return None
            grid.append(line)
        # 检查每个字符是否有效
        for row in grid:
            for c in row:
                if c.upper() not in {'A', 'B', 'C', 'D'}:
                    return None
        return {
            'n': n,
            'm': m,
            'grid': grid
        }

    @classmethod
    def _verify_correction(cls, solution, identity):
        solution_grid = solution['grid']
        a = cls.count_components(solution_grid, 'A')
        b = cls.count_components(solution_grid, 'B')
        c = cls.count_components(solution_grid, 'C')
        d = cls.count_components(solution_grid, 'D')
        expected_a = identity['a']
        expected_b = identity['b']
        expected_c = identity['c']
        expected_d = identity['d']
        # 验证各连通块数目是否正确
        return a == expected_a and b == expected_b and c == expected_c and d == expected_d

    def generate_grid(self, n, m):
        # 初始化网格，所有位置填充为'A'
        grid = [['A' for _ in range(m)] for _ in range(n)]
        # 从中间开始，将右侧填充为'B'
        mid = m // 2
        for i in range(n):
            for j in range(mid, m):
                grid[i][j] = 'B'
        # 生成各花种的连通块数目
        # 确保a, b, c, d的值在合理范围内
        a = 1
        b = 1
        c = 0
        d = 0
        # 在左侧填充'C'和'D'
        for i in range(1, n, 2):
            for j in range(1, mid, 2):
                if c < 100:
                    grid[i][j] = 'C'
                    c += 1
                elif d < 100:
                    grid[i][j] = 'D'
                    d += 1
        # 在右侧填充'A', 'C', 'D'
        for i in range(1, n, 2):
            for j in range(mid + 1, m, 2):
                if a < 100:
                    grid[i][j] = 'A'
                    a += 1
                elif c < 100:
                    grid[i][j] = 'C'
                    c += 1
                elif d < 100:
                    grid[i][j] = 'D'
                    d += 1
        return grid

    @staticmethod
    def count_components(grid, char):
        rows = len(grid)
        if rows == 0:
            return 0
        cols = len(grid[0])
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        count = 0
        for i in range(rows):
            for j in range(cols):
                if not visited[i][j] and grid[i][j].upper() == char.upper():
                    queue = [(i, j)]
                    visited[i][j] = True
                    while queue:
                        x, y = queue.pop(0)
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx = x + dx
                            ny = y + dy
                            if 0 <= nx < rows and 0 <= ny < cols:
                                if not visited[nx][ny] and grid[nx][ny].upper() == char.upper():
                                    visited[nx][ny] = True
                                    queue.append((nx, ny))
                    count += 1
        return count
