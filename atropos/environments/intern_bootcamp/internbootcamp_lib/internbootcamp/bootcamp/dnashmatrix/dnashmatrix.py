"""# 

### 谜题描述
Nash designed an interesting yet simple board game where a player is simply required to follow instructions written on the cell where the player currently stands. 

This board game is played on the n× n board. Rows and columns of this board are numbered from 1 to n. The cell on the intersection of the r-th row and c-th column is denoted by (r, c).

Some cells on the board are called blocked zones. On each cell of the board, there is written one of the following 5 characters — U, D, L, R or X — instructions for the player. Suppose that the current cell is (r, c). If the character is R, the player should move to the right cell (r, c+1), for L the player should move to the left cell (r, c-1), for U the player should move to the top cell (r-1, c), for D the player should move to the bottom cell (r+1, c). Finally, if the character in the cell is X, then this cell is the blocked zone. The player should remain in this cell (the game for him isn't very interesting from now on).

It is guaranteed that the characters are written in a way that the player will never have to step outside of the board, no matter at which cell he starts.

As a player starts from a cell, he moves according to the character in the current cell. The player keeps moving until he lands in a blocked zone. It is also possible that the player will keep moving infinitely long.

For every of the n^2 cells of the board Alice, your friend, wants to know, how will the game go, if the player starts in this cell. For each starting cell of the board, she writes down the cell that the player stops at, or that the player never stops at all. She gives you the information she has written: for each cell (r, c) she wrote: 

  * a pair (x,y), meaning if a player had started at (r, c), he would end up at cell (x,y). 
  * or a pair (-1,-1), meaning if a player had started at (r, c), he would keep moving infinitely long and would never enter the blocked zone. 



It might be possible that Alice is trying to fool you and there's no possible grid that satisfies all the constraints Alice gave you. For the given information Alice provided you, you are required to decipher a possible board, or to determine that such a board doesn't exist. If there exist several different boards that satisfy the provided information, you can find any of them.

Input

The first line of the input contains a single integer n (1 ≤ n ≤ 10^{3}) — the side of the board.

The i-th of the next n lines of the input contains 2n integers x_1, y_1, x_2, y_2, ..., x_n, y_n, where (x_j, y_j) (1 ≤ x_j ≤ n, 1 ≤ y_j ≤ n, or (x_j,y_j)=(-1,-1)) is the pair written by Alice for the cell (i, j). 

Output

If there doesn't exist a board satisfying the information that Alice gave you, print a single line containing INVALID. 

Otherwise, in the first line print VALID. In the i-th of the next n lines, print the string of n characters, corresponding to the characters in the i-th row of the suitable board you found. Each character of a string can either be U, D, L, R or X. If there exist several different boards that satisfy the provided information, you can find any of them.

Examples

Input


2
1 1 1 1
2 2 2 2


Output


VALID
XL
RX


Input


3
-1 -1 -1 -1 -1 -1
-1 -1 2 2 -1 -1
-1 -1 -1 -1 -1 -1


Output


VALID
RRD
UXD
ULL

Note

For the sample test 1 :

The given grid in output is a valid one. 

  * If the player starts at (1,1), he doesn't move any further following X and stops there. 
  * If the player starts at (1,2), he moves to left following L and stops at (1,1). 
  * If the player starts at (2,1), he moves to right following R and stops at (2,2). 
  * If the player starts at (2,2), he doesn't move any further following X and stops there. 



The simulation can be seen below : 

<image>

For the sample test 2 : 

The given grid in output is a valid one, as a player starting at any cell other than the one at center (2,2), keeps moving in an infinitely long cycle and never stops. Had he started at (2,2), he wouldn't have moved further following instruction X .

The simulation can be seen below : 

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division, print_function

import os
import sys
from io import BytesIO, IOBase

if sys.version_info[0] < 3:
    from __builtin__ import xrange as range
    from future_builtins import ascii, filter, hex, map, oct, zip

TEST_CASES = False
def main():
    n = get_int()
    mat = [[] for _ in range(n)]
    res = [['A' for _ in range(n)] for _ in range(n)]
    for j in range(n):
        li = get_list()
        for i in range(0, 2 * n, 2):
            x, y = li[i] - 1, li[i + 1] - 1
            mat[j].append([x, y])

    def nbrs(i, j):
        res = []
        if i > 0: res.append([i - 1, j, 'U', 'D'])
        if i < n - 1: res.append([i + 1, j, 'D', 'U'])
        if j > 0: res.append([i, j - 1, 'L', 'R'])
        if j < n - 1: res.append([i, j + 1, 'R', 'L'])
        return res

    for i in range(n):
        for j in range(n):
            x, y = mat[i][j]
            if x < 0:
                for x1, y1, loc, _ in nbrs(i, j):
                    if mat[x1][y1][0] < 0:
                        res[i][j] = loc
                        break
            else:
                if x == i and y == j:
                    queue = [[x, y]]
                    res[x][y] = 'X'
                    while len(queue) > 0:
                        new_q = []
                        for qx1, qx2 in queue:
                            for nx1, nx2, _, loc in nbrs(qx1, qx2):
                                if res[nx1][nx2] == 'A' and mat[qx1][qx2] == mat[nx1][nx2]:
                                    res[nx1][nx2] = loc
                                    new_q.append([nx1, nx2])
                        queue = new_q
                    # BFS Backward

    for i in range(n):
        for j in range(n):
            if res[i][j] == 'A':
                print(\"INVALID\")
                return
    print(\"VALID\")
    for li in res:
        print(\"\".join(li))


# region fastio

BUFSIZE = 8192


class FastIO(IOBase):
    newlines = 0

    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = \"x\" in file.mode or \"r\" not in file.mode
        self.write = self.buffer.write if self.writable else None

    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()

    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b\"\n\") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()

    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)


class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode(\"ascii\"))
        self.read = lambda: self.buffer.read().decode(\"ascii\")
        self.readline = lambda: self.buffer.readline().decode(\"ascii\")


def print(*args, **kwargs):
    \"\"\"Prints the values to a stream, or to sys.stdout by default.\"\"\"
    sep, file = kwargs.pop(\"sep\", \" \"), kwargs.pop(\"file\", sys.stdout)
    at_start = True
    for x in args:
        if not at_start:
            file.write(sep)
        file.write(str(x))
        at_start = False
    file.write(kwargs.pop(\"end\", \"\n\"))
    if kwargs.pop(\"flush\", False):
        file.flush()


if sys.version_info[0] < 3:
    sys.stdin, sys.stdout = FastIO(sys.stdin), FastIO(sys.stdout)
else:
    sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)

input = lambda: sys.stdin.readline().rstrip(\"\r\n\")
get_int = lambda: int(input())
get_list = lambda: list(map(int, input().split()))
# endregion

if __name__ == \"__main__\":
    if TEST_CASES: [main() for _ in range(int(input()))]
    else: main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Dnashmatrixbootcamp(Basebootcamp):
    def __init__(self, n=3, x_prob=0.2, invalid_prob=0.3):
        self.n = n
        self.x_prob = x_prob
        self.invalid_prob = invalid_prob  # 生成无效案例的概率

    def case_generator(self):
        if random.random() < self.invalid_prob:
            return self.generate_invalid_case()
        else:
            return self.generate_valid_case()

    def generate_valid_case(self):
        """生成有效案例并标记is_valid=True"""
        grid = self.generate_valid_grid()
        valid_cells = self.simulate_grid(grid)
        return {
            "n": self.n,
            "cells": valid_cells,
            "is_valid": True
        }

    def generate_invalid_case(self):
        """生成无效案例并标记is_valid=False"""
        n = self.n
        # 创建一个必定矛盾的案例：所有单元格要求最终到达同一个X但路径冲突
        valid_grid = self.generate_valid_grid()
        cells = self.simulate_grid(valid_grid)
        # 随机选择一个单元格，强制其终止点为另一个单元格，但该单元格并非X且路径无法到达
        i, j = random.randint(0, n-1), random.randint(0, n-1)
        target = (random.randint(1, n), random.randint(1, n))
        while target == (i+1, j+1) or valid_grid[i][j] == 'X':
            i, j = random.randint(0, n-1), random.randint(0, n-1)
            target = (random.randint(1, n), random.randint(1, n))
        cells[i][j] = target
        return {
            "n": n,
            "cells": cells,
            "is_valid": False  # 强制标记为无效
        }

    def generate_valid_grid(self):
        """生成合法网格，确保指令不会导致越界"""
        grid = []
        for i in range(self.n):
            row = []
            for j in range(self.n):
                possible = []
                if i > 0:
                    possible.append('U')
                if i < self.n - 1:
                    possible.append('D')
                if j > 0:
                    possible.append('L')
                if j < self.n - 1:
                    possible.append('R')
                possible.append('X')
                # 优先设置X的概率
                if random.random() < self.x_prob:
                    char = 'X'
                else:
                    char = random.choice(possible)
                row.append(char)
            grid.append(row)
        return grid

    def simulate_grid(self, grid):
        """计算每个单元格的终止点"""
        cells = []
        for i in range(self.n):
            row = []
            for j in range(self.n):
                termination = self.simulate_cell(i, j, grid)
                row.append(termination)
            cells.append(row)
        return cells

    def simulate_cell(self, r, c, grid):
        """模拟玩家移动，返回终止点或(-1,-1)"""
        visited = set()
        current_r, current_c = r, c
        while True:
            if (current_r, current_c) in visited:
                return (-1, -1)
            visited.add((current_r, current_c))
            char = grid[current_r][current_c]
            if char == 'X':
                return (current_r + 1, current_c + 1)
            elif char == 'U':
                current_r -= 1
            elif char == 'D':
                current_r += 1
            elif char == 'L':
                current_c -= 1
            elif char == 'R':
                current_c += 1

    @staticmethod
    def prompt_func(question_case):
        """构造完整的谜题描述"""
        n = question_case['n']
        cells = question_case['cells']
        input_lines = [str(n)]
        for row in cells:
            parts = []
            for cell in row:
                if cell == (-1, -1):
                    parts.extend([-1, -1])
                else:
                    x, y = cell
                    parts.extend([x, y])
            input_lines.append(' '.join(map(str, parts)))
        problem = f"""Alice设计了一个棋盘游戏，其中每个单元格包含一个指令（U、D、L、R或X）。玩家根据指令移动，直到进入X（阻塞区）或无限循环。给定每个单元格的终止点或标记为无限循环的信息，请判断是否存在有效的棋盘。如果存在，输出VALID并给出一个可能的解；否则，输出INVALID。

输入数据：
第一行是一个整数n，表示棋盘的大小。
接下来n行，每行包含2n个整数，按顺序表示每个单元格的终止点坐标或(-1,-1)。

输入示例：
{chr(10).join(input_lines)}

将答案放置在[answer]标签内。例如：
[answer]
VALID
XL
RX
[/answer]
或：
[answer]
INVALID
[/answer]"""
        return problem

    @staticmethod
    def extract_output(output):
        """从模型输出中提取最后一个答案块"""
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """验证答案正确性，根据identity中的有效性标记"""
        solution_lines = solution.strip().splitlines()
        if not solution_lines:
            return False
        first_line = solution_lines[0].strip().upper()
        is_valid_case = identity.get("is_valid", True)

        if is_valid_case:
            # 案例有效，模型应返回VALID并给出正确网格
            if first_line != "VALID":
                return False
            return cls.check_valid_solution(solution_lines, identity)
        else:
            # 案例无效，模型应返回INVALID
            return first_line == "INVALID"

    @classmethod
    def check_valid_solution(cls, solution_lines, identity):
        """验证有效案例的网格正确性"""
        n = identity["n"]
        if len(solution_lines) != n + 1:
            return False
        grid = solution_lines[1:]
        # 格式检查
        for row in grid:
            if len(row) != n or any(c not in "UDLRX" for c in row):
                return False
        # 指令合法性检查
        for i in range(n):
            for j in range(n):
                c = grid[i][j]
                if (c == 'U' and i == 0) or (c == 'D' and i == n-1) or \
                   (c == 'L' and j == 0) or (c == 'R' and j == n-1):
                    return False
        # 终止点一致性检查
        for i in range(n):
            for j in range(n):
                simulated = cls.static_simulate_cell(i, j, grid)
                expected = identity["cells"][i][j]
                if simulated != expected:
                    return False
        return True

    @staticmethod
    def static_simulate_cell(r, c, grid):
        """静态方法：模拟单元格移动"""
        n = len(grid)
        visited = set()
        current_r, current_c = r, c
        while True:
            if (current_r, current_c) in visited:
                return (-1, -1)
            visited.add((current_r, current_c))
            char = grid[current_r][current_c]
            if char == 'X':
                return (current_r + 1, current_c + 1)
            elif char == 'U':
                current_r -= 1
            elif char == 'D':
                current_r += 1
            elif char == 'L':
                current_c -= 1
            elif char == 'R':
                current_c += 1
