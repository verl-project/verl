"""# 

### 谜题描述
This is an interactive problem

You are given a grid n× n, where n is odd. Rows are enumerated from 1 to n from up to down, columns are enumerated from 1 to n from left to right. Cell, standing on the intersection of row x and column y, is denoted by (x, y).

Every cell contains 0 or 1. It is known that the top-left cell contains 1, and the bottom-right cell contains 0.

We want to know numbers in all cells of the grid. To do so we can ask the following questions: 

\"? x_1 y_1 x_2 y_2\", where 1 ≤ x_1 ≤ x_2 ≤ n, 1 ≤ y_1 ≤ y_2 ≤ n, and x_1 + y_1 + 2 ≤ x_2 + y_2. In other words, we output two different cells (x_1, y_1), (x_2, y_2) of the grid such that we can get from the first to the second by moving only to the right and down, and they aren't adjacent.

As a response to such question you will be told if there exists a path between (x_1, y_1) and (x_2, y_2), going only to the right or down, numbers in cells of which form a palindrome.

For example, paths, shown in green, are palindromic, so answer for \"? 1 1 2 3\" and \"? 1 2 3 3\" would be that there exists such path. However, there is no palindromic path between (1, 1) and (3, 1).

<image>

Determine all cells of the grid by asking not more than n^2 questions. It can be shown that the answer always exists.

Input

The first line contains odd integer (3 ≤ n < 50) — the side of the grid.

Interaction

You begin the interaction by reading n.

To ask a question about cells (x_1, y_1), (x_2, y_2), in a separate line output \"? x_1 y_1 x_2 y_2\".

Numbers in the query have to satisfy 1 ≤ x_1 ≤ x_2 ≤ n, 1 ≤ y_1 ≤ y_2 ≤ n, and x_1 + y_1 + 2 ≤ x_2 + y_2. Don't forget to 'flush', to get the answer.

In response, you will receive 1, if there exists a path going from (x_1, y_1) to (x_2, y_2) only to the right or down, numbers in cells of which form a palindrome, and 0 otherwise.

In case your query is invalid or you asked more than n^2 queries, program will print -1 and will finish interaction. You will receive Wrong answer verdict. Make sure to exit immediately to avoid getting other verdicts.

When you determine numbers in all cells, output \"!\".

Then output n lines, the i-th of which is a string of length n, corresponding to numbers in the i-th row of the grid.

After printing a query do not forget to output end of line and flush the output. Otherwise, you will get Idleness limit exceeded. To do this, use:

  * fflush(stdout) or cout.flush() in C++;
  * System.out.flush() in Java;
  * flush(output) in Pascal;
  * stdout.flush() in Python;
  * see documentation for other languages.



Hack Format

To hack, use the following format.

The first line should contain a single odd integer n (side of your grid).

The i-th of n following lines should contain a string of length n corresponding to the i-th row of the grid. Top left element of the grid has to be equal to 1, bottom right has to be equal to 0.

Example

Input


3
0
1
0
1
1
1
1

Output


? 1 1 1 3
? 1 1 2 3
? 2 1 2 3
? 3 1 3 3
? 2 2 3 3
? 1 2 3 2
? 1 2 3 3
!
100
001
000

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdout
def Q(x1, y1, x2, y2):
    print '?', x1 + 1, y1 + 1, x2 + 1, y2 + 1
    stdout.flush()
    return raw_input()[0] == '1'
def A(a):
    n = len(a)
    print \"!\"
    for i in xrange(n):
        print ''.join(map(str, a[i]))
    stdout.flush()
    quit()
def q(x1, y1, x2, y2, a, b, rev=0):
    x = Q(x1, y1, x2, y2) ^ 1
    if rev:
        x1, y1, x2, y2 = x2, y2, x1, y1
    a[x2][y2] = a[x1][y1] ^ x
    b[x2][y2] = b[x1][y1] ^ x
def ok(a, x1, y1, x2, y2):
    x = Q(x1, y1, x2, y2)
    s = {}
    f = 0
    if x2 - x1 == 1:
        s[y1] = str(a[x1][y1])
        for i in xrange(y1 + 1, y2 + 1):
            s[i] = s[i-1] + str(a[x1][i])
        for i in xrange(y1, y2 + 1):
            t = s[i]
            for j in xrange(i, y2 + 1):
                t += str(a[x1+1][j])
            if t == t[::-1]:
                f = 1
    else:
        s[x1] = str(a[x1][y1])
        for i in xrange(x1 + 1, x2 + 1):
            s[i] = s[i-1] + str(a[i][y1])
        for i in xrange(x1, x2 + 1):
            t = s[i]
            for j in xrange(i, x2 + 1):
                t += str(a[j][y1+1])
            if t == t[::-1]:
                f = 1
    return x == f
 
def main():
    n = int(raw_input())
    a = [[-1] * n for _ in xrange(n)]
    a[0][0] = 1
    a[-1][-1] = 0
    b = [l[:] for l in a]
    a[0][1] = 0
    b[0][1] = 1
    q(0, 0, 1, 1, a, b)
    for i in xrange(n):
        j = i % 2
        if i >= 2:
            q(i - 2, j, i, j, a, b)
        nd = n
        if i == n - 1:
            nd -= 2
        while j + 2 < nd:
            q(i, j, i, j + 2, a, b)
            j += 2
    for i in xrange(1, n - 3, 2):
        q(0, i, 0, i + 2, a, b)
    q(0, n - 2, 1, n - 1, a, b)
    for i in xrange(n - 1, 1, -2):
        q(1, i - 2, 1, i, a, b, 1)
    for i in xrange(2, n):
        j = (i % 2) ^ 1
        q(i - 2, j, i, j, a, b)
        while j + 2 < n:
            q(i, j, i, j + 2, a, b)
            j += 2
    for i in xrange(0, n - 1, 2):
        for j in xrange(0, n - 1, 2):
            if a[i][j] != a[i+2][j+2]:
                if ok(a, i, j, i + 1, j + 2) and ok(a, i, j, i + 2, j + 1) and ok(a, i + 1, j, i + 2, j + 2):# and ok(a, i, j + 1, i + 2, j + 2):
                    A(a)
                else:
                    A(b)
main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random

class Cpalindromicpathsbootcamp(Basebootcamp):
    def __init__(self, n=3):
        if (n % 2) == 0:
            n += 1  # 确保n为奇数
        self.n = n
    
    def case_generator(self):
        n = self.n
        grid = [[0] * n for _ in range(n)]
        grid[0][0] = 1  # 左上角为1
        grid[-1][-1] = 0  # 右下角为0
        
        # 随机生成其他单元格的值
        for i in range(n):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                if i == n-1 and j == n-1:
                    continue
                grid[i][j] = random.choice([0, 1])
        
        return {
            'n': n,
            'grid': grid
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        grid = question_case['grid']
        example = "\n".join(["".join(map(str, row)) for row in grid])
        
        prompt = f"你有一个{n}×{n}的网格。已知左上角（1,1）的值是1，右下角（{n},{n}）的值是0。其他单元格的值是0或1。你需要通过询问问题来确定整个网格的值。每次询问的形式是“? x1 y1 x2 y2”，并得到0或1的回答。你的任务是输出整个网格的值，每行是一个由0和1组成的字符串，例如：\n\n"
        prompt += f"示例网格：\n{example}\n\n"
        prompt += "请将答案放在[answer]标签中，格式如下：\n"
        prompt += "[answer]\n"
        prompt += "100\n"
        prompt += "001\n"
        prompt += "000\n"
        prompt += "[/answer]\n"
        return prompt
    
    @staticmethod
    def extract_output(output):
        import re
        pattern = r'\[answer\](.*?)\[\/answer\]'
        matches = re.findall(pattern, output, re.DOTALL)
        if not matches:
            return None
        answer = matches[-1].strip()
        rows = answer.split('\n')
        rows = [row.strip() for row in rows if row.strip() != '']
        for row in rows:
            if not all(c in '01' for c in row):
                return None
        return rows
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        expected_grid = identity['grid']
        
        if len(solution) != n:
            return False
        for row in solution:
            if len(row) != n:
                return False
        
        actual_grid = []
        for row in solution:
            actual_grid.append([int(c) for c in row])
        
        for i in range(n):
            for j in range(n):
                if actual_grid[i][j] != expected_grid[i][j]:
                    return False
        return True
