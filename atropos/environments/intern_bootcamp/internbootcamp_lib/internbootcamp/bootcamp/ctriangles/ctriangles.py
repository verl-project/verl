"""# 

### 谜题描述
Gildong has a square board consisting of n rows and n columns of square cells, each consisting of a single digit (from 0 to 9). The cell at the j-th column of the i-th row can be represented as (i, j), and the length of the side of each cell is 1. Gildong likes big things, so for each digit d, he wants to find a triangle such that:

  * Each vertex of the triangle is in the center of a cell. 
  * The digit of every vertex of the triangle is d. 
  * At least one side of the triangle is parallel to one of the sides of the board. You may assume that a side of length 0 is parallel to both sides of the board. 
  * The area of the triangle is maximized. 



Of course, he can't just be happy with finding these triangles as is. Therefore, for each digit d, he's going to change the digit of exactly one cell of the board to d, then find such a triangle. He changes it back to its original digit after he is done with each digit. Find the maximum area of the triangle he can make for each digit.

Note that he can put multiple vertices of the triangle on the same cell, and the triangle can be a [degenerate triangle](https://cutt.ly/NhbjZ2l); i.e. the area of the triangle can be 0. Also, note that he is allowed to change the digit of a cell from d to d.

Input

Each test contains one or more test cases. The first line contains the number of test cases t (1 ≤ t ≤ 1000).

The first line of each test case contains one integer n (1 ≤ n ≤ 2000) — the number of rows and columns of the board.

The next n lines of each test case each contain a string of n digits without spaces. The j-th digit of the i-th line is the digit of the cell at (i, j). Each digit is one of the characters from 0 to 9.

It is guaranteed that the sum of n^2 in all test cases doesn't exceed 4 ⋅ 10^6.

Output

For each test case, print one line with 10 integers. The i-th integer is the maximum area of triangle Gildong can make when d = i-1, multiplied by 2.

Example

Input


5
3
000
122
001
2
57
75
4
0123
4012
3401
2340
1
9
8
42987101
98289412
38949562
87599023
92834718
83917348
19823743
38947912


Output


4 4 1 0 0 0 0 0 0 0
0 0 0 0 0 1 0 1 0 0
9 6 9 9 6 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
18 49 49 49 49 15 0 30 42 42

Note

In the first case, for d=0, no matter which cell he chooses to use, the triangle with vertices at (1, 1), (1, 3), and (3, 1) is the biggest triangle with area of \cfrac{2 ⋅ 2}{2} = 2. Since we should print it multiplied by 2, the answer for d=0 is 4.

For d=1, Gildong can change the digit of the cell at (1, 3) into 1, making a triangle with vertices on all three 1's that has an area of 2.

For d=2, Gildong can change the digit of one of the following six cells into 2 to make a triangle with an area of \cfrac{1}{2}: (1, 1), (1, 2), (1, 3), (3, 1), (3, 2), and (3, 3).

For the remaining digits (from 3 to 9), the cell Gildong chooses to change will be the only cell that contains that digit. Therefore the triangle will always be a degenerate triangle with an area of 0.

In the third case, for d=4, note that the triangle will be bigger than the answer if Gildong changes the digit of the cell at (1, 4) and use it along with the cells at (2, 1) and (4, 3), but this is invalid because it violates the condition that at least one side of the triangle must be parallel to one of the sides of the board.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from collections import Counter, defaultdict, deque
import bisect
from sys import stdin, stdout
from itertools import repeat
import math


def inp(force_list=False):
    re = map(int, raw_input().split())
    if len(re) == 1 and not force_list:
        return re[0]
    return re

def inst():
    return raw_input().strip()

def gcd(x, y):
   while(y):
       x, y = y, x % y
   return x

mod = 1000000007

def my_main():
    kase = inp()
    pans = []
    for i in range(kase):
        n = inp()
        da = []
        for i in range(n):
            da.append(inst())
        l,r,u,d = [], [], [], []
        for i in range(10):
            l.append(n)
            r.append(-1)
            u.append(n)
            d.append(-1)
        ans = [0 for i in range(10)]
        for i in range(n):
            for j in range(n):
                idx = int(da[i][j])
                l[idx] = min(l[idx], j)
                r[idx] = max(r[idx], j)
                u[idx] = min(u[idx], i)
                d[idx] = max(d[idx], i)
        for i in range(n):
            for j in range(n):
                idx = int(da[i][j])
                ans[idx] = max(ans[idx], (n-1-i)*(r[idx]-j))
                ans[idx] = max(ans[idx], (n-1-i)*(j-l[idx]))
                ans[idx] = max(ans[idx], (i)*(r[idx]-j))
                ans[idx] = max(ans[idx], (i)*(j-l[idx]))

                ans[idx] = max(ans[idx], (d[idx]-i)*(n-1-j))
                ans[idx] = max(ans[idx], (i-u[idx])*(n-1-j))
                ans[idx] = max(ans[idx], (d[idx]-i)*(j))
                ans[idx] = max(ans[idx], (i-u[idx])*(j))
        pans.append(' '.join(map(str, ans)))

    print '\n'.join(pans)

my_main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Ctrianglesbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=5, **kwargs):
        super().__init__(**kwargs)
        self.min_n = min_n
        self.max_n = max_n

    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        board = [''.join(random.choices('0123456789', k=n)) for _ in range(n)]
        expected = self.calculate_answer(n, board)
        return {'n': n, 'board': board, 'expected': expected}

    @staticmethod
    def calculate_answer(n, board):
        ans = [0] * 10
        for d in range(10):
            min_row = max_row = min_col = max_col = None
            # Find original extents for d
            original_exists = False
            for i in range(n):
                for j in range(n):
                    if int(board[i][j]) == d:
                        if not original_exists:
                            min_row = max_row = i
                            min_col = max_col = j
                            original_exists = True
                        else:
                            min_row = min(min_row, i)
                            max_row = max(max_row, i)
                            min_col = min(min_col, j)
                            max_col = max(max_col, j)
            
            if not original_exists:
                # Must create one cell of d (area remains 0)
                ans[d] = 0
                continue

            max_area = (max_row - min_row) * (max_col - min_col)
            # Check all possible cell modifications
            for i in range(n):
                for j in range(n):
                    new_min_row = min(min_row, i)
                    new_max_row = max(max_row, i)
                    new_min_col = min(min_col, j)
                    new_max_col = max(max_col, j)
                    candidate = (new_max_row - new_min_row) * (new_max_col - new_min_col)
                    if candidate > max_area:
                        max_area = candidate
            ans[d] = max_area
        return ans

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        board = question_case['board']
        return (
            "你是一个编程竞赛选手，需要解决以下问题。\n\n"
            "问题描述：\n"
            "给定一个n×n的棋盘，每个单元格包含一个0-9的数字。对于每个数字d（0-9），你必须将恰好一个单元格的数值改为d，然后找出满足以下条件的最大三角形的面积的两倍：\n"
            "1. 三角形的三个顶点都是d。\n"
            "2. 至少有一条边与棋盘的边平行。\n\n"
            "输入格式：\n"
            f"n = {n}\n棋盘内容如下：\n" + "\n".join(board) + "\n\n"
            "输出格式：\n"
            "输出10个整数，分别对应d=0到d=9的结果，每个整数为最大面积的两倍。\n"
            "答案必须用[answer]和[/answer]标签包裹，例如：[answer]0 0 0 0 0 0 0 0 0 0[/answer]。"
        )

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            solution = list(map(int, last_match.split()))
            return solution if len(solution) == 10 else None
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected']
