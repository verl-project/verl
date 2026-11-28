"""# 

### 谜题描述
You are given a matrix a consisting of positive integers. It has n rows and m columns.

Construct a matrix b consisting of positive integers. It should have the same size as a, and the following conditions should be met: 

  * 1 ≤ b_{i,j} ≤ 10^6; 
  * b_{i,j} is a multiple of a_{i,j}; 
  * the absolute value of the difference between numbers in any adjacent pair of cells (two cells that share the same side) in b is equal to k^4 for some integer k ≥ 1 (k is not necessarily the same for all pairs, it is own for each pair). 



We can show that the answer always exists.

Input

The first line contains two integers n and m (2 ≤ n,m ≤ 500).

Each of the following n lines contains m integers. The j-th integer in the i-th line is a_{i,j} (1 ≤ a_{i,j} ≤ 16).

Output

The output should contain n lines each containing m integers. The j-th integer in the i-th line should be b_{i,j}.

Examples

Input


2 2
1 2
2 3


Output


1 2
2 3


Input


2 3
16 16 16
16 16 16


Output


16 32 48
32 48 64


Input


2 2
3 11
12 8


Output


327 583
408 664

Note

In the first example, the matrix a can be used as the matrix b, because the absolute value of the difference between numbers in any adjacent pair of cells is 1 = 1^4.

In the third example: 

  * 327 is a multiple of 3, 583 is a multiple of 11, 408 is a multiple of 12, 664 is a multiple of 8; 
  * |408 - 327| = 3^4, |583 - 327| = 4^4, |664 - 408| = 4^4, |664 - 583| = 3^4. 

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


mod = int(1e9)+7

def quickm(a, b):
    base = a
    re = 1
    while b:
        if b&1:
            re *= base
            re %= mod
        b >>= 1
        base *= base
        base %= mod
    return re

def inv(num):
    return quickm(num, mod-2)


def my_main():
    kase = 1 #inp()
    pans = []
    for _ in range(kase):
        n, m = inp()
        da = []
        for i in range(n):
            da.append(inp())
        for i in range(n):
            ans = []
            for j in range(m):
                if (i+j)%2:
                    ans.append(720720)
                else:
                    ans.append(720720+da[i][j]**4)
            pans.append(' '.join(map(str, ans)))
    print '\n'.join(pans)





my_main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Dmultiplesandpowerdifferencesbootcamp(Basebootcamp):
    def __init__(self, n=2, m=2, min_a=1, max_a=16):
        self.n = n
        self.m = m
        self.min_a = min_a
        self.max_a = max_a
    
    def case_generator(self):
        a_matrix = [
            [random.randint(self.min_a, self.max_a) for _ in range(self.m)]
            for _ in range(self.n)
        ]
        return {
            'n': self.n,
            'm': self.m,
            'a_matrix': a_matrix
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        a_matrix = question_case['a_matrix']
        problem = (
            "你是一名数学谜题的解谜者，需要构造一个满足特定条件的矩阵。给定一个n行m列的矩阵a，其中每个元素都是正整数。你的任务是根据以下规则构造矩阵b：\n\n"
            "规则说明：\n"
            "1. 矩阵b的大小必须与a相同，并且每个元素b[i][j]必须是a[i][j]的正整数倍。\n"
            "2. 矩阵b中任意两个相邻元素（相邻指的是上下左右相邻，即共享同一条边的单元格）的绝对差必须是某个整数的四次方。例如，差可以是1⁴=1，2⁴=16，3⁴=81，依此类推。\n"
            "3. 每个元素b[i][j]的取值范围必须在1到1,000,000之间（包含1和1,000,000）。\n\n"
            "输入格式：\n"
            f"第一行包含两个整数n和m（此处n={n}，m={m}）。\n"
            "接下来的n行每行包含m个整数，表示矩阵a的元素。\n\n"
            "输出格式：\n"
            "输出n行，每行m个整数，表示满足条件的矩阵b。\n\n"
            "矩阵a的输入如下：\n"
            f"{n} {m}\n"
        )
        for row in a_matrix:
            problem += " ".join(map(str, row)) + "\n"
        problem += (
            "\n请按照输出格式要求，将正确的矩阵b输出，并将答案放置在[answer]标签内。例如：\n"
            "[answer]\n"
            "16 32\n"
            "48 64\n"
            "[/answer]"
        )
        return problem
    
    @staticmethod
    def extract_output(output):
        import re
        pattern = re.compile(r'\[answer\](.*?)\[/answer\]', re.DOTALL)
        matches = pattern.findall(output)
        if not matches:
            return None
        last_answer = matches[-1].strip()
        matrix = []
        for line in last_answer.split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                elements = list(map(int, line.split()))
                matrix.append(elements)
            except ValueError:
                return None
        if not matrix:
            return None
        cols = len(matrix[0])
        for row in matrix:
            if len(row) != cols:
                return None
        return matrix
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or not isinstance(solution, list) or not all(isinstance(row, list) for row in solution):
            return False
        n = identity['n']
        m = identity['m']
        a_matrix = identity['a_matrix']
        if len(solution) != n or any(len(row) != m for row in solution):
            return False
        for i in range(n):
            for j in range(m):
                b_val = solution[i][j]
                a_val = a_matrix[i][j]
                if not (1 <= b_val <= 10**6) or b_val % a_val != 0:
                    return False
        for i in range(n):
            for j in range(m):
                current = solution[i][j]
                if j + 1 < m:
                    right = solution[i][j+1]
                    if not cls.is_k4(abs(current - right)):
                        return False
                if i + 1 < n:
                    down = solution[i+1][j]
                    if not cls.is_k4(abs(current - down)):
                        return False
        return True
    
    @staticmethod
    def is_k4(x):
        if x < 1:
            return False
        k = 1
        while True:
            k4 = k ** 4
            if k4 == x:
                return True
            if k4 > x:
                return False
            k += 1
