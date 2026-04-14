"""# 

### 谜题描述
They say \"years are like dominoes, tumbling one after the other\". But would a year fit into a grid? I don't think so.

Limak is a little polar bear who loves to play. He has recently got a rectangular grid with h rows and w columns. Each cell is a square, either empty (denoted by '.') or forbidden (denoted by '#'). Rows are numbered 1 through h from top to bottom. Columns are numbered 1 through w from left to right.

Also, Limak has a single domino. He wants to put it somewhere in a grid. A domino will occupy exactly two adjacent cells, located either in one row or in one column. Both adjacent cells must be empty and must be inside a grid.

Limak needs more fun and thus he is going to consider some queries. In each query he chooses some rectangle and wonders, how many way are there to put a single domino inside of the chosen rectangle?

Input

The first line of the input contains two integers h and w (1 ≤ h, w ≤ 500) – the number of rows and the number of columns, respectively.

The next h lines describe a grid. Each line contains a string of the length w. Each character is either '.' or '#' — denoting an empty or forbidden cell, respectively.

The next line contains a single integer q (1 ≤ q ≤ 100 000) — the number of queries.

Each of the next q lines contains four integers r1i, c1i, r2i, c2i (1 ≤ r1i ≤ r2i ≤ h, 1 ≤ c1i ≤ c2i ≤ w) — the i-th query. Numbers r1i and c1i denote the row and the column (respectively) of the upper left cell of the rectangle. Numbers r2i and c2i denote the row and the column (respectively) of the bottom right cell of the rectangle.

Output

Print q integers, i-th should be equal to the number of ways to put a single domino inside the i-th rectangle.

Examples

Input

5 8
....#..#
.#......
##.#....
##..#.##
........
4
1 1 2 3
4 1 4 1
1 2 4 5
2 5 5 8


Output

4
0
10
15


Input

7 39
.......................................
.###..###..#..###.....###..###..#..###.
...#..#.#..#..#.........#..#.#..#..#...
.###..#.#..#..###.....###..#.#..#..###.
.#....#.#..#....#.....#....#.#..#..#.#.
.###..###..#..###.....###..###..#..###.
.......................................
6
1 1 3 20
2 10 6 30
2 10 7 30
2 2 7 7
1 7 7 7
1 8 7 8


Output

53
89
120
23
0
2

Note

A red frame below corresponds to the first query of the first sample. A domino can be placed in 4 possible ways.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def intarr():
    return map(int, raw_input().split())


def js(mp, d, i, j):
    if (i, j) in d:
        return d[(i,j)]
    if i <= 1 or j == 0 :
        d[(i, j)] = 0
        return 0
    d[(i, j)] = d[(i - 1, j)] + d[(i, j - 1)] - d[(i - 1, j - 1)] + (1 if mp[i - 1][j - 1] == '.' and mp[i - 2][j - 1] == '.' else 0)
    return d[(i, j)]

def js2(mp, d, i, j):
    if (i, j) in d:
        return d[(i,j)]
    if i == 0 or j <= 1 :
        d[(i, j)] = 0
        return 0
    d[(i, j)] = d[(i - 1, j)] + d[(i, j - 1)] - d[(i - 1, j - 1)] + (1 if mp[i - 1][j - 1] == '.' and mp[i - 1][j - 2] == '.' else 0)
    return d[(i, j)]

def sumf(r1, c1, r2, c2, d):
    r1 -= 1
    c1 -= 1
    return d[(r1, c1)] + d[(r2, c2)] - d[(r1, c2)] - d[(r2, c1)]

h, w = intarr()
mp =[raw_input() for i in xrange(h)]
d1 = {(0,0):0}
for i in xrange(h + 1):
    for j in xrange(w + 1):
        js(mp, d1, i, j)

d2 = {(0,0):0}
for i in xrange(h + 1):
    for j in xrange(w + 1):
        js2(mp, d2, i, j)

q, = intarr()
for i in xrange(q):
    r1, c1, r2, c2 = intarr()
    print sum([sumf(r1 + k1, c1 + k2, r2, c2, d) for d, k1, k2 in [(d1, 1, 0), (d2, 0, 1)]])
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from collections import defaultdict
from bootcamp import Basebootcamp

class Cnewyearanddominobootcamp(Basebootcamp):
    def __init__(self, h=5, w=5, p=0.7, max_queries=5):
        self.h = h
        self.w = w
        self.p = p  # Probability of empty cell
        self.max_queries = max_queries  # Number of queries per case

    def case_generator(self):
        # Generate valid grid
        while True:
            grid = [['.' if random.random() < self.p else '#' for _ in range(self.w)] for _ in range(self.h)]
            if sum(cell == '.' for row in grid for cell in row) >= 2:  # Ensure at least 2 empty cells
                break

        # Generate multiple queries
        queries = []
        correct_answers = []
        for _ in range(random.randint(1, self.max_queries)):
            r1 = random.randint(1, self.h)
            r2 = random.randint(r1, self.h)
            c1 = random.randint(1, self.w)
            c2 = random.randint(c1, self.w)
            query = (r1, c1, r2, c2)
            answer = self.compute_answer(grid, query)
            queries.append(query)
            correct_answers.append(answer)

        return {
            'h': self.h,
            'w': self.w,
            'grid': [''.join(row) for row in grid],
            'queries': queries,
            'correct_answers': correct_answers
        }

    def compute_answer(self, grid, query):
        h, w = len(grid), len(grid[0])
        r1, c1, r2, c2 = query

        # Build prefix sums for vertical dominoes
        d1 = defaultdict(int)
        for i in range(h+1):
            for j in range(w+1):
                if i <= 1 or j == 0:
                    d1[(i, j)] = 0
                else:
                    term = 1 if (i >= 2 and 
                                grid[i-1][j-1] == '.' and 
                                grid[i-2][j-1] == '.') else 0
                    d1[(i, j)] = d1[(i-1, j)] + d1[(i, j-1)] - d1[(i-1, j-1)] + term

        # Build prefix sums for horizontal dominoes
        d2 = defaultdict(int)
        for i in range(h+1):
            for j in range(w+1):
                if j <= 1 or i == 0:
                    d2[(i, j)] = 0
                else:
                    term = 1 if (j >= 2 and 
                                grid[i-1][j-1] == '.' and 
                                grid[i-1][j-2] == '.') else 0
                    d2[(i, j)] = d2[(i-1, j)] + d2[(i, j-1)] - d2[(i-1, j-1)] + term

        # Calculate sum for vertical dominoes
        def sum_vertical(r1, c1, r2, c2):
            a = d1.get((r1-1, c1-1), 0)
            b = d1.get((r1-1, c2), 0)
            c_val = d1.get((r2, c1-1), 0)
            d_val = d1.get((r2, c2), 0)
            return d_val - b - c_val + a

        # Calculate sum for horizontal dominoes
        def sum_horizontal(r1, c1, r2, c2):
            a = d2.get((r1-1, c1-1), 0)
            b = d2.get((r1-1, c2), 0)
            c_val = d2.get((r2, c1-1), 0)
            d_val = d2.get((r2, c2), 0)
            return d_val - b - c_val + a

        total = 0
        # Vertical dominoes (need at least 2 rows)
        if r2 >= r1 + 1:
            total += sum_vertical(r1+1, c1, r2, c2)
        # Horizontal dominoes (need at least 2 columns)
        if c2 >= c1 + 1:
            total += sum_horizontal(r1, c1+1, r2, c2)
        
        return total

    @staticmethod
    def prompt_func(question_case):
        grid = question_case['grid']
        queries = question_case['queries']
        prompt = f"""Limak needs help counting domino placements in a {len(grid)}x{len(grid[0])} grid:

Grid (rows 1-{len(grid)}):
""" + '\n'.join(f"Row {i+1}: {row}" for i, row in enumerate(grid)) + """

Answer these queries (format as space-separated numbers in [answer] tags):
"""
        for i, (r1, c1, r2, c2) in enumerate(queries, 1):
            prompt += f"\nQuery {i}: Rectangle from ({r1}, {c1}) to ({r2}, {c2})"
        
        prompt += "\n\nPlace your final answer between [answer] and [/answer], e.g.: [answer]1 4 0 5[/answer]"
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return list(map(int, matches[-1].strip().split()))
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not isinstance(solution, list) or len(solution) != len(identity['correct_answers']):
            return False
        return all(s == a for s, a in zip(solution, identity['correct_answers']))
