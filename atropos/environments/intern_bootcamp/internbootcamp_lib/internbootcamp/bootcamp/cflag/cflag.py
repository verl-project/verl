"""# 

### 谜题描述
Innokenty works at a flea market and sells some random stuff rare items. Recently he found an old rectangular blanket. It turned out that the blanket is split in n ⋅ m colored pieces that form a rectangle with n rows and m columns. 

The colored pieces attracted Innokenty's attention so he immediately came up with the following business plan. If he cuts out a subrectangle consisting of three colored stripes, he can sell it as a flag of some country. Innokenty decided that a subrectangle is similar enough to a flag of some country if it consists of three stripes of equal heights placed one above another, where each stripe consists of cells of equal color. Of course, the color of the top stripe must be different from the color of the middle stripe; and the color of the middle stripe must be different from the color of the bottom stripe.

Innokenty has not yet decided what part he will cut out, but he is sure that the flag's boundaries should go along grid lines. Also, Innokenty won't rotate the blanket. Please help Innokenty and count the number of different subrectangles Innokenty can cut out and sell as a flag. Two subrectangles located in different places but forming the same flag are still considered different.

<image> <image> <image>

These subrectangles are flags.

<image> <image> <image> <image> <image> <image>

These subrectangles are not flags.

Input

The first line contains two integers n and m (1 ≤ n, m ≤ 1 000) — the number of rows and the number of columns on the blanket.

Each of the next n lines contains m lowercase English letters from 'a' to 'z' and describes a row of the blanket. Equal letters correspond to equal colors, different letters correspond to different colors.

Output

In the only line print the number of subrectangles which form valid flags.

Examples

Input


4 3
aaa
bbb
ccb
ddd


Output


6


Input


6 1
a
a
b
b
c
c


Output


1

Note

<image> <image>

The selected subrectangles are flags in the first example.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
R,C = map(int,raw_input().split())
grid = [raw_input() for _ in range(R)]
down = [[None]*C for _ in range(R)]
for c in range(C):
    szs = []
    cnt = 1
    cur = 1
    while cur < R:
        if grid[cur][c] == grid[cur-1][c]:
            cnt += 1
        else:
            szs.append(cnt)
            cnt = 1
        cur += 1
    szs.append(cnt)
    st = 0
    for i in range(1,len(szs)-1):
        if szs[i] > min(szs[i-1],szs[i+1]):
            st += szs[i-1]
            continue
        sz = szs[i]
        down[st+szs[i-1]-sz][c] = (sz,grid[st][c],grid[st+szs[i-1]][c],grid[st+szs[i-1]+sz][c])
        st += szs[i-1]
out = 0
for r in range(R):
    st = 0
    cnt = 0
    cur = None
    while st < C:
        if down[r][st] == None:
            out += (cnt+1)*cnt/2
            cnt = 0
            cur = None
        elif cur == down[r][st]:
            cnt += 1
        else:
            out += (cnt+1)*cnt/2
            cnt = 1
            cur = down[r][st]

        st += 1
    out += (cnt+1)*cnt/2
print out
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from string import ascii_lowercase
from bootcamp import Basebootcamp

class Cflagbootcamp(Basebootcamp):
    def __init__(self, n=4, m=3):
        self.n = n
        self.m = m

    def case_generator(self):
        grid = []
        for _ in range(self.n):
            row = ''.join(random.choice(ascii_lowercase) for _ in range(self.m))
            grid.append(row)
        
        correct_answer = self._calculate_correct_answer(grid)
        return {
            'grid': grid,
            'correct_answer': correct_answer
        }

    @staticmethod
    def prompt_func(question_case):
        grid = question_case['grid']
        problem = (
            "Innokenty has a rectangular blanket divided into colored pieces. He wants to cut out subrectangles that form valid country flags. "
            "A valid flag must have three equal-height horizontal stripes, each with a uniform color, and adjacent stripes must have different colors. "
            "Your task is to count how many such subrectangles exist in the given blanket.\n\n"
            "The blanket is as follows:\n"
        )
        for i, row in enumerate(grid):
            problem += f"Row {i+1}: {row}\n"
        problem += (
            "Please provide the number of valid flag subrectangles in the format: [answer]X[/answer], where X is the count."
        )
        return problem

    @staticmethod
    def extract_output(output):
        matches = list(re.finditer(r'\[answer\](\d+)\[\/answer\]', output))
        if matches:
            return int(matches[-1].group(1))
        return None

    @classmethod
    def _calculate_correct_answer(cls, grid):
        n = len(grid)
        if n == 0:
            return 0
        m = len(grid[0])
        down = [[None] * m for _ in range(n)]
        
        for c in range(m):
            szs = []
            cnt = 1
            for r in range(1, n):
                if grid[r][c] == grid[r-1][c]:
                    cnt += 1
                else:
                    szs.append(cnt)
                    cnt = 1
            szs.append(cnt)
            
            st = 0
            for i in range(1, len(szs)-1):
                if szs[i] > min(szs[i-1], szs[i+1]):
                    st += szs[i-1]
                    continue
                sz = szs[i]
                top_start = st
                top_end = st + szs[i-1] - 1
                mid_start = top_end + 1
                mid_end = mid_start + sz - 1
                if mid_end >= n:
                    st += szs[i-1]
                    continue
                bot_start = mid_end + 1
                bot_end = bot_start + sz - 1
                if bot_end >= n:
                    st += szs[i-1]
                    continue
                top_color = grid[top_start][c]
                mid_color = grid[mid_start][c]
                bot_color = grid[bot_start][c]
                if top_color != mid_color and mid_color != bot_color:
                    for r in range(top_start, top_end + 1):
                        down[r][c] = (sz, top_color, mid_color, bot_color)
                st += szs[i-1]
        
        out = 0
        for r in range(n):
            st = 0
            cnt = 0
            cur = None
            while st < m:
                cell = down[r][st]
                if cell is None:
                    if cnt > 0:
                        out += (cnt + 1) * cnt // 2
                        cnt = 0
                    st += 1
                else:
                    if cell == cur:
                        cnt += 1
                    else:
                        if cnt > 0:
                            out += (cnt + 1) * cnt // 2
                        cur = cell
                        cnt = 1
                    st += 1
            if cnt > 0:
                out += (cnt + 1) * cnt // 2
        return out

    @classmethod
    def _verify_correction(cls, solution, identity):
        correct_answer = identity['correct_answer']
        return solution == correct_answer
