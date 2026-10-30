"""# 

### 谜题描述
Gargari is jealous that his friend Caisa won the game from the previous problem. He wants to prove that he is a genius.

He has a n × n chessboard. Each cell of the chessboard has a number written on it. Gargari wants to place two bishops on the chessboard in such a way that there is no cell that is attacked by both of them. Consider a cell with number x written on it, if this cell is attacked by one of the bishops Gargari will get x dollars for it. Tell Gargari, how to place bishops on the chessboard to get maximum amount of money.

We assume a cell is attacked by a bishop, if the cell is located on the same diagonal with the bishop (the cell, where the bishop is, also considered attacked by it).

Input

The first line contains a single integer n (2 ≤ n ≤ 2000). Each of the next n lines contains n integers aij (0 ≤ aij ≤ 109) — description of the chessboard.

Output

On the first line print the maximal number of dollars Gargari will get. On the next line print four integers: x1, y1, x2, y2 (1 ≤ x1, y1, x2, y2 ≤ n), where xi is the number of the row where the i-th bishop should be placed, yi is the number of the column where the i-th bishop should be placed. Consider rows are numbered from 1 to n from top to bottom, and columns are numbered from 1 to n from left to right.

If there are several optimal solutions, you can print any of them.

Examples

Input

4
1 1 1 1
2 1 1 0
1 1 1 0
1 0 0 1


Output

12
2 2 3 2

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys 
input=sys.stdin.readline 
n=int(input())
mat=[]

for i in range(n):
    mat.append([int(i) for i in input().split()])
pri=[0]*(2*n+1)
sec=[0]*(2*n+1)
for i in range(n):
    for j in range(n):
        pri[i-j+n]+=mat[i][j]
        sec[i+j]+=mat[i][j]
resw=-1 
resb=-1 
indw=[]
indb=[]
for i in range(n):
    for j in range(n):
        curr=pri[i-j+n]+sec[i+j]-mat[i][j]
        if (i+j)&1:
            #i am blck 
            if curr>resb:
                resb=curr 
                indb=[i+1,j+1]
        else: 
            if curr>resw:
                resw=curr 
                indw=[i+1,j+1]
print(resw+resb)
print indb[0],indb[1],
print indw[0],indw[1]
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cgargariandbishopsbootcamp(Basebootcamp):
    def __init__(self, n=4, min_val=0, max_val=10):
        if n < 2:
            raise ValueError("n must be at least 2")
        self.n = n
        self.min_val = min_val
        self.max_val = max_val

    def case_generator(self):
        n = self.n
        mat = [[random.randint(self.min_val, self.max_val) for _ in range(n)] for _ in range(n)]
        
        # 正确初始化对角线数组
        pri = [0] * (2 * n)     # 主对角线 (i-j) 范围：-(n-1) 到 (n-1) → 索引范围：1 到 2n-1
        sec = [0] * (2 * n - 1)  # 副对角线 (i+j) 范围：0 到 2n-2
        
        for i in range(n):
            for j in range(n):
                pri_index = i - j + n
                sec_index = i + j
                pri[pri_index] += mat[i][j]
                sec[sec_index] += mat[i][j]
        
        # 计算各颜色最大贡献
        white_max = -1
        black_max = -1
        white_positions = []
        black_positions = []
        
        for i in range(n):
            for j in range(n):
                total = pri[i-j+n] + sec[i+j] - mat[i][j]
                if (i + j) % 2 == 0:  # 白格
                    if total > white_max:
                        white_max = total
                        white_positions = [(i+1, j+1)]
                    elif total == white_max:
                        white_positions.append((i+1, j+1))
                else:                 # 黑格
                    if total > black_max:
                        black_max = total
                        black_positions = [(i+1, j+1)]
                    elif total == black_max:
                        black_positions.append((i+1, j+1))
        
        return {
            'n': n,
            'mat': mat,
            'white_positions': white_positions,
            'black_positions': black_positions,
            'correct_total': white_max + black_max
        }

    @staticmethod
    def prompt_func(question_case):
        mat_str = '\n'.join(' '.join(map(str, row)) for row in question_case['mat'])
        return f"""Place two bishops on a {question_case['n']}x{question_case['n']} chessboard to maximize attack value without overlapping coverage. 

**Rules**:
- Bishops attack diagonally (including current cell)
- Overlapping cells count only once
- Output format:
[answer]
{{total_value}}
{{x1 y1 x2 y2}}
[/answer]

**Chessboard**:
{mat_str}"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            parts = matches[-1].strip().split()
            total = int(parts[0])
            coords = list(map(int, parts[1:5]))
            if len(coords) != 4:
                return None
            return (total, *coords)
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            total, x1, y1, x2, y2 = solution
        except:
            return False
        
        # 基础验证
        if total != identity['correct_total']:
            return False
        if not all(1 <= v <= identity['n'] for v in [x1, y1, x2, y2]):
            return False
        if (x1, y1) == (x2, y2):
            return False
        
        # 转换为0-based坐标
        i1, j1 = x1-1, y1-1
        i2, j2 = x2-1, y2-1
        
        # 颜色校验
        if (i1 + j1) % 2 == (i2 + j2) % 2:
            return False
        
        # 候选位置校验
        pos1, pos2 = (x1, y1), (x2, y2)
        white_valid = pos1 in identity['white_positions'] and pos2 in identity['black_positions']
        black_valid = pos1 in identity['black_positions'] and pos2 in identity['white_positions']
        
        return white_valid or black_valid
