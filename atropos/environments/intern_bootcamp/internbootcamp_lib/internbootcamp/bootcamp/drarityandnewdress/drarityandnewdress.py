"""# 

### 谜题描述
Carousel Boutique is busy again! Rarity has decided to visit the pony ball and she surely needs a new dress, because going out in the same dress several times is a sign of bad manners. First of all, she needs a dress pattern, which she is going to cut out from the rectangular piece of the multicolored fabric.

The piece of the multicolored fabric consists of n × m separate square scraps. Since Rarity likes dresses in style, a dress pattern must only include scraps sharing the same color. A dress pattern must be the square, and since Rarity is fond of rhombuses, the sides of a pattern must form a 45^{\circ} angle with sides of a piece of fabric (that way it will be resembling the traditional picture of a rhombus).

Examples of proper dress patterns: <image> Examples of improper dress patterns: <image> The first one consists of multi-colored scraps, the second one goes beyond the bounds of the piece of fabric, the third one is not a square with sides forming a 45^{\circ} angle with sides of the piece of fabric.

Rarity wonders how many ways to cut out a dress pattern that satisfies all the conditions that do exist. Please help her and satisfy her curiosity so she can continue working on her new masterpiece!

Input

The first line contains two integers n and m (1 ≤ n, m ≤ 2000). Each of the next n lines contains m characters: lowercase English letters, the j-th of which corresponds to scrap in the current line and in the j-th column. Scraps having the same letter share the same color, scraps having different letters have different colors.

Output

Print a single integer: the number of ways to cut out a dress pattern to satisfy all of Rarity's conditions.

Examples

Input


3 3
aaa
aaa
aaa


Output


10


Input


3 4
abab
baba
abab


Output


12


Input


5 5
zbacg
baaac
aaaaa
eaaad
weadd


Output


31

Note

In the first example, all the dress patterns of size 1 and one of size 2 are satisfactory.

In the second example, only the dress patterns of size 1 are satisfactory.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
if sys.subversion[0] == \"PyPy\":
    import io, atexit
    sys.stdout = io.BytesIO()
    atexit.register(lambda: sys.__stdout__.write(sys.stdout.getvalue()))
    
    sys.stdin = io.BytesIO(sys.stdin.read())
    input = lambda: sys.stdin.readline().rstrip()

RS = raw_input
RI = lambda x=int: map(x,RS().split())
RN = lambda x=int: x(RS())
''' ...................................................................... '''

n,m = RI()
wall = ['W'*(m+2)]
mat = wall + ['W'+RS()+'W' for i in xrange(n)] + wall

dp = [[1]*(m+2) for i in xrange(n+2)]
for i in xrange(n+2):
    dp[i][0] = dp[i][m+1] = 0
for j in xrange(m+2):
    dp[0][j] = dp[n+1][j] = 0

for i in xrange(1,n+1):
    for j in xrange(1,m+1):
        if mat[i][j]==mat[i-1][j-1]==mat[i-1][j]==mat[i-1][j+1]:
            pass
        else: continue
        depth = min(dp[i-1][j-1:j+2])
        height = 2*depth
        if mat[i][j]==mat[i-height][j]:
            dp[i][j] = depth+1
        else: dp[i][j] = depth

ans = sum([sum(r) for r in dp])
print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import deque
from bootcamp import Basebootcamp

class Drarityandnewdressbootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_m=10):
        self.max_n = max(max_n, 1)
        self.max_m = max(max_m, 1)
    
    def case_generator(self):
        n = random.randint(1, self.max_n)
        m = random.randint(1, self.max_m)
        grid = []
        for _ in range(n):
            row = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=m))
            grid.append(row)
        return {'n': n, 'm': m, 'grid': grid}
    
    @staticmethod
    def prompt_func(question_case):
        grid = question_case['grid']
        grid_str = '\n'.join(grid)
        prompt = f"""你是小马谷的服装设计师，需要帮助Rarity计算满足条件的菱形图案数目。布料由{question_case['n']}行{question_case['m']}列的彩色方块组成，每个方块是小写字母表示的顏色。菱形必须满足以下条件：

1. 菱形由同一颜色的方块组成。
2. 菱形的边必须与布料的边成45度角（即呈菱形形状）。
3. 菱形不能超出布料边界。

输入格式：
- 第一行是两个整数n和m。
- 接下来n行，每行m个小写字母。

输出格式：
- 一个整数，表示符合条件的菱形数目。

例如，输入：
3 3
aaa
aaa
aaa
正确输出为10，因为有10个符合条件的菱形。

现在请解决以下谜题实例：
{question_case['n']} {question_case['m']}
{grid_str}

请将最终答案放入[answer]和[/answer]标签之间，例如：[answer]10[/answer]。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output, re.DOTALL)
        return int(matches[-1]) if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            n = identity['n']
            m = identity['m']
            grid = identity['grid']
            
            # 保证输入有效性
            if n <= 0 or m <= 0 or len(grid) != n or any(len(row)!=m for row in grid):
                return False
            if not all(c.islower() and c.isalpha() for row in grid for c in row):
                return False

            # 构造带保护圈的矩阵
            wall = 'W' * (m + 2)
            mat = [wall]  # 上保护墙
            mat.extend(f'W{row}W' for row in grid)
            mat.append(wall)  # 下保护墙

            # 初始化DP表
            dp = [[1]*(m+2) for _ in range(n+2)]
            
            # 设置边界条件
            for i in range(n+2):
                dp[i][0] = dp[i][m+1] = 0
            for j in range(m+2):
                dp[0][j] = dp[n+1][j] = 0

            # 动态规划计算
            for i in range(1, n+1):
                for j in range(1, m+1):
                    current_color = mat[i][j]
                    neighbors = [
                        mat[i-1][j-1],
                        mat[i-1][j],
                        mat[i-1][j+1]
                    ]
                    
                    if all(c == current_color for c in neighbors):
                        depth = min(dp[i-1][j-1], dp[i-1][j], dp[i-1][j+1])
                        height = 2 * depth
                        
                        if i > height and mat[i - height][j] == current_color:
                            dp[i][j] = depth + 1
                        else:
                            dp[i][j] = depth
                    else:
                        dp[i][j] = 1  # 重置为基本单元

            correct = sum(sum(row[1:-1]) for row in dp[1:-1])
            return solution == correct
            
        except Exception:
            return False
