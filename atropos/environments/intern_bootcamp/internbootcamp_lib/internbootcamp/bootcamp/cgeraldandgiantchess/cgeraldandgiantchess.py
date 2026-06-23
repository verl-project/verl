"""# 

### 谜题描述
Giant chess is quite common in Geraldion. We will not delve into the rules of the game, we'll just say that the game takes place on an h × w field, and it is painted in two colors, but not like in chess. Almost all cells of the field are white and only some of them are black. Currently Gerald is finishing a game of giant chess against his friend Pollard. Gerald has almost won, and the only thing he needs to win is to bring the pawn from the upper left corner of the board, where it is now standing, to the lower right corner. Gerald is so confident of victory that he became interested, in how many ways can he win?

The pawn, which Gerald has got left can go in two ways: one cell down or one cell to the right. In addition, it can not go to the black cells, otherwise the Gerald still loses. There are no other pawns or pieces left on the field, so that, according to the rules of giant chess Gerald moves his pawn until the game is over, and Pollard is just watching this process.

Input

The first line of the input contains three integers: h, w, n — the sides of the board and the number of black cells (1 ≤ h, w ≤ 105, 1 ≤ n ≤ 2000). 

Next n lines contain the description of black cells. The i-th of these lines contains numbers ri, ci (1 ≤ ri ≤ h, 1 ≤ ci ≤ w) — the number of the row and column of the i-th cell.

It is guaranteed that the upper left and lower right cell are white and all cells in the description are distinct.

Output

Print a single line — the remainder of the number of ways to move Gerald's pawn from the upper left to the lower right corner modulo 109 + 7.

Examples

Input

3 4 2
2 2
2 3


Output

2


Input

100 100 3
15 16
16 15
99 88


Output

545732279

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#from sys import setrecursionlimit as srl
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
mod = 10**9+7
N = 200000
fact = [1]*N
inv = [1]*N     # Precompute when queries of order N

for i in xrange(1,N):
    fact[i] = (fact[i-1]*i)%mod
    inv[i] = (inv[i-1]*pow(i,mod-2,mod))%mod

class dot:
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __cmp__(self,other):
        if self.x==other.x:
            if self.y==other.y:
                return 0
            elif self.y<other.y:
                return -1
            else:
                return 1
        elif self.x<other.x: return -1
        else: return 1

##    def __lt__(self,other):
##        return(self.x<=other.x and self.y<=other.y)

    def __xor__(self,other):
        dx = self.x - other.x
        dy = self.y - other.y
        return (fact[dx+dy]*inv[dx]*inv[dy])%mod
       
n,m,k = RI()
grid = [dot(*RI()) for i in xrange(k)]
grid.sort()
grid.append(dot(n,m))
    
dp = [0]*(k+1)
top = dot(1,1)

for i in xrange(k+1):
    tot = grid[i]^top
    for j in xrange(i):
        if grid[j].x<=grid[i].x and grid[j].y<=grid[i].y:
            rem = (dp[j]*(grid[i]^grid[j]))%mod
            tot = (tot-rem)%mod
    dp[i] = tot
print dp[k]
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

MOD = 10**9 + 7

class Cgeraldandgiantchessbootcamp(Basebootcamp):
    def __init__(self, h_min=1, h_max=20, w_min=1, w_max=20, max_black=10):
        self.h_min = h_min
        self.h_max = h_max
        self.w_min = w_min
        self.w_max = w_max
        self.max_black = max_black
    
    def case_generator(self):
        # 生成合法棋盘尺寸
        h = random.randint(self.h_min, self.h_max)
        w = random.randint(self.w_min, self.w_max)
        
        # 生成所有可能黑格（排除起点和终点）
        forbidden = {(1, 1), (h, w)}
        all_cells = [
            (r, c)
            for r in range(1, h+1)
            for c in range(1, w+1)
            if (r, c) not in forbidden
        ]
        
        # 确定实际黑格数量
        max_possible = min(len(all_cells), self.max_black)
        n = random.randint(0, max_possible)
        black_cells = random.sample(all_cells, n) if n > 0 else []
        
        return {
            'h': h,
            'w': w,
            'n': n,
            'black_cells': sorted(black_cells, key=lambda x: (x[0], x[1]))
        }
    
    @staticmethod
    def prompt_func(question_case):
        h = question_case['h']
        w = question_case['w']
        n = question_case['n']
        cells = question_case['black_cells']
        
        problem = (
            "## Giant Chess Path Counting Problem\n\n"
            "### Background\n"
            "In Geraldion, a special chess variant is played on an h×w grid. The pawn starts at the top-left corner (1,1) "
            "and must reach the bottom-right corner ({h},{w}). The pawn can only move right or down, and cannot step on "
            "black cells. Your task is to calculate the number of valid paths modulo 10^9+7.\n\n"
            "### Problem Instance\n"
            "- Grid dimensions: {h} rows × {w} columns\n"
            "- Black cells: {n}\n".format(h=h, w=w, n=n)
        )
        
        if n > 0:
            problem += "- Coordinates of black cells:\n"
            for r, c in cells:
                problem += f"  ({r}, {c})\n"
        
        problem += (
            "\n### Answer Requirements\n"
            "Calculate the number of valid paths modulo 10^9+7.\n"
            "Enclose your final answer within [answer] tags like: [answer]12345[/answer]"
        )
        return problem
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            # 解析输入参数
            h = identity['h']
            w = identity['w']
            blocks = [(r, c) for r, c in identity['black_cells']]
            target = (h, w)
            
            # 包含终点并排序障碍点
            points = sorted(blocks + [target], key=lambda p: (p[0], p[1]))
            
            # 动态计算组合数
            max_n = h + w
            fact = [1]*(max_n+1)
            for i in range(1, max_n+1):
                fact[i] = fact[i-1] * i % MOD
                
            inv_fact = [1]*(max_n+1)
            inv_fact[max_n] = pow(fact[max_n], MOD-2, MOD)
            for i in range(max_n-1, -1, -1):
                inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
            
            def comb(n, k):
                if n < 0 or k < 0 or n < k:
                    return 0
                return fact[n] * inv_fact[k] % MOD * inv_fact[n-k] % MOD
            
            # 递推计算路径数
            dp = []
            for i, (x, y) in enumerate(points):
                # 到当前点的总路径数
                total = comb(x+y-2, x-1)
                
                # 减去经过前面障碍点的路径
                for j in range(i):
                    px, py = points[j]
                    if px <= x and py <= y:
                        dx = x - px
                        dy = y - py
                        subtract = dp[j] * comb(dx + dy, dx) % MOD
                        total = (total - subtract) % MOD
                
                dp.append(total)
            
            expected = dp[-1] % MOD
            actual = int(solution.strip()) % MOD
            return actual == expected
        except:
            return False
