"""# 

### 谜题描述
Bashar was practicing for the national programming contest. Because of sitting too much in front of the computer without doing physical movements and eating a lot Bashar became much fatter. Bashar is going to quit programming after the national contest and he is going to become an actor (just like his father), so he should lose weight.

In order to lose weight, Bashar is going to run for k kilometers. Bashar is going to run in a place that looks like a grid of n rows and m columns. In this grid there are two one-way roads of one-kilometer length between each pair of adjacent by side cells, one road is going from the first cell to the second one, and the other road is going from the second cell to the first one. So, there are exactly (4 n m - 2n - 2m) roads.

Let's take, for example, n = 3 and m = 4. In this case, there are 34 roads. It is the picture of this case (arrows describe roads):

<image>

Bashar wants to run by these rules:

  * He starts at the top-left cell in the grid; 
  * In one move Bashar may go up (the symbol 'U'), down (the symbol 'D'), left (the symbol 'L') or right (the symbol 'R'). More formally, if he stands in the cell in the row i and in the column j, i.e. in the cell (i, j) he will move to: 
    * in the case 'U' to the cell (i-1, j); 
    * in the case 'D' to the cell (i+1, j); 
    * in the case 'L' to the cell (i, j-1); 
    * in the case 'R' to the cell (i, j+1); 
  * He wants to run exactly k kilometers, so he wants to make exactly k moves; 
  * Bashar can finish in any cell of the grid; 
  * He can't go out of the grid so at any moment of the time he should be on some cell; 
  * Bashar doesn't want to get bored while running so he must not visit the same road twice. But he can visit the same cell any number of times. 



Bashar asks you if it is possible to run by such rules. If it is possible, you should tell him how should he run.

You should give him a steps to do and since Bashar can't remember too many steps, a should not exceed 3000. In every step, you should give him an integer f and a string of moves s of length at most 4 which means that he should repeat the moves in the string s for f times. He will perform the steps in the order you print them.

For example, if the steps are 2 RUD, 3 UUL then the moves he is going to move are RUD + RUD + UUL + UUL + UUL = RUDRUDUULUULUUL.

Can you help him and give him a correct sequence of moves such that the total distance he will run is equal to k kilometers or say, that it is impossible?

Input

The only line contains three integers n, m and k (1 ≤ n, m ≤ 500, 1 ≤ k ≤ 10 ^{9}), which are the number of rows and the number of columns in the grid and the total distance Bashar wants to run.

Output

If there is no possible way to run k kilometers, print \"NO\" (without quotes), otherwise print \"YES\" (without quotes) in the first line.

If the answer is \"YES\", on the second line print an integer a (1 ≤ a ≤ 3000) — the number of steps, then print a lines describing the steps.

To describe a step, print an integer f (1 ≤ f ≤ 10^{9}) and a string of moves s of length at most 4. Every character in s should be 'U', 'D', 'L' or 'R'.

Bashar will start from the top-left cell. Make sure to move exactly k moves without visiting the same road twice and without going outside the grid. He can finish at any cell.

We can show that if it is possible to run exactly k kilometers, then it is possible to describe the path under such output constraints.

Examples

Input


3 3 4


Output


YES
2
2 R
2 L


Input


3 3 1000000000


Output


NO


Input


3 3 8


Output


YES
3
2 R
2 D
1 LLRR


Input


4 4 9


Output


YES
1
3 RLD


Input


3 4 16


Output


YES
8
3 R
3 L
1 D
3 R
1 D
1 U
3 L
1 D

Note

The moves Bashar is going to move in the first example are: \"RRLL\".

It is not possible to run 1000000000 kilometers in the second example because the total length of the roads is smaller and Bashar can't run the same road twice.

The moves Bashar is going to move in the third example are: \"RRDDLLRR\".

The moves Bashar is going to move in the fifth example are: \"RRRLLLDRRRDULLLD\". It is the picture of his run (the roads on this way are marked with red and numbered in the order of his running):

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def get_sol(R,C,K):
    dir = solve(R,C)
    out = []
    left = K
    for f,d in dir:
        if left >= f * len(d):
            out.append((f,d))
            left -= f * len(d)
        elif left == 0: break
        else:
            out.append((left/len(d),d))
            left -= (left/len(d)) * len(d)
            if left:
                out.append((1,d[:left]))
            left = 0
    out2 = []
    for f,s in out:
        if f: out2.append((f,s))
    return out2
def solve(r,c):
    out = []
    out.append((c-1,'R'))
    for i in range(1,r):
        out.append((1,'D'))
        out.append((c-1,'L' if i%2 else 'R'))
    out.append((c-1,'R' if (r-1)%2 else 'L'))
    for i in range(r-2,-1,-1):
        out.append((1,'U'))
        out.append((c-1,'RDU' if i%2 else 'LDU'))
    out2 = []
    for f,s in out:
        if f: out2.append((f,s))
    return out2
R,C,K = map(int,raw_input().split())
if 4*R*C-2*R-2*C < K:
    print 'NO'
    exit()
out = get_sol(R,C,K)
print 'YES'
print len(out)
test = 0
for o in out:
    print o[0],o[1]
    test += o[0] * len(o[1])
assert test == K
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Dtimetorunbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=50, min_m=1, max_m=50, possible_prob=0.5, max_k=10**4):
        self.min_n = min_n
        self.max_n = max_n
        self.min_m = min_m
        self.max_m = max_m
        self.possible_prob = possible_prob
        self.max_k = max_k  # 控制可验证的k上限

    def case_generator(self):
        """生成有效的谜题实例，确保k在可验证范围内"""
        while True:
            n = random.randint(self.min_n, self.max_n)
            m = random.randint(self.min_m, self.max_m)
            max_roads = 4 * n * m - 2 * n - 2 * m
            max_roads = max(max_roads, 0)  # 确保非负
            
            if max_roads == 0:
                # 无法生成可行案例
                k = random.randint(1, 10**9)
            else:
                if random.random() < self.possible_prob:
                    # 生成可行案例
                    k = random.randint(1, min(max_roads, self.max_k))
                else:
                    # 生成不可行案例
                    k = random.randint(max_roads + 1, 10**9)
            return {'n': n, 'm': m, 'k': k}

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        m = question_case['m']
        k = question_case['k']
        return f"""Dtimetorun needs to run exactly {k} km in a {n}x{m} grid. Roads are one-way between adjacent cells. Rules:
- Start at top-left (1,1)
- Each move is U/D/L/R
- No road reuse
- Exactly {k} moves

Format answer as:
[answer]
YES/NO
[if YES]
a
f₁ s₁
...
fₐ sₐ
[/answer]

Example (n=3, m=3, k=4):
[answer]
YES
2
2 R
2 L
[/answer]"""

    @staticmethod
    def extract_output(output):
        answer_block = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_block:
            return None
        lines = [l.strip() for l in answer_block[-1].strip().split('\n')]
        
        if not lines:
            return None
        
        if lines[0].upper() == 'NO':
            return {'answer': 'NO'} if len(lines) == 1 else None
        
        if lines[0].upper() != 'YES' or len(lines) < 2:
            return None
        
        try:
            a = int(lines[1])
            if not (1 <= a <= 3000) or len(lines) < 2 + a:
                return None
        except:
            return None
        
        steps = []
        for line in lines[2:2+a]:
            parts = line.split()
            if len(parts) < 2:
                return None
            try:
                f = int(parts[0])
                s = ''.join(parts[1:]).upper()
                if not (1 <= f <= 1e9) or not (1 <= len(s) <=4) or any(c not in 'UDLR' for c in s):
                    return None
                steps.append((f, s))
            except:
                return None
        
        return {'answer': 'YES', 'steps': steps}

    @classmethod
    def _verify_correction(cls, solution, identity):
        n, m, k = identity['n'], identity['m'], identity['k']
        max_roads = 4 * n * m - 2 * n - 2 * m
        
        # 快速判断不可能情形
        if k > max_roads:
            return solution.get('answer') == 'NO'
        if solution.get('answer') != 'YES':
            return False
        
        steps = solution.get('steps', [])
        if len(steps) == 0 or len(steps) > 3000:
            return False
        
        total_moves = sum(f * len(s) for f, s in steps)
        if total_moves != k:
            return False
        
        # 路径模拟优化
        current = (1, 1)
        used = set()
        
        for f, s in steps:
            s = s.upper()
            # 处理单方向连续移动
            if len(set(s)) == 1:
                dir = s[0]
                dx, dy = {'U': (-1,0), 'D':(1,0), 'L':(0,-1), 'R':(0,1)}[dir]
                steps_needed = f * len(s)
                
                # 批量检查越界和道路重复
                x, y = current
                road_chain = []
                for _ in range(steps_needed):
                    nx, ny = x + dx, y + dy
                    if not (1 <= nx <= n and 1 <= ny <= m):
                        return False
                    road = ((x, y), (nx, ny))
                    if road in used or road in road_chain:
                        return False
                    road_chain.append(road)
                    x, y = nx, ny
                used.update(road_chain)
                current = (x, y)
            else:
                # 处理复杂路径
                for _ in range(f):
                    pos = current
                    step_roads = []
                    for move in s:
                        x, y = pos
                        dx, dy = {'U': (-1,0), 'D':(1,0), 'L':(0,-1), 'R':(0,1)}[move]
                        nx, ny = x + dx, y + dy
                        if not (1 <= nx <= n and 1 <= ny <= m):
                            return False
                        road = ((x, y), (nx, ny))
                        if road in used or road in step_roads:
                            return False
                        step_roads.append(road)
                        pos = (nx, ny)
                    used.update(step_roads)
                    current = pos
        return True
