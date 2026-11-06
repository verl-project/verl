"""# 

### 谜题描述
Piet is one of the most known visual esoteric programming languages. The programs in Piet are constructed from colorful blocks of pixels and interpreted using pretty complicated rules. In this problem we will use a subset of Piet language with simplified rules.

The program will be a rectangular image consisting of colored and black pixels. The color of each pixel will be given by an integer number between 0 and 9, inclusive, with 0 denoting black. A block of pixels is defined as a rectangle of pixels of the same color (not black). It is guaranteed that all connected groups of colored pixels of the same color will form rectangular blocks. Groups of black pixels can form arbitrary shapes.

The program is interpreted using movement of instruction pointer (IP) which consists of three parts:

  * current block pointer (BP); note that there is no concept of current pixel within the block;
  * direction pointer (DP) which can point left, right, up or down;
  * block chooser (CP) which can point to the left or to the right from the direction given by DP; in absolute values CP can differ from DP by 90 degrees counterclockwise or clockwise, respectively.



Initially BP points to the block which contains the top-left corner of the program, DP points to the right, and CP points to the left (see the orange square on the image below).

One step of program interpretation changes the state of IP in a following way. The interpreter finds the furthest edge of the current color block in the direction of the DP. From all pixels that form this edge, the interpreter selects the furthest one in the direction of CP. After this, BP attempts to move from this pixel into the next one in the direction of DP. If the next pixel belongs to a colored block, this block becomes the current one, and two other parts of IP stay the same. It the next pixel is black or outside of the program, BP stays the same but two other parts of IP change. If CP was pointing to the left, now it points to the right, and DP stays the same. If CP was pointing to the right, now it points to the left, and DP is rotated 90 degrees clockwise.

This way BP will never point to a black block (it is guaranteed that top-left pixel of the program will not be black).

You are given a Piet program. You have to figure out which block of the program will be current after n steps.

Input

The first line of the input contains two integer numbers m (1 ≤ m ≤ 50) and n (1 ≤ n ≤ 5·107). Next m lines contain the rows of the program. All the lines have the same length between 1 and 50 pixels, and consist of characters 0-9. The first character of the first line will not be equal to 0.

Output

Output the color of the block which will be current after n steps of program interpretation.

Examples

Input

2 10
12
43


Output

1


Input

3 12
1423
6624
6625


Output

6


Input

5 9
10345
23456
34567
45678
56789


Output

5

Note

In the first example IP changes in the following way. After step 1 block 2 becomes current one and stays it after two more steps. After step 4 BP moves to block 3, after step 7 — to block 4, and finally after step 10 BP returns to block 1.

<image>

The sequence of states of IP is shown on the image: the arrows are traversed clockwise, the main arrow shows direction of DP, the side one — the direction of CP.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
class Piet:
    inc = [{'x':0,'y':-1},{'x':1,'y':0},{'x':0,'y':1},{'x':-1,'y':0}]
    def __init__(self):
        self.BP = {'x':0,'y':0}
        self.DP = 1
        self.CP = 0
        self.getdata()
        self.go()
    def getdata(self):
        in_line = raw_input().split()
        self.m = int(in_line[0])
        self.n = int(in_line[1])
        self.pixels = []
        for i in range(self.m):
            self.pixels.append(raw_input())
    def is_out_limit(self,x,y):
        if x >= len(self.pixels[0]) or y >= self.m or x<0 or y<0:
            return True
        else:
            return False
    def go(self):
        ans = []
        info = []
        for t in xrange(self.n):
            while True:
                if self.is_out_limit(self.BP['x']+Piet.inc[self.DP]['x'],self.BP['y']+Piet.inc[self.DP]['y']):
                    break
                curr_color = self.pixels[self.BP['y']][self.BP['x']]
                new_color = self.pixels[self.BP['y']+Piet.inc[self.DP]['y']][self.BP['x']+Piet.inc[self.DP]['x']]
                if curr_color == new_color:
                    self.BP['x'] += Piet.inc[self.DP]['x']
                    self.BP['y'] += Piet.inc[self.DP]['y']
                else:
                    break
            while True:
                if self.is_out_limit(self.BP['x']+Piet.inc[self.CP]['x'],self.BP['y']+Piet.inc[self.CP]['y']):
                    break
                curr_color = self.pixels[self.BP['y']][self.BP['x']]
                new_color = self.pixels[self.BP['y']+Piet.inc[self.CP]['y']][self.BP['x']+Piet.inc[self.CP]['x']]
                if curr_color == new_color:
                    self.BP['x'] += Piet.inc[self.CP]['x']
                    self.BP['y'] += Piet.inc[self.CP]['y']
                else:
                    break
            if  self.is_out_limit(self.BP['x']+Piet.inc[self.DP]['x'],self.BP['y']+Piet.inc[self.DP]['y']) or self.pixels[self.BP['y']+Piet.inc[self.DP]['y']][self.BP['x']+Piet.inc[self.DP]['x']] == '0':
                if self.DP == (self.CP + 1)%4:
                    self.CP = (self.CP + 2)%4
                else:
                    self.DP = (self.DP + 1)%4
                    self.CP = (self.DP - 1)%4
            else:
                self.BP['x'] += Piet.inc[self.DP]['x']
                self.BP['y'] += Piet.inc[self.DP]['y']
            #print(self.BP['x'],self.BP['y'],self.DP,self.CP)
            if [self.BP['x'],self.BP['y'],self.DP,self.CP] in info:
                dot = info.index( [self.BP['x'],self.BP['y'],self.DP,self.CP] )
                print ans[dot-1+(self.n-dot)%(len(ans)-dot)]
                break
            else:
                ans.append(self.pixels[self.BP['y']][self.BP['x']])
                info.append( [self.BP['x'],self.BP['y'],self.DP,self.CP] )
        else:
            print ans[-1]

def main():
    p = Piet()

if __name__ == '__main__':
    main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import deque
from bootcamp import Basebootcamp

class Dpietbootcamp(Basebootcamp):
    def __init__(self, max_rows=5, max_cols=5, max_steps=1_000_000):
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.max_steps = max_steps

    def case_generator(self):
        while True:
            try:
                m = random.randint(1, self.max_rows)
                cols = random.randint(1, self.max_cols)
                pixels = self._generate_valid_piet_grid(m, cols)
                n = random.randint(1, self.max_steps)
                expected = self._simulate_piet(m, n, pixels)
                return {
                    'm': m,
                    'n': n,
                    'pixels': pixels,
                    'expected': str(expected)
                }
            except Exception as e:
                continue

    def _generate_valid_piet_grid(self, m, cols):
        grid = [['0']*cols for _ in range(m)]
        colors = deque(random.sample('123456789', k=9))
        visited = [[False]*cols for _ in range(m)]

        # 生成初始块（包含(0,0)）
        color = colors.popleft()
        max_h = min(random.randint(1, m), m)
        max_w = min(random.randint(1, cols), cols)
        for i in range(max_h):
            for j in range(max_w):
                grid[i][j] = color
                visited[i][j] = True

        # 生成后续色块
        while colors:
            candidates = []
            for i in range(m):
                for j in range(cols):
                    if not visited[i][j]:
                        if (i == 0 or visited[i-1][j]) and (j == 0 or visited[i][j-1]):
                            max_h_block = 1
                            while i+max_h_block < m and not visited[i+max_h_block][j]:
                                max_h_block += 1
                            max_w_block = 1
                            while j+max_w_block < cols and not visited[i][j+max_w_block]:
                                max_w_block += 1
                            if max_h_block >=1 and max_w_block >=1:
                                candidates.append((i,j,max_h_block,max_w_block))
            
            if not candidates:
                break
            
            i,j,h_max,w_max = random.choice(candidates)
            color = colors.popleft()
            h = random.randint(1, h_max)
            w = random.randint(1, w_max)
            
            for di in range(h):
                for dj in range(w):
                    if i+di < m and j+dj < cols:
                        grid[i+di][j+dj] = color
                        visited[i+di][j+dj] = True
        
        return [''.join(row) for row in grid]

    @staticmethod
    def _simulate_piet(m, n, pixels):
        simulator = DpietSimulator(m, n, pixels)
        return simulator.simulate()

    @staticmethod
    def prompt_func(question_case):
        grid = '\n'.join(question_case['pixels'])
        return f"""Solve this Dpiet programming puzzle. After exactly {question_case['n']} steps, what is the current color block?

Rules:
- Program is a {question_case['m']}x{len(question_case['pixels'][0])} grid (0=black)
- Blocks are rectangular non-black regions
- IP starts at top-left block, DP=right, CP=left
- Each step:
  1. Move to DP edge within current block
  2. Move to CP edge within current block 
  3. Attempt DP move:
     - Success: Enter new block
     - Failure: 
       If CP was left → switch to right
       If CP was right → rotate DP 90° clockwise and reset CP to left

Input:
{question_case['m']} {question_case['n']}
{grid}

Output your answer between [answer] and [/answer], e.g. [answer]3[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d)\s*\[/answer\]', output, re.IGNORECASE)
        return matches[-1] if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected']

class DpietSimulator:
    DIRS = [{'x':0,'y':-1}, {'x':1,'y':0}, {'x':0,'y':1}, {'x':-1,'y':0}]  # 上下左右
    
    def __init__(self, m, n, pixels):
        self.m = m
        self.n = n
        self.pixels = pixels
        self.cols = len(pixels[0])
        self.bp = {'x':0, 'y':0}
        self.dp = 1  # 初始方向：右
        self.cp = 0  # 初始选择器：左
    
    def simulate(self):
        history = []
        colors = []
        
        for _ in range(self.n):
            # 循环检测
            state = (self.bp['x'], self.bp['y'], self.dp, self.cp)
            if state in history:
                idx = history.index(state)
                cycle = colors[idx:]
                return cycle[(self.n - idx) % len(cycle)]
            history.append(state)
            
            # 步骤1：移动到DP方向边缘
            self.move_to_edge(self.dp)
            # 步骤2：移动到CP方向边缘
            self.move_to_edge(self.cp)
            
            # 步骤3：尝试移动
            next_x = self.bp['x'] + self.DIRS[self.dp]['x']
            next_y = self.bp['y'] + self.DIRS[self.dp]['y']
            
            if self.is_out_of_bounds(next_x, next_y) or self.pixels[next_y][next_x] == '0':
                # 处理方向调整
                if self.cp == (self.dp - 1) % 4:
                    self.cp = (self.cp + 2) % 4
                else:
                    self.dp = (self.dp + 1) % 4
                    self.cp = (self.dp - 1) % 4
            else:
                self.bp = {'x': next_x, 'y': next_y}
            
            colors.append(self.pixels[self.bp['y']][self.bp['x']])
        
        return colors[-1]

    def move_to_edge(self, direction):
        current_color = self.pixels[self.bp['y']][self.bp['x']]
        while True:
            next_x = self.bp['x'] + self.DIRS[direction]['x']
            next_y = self.bp['y'] + self.DIRS[direction]['y']
            if self.is_out_of_bounds(next_x, next_y):
                break
            if self.pixels[next_y][next_x] != current_color:
                break
            self.bp = {'x': next_x, 'y': next_y}
    
    def is_out_of_bounds(self, x, y):
        return not (0 <= x < self.cols and 0 <= y < self.m)
