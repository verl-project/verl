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
# CP = [left, right] right = +1 , left = -1

# DP = [left,right,top,down]
#       0     2     1   3
\"\"\"
      1
--- 0   2
      3
\"\"\"

gather ={}

def move(i):
    global BP
    global DP
    global CP
    global skip
    
    if (BP[0],BP[1],DP,CP) not in gather:
        gather[(BP[0],BP[1],DP,CP)] = i
    else:
        return False
    
    color = store[BP[1]][BP[0]]
    if DP % 2 == 0: 
        while BP[0] + (DP - 1) >= 0 and BP[0] + (DP - 1) < width   and store[BP[1]][BP[0] + (DP - 1)] == color:
            BP[0] = BP[0] + (DP - 1)
    else:
        while BP[1] + (DP - 2) >= 0 and BP[1] + (DP - 2) < length  and store[BP[1] + (DP - 2)][BP[0]] == color:
            BP[1] = BP[1] + (DP - 2)

    direction = (DP+CP)%4

    if direction == 0:
        
        while BP[0] - 1 >= 0 and store[BP[1]][BP[0]- 1] == color:
            BP[0] -= 1
            
    elif direction == 2:
         while BP[0] + 1 < width and store[BP[1]][BP[0]+1] == color:
            BP[0] += 1

    elif direction == 1:
         while BP[1] - 1 >= 0 and store[BP[1]-1][BP[0]] == color:
            BP[1] -= 1

    elif direction == 3:
         while BP[1] + 1 < length and store[BP[1]+1][BP[0]] == color:
            BP[1] += 1
    if DP % 2 == 0:
        if BP[0] + (DP - 1) < 0 or  BP[0] + (DP - 1) >= width or store[BP[1]][BP[0] + (DP - 1)] == '0':
            if CP == -1:
                CP = 1
            else:
                CP = -1
                DP = (DP+1)%4
        else:
            BP[0] += (DP-1)
    else:
        if BP[1] + (DP - 2) < 0 or  BP[1] + (DP - 2) >= length or store[BP[1] + (DP - 2)][BP[0]] == '0':
            if CP == -1:
                CP = 1
            else:
                CP = -1
                DP = (DP+1)%4
        else:
            BP[1] += (DP-2)
    return True

(m,n) = raw_input().split()

m = int(m)

n = int(n)


store = []

for i in range(m):
    store.append(raw_input())
    
width = len(store[0])
length = m

BP = [0,0] # current pixel ( although ...)
DP = 2 # right
CP = -1 # left of DP

i = 0

while i<n:
    x = move(i)
    if x == False:
        t = gather[(BP[0],BP[1],DP,CP)]
        i += ((n-i)//(i-t))*(i-t)
        gather.pop((BP[0],BP[1],DP,CP))
    else:
        i+=1

print store[BP[1]][BP[0]]
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from collections import OrderedDict
from bootcamp import Basebootcamp

def generate_valid_grid(m, k):
    """Generate valid grid with rectangular color blocks"""
    colors = [str(i) for i in range(1, 10)]
    grid = [['0'] * k for _ in range(m)]
    
    # Create main block (ensures starting point)
    grid[0][0] = random.choice(colors)
    main_color = grid[0][0]
    colors.remove(main_color)
    
    # Expand main block randomly
    max_w = 1
    while max_w < k and grid[0][max_w] == '0':
        max_w += 1
    max_h = 1
    while max_h < m and grid[max_h][0] == main_color:
        max_h += 1
    
    w = random.randint(1, max_w)
    h = random.randint(1, max_h)
    for y in range(h):
        for x in range(w):
            grid[y][x] = main_color
    
    # Add other blocks with simple shapes
    for color in colors:
        placed = False
        for _ in range(10):  # Placement attempts
            y = random.randint(0, m-1)
            x = random.randint(0, k-1)
            if grid[y][x] != '0':
                continue
                
            # Find maximum available rectangle
            max_w = 1
            while x + max_w < k and grid[y][x+max_w] == '0':
                max_w += 1
            max_h = 1
            while y + max_h < m:
                if all(c == '0' for c in grid[y+max_h][x:x+max_w]):
                    max_h += 1
                else:
                    break
            
            if max_w > 0 and max_h > 0:
                bw = random.randint(1, min(3, max_w))  # Limit block size
                bh = random.randint(1, min(3, max_h))
                for dy in range(bh):
                    for dx in range(bw):
                        grid[y+dy][x+dx] = color
                placed = True
                break
        if not placed:
            break
    
    return [''.join(row) for row in grid]

def calculate_block_bounds(grid, x, y):
    """Calculate rectangular bounds for the block containing (x,y)"""
    color = grid[y][x]
    # Horizontal expansion
    min_x = x
    while min_x > 0 and grid[y][min_x-1] == color:
        min_x -= 1
    max_x = x
    while max_x < len(grid[0])-1 and grid[y][max_x+1] == color:
        max_x += 1
    # Vertical expansion
    min_y = y
    while min_y > 0 and all(grid[min_y-1][cx] == color for cx in range(min_x, max_x+1)):
        min_y -= 1
    max_y = y
    while max_y < len(grid)-1 and all(grid[max_y+1][cx] == color for cx in range(min_x, max_x+1)):
        max_y += 1
    return (min_x, max_x, min_y, max_y)

def simulate_piet(grid, n_steps):
    if not grid or not grid[0]:
        return '0'
    
    m = len(grid)
    k = len(grid[0])
    DP = 2  # Initial direction: right
    CP = -1  # Initial chooser: left
    x, y = 0, 0
    
    state_cache = OrderedDict()
    step = 0
    
    while step < n_steps:
        # Check for cycles using LRU cache
        state = (x, y, DP, CP)
        if state in state_cache:
            cycle_start = state_cache[state]
            cycle_length = step - cycle_start
            if cycle_length > 0:
                remaining = n_steps - step
                step += (remaining // cycle_length) * cycle_length
                if step >= n_steps:
                    break
                # Reset cache after cycle skip
                state_cache.clear()
        state_cache[state] = step
        if len(state_cache) > 100:
            state_cache.popitem(last=False)
        
        # Get current block bounds
        min_x, max_x, min_y, max_y = calculate_block_bounds(grid, x, y)
        current_color = grid[y][x]
        
        # Move to DP edge using block bounds
        if DP == 0:    # Left
            x = min_x
        elif DP == 2:  # Right
            x = max_x
        elif DP == 1:  # Up
            y = min_y
        elif DP == 3:  # Down
            y = max_y
        
        # Move in CP direction using block bounds
        cp_dir = (DP + CP) % 4
        if cp_dir == 0:    # Left
            x = min_x
        elif cp_dir == 2:  # Right
            x = max_x
        elif cp_dir == 1:  # Up
            y = min_y
        elif cp_dir == 3:  # Down
            y = max_y
        
        # Attempt to move in DP direction
        new_x, new_y = x, y
        if DP == 0:    # Left
            new_x = x - 1
        elif DP == 2:  # Right
            new_x = x + 1
        elif DP == 1:  # Up
            new_y = y - 1
        elif DP == 3:  # Down
            new_y = y + 1
        
        valid = False
        if 0 <= new_x < k and 0 <= new_y < m:
            if grid[new_y][new_x] != '0':
                x, y = new_x, new_y
                valid = True
        
        if not valid:
            if CP == -1:
                CP = 1
            else:
                CP = -1
                DP = (DP + 1) % 4
        
        step += 1
    
    return grid[y][x]

class Bpietbootcamp(Basebootcamp):
    def __init__(self, max_m=4, max_k=4, max_n=10000):
        self.max_m = max_m
        self.max_k = max_k
        self.max_n = max_n
    
    def case_generator(self):
        # Generate small grids for fast simulation
        m = random.randint(1, min(3, self.max_m))
        k = random.randint(1, min(3, self.max_k))
        
        # Limit steps for large grids
        max_steps = self.max_n // (m * k)
        n = random.randint(1, max_steps)
        
        grid = generate_valid_grid(m, k)
        answer = simulate_piet(grid, n)
        
        return {
            'm': m,
            'n': n,
            'grid': grid,
            'answer': answer
        }
    
    @staticmethod
    def prompt_func(case) -> str:
        return f"""Solve this Bpiet puzzle after {case['n']} steps:
Grid:
{chr(10).join(case['grid'])}
Format answer as [answer]{{digit}}[/answer]"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\s*\]\s*(\d)\s*\[/answer\s*\]', output, re.IGNORECASE)
        return matches[-1] if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return str(solution) == str(identity['answer'])
