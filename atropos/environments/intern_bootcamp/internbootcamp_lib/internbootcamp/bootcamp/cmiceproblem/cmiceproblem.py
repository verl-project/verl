"""# 

### 谜题描述
Igor the analyst fell asleep on the work and had a strange dream. In the dream his desk was crowded with computer mice, so he bought a mousetrap to catch them.

The desk can be considered as an infinite plane, then the mousetrap is a rectangle which sides are parallel to the axes, and which opposite sides are located in points (x1, y1) and (x2, y2).

Igor wants to catch all mice. Igor has analysed their behavior and discovered that each mouse is moving along a straight line with constant speed, the speed of the i-th mouse is equal to (vix, viy), that means that the x coordinate of the mouse increases by vix units per second, while the y coordinates increases by viy units. The mousetrap is open initially so that the mice are able to move freely on the desk. Igor can close the mousetrap at any moment catching all the mice that are strictly inside the mousetrap.

Igor works a lot, so he is busy in the dream as well, and he asks you to write a program that by given mousetrap's coordinates, the initial coordinates of the mice and their speeds determines the earliest time moment in which he is able to catch all the mice. Please note that Igor can close the mousetrap only once.

Input

The first line contains single integer n (1 ≤ n ≤ 100 000) — the number of computer mice on the desk.

The second line contains four integers x1, y1, x2 and y2 (0 ≤ x1 ≤ x2 ≤ 100 000), (0 ≤ y1 ≤ y2 ≤ 100 000) — the coordinates of the opposite corners of the mousetrap.

The next n lines contain the information about mice.

The i-th of these lines contains four integers rix, riy, vix and viy, (0 ≤ rix, riy ≤ 100 000,  - 100 000 ≤ vix, viy ≤ 100 000), where (rix, riy) is the initial position of the mouse, and (vix, viy) is its speed.

Output

In the only line print minimum possible non-negative number t such that if Igor closes the mousetrap at t seconds from the beginning, then all the mice are strictly inside the mousetrap. If there is no such t, print -1.

Your answer is considered correct if its absolute or relative error doesn't exceed 10 - 6. 

Formally, let your answer be a, and the jury's answer be b. Your answer is considered correct if <image>.

Examples

Input

4
7 7 9 8
3 5 7 5
7 5 2 4
3 3 7 8
6 6 3 2


Output

0.57142857142857139685


Input

4
7 7 9 8
0 3 -5 4
5 0 5 4
9 9 -1 -6
10 5 -7 -10


Output

-1

Note

Here is a picture of the first sample

Points A, B, C, D - start mice positions, segments are their paths.

<image>

Then, at first time when all mice will be in rectangle it will be looks like this:

<image>

Here is a picture of the second sample

<image>

Points A, D, B will never enter rectangle.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def comp(rx,ry,vx,vy,x1,y1,x2,y2):

    \"\"\"
    if x1 == rx and vx <= 0  :
        return None,None

    if x2 == rx and vx >= 0:
        return None,None

    if y1 == ry and vy <= 0:
        return None,None

    if y2 == ry and vy >= 0:
        return None,None

    #if x1 == rx and or x2 == rx  or y1 == ry or y2 == ry:
    #    ta = 0 # at border in start
    if x1 < rx < x2 and y1 < ry < y2:
        ta = 0
    else:
        ta = -1
    \"\"\"
    \"\"\"
    if ta == 0: # compute tb, leavetime
        time = []
        # meet x = x1
        if vx != 0:
            t = (x1-rx)
            if y1*vx <= ry + t * vy <= y2*vx:
                time.append(float(t)/vx)


        # meet x = x2
        if vx != 0:
            t = (x2-rx)
            if y1*vx <= ry + t * vy <= y2*vx:
                time.append(float(t)/vx)

        # meet y = y1
        if vy != 0:
            t = (y1-ry)
            if x1*vy <= rx + t * vx <= x2*vy:
                time.append(float(t)/vy)

        # meet y = y2
        if vy != 0:
            t = (y2-ry)
            if x1*vy <= rx + t * vx <= x2*vy:
                time.append(float(t)/vy)

        if len(time) == 0:
            print \"=========ERROR\"
        else:
            return [0,min(time)]
    \"\"\"
    
    time = []
    # meet x = x1
    if vx != 0: # ------->
        t = (x1-rx)
  
        if y1*vx < ry*vx + t * vy < y2*vx or y1*vx > ry*vx + t * vy > y2*vx:
            time.append(float(t)/vx)
        elif y1*vx == ry*vx + t * vy or  y2*vx == ry*vx + t * vy:
            if vy != 0:
                time.append(float(t)/vx)


    # meet x = x2
    if vx != 0:
        t = (x2-rx)
        if y1*vx < ry*vx + t * vy < y2*vx or y1*vx > ry*vx + t * vy > y2*vx:
            time.append(float(t)/vx)
        elif y1*vx == ry*vx + t * vy or  y2*vx == ry*vx + t * vy:
            if vy != 0:
                time.append(float(t)/vx)

    # meet y = y1
    if vy != 0:
        t = (y1-ry)

        if x1*vy < rx*vy + t * vx < x2*vy or x1*vy > rx*vy + t * vx > x2*vy:
            time.append(float(t)/vy)
        elif x1*vy == rx*vy + t * vx or  x2*vy == rx*vy + t * vx:
            if vx != 0:
                time.append(float(t)/vy)

    # meet y = y2
    if vy != 0:
        t = (y2-ry)

        if x1*vy < rx*vy + t * vx < x2*vy or x1*vy > rx*vy + t * vx > x2*vy:
            time.append(float(t)/vy)
        elif x1*vy == rx*vy + t * vx or  x2*vy == rx*vy + t * vx:
            if vx != 0:
                time.append(float(t)/vy)

    time = [i for i in time if i >= 0]
  
    if len(time) == 0:
        if x1 < rx < x2 and y1 < ry < y2:
            return True,True
        else:
            return None,None

    elif len(time) == 1:
        return [0,time[0]]
    
    elif len(time) == 2:
        if x1 < rx < x2 and y1 < ry < y2:
            if time[0] > 0:
                return [0,time[0]]
            else:
                return [0,time[1]]

        else:
            if time[0] == time[1]:
                return None,None
            else:
                return min(time),max(time)
    else:
        
        return min(time),max(time)
            
        




        
n  = input()
x1,y1,x2,y2 = [int(_) for _ in raw_input().split()]

store = [[int(_) for _ in raw_input().split()] for _ in range(n)]

if x1 == x2 or y1 == y2:
    print -1
else:

    ans = [-1,float('inf')]

    for rx,ry,vx,vy in store:

        ta,tb = comp(rx,ry,vx,vy,x1,y1,x2,y2)

        #print ta,tb
        #print ta,tb, rx,ry,vx,vy,x1,y1,x2,y2
        
        if ta is None:
            print -1
            break

        if ta is True:
            continue # always satisfy
        
        ans = [max(ta,ans[0]),min(tb,ans[1])]

        if ans[0] >= ans[1]:
            print -1
            break

    else:
        print max(0,ans[0])
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import json
from bootcamp import Basebootcamp

class Cmiceproblembootcamp(Basebootcamp):
    def __init__(self, x1=7, y1=7, x2=9, y2=8, n=4, has_solution=True, seed=None):
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)
        self.n = n
        self.has_solution = has_solution
        self.rng = random.Random(seed)
        
        # 参数合法性校验
        if not (0 <= self.x1 < self.x2 <= 1e5 and 0 <= self.y1 < self.y2 <= 1e5):
            raise ValueError("Invalid rectangle coordinates")

    def case_generator(self):
        mice = []
        x1, y1, x2, y2 = self.x1, self.y1, self.x2, self.y2

        if self.has_solution:
            # 生成保证有解的案例
            t_solution = self.rng.uniform(0.1, 10.0)  # 公共解时间
            for _ in range(self.n):
                # 生成在t_solution时位于矩形内部的参数
                rx_at_t = self.rng.uniform(x1 + 1e-9, x2 - 1e-9)
                ry_at_t = self.rng.uniform(y1 + 1e-9, y2 - 1e-9)
                
                # 初始位置可能在内部或外部
                if self.rng.random() < 0.5:  # 50%概率生成初始在内部的老鼠
                    rx = rx_at_t
                    ry = ry_at_t
                    vx = 0.0
                    vy = 0.0
                else:
                    # 初始在外部但会在t_solution时进入
                    dx = self.rng.uniform(1.0, 5.0)
                    dy = self.rng.uniform(1.0, 5.0)
                    # 随机选择进入方向（左/右/下/上）
                    direction = self.rng.choice(['left', 'right', 'bottom', 'top'])
                    if direction == 'left':
                        rx = x1 - dx
                        ry = self.rng.uniform(y1, y2)
                        vx = (x1 - dx - rx) / t_solution  # 确保在t时到达x1边界
                    elif direction == 'right':
                        rx = x2 + dx
                        ry = self.rng.uniform(y1, y2)
                        vx = (x1 - dx - rx) / t_solution
                    elif direction == 'bottom':
                        rx = self.rng.uniform(x1, x2)
                        ry = y1 - dy
                        vy = (y1 - dy - ry) / t_solution
                    else:
                        rx = self.rng.uniform(x1, x2)
                        ry = y2 + dy
                        vy = (y1 - dy - ry) / t_solution
                    # 调整速度确保在t时准确进入
                    vx = (rx_at_t - rx) / t_solution
                    vy = (ry_at_t - ry) / t_solution
                
                mice.append((rx, ry, vx, vy))
        else:
            # 生成保证至少有一个无法捕获的老鼠
            direction = self.rng.choice(['left', 'right', 'bottom', 'top'])
            # 生成永远无法进入的老鼠
            if direction == 'left':
                rx = self.rng.uniform(0, x1 - 1e-9)
                ry = self.rng.uniform(y1, y2)
                vx = self.rng.uniform(-1e5, -1.0)  # 持续左移
                vy = 0.0
            elif direction == 'right':
                rx = self.rng.uniform(x2 + 1e-9, 1e5)
                ry = self.rng.uniform(y1, y2)
                vx = self.rng.uniform(1.0, 1e5)   # 持续右移
                vy = 0.0
            elif direction == 'bottom':
                ry = self.rng.uniform(0, y1 - 1e-9)
                rx = self.rng.uniform(x1, x2)
                vy = self.rng.uniform(-1e5, -1.0) # 持续下移
                vx = 0.0
            else:
                ry = self.rng.uniform(y2 + 1e-9, 1e5)
                rx = self.rng.uniform(x1, x2)
                vy = self.rng.uniform(1.0, 1e5)   # 持续上移
                vx = 0.0
            mice.append((rx, ry, vx, vy))
            
            # 其余老鼠生成可捕获的参数
            for _ in range(self.n-1):
                rx = self.rng.uniform(x1 + 1e-9, x2 - 1e-9)
                ry = self.rng.uniform(y1 + 1e-9, y2 - 1e-9)
                mice.append((rx, ry, 0.0, 0.0))

        return {
            "n": self.n,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "mice": [[round(v, 6) for v in mouse] for mouse in mice]
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        mice_desc = "\n".join(
            f"- Mouse {i+1}: Starts at ({m[0]:.2f}, {m[1]:.2f}), Velocity ({m[2]:.2f}, {m[3]:.2f})"
            for i, m in enumerate(question_case['mice'])
        )
        return f"""Igor needs to catch all computer mice moving on an infinite desk. The mousetrap is a rectangle with corners at ({question_case['x1']}, {question_case['y1']}) and ({question_case['x2']}, {question_case['y2']}).

**Movement Rules:**
- Each mouse moves at a constant velocity (vx, vy)
- A mouse is caught if strictly INSIDE the rectangle when the trap closes
- Trap can be closed only ONCE at time t ≥ 0

**Given Mouse Data (n={question_case['n']}):**
{mice_desc}

**Output Requirements:**
1. If possible: Print the earliest t with at least 10 decimal digits
2. If impossible: Print -1
3. Enclose your answer with [answer] and [/answer]

**Example Format:**
[answer]0.5714285714285714[/answer]"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return -1 if last_match == "-1" else float(last_match)
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        x1, y1, x2, y2 = identity['x1'], identity['y1'], identity['x2'], identity['y2']
        if x1 >= x2 or y1 >= y2:
            return solution == -1
        
        time_windows = []
        for mouse in identity['mice']:
            rx, ry, vx, vy = mouse
            enter, exit = cls.calculate_time_window(rx, ry, vx, vy, x1, y1, x2, y2)
            
            if enter is None:
                return solution == -1  # 当前老鼠无法被捕获
            
            time_windows.append((enter, exit))
        
        # 计算全局时间窗口交集
        global_enter = max(t[0] for t in time_windows)
        global_exit = min(t[1] for t in time_windows)
        
        if global_enter > global_exit:
            return solution == -1
        
        expected_t = max(0.0, global_enter)
        
        # 精度校验
        if solution == -1:
            return False
        return abs(solution - expected_t) < 1e-6 or abs(solution - expected_t)/expected_t < 1e-6

    @staticmethod
    def calculate_time_window(rx, ry, vx, vy, x1, y1, x2, y2):
        epsilon = 1e-12  # 处理浮点误差
        initial_inside = (x1 < rx < x2) and (y1 < ry < y2)
        time_points = []
        
        # X方向边界计算
        if vx != 0:
            tx1 = (x1 - rx) / vx
            if tx1 >= -epsilon:
                y_at_tx1 = ry + vy * tx1
                if y1 - epsilon < y_at_tx1 < y2 + epsilon:
                    time_points.append(tx1)
            
            tx2 = (x2 - rx) / vx
            if tx2 >= -epsilon:
                y_at_tx2 = ry + vy * tx2
                if y1 - epsilon < y_at_tx2 < y2 + epsilon:
                    time_points.append(tx2)
        
        # Y方向边界计算
        if vy != 0:
            ty1 = (y1 - ry) / vy
            if ty1 >= -epsilon:
                x_at_ty1 = rx + vx * ty1
                if x1 - epsilon < x_at_ty1 < x2 + epsilon:
                    time_points.append(ty1)
            
            ty2 = (y2 - ry) / vy
            if ty2 >= -epsilon:
                x_at_ty2 = rx + vx * ty2
                if x1 - epsilon < x_at_ty2 < x2 + epsilon:
                    time_points.append(ty2)
        
        # 处理初始在内部的情况
        if initial_inside:
            time_points.append(0.0)
        
        # 筛选有效时间点
        valid_times = []
        for t in time_points:
            x = rx + vx * t
            y = ry + vy * t
            if (x1 < x < x2) and (y1 < y < y2):
                valid_times.append(t)
        
        if not valid_times:
            return (None, None) if not initial_inside else (0.0, float('inf'))
        
        # 确定时间窗口
        valid_times = sorted(valid_times)
        enter = valid_times[0]
        exit = valid_times[-1]
        
        if initial_inside:
            # 处理持续停留的情况
            if vx == 0 and vy == 0:
                return (0.0, float('inf'))
            # 找到离开时间
            exit_times = [t for t in valid_times if t > 0]
            if not exit_times:
                return (0.0, float('inf'))
            exit = min(exit_times)
        
        return (max(0.0, enter), exit)
