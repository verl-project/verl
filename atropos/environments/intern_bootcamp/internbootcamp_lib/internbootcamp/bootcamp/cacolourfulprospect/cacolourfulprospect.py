"""# 

### 谜题描述
Firecrackers scare Nian the monster, but they're wayyyyy too noisy! Maybe fireworks make a nice complement.

Little Tommy is watching a firework show. As circular shapes spread across the sky, a splendid view unfolds on the night of Lunar New Year's eve.

A wonder strikes Tommy. How many regions are formed by the circles on the sky? We consider the sky as a flat plane. A region is a connected part of the plane with positive area, whose bound consists of parts of bounds of the circles and is a curve or several curves without self-intersections, and that does not contain any curve other than its boundaries. Note that exactly one of the regions extends infinitely.

Input

The first line of input contains one integer n (1 ≤ n ≤ 3), denoting the number of circles.

The following n lines each contains three space-separated integers x, y and r ( - 10 ≤ x, y ≤ 10, 1 ≤ r ≤ 10), describing a circle whose center is (x, y) and the radius is r. No two circles have the same x, y and r at the same time.

Output

Print a single integer — the number of regions on the plane.

Examples

Input

3
0 0 1
2 0 1
4 0 1


Output

4


Input

3
0 0 2
3 0 2
6 0 2


Output

6


Input

3
0 0 2
2 0 2
1 1 2


Output

8

Note

For the first example,

<image>

For the second example,

<image>

For the third example,

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
import math

def stoc(s):
    return tuple(map(int, s.split()))

def ncut(c0, c1):
    d2 = (c0[0]-c1[0])**2 + (c0[1]-c1[1])**2
    rsum2 = (c0[2]+c1[2])**2
    if rsum2 < d2:
        return 0
    elif rsum2 == d2:
        return 1
    else:
        #d = |r0 - r1| ?
        rdiff2 = (c0[2]-c1[2])**2
        if d2 < rdiff2:
            return 0
        elif d2 == rdiff2:
            return 1
        else:        
            return 2

def testcuts(cs):
    N=10**10
    eps=1e-8
    ai = 0
    while ai < N:
        a = 2*math.pi*ai/N+.1
        x = cs[0][0] + cs[0][2]*math.cos(a)
        y = cs[0][1] + cs[0][2]*math.sin(a)
        d1 = abs(((x-cs[1][0])**2 + (y-cs[1][1])**2)**.5 - cs[1][2])
        d2 = abs(((x-cs[2][0])**2 + (y-cs[2][1])**2)**.5 - cs[2][2])
        #print x, y, d1, d2
        if abs(d1) + abs(d2) < eps:
            #print x, y, d1, d2
            return True
        steps = min(d1, d2) / cs[0][2]
        ai += 1
        ai += int(N*steps/10)
    return False
        
xx = sys.stdin.readlines()
n = int(xx[0])
cs = map(stoc, xx[1:])
#print cs

if n == 1:
    print 2
    sys.exit(0)
if n == 2:
    print 2 + max(ncut(cs[0], cs[1]), 1)
    sys.exit(0)

ncuts = ncut(cs[0],cs[1]) + ncut(cs[1],cs[2]) + ncut(cs[2], cs[0])

if ncuts == 0:
    print 4
    sys.exit(0)
    
#print ncut(cs[0],cs[1]), ncut(cs[1],cs[2]), ncut(cs[2], cs[0])

if ncut(cs[0], cs[1]) == 0 and ncut(cs[1],cs[2]) == 0:
    ncuts += 1
if ncut(cs[0], cs[1]) == 0 and ncut(cs[2],cs[0]) == 0:
    ncuts += 1
if ncut(cs[0], cs[2]) == 0 and ncut(cs[1],cs[2]) == 0:
    ncuts += 1

#Do all three intersect at the same point?
if ncuts >=3 and testcuts(cs):
    ncuts -= 1
    #two cuts and all three in same line?
    if ncuts>3 and (cs[0][0] - cs[1][0])*(cs[0][1] - cs[2][1]) == (cs[0][1] - cs[1][1])*(cs[0][0] - cs[2][0]):
        ncuts -= 1

print 2 + ncuts
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import re
import random
from bootcamp import Basebootcamp

class Cacolourfulprospectbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=3, x_range=(-10, 10), y_range=(-10, 10), r_range=(1, 10)):
        self.min_n = min_n
        self.max_n = max_n
        self.x_range = x_range
        self.y_range = y_range
        self.r_range = r_range

    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        circles = []
        for _ in range(n):
            while True:
                x = random.randint(*self.x_range)
                y = random.randint(*self.y_range)
                r = random.randint(*self.r_range)
                if not any((x, y, r) == c for c in circles):
                    circles.append((x, y, r))
                    break
        return {'n': n, 'circles': circles}

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        circles = question_case['circles']
        problem = (
            "在春节烟花表演中，计算圆形成的平面区域数。规则：\n"
            "1. 区域是连通的且面积大于0\n"
            "2. 每个区域由圆弧围成且无自交\n"
            "3. 恰好一个区域无限延伸\n\n"
            f"输入：\n{n}\n" + 
            "\n".join(f"{x} {y} {r}" for x, y, r in circles) +
            "\n答案格式：[answer]整数[/answer]"
        )
        return problem

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        try:
            return int(matches[-1].strip()) if matches else None
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return solution == cls.compute_regions(identity['n'], identity['circles'])
        except:
            return False

    @classmethod
    def compute_regions(cls, n, circles):
        circles = [tuple(c) for c in circles]
        if n == 1:
            return 2
        if n == 2:
            return 2 + max(cls.ncut(*circles), 1)
        
        # 处理三个圆的情况
        c1, c2, c3 = circles
        cuts = (
            cls.ncut(c1, c2) 
            + cls.ncut(c2, c3) 
            + cls.ncut(c3, c1)
        )
        
        # 处理不相交圆对
        non_intersect_pairs = [
            (cls.ncut(c1, c2) == 0),
            (cls.ncut(c2, c3) == 0),
            (cls.ncut(c3, c1) == 0)
        ]
        if sum(non_intersect_pairs) >= 2:
            cuts += 1
        
        # 检测三圆公共交点
        if cuts >= 3 and cls.triple_intersection(circles):
            cuts -= 1
            if cls.collinear(c1[:2], c2[:2], c3[:2]):
                cuts -= 1
                
        return 2 + cuts

    @classmethod
    def ncut(cls, c1, c2):
        dx, dy = c1[0]-c2[0], c1[1]-c2[1]
        d_sq = dx**2 + dy**2
        r_sum = c1[2] + c2[2]
        r_diff = abs(c1[2] - c2[2])
        
        if d_sq > r_sum**2: return 0     # 外离
        if d_sq == r_sum**2: return 1    # 外切
        if d_sq < r_diff**2: return 0    # 内含
        if d_sq == r_diff**2: return 1   # 内切
        return 2                         # 相交

    @classmethod
    def triple_intersection(cls, circles):
        """精确检测三圆公共交点"""
        for i in range(3):
            a, b, c = circles[i], circles[(i+1)%3], circles[(i+2)%3]
            points = cls.get_intersections(a, b)
            for p in points:
                if cls.point_on_circle(p, c):
                    return True
        return False

    @staticmethod
    def get_intersections(c0, c1):
        """计算两圆精确交点"""
        x0, y0, r0 = c0
        x1, y1, r1 = c1
        
        d = math.hypot(x1-x0, y1-y0)
        if d > r0 + r1 or d < abs(r0 - r1):
            return []
        
        a = (r0**2 - r1**2 + d**2) / (2*d)
        h = math.sqrt(r0**2 - a**2)
        x2 = x0 + a*(x1 - x0)/d
        y2 = y0 + a*(y1 - y0)/d
        
        return [
            (x2 + h*(y1-y0)/d, y2 - h*(x1-x0)/d),
            (x2 - h*(y1-y0)/d, y2 + h*(x1-x0)/d)
        ] if h != 0 else [(x2, y2)]

    @staticmethod
    def point_on_circle(point, circle, eps=1e-8):
        """精确到1e-8的浮点误差判断"""
        x, y = point
        cx, cy, r = circle
        return abs((x - cx)**2 + (y - cy)**2 - r**2) < eps

    @staticmethod
    def collinear(p1, p2, p3):
        """三点共线检测优化版"""
        area = (p2[0] - p1[0])*(p3[1] - p1[1]) - (p2[1] - p1[1])*(p3[0] - p1[0])
        return abs(area) < 1e-8  # 允许浮点误差
