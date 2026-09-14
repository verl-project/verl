"""# 

### 谜题描述
There are three points marked on the coordinate plane. The goal is to make a simple polyline, without self-intersections and self-touches, such that it passes through all these points. Also, the polyline must consist of only segments parallel to the coordinate axes. You are to find the minimum number of segments this polyline may consist of.

Input

Each of the three lines of the input contains two integers. The i-th line contains integers xi and yi ( - 109 ≤ xi, yi ≤ 109) — the coordinates of the i-th point. It is guaranteed that all points are distinct.

Output

Print a single number — the minimum possible number of segments of the polyline.

Examples

Input

1 -1
1 1
1 2


Output

1


Input

-1 -1
-1 3
4 3


Output

2


Input

1 1
2 3
3 2


Output

3

Note

The variant of the polyline in the first sample: <image> The variant of the polyline in the second sample: <image> The variant of the polyline in the third sample: <image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
iai = lambda: map(int, raw_input().split())
p1=x1,y1=iai()
p2=x2,y2=iai()
p3=x3,y3=iai()

sp = [p1,p2,p3]

sx = [x1,x2,x3]
sy = [y1,y2,y3]
ssx = len(set([x1,x2,x3]))
ssy = len(set([y1,y2,y3]))
if 1 in (ssx, ssy):
    print 1
elif ssx==ssy==2:
    print 2
elif ssx==2:
    diff = [p for p in sp if sx.count(p[0]) == 1][0]
    print 2 + (sorted(sy).index(diff[1]) == 1)
elif ssy==2:
    diff = [p for p in sp if sy.count(p[1]) == 1][0]
    print 2 + (sorted(sx).index(diff[0]) == 1)
else:
    print 3
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Dpolylinebootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__(**params)
        self.x_range = params.get('x_range', (-10**9, 10**9))
        self.y_range = params.get('y_range', (-10**9, 10**9))
    
    def case_generator(self):
        # 主动生成覆盖所有逻辑分支的测试案例
        case_type = random.choice(['colinear', 'two_x', 'two_y', 'full'])
        
        while True:
            if case_type == 'colinear':
                # 三点水平或垂直线
                axis = random.choice(['x', 'y'])
                val = random.randint(*self.x_range if axis == 'x' else self.y_range)
                points = []
                for _ in range(3):
                    if axis == 'x':
                        x = val
                        y = random.randint(*self.y_range)
                    else:
                        y = val
                        x = random.randint(*self.x_range)
                    points.append([x, y])
                    while len(points) < 3 and any(p == points[-1] for p in points[:-1]):
                        points[-1][0 if axis == 'x' else 1] += 1  # 确保不重复
            
            elif case_type == 'two_x':
                # 两个相同x的情况
                x_common = random.randint(*self.x_range)
                x_diff = x_common
                while x_diff == x_common:
                    x_diff = random.randint(*self.x_range)
                    
                points = [
                    [x_common, random.randint(*self.y_range)],
                    [x_common, random.randint(*self.y_range)],
                    [x_diff, random.randint(*self.y_range)]
                ]
            
            elif case_type == 'two_y':
                # 两个相同y的情况
                y_common = random.randint(*self.y_range)
                y_diff = y_common
                while y_diff == y_common:
                    y_diff = random.randint(*self.y_range)
                    
                points = [
                    [random.randint(*self.x_range), y_common],
                    [random.randint(*self.x_range), y_common],
                    [random.randint(*self.x_range), y_diff]
                ]
            
            else:  # full case
                points = []
                for _ in range(3):
                    x = random.randint(*self.x_range)
                    y = random.randint(*self.y_range)
                    while [x, y] in points:
                        x += 1
                        y += 1
                    points.append([x, y])
            
            # 去重检查
            if len({tuple(p) for p in points}) == 3:
                return {'points': points}
    
    @staticmethod
    def prompt_func(question_case):
        points = question_case['points']
        return f"""在平面坐标系上有三个点：{"、".join(f"({x},{y})" for x,y in points)}
请构造仅由坐标轴平行线段组成的简单折线（无自交/自触）连接所有点，求最小线段数。

答案应为1到3之间的整数，放在[answer]标签内。例如：[answer]2[/answer]"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        points = identity['points']
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        sx, sy = len(set(x)), len(set(y))
        
        # 核心验证逻辑
        if sx == 1 or sy == 1:
            return solution == 1
        elif sx == 2 and sy == 2:
            return solution == 2
        elif sx == 2:
            mid_y = sorted(y)[1]
            return solution == (3 if any(p[1] == mid_y for p in points) else 2)
        elif sy == 2:
            mid_x = sorted(x)[1]
            return solution == (3 if any(p[0] == mid_x for p in points) else 2)
        else:
            return solution == 3
