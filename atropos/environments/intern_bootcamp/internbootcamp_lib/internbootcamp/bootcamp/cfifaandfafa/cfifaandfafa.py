"""# 

### 谜题描述
Fifa and Fafa are sharing a flat. Fifa loves video games and wants to download a new soccer game. Unfortunately, Fafa heavily uses the internet which consumes the quota. Fifa can access the internet through his Wi-Fi access point. This access point can be accessed within a range of r meters (this range can be chosen by Fifa) from its position. Fifa must put the access point inside the flat which has a circular shape of radius R. Fifa wants to minimize the area that is not covered by the access point inside the flat without letting Fafa or anyone outside the flat to get access to the internet.

The world is represented as an infinite 2D plane. The flat is centered at (x1, y1) and has radius R and Fafa's laptop is located at (x2, y2), not necessarily inside the flat. Find the position and the radius chosen by Fifa for his access point which minimizes the uncovered area.

Input

The single line of the input contains 5 space-separated integers R, x1, y1, x2, y2 (1 ≤ R ≤ 105, |x1|, |y1|, |x2|, |y2| ≤ 105).

Output

Print three space-separated numbers xap, yap, r where (xap, yap) is the position which Fifa chose for the access point and r is the radius of its range. 

Your answer will be considered correct if the radius does not differ from optimal more than 10 - 6 absolutely or relatively, and also the radius you printed can be changed by no more than 10 - 6 (absolutely or relatively) in such a way that all points outside the flat and Fafa's laptop position are outside circle of the access point range.

Examples

Input

5 3 3 1 1


Output

3.7677669529663684 3.7677669529663684 3.914213562373095


Input

10 5 5 5 15


Output

5.0 5.0 10.0

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
R,x1,y1,x2,y2=map(float,raw_input().split())
if (x2-x1)**2+(y2-y1)**2>R**2:
	print x1,y1,R
	exit(0)
xx=yy=0.0
if [x1,y1]==[x2,y2]:
	xx=x1+R
	yy=y1
else:
	tmp=[x1-x2,y1-y2]
	dist=((x1-x2)**2+(y1-y2)**2)**0.5
	dist=R/dist
	tmp[0]*=dist
	tmp[1]*=dist
	xx=tmp[0]+x1
	yy=tmp[1]+y1
	#print xx,yy
xx=(xx+x2)/2
yy=(yy+y2)/2
print xx,yy,((xx-x2)**2+(yy-y2)**2)**0.5
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
import math
from bootcamp import Basebootcamp

class Cfifaandfafabootcamp(Basebootcamp):
    def __init__(self, R_max=10**5, coord_max=10**5, **kwargs):
        super().__init__(**kwargs)
        self.R_max = R_max
        self.coord_max = coord_max

    def case_generator(self):
        R = random.randint(1, self.R_max)
        x1 = random.randint(-self.coord_max, self.coord_max)
        y1 = random.randint(-self.coord_max, self.coord_max)
        x2 = random.randint(-self.coord_max, self.coord_max)
        y2 = random.randint(-self.coord_max, self.coord_max)
        return {
            'R': R,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2
        }

    @staticmethod
    def prompt_func(question_case):
        R = question_case['R']
        x1 = question_case['x1']
        y1 = question_case['y1']
        x2 = question_case['x2']
        y2 = question_case['y2']
        prompt = f"""You are Fifa trying to set up a Wi-Fi access point in your circular flat. The flat has a radius of {R} meters and is centered at coordinates ({x1}, {y1}). Fafa's laptop is located at ({x2}, {y2}), which you must exclude from the access point's coverage. The access point must be placed entirely within the flat. Your goal is to determine the optimal position and radius of the access point to minimize the uncovered area within the flat.

Requirements:
1. The access point's coverage circle must be entirely inside the flat.
2. Fafa's laptop must not be within the access point's coverage area.
3. The radius of the access point should be as large as possible to minimize the uncovered area.

Input parameters:
- Flat radius (R): {R}
- Flat center coordinates: ({x1}, {y1})
- Fafa's laptop coordinates: ({x2}, {y2})

Please provide the access point's coordinates (xap, yap) and radius (r), formatted as three space-separated floating-point numbers with at least six decimal places. Enclose your final answer within [answer] and [/answer] tags. Example: [answer]3.767767 3.767767 3.914214[/answer]"""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        parts = re.split(r'\s+', last_match)
        if len(parts) != 3:
            return None
        try:
            x = float(parts[0])
            y = float(parts[1])
            r = float(parts[2])
            return f"{x} {y} {r}"
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        parts = solution.strip().split()
        if len(parts) != 3:
            return False
        try:
            xap_sol = float(parts[0])
            yap_sol = float(parts[1])
            r_sol = float(parts[2])
        except ValueError:
            return False

        R = identity['R']
        x1 = identity['x1']
        y1 = identity['y1']
        x2 = identity['x2']
        y2 = identity['y2']

        xap_true, yap_true, r_true = cls.compute_true_solution(R, x1, y1, x2, y2)

        epsilon = 1e-6
        if abs(xap_sol - xap_true) > epsilon:
            return False
        if abs(yap_sol - yap_true) > epsilon:
            return False

        delta_r = abs(r_sol - r_true)
        if delta_r > epsilon and delta_r / max(abs(r_true), 1e-20) > epsilon:
            return False

        return True

    @staticmethod
    def compute_true_solution(R, x1, y1, x2, y2):
        R = float(R)
        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)

        dx = x2 - x1
        dy = y2 - y1
        d_squared = dx ** 2 + dy ** 2

        if d_squared > R ** 2:
            return (x1, y1, R)
        else:
            if x1 == x2 and y1 == y2:
                edge_x = x1 + R
                edge_y = y1
            else:
                distance = math.sqrt(d_squared)
                direction_x = (x1 - x2) / distance
                direction_y = (y1 - y2) / distance
                edge_x = x1 + direction_x * R
                edge_y = y1 + direction_y * R

            ap_x = (edge_x + x2) / 2
            ap_y = (edge_y + y2) / 2
            r = math.hypot(ap_x - x2, ap_y - y2)
            return (ap_x, ap_y, r)
