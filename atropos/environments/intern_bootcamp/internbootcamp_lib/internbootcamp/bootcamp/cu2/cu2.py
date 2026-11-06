"""# 

### 谜题描述
Recently Vasya learned that, given two points with different x coordinates, you can draw through them exactly one parabola with equation of type y = x^2 + bx + c, where b and c are reals. Let's call such a parabola an U-shaped one.

Vasya drew several distinct points with integer coordinates on a plane and then drew an U-shaped parabola through each pair of the points that have different x coordinates. The picture became somewhat messy, but Vasya still wants to count how many of the parabolas drawn don't have any drawn point inside their internal area. Help Vasya.

The internal area of an U-shaped parabola is the part of the plane that lies strictly above the parabola when the y axis is directed upwards.

Input

The first line contains a single integer n (1 ≤ n ≤ 100 000) — the number of points.

The next n lines describe the points, the i-th of them contains two integers x_i and y_i — the coordinates of the i-th point. It is guaranteed that all points are distinct and that the coordinates do not exceed 10^6 by absolute value.

Output

In the only line print a single integer — the number of U-shaped parabolas that pass through at least two of the given points and do not contain any of the given points inside their internal area (excluding the parabola itself).

Examples

Input


3
-1 0
0 2
1 0


Output


2


Input


5
1 0
1 -1
0 -1
-1 0
-1 -1


Output


1

Note

On the pictures below all U-shaped parabolas that pass through at least two given points are drawn for each of the examples. The U-shaped parabolas that do not have any given point inside their internal area are drawn in red. 

<image> The first example.  <image> The second example. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys

def readl():
    return sys.stdin.readline()

def read(tp=int):
    return tp(readl())

def readn(tp=int):
    ln = readl().split()
    return [tp(x) for x in ln]

def readf(*tp):
    ln = readl().split()
    return [x(y) for x,y in zip(tp,ln)]

################################################################################

def ori(p, q, r):
    x1 = q[0] - p[0]
    y1 = q[1] - p[1]
    x2 = r[0] - p[0]
    y2 = r[1] - p[1]
    return x1 * y2 - x2 * y1

def main():
    n = read()
    lns = sys.stdin.readlines()
    pt = map(lambda ln: map(int,ln.split()), lns)
    pt = map(lambda p: (p[0], p[1]-p[0]*p[0]), pt)
    pt.sort(reverse=True)

    qt = []
    for p in pt:
        while len(qt) >= 2 and ori(qt[-2], qt[-1], p) <= 0:
            qt.pop()
        qt.append(p)

    if len(qt) >= 2 and qt[-1][0] == qt[-2][0]:
        qt.pop()

    if len(qt) < 2:
        print 0
        sys.exit(0)

    p = qt[0]
    q = qt[1]
    s = 1
    d = 1
    for r in qt[2:]:
        if ori(p, q, r) == 0:
            d += 1
            q = r
        else:
            d = 1
            p = q
            q = r
        s += d

    print s

main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import json
import random
import re
from typing import Dict, Any
from bootcamp import Basebootcamp

class Cu2bootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__()
        self.params = {
            'n': params.get('n', 3),
            'x_range': params.get('x_range', (-1000, 1000)),
            'y_range': params.get('y_range', (-1000, 1000))
        }
    
    def case_generator(self) -> Dict[str, Any]:
        n = self.params['n']
        x_min, x_max = self.params['x_range']
        y_min, y_max = self.params['y_range']
        
        points = []
        while len(points) < n:
            x = random.randint(x_min, x_max)
            y = random.randint(y_min, y_max)
            if (x, y) not in points:
                points.append((x, y))
        
        def ori(p, q, r):
            x1 = q[0] - p[0]
            y1 = q[1] - p[1]
            x2 = r[0] - p[0]
            y2 = r[1] - p[1]
            return x1 * y2 - x2 * y1
        
        converted = sorted([(x, y - x**2) for x, y in points], key=lambda p: (-p[0], p[1]))
        qt = []
        for p in converted:
            while len(qt) >= 2 and ori(qt[-2], qt[-1], p) <= 0:
                qt.pop()
            qt.append(p)
        
        if len(qt) >= 2 and qt[-1][0] == qt[-2][0]:
            qt.pop()
        
        if len(qt) < 2:
            correct_answer = 0
        else:
            p = qt[0]
            q = qt[1]
            s = 1
            d = 1
            for r in qt[2:]:
                if ori(p, q, r) == 0:
                    d += 1
                    q = r
                else:
                    d = 1
                    p = q
                    q = r
                s += d
            correct_answer = s
        
        return {
            'points': points,
            'correct_answer': correct_answer
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        points = question_case['points']
        point_list = '\n'.join([f'({x}, {y})' for x, y in points])
        
        prompt = f"Given the following {len(points)} distinct points with integer coordinates:\n{point_list}\n\nYou need to determine the number of U-shaped parabolas that can be drawn through any two of these points such that no other point lies inside the internal area of the parabola (the area strictly above the parabola).\n\nA U-shaped parabola is defined by the equation y = x² + bx + c for some real numbers b and c, where each parabola is uniquely determined by any two points with different x-coordinates.\n\nOutput the count of such valid parabolas. Place your final answer within [answer] tags as follows:\n\n[answer]your_answer[/answer]"
        
        return prompt
    
    @staticmethod
    def extract_output(output: str) -> str:
        matches = list(re.finditer(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL))
        if matches:
            return matches[-1].group(1).strip()
        return None
    
    @classmethod
    def _verify_correction(cls, solution: str, identity: Dict[str, Any]) -> bool:
        try:
            solution_int = int(solution.strip())
            correct_answer = identity['correct_answer']
            return solution_int == correct_answer
        except:
            return False
