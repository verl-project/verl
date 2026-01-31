"""# 

### 谜题描述
You are given n points on the plane. The polygon formed from all the n points is strictly convex, that is, the polygon is convex, and there are no three collinear points (i.e. lying in the same straight line). The points are numbered from 1 to n, in clockwise order.

We define the distance between two points p_1 = (x_1, y_1) and p_2 = (x_2, y_2) as their Manhattan distance: $$$d(p_1, p_2) = |x_1 - x_2| + |y_1 - y_2|.$$$

Furthermore, we define the perimeter of a polygon, as the sum of Manhattan distances between all adjacent pairs of points on it; if the points on the polygon are ordered as p_1, p_2, …, p_k (k ≥ 3), then the perimeter of the polygon is d(p_1, p_2) + d(p_2, p_3) + … + d(p_k, p_1).

For some parameter k, let's consider all the polygons that can be formed from the given set of points, having any k vertices, such that the polygon is not self-intersecting. For each such polygon, let's consider its perimeter. Over all such perimeters, we define f(k) to be the maximal perimeter.

Please note, when checking whether a polygon is self-intersecting, that the edges of a polygon are still drawn as straight lines. For instance, in the following pictures:

<image>

In the middle polygon, the order of points (p_1, p_3, p_2, p_4) is not valid, since it is a self-intersecting polygon. The right polygon (whose edges resemble the Manhattan distance) has the same order and is not self-intersecting, but we consider edges as straight lines. The correct way to draw this polygon is (p_1, p_2, p_3, p_4), which is the left polygon.

Your task is to compute f(3), f(4), …, f(n). In other words, find the maximum possible perimeter for each possible number of points (i.e. 3 to n).

Input

The first line contains a single integer n (3 ≤ n ≤ 3⋅ 10^5) — the number of points. 

Each of the next n lines contains two integers x_i and y_i (-10^8 ≤ x_i, y_i ≤ 10^8) — the coordinates of point p_i.

The set of points is guaranteed to be convex, all points are distinct, the points are ordered in clockwise order, and there will be no three collinear points.

Output

For each i (3≤ i≤ n), output f(i).

Examples

Input

4
2 4
4 3
3 0
1 3


Output

12 14 

Input

3
0 0
0 2
2 0


Output

8 

Note

In the first example, for f(3), we consider four possible polygons: 

  * (p_1, p_2, p_3), with perimeter 12. 
  * (p_1, p_2, p_4), with perimeter 8. 
  * (p_1, p_3, p_4), with perimeter 12. 
  * (p_2, p_3, p_4), with perimeter 12. 



For f(4), there is only one option, taking all the given points. Its perimeter 14.

In the second example, there is only one possible polygon. Its perimeter is 8.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = int(raw_input())
X = []
Y = []
for _ in range(n):
    x,y = map(int,raw_input().strip(' ').strip('\n').split(' '))
    X.append(x)
    Y.append(y)
maxx,minx = max(X),min(X)
maxy,miny = max(Y),min(Y)

ans = -10e18
for i in range(n):
    dx = max(maxx-X[i],X[i]-minx)
    dy = max(maxy-Y[i],Y[i]-miny)
    ans = max(ans,dx+dy)
ans = str(2*ans)+' '
rec = 2*(maxx+maxy-minx-miny)
for i in range(3,n):
    ans += str(rec)+' '
ans = ans.strip('')
print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Coptimalpolygonperimeterbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.min_points = params.get('min_points', 3)
        self.max_points = params.get('max_points', 10)
        self.coord_range = params.get('coord_range', (-100, 100))
        # Ensure min_points is at least 3
        self.min_points = max(3, self.min_points)

    def case_generator(self):
        # Generate n in specified range
        n = random.randint(self.min_points, self.max_points)
        
        # Generate bounding box with sufficient size
        minx = random.randint(self.coord_range[0], self.coord_range[1] - 10)
        maxx = minx + 10  # Ensure width is at least 10
        miny = random.randint(self.coord_range[0], self.coord_range[1] - 10)
        maxy = miny + 10  # Ensure height is at least 10
        
        # Generate points in strict convex polygon (simulating convex hull)
        points = []
        edges = ['bottom', 'right', 'top', 'left']
        edge_idx = 0
        
        # Generate points in clockwise order without colinear points
        for _ in range(n):
            edge = edges[edge_idx % 4]
            edge_idx += 1
        
            if edge == 'bottom':
                x = random.randint(minx, maxx - 1)  # Avoid rightmost point
                y = miny
            elif edge == 'right':
                x = maxx
                y = random.randint(miny + 1, maxy - 1)  # Avoid top/bottom extremes
            elif edge == 'top':
                x = random.randint(minx + 1, maxx)  # Avoid leftmost point
                y = maxy
            else:  # left
                x = minx
                y = random.randint(miny + 1, maxy - 1)
        
            points.append((x, y))
        
        # Calculate expected output using problem logic
        X = [p[0] for p in points]
        Y = [p[1] for p in points]
        maxx_val = max(X)
        minx_val = min(X)
        maxy_val = max(Y)
        miny_val = min(Y)
        
        ans_3 = 0
        for x, y in zip(X, Y):
            dx = max(maxx_val - x, x - minx_val)
            dy = max(maxy_val - y, y - miny_val)
            ans_3 = max(ans_3, dx + dy)
        ans_3 *= 2
        rec = 2 * ((maxx_val - minx_val) + (maxy_val - miny_val))
        
        # Generate expected output sequence
        expected_output = [ans_3] + [rec] * (n - 3)
        
        return {
            'n': n,
            'points': points,
            'expected_output': expected_output
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        input_lines = [str(question_case['n'])] + [
            f"{x} {y}" for x, y in question_case['points']
        ]
        input_example = '\n'.join(input_lines)
        
        return f"""\
Given a strictly convex polygon with {question_case['n']} vertices arranged clockwise. Compute maximum perimeter for k=3..n using Manhattan distances.

Input:
{input_example}

Output format: Space-separated integers f(3) to f(n).

Place your answer within [answer][/answer] tags. Example: [answer]12 14[/answer]"""

    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        return answer_blocks[-1].strip()

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            solution_values = list(map(int, solution.split()))
            return solution_values == identity['expected_output']
        except:
            return False
