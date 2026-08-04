"""# 

### 谜题描述
Convexity of a set of points on the plane is the size of the largest subset of points that form a convex polygon. Your task is to build a set of n points with the convexity of exactly m. Your set of points should not contain three points that lie on a straight line.

Input

The single line contains two integers n and m (3 ≤ m ≤ 100, m ≤ n ≤ 2m).

Output

If there is no solution, print \"-1\". Otherwise, print n pairs of integers — the coordinates of points of any set with the convexity of m. The coordinates shouldn't exceed 108 in their absolute value.

Examples

Input

4 3


Output

0 0
3 0
0 3
1 1


Input

6 3


Output

-1


Input

6 6


Output

10 0
-10 0
10 1
9 1
9 -1
0 -2


Input

7 4


Output

176166 6377
709276 539564
654734 174109
910147 434207
790497 366519
606663 21061
859328 886001

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n,m=map(int,raw_input().split())

if m==3 and n >= 5:

	print -1

else:

	for i in range(m):

		print i,i*i

	for i in range(n-m):

		print i*i+10001,i



# Made By Mostafa_Khaled
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Bsetofpointsbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.m_min = params.get('m_min', 3)
        self.m_max = params.get('m_max', 100)
        self.max_n_multiplier = params.get('max_n_multiplier', 2)
    
    def case_generator(self):
        m = random.randint(self.m_min, self.m_max)
        max_n = m * self.max_n_multiplier
        min_n = m
        n = random.randint(min_n, max_n)
        
        solution = None
        if m == 3 and n >= 5:
            solution = -1
        else:
            solution = []
            # Generate points according to the reference solution structure
            for i in range(m):
                solution.append((i, i * i))
            for i in range(n - m):
                x = i * i + 10001
                y = i
                solution.append((x, y))
        
        return {
            'n': n,
            'm': m,
            'solution': solution  # Stored for potential debugging, not used in verification
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        m = question_case['m']
        problem_desc = f"""You are a mathematician working on the convexity of planar point sets. Your task is to construct a set of {n} points such that the convexity of the set is exactly {m}, and no three points lie on a straight line. If it's impossible, output "-1". Otherwise, output the coordinates of the points, each with absolute values not exceeding 1e8.

Convexity is defined as the size of the largest subset of points that form a convex polygon.

Input: n = {n}, m = {m}

Please provide your answer within [answer] and [/answer] tags. For example:
[answer]
0 0
1 1
2 4
1 2
[/answer]
"""
        return problem_desc
    
    @staticmethod
    def extract_output(output):
        pattern = r'\[answer\](.*?)\[/answer\]'
        matches = re.findall(pattern, output, re.DOTALL)
        if not matches:
            return None
        content = matches[-1].strip()
        if content.strip() == '-1':
            return -1
        points = []
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                x = int(parts[0])
                y = int(parts[1])
                points.append((x, y))
            except ValueError:
                continue
        return points if points else None
    
    @staticmethod
    def has_collinear_triples(points):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    # Calculate the area of the triangle formed by the three points
                    area = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                    if area == 0:
                        return True
        return False
    
    @staticmethod
    def compute_convex_hull(points):
        if len(points) <= 1:
            return points.copy()
        
        # Sort the points lexographically
        points = sorted(points)
        lower = []
        for p in points:
            while len(lower) >= 2 and (lower[-1][0] - lower[-2][0]) * (p[1] - lower[-2][1]) - (lower[-1][1] - lower[-2][1]) * (p[0] - lower[-2][0]) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and (upper[-1][0] - upper[-2][0]) * (p[1] - upper[-2][1]) - (upper[-1][1] - upper[-2][1]) * (p[0] - upper[-2][0]) <= 0:
                upper.pop()
            upper.append(p)
        # Remove duplicates
        convex_hull = lower[:-1] + upper[:-1]
        return convex_hull
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        m = identity['m']
        
        # Handle the case where solution is -1
        if solution == -1:
            return m == 3 and n >= 5
        
        # Check if solution is a valid list of points
        if not isinstance(solution, list) or len(solution) != n:
            return False
        
        # Validate each point's format and coordinates
        for point in solution:
            if len(point) != 2:
                return False
            x, y = point
            if abs(x) > 10**8 or abs(y) > 10**8:
                return False
        
        # Check for any collinear triplets
        if cls.has_collinear_triples(solution):
            return False
        
        # Compute convex hull and check its size
        convex_hull = cls.compute_convex_hull(solution)
        convexity = len(convex_hull)
        return convexity == m
