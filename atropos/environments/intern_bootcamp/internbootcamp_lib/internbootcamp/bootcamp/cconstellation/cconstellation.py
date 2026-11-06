"""# 

### 谜题描述
Cat Noku has obtained a map of the night sky. On this map, he found a constellation with n stars numbered from 1 to n. For each i, the i-th star is located at coordinates (xi, yi). No two stars are located at the same position.

In the evening Noku is going to take a look at the night sky. He would like to find three distinct stars and form a triangle. The triangle must have positive area. In addition, all other stars must lie strictly outside of this triangle. He is having trouble finding the answer and would like your help. Your job is to find the indices of three stars that would form a triangle that satisfies all the conditions. 

It is guaranteed that there is no line such that all stars lie on that line. It can be proven that if the previous condition is satisfied, there exists a solution to this problem.

Input

The first line of the input contains a single integer n (3 ≤ n ≤ 100 000).

Each of the next n lines contains two integers xi and yi ( - 109 ≤ xi, yi ≤ 109).

It is guaranteed that no two stars lie at the same point, and there does not exist a line such that all stars lie on that line.

Output

Print three distinct integers on a single line — the indices of the three points that form a triangle that satisfies the conditions stated in the problem.

If there are multiple possible answers, you may print any of them.

Examples

Input

3
0 1
1 0
1 1


Output

1 2 3


Input

5
0 0
0 2
2 0
2 2
1 1


Output

1 3 5

Note

In the first sample, we can print the three indices in any order.

In the second sample, we have the following picture. 

<image>

Note that the triangle formed by starts 1, 4 and 3 doesn't satisfy the conditions stated in the problem, as point 5 is not strictly outside of this triangle (it lies on it's border).

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
import random
from fractions import gcd

f = sys.stdin
n = int(f.readline())
x,y = zip(*[map(int, f.readline().split()) for i in range(n)])
st = random.randint(0, n-1)
dist = [(x[i]-x[st])*(x[i]-x[st])+(y[i]-y[st])*(y[i]-y[st]) for i in range(n)]

def getSlope(x1,y1):
	g = gcd(abs(x1),abs(y1))
	x1,y1 = x1/g,y1/g
	if x1 < 0 or (x1 == 0 and y1 < 0):
		x1,y1 = -x1,-y1
	return (x1,y1)

s = {}
for i in xrange(n):
	if i == st: continue
	slope = getSlope(x[i] - x[st], y[i] - y[st])
	if slope not in s or dist[i] < dist[s[slope]]:
		s[slope] = i

lst = sorted(s.values(), key=lambda x: dist[x])
print st+1, lst[0]+1, lst[1]+1
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from math import gcd
from bootcamp import Basebootcamp

class Cconstellationbootcamp(Basebootcamp):
    def __init__(self, min_n=3, max_n=10, coord_range=(-10**9, 10**9)):
        self.min_n = max(3, min_n)
        self.max_n = max(self.min_n, max_n)
        self.coord_range = coord_range

    def case_generator(self):
        # Generate solution triangle ABC ensuring all other points are strictly outside
        while True:
            # Generate valid solution triangle
            A = (random.randint(*self.coord_range), random.randint(*self.coord_range))
            B = (random.randint(*self.coord_range), random.randint(*self.coord_range))
            C = (random.randint(*self.coord_range), random.randint(*self.coord_range))
            if not self._are_collinear(A, B, C):
                break
        
        n = random.randint(self.min_n, self.max_n)
        points = [A, B, C]
        
        # Generate other points strictly outside ABC
        for _ in range(n - 3):
            while True:
                # Generate points in regions guaranteed to be outside
                region = random.choice(['left', 'right', 'top', 'bottom', 'far'])
                dx = random.randint(10, 100)  # Offset for robustness
                dy = random.randint(10, 100)
                
                # Calculate bounding box of ABC
                min_x = min(A[0], B[0], C[0])
                max_x = max(A[0], B[0], C[0])
                min_y = min(A[1], B[1], C[1])
                max_y = max(A[1], B[1], C[1])
                
                # Generate point in specified region
                if region == 'left':
                    x = min_x - dx
                    y = random.randint(min_y - dy, max_y + dy)
                elif region == 'right':
                    x = max_x + dx
                    y = random.randint(min_y - dy, max_y + dy)
                elif region == 'top':
                    y = max_y + dy
                    x = random.randint(min_x - dx, max_x + dx)
                elif region == 'bottom':
                    y = min_y - dy
                    x = random.randint(min_x - dx, max_x + dx)
                else:  # far region
                    x = random.choice([min_x - dx*10, max_x + dx*10])
                    y = random.choice([min_y - dy*10, max_y + dy*10])
                
                p = (x, y)
                if p not in points and not self.is_inside_or_on_edge(p, A, B, C):
                    points.append(p)
                    break
        
        # Shuffle and record solution indices
        random.shuffle(points)
        solution_indices = [
            points.index(A) + 1,
            points.index(B) + 1,
            points.index(C) + 1
        ]
        
        return {
            'n': n,
            'points': points,
            'solution': solution_indices  # Hidden verification hint
        }

    @staticmethod
    def _are_collinear(a, b, c):
        # Robust collinearity check using cross product
        return (b[0] - a[0]) * (c[1] - a[1]) == (b[1] - a[1]) * (c[0] - a[0])

    @staticmethod
    def prompt_func(question_case):
        points = question_case['points']
        n = question_case['n']
        problem = [
            f"Cat Noku's star map contains {n} distinct stars:",
            "Coordinates (1-based index):"
        ]
        problem += [f"{i+1}: ({x}, {y})" for i, (x, y) in enumerate(points)]
        problem += [
            "\nFind three stars forming a non-degenerate triangle with ALL other stars strictly outside.",
            "Output three distinct 1-based indices within [answer]...[/answer]."
        ]
        return '\n'.join(problem)

    @staticmethod
    def extract_output(output):
        # Multi-stage extraction with priority on tagged content
        tagged_answers = re.findall(
            r'\[answer\](.*?)\[/answer\]', 
            output, 
            re.DOTALL | re.IGNORECASE
        )
        
        if tagged_answers:
            # Extract last tagged answer
            last_tagged = tagged_answers[-1]
            candidates = re.findall(r'\b(\d+)\s+(\d+)\s+(\d+)\b', last_tagged)
            if candidates:
                return list(map(int, candidates[-1]))
        
        # Fallback to find last three integers in whole output
        all_nums = re.findall(r'\b\d+\b', output)
        if len(all_nums) >= 3:
            return list(map(int, all_nums[-3:]))
            
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # Basic format validation
        try:
            if len(solution) != 3:
                return False
            a, b, c = map(int, solution)
        except:
            return False
        
        points = identity['points']
        n = identity['n']
        
        # Validate index range and uniqueness
        if not all(1 <= i <= n for i in [a, b, c]) or len({a, b, c}) != 3:
            return False
        
        # Get coordinates
        A = points[a-1]
        B = points[b-1]
        C = points[c-1]
        
        # Check for degenerate triangle
        if cls._are_collinear(A, B, C):
            return False
        
        # Verify all other points are strictly outside
        for i, p in enumerate(points):
            if i+1 in {a, b, c}:
                continue
            if cls.is_inside_or_on_edge(p, A, B, C):
                return False
                
        return True

    @classmethod
    def is_inside_or_on_edge(cls, p, a, b, c):
        # Cross product based containment check
        def cross(o, x, y):
            return (x[0]-o[0])*(y[1]-o[1]) - (x[1]-o[1])*(y[0]-o[0])
        
        area = cross(a, b, c)
        if area == 0:
            return False  # Should never happen due to generation constraints
        
        c1 = cross(a, b, p)
        c2 = cross(b, c, p)
        c3 = cross(c, a, p)
        
        # Check if any sub-area has different sign
        if (c1 * area < 0) or (c2 * area < 0) or (c3 * area < 0):
            return False
        
        # Check edge cases
        def on_segment(i, j, k):
            if cross(i, j, k) != 0:
                return False
            return (min(i[0], j[0]) <= k[0] <= max(i[0], j[0])) and \
                   (min(i[1], j[1]) <= k[1] <= max(i[1], j[1]))
        
        return on_segment(a, b, p) or on_segment(b, c, p) or on_segment(c, a, p)
