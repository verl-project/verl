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
from math import sqrt

class vector:
	def __init__(self, _x = 0, _y = 0):
		self.x = _x
		self.y = _y
	def len(self):
		return sqrt(self.x ** 2 + self.y ** 2)
	def len_sq(self):
		return self.x ** 2 + self.y ** 2
	def __mul__(self, other):
		if (type(self) == type(other)):
			return self.x * other.x + self.y * other.y
		return vector(self.x * other, self.y * other)
	def __mod__(self, other):
		return self.x * other.y - self.y * other.x
	def normed(self):
		length = self.len()
		return vector(self.x / length, self.y / length)
	def normate(self):
		self = self.normed()
	def __str__(self):
		return \"(\" + str(self.x) + \", \" + str(self.y) + \")\"
	def __add__(self, other):
		return vector(self.x + other.x, self.y + other.y);
	def __sub__(self, other):
		return vector(self.x - other.x, self.y - other.y);
	def __eq__(self, other):
		return self.x == other.x and self.y == other.y
	def rot(self):
		return vector(self.y, -self.x)

class line:
	def __init__(self, a = 0, b = 0, c = 0):
		self.a = a
		self.b = b
		self.c = c
	def intersect(self, other):
		d = self.a * other.b - self.b * other.a
		dx = self.c * other.b - self.b * other.c
		dy = self.a * other.c - self.c * other.a
		return vector(dx / d, dy / d)
	def fake(self, other):
		d = self.a * other.b - self.b * other.a
		return d
	def __str__(self):
		return str(self.a) + \"*x + \" + str(self.b) + \"*y = \" + str(self.c) 

def line_pt(A, B):
		d = (A - B).rot()
		return line(d.x, d.y, d * A)

class circle:
	def __init__(self, O = vector(0, 0), r = 0):
		self.O = O
		self.r = r
	def intersect(self, other):
		O1 = self.O
		O2 = other.O
		r1 = self.r
		r2 = other.r
		if (O1 == O2):
			return []
		if ((O1 - O2).len_sq() > r1 ** 2 + r2 ** 2 + 2 * r1 * r2):
			return []
		rad_line = line(2 * (O2.x - O1.x), 2 * (O2.y - O1.y), r1 ** 2 - O1.len_sq() - r2 ** 2 + O2.len_sq())
		central = line_pt(O1, O2)
		M = rad_line.intersect(central)
		if ((O1 - O2).len_sq() == r1 ** 2 + r2 ** 2 + 2 * r1 * r2):
			return [M]
		d = (O2 - O1).normed().rot()
		if (r1 ** 2 - (O1 - M).len_sq() < 0):
			return []
		d = d * (sqrt(r1 ** 2 - (O1 - M).len_sq()))
		return [M + d, M - d]
	def fake(self, other):
		O1 = self.O
		O2 = other.O
		r1 = self.r
		r2 = other.r
		if (O1 == O2):
			return 1
		if ((O1 - O2).len_sq() > r1 ** 2 + r2 ** 2 + 2 * r1 * r2):
			return 1
		rad_line = line(2 * (O2.x - O1.x), 2 * (O2.y - O1.y), r1 ** 2 - O1.len_sq() - r2 ** 2 + O2.len_sq())
		central = line_pt(O1, O2)
		return rad_line.fake(central)
n = input()
arr = []
m = 1
for i in range(n):
	x, y, r = map(int, raw_input().split())
	arr.append(circle(vector(x, y), r))
for i in range(n):
	for j in range(i + 1, n):
		m *= arr[i].fake(arr[j])
for i in range(n):
	arr[i].O = arr[i].O * m
	arr[i].r = arr[i].r * m
s = set()
V = 0
for i in range(n):
	for j in range(i + 1, n):
		tmp = arr[i].intersect(arr[j])
		for e in tmp:
			s.add((round(e.x, 6), round(e.y, 6)))
V += len(s)
E = 0

par = [i for i in range(n)]

def get_par(v):
	if (par[v] != v):
		par[v] = get_par(par[v])
	return par[v]
def unite(v, u):
	par[get_par(v)] = get_par(u)
for i in range(n):
	s = set()
	for j in range(n):	
		tmp = arr[i].intersect(arr[j])
		if (len(tmp)):
			unite(i, j)
		for e in tmp:
			s.add((round(e.x, 	), round(e.y, 	)))
	E += len(s)
print(E - V + 1 + len({get_par(i) for i in range(n)}))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from math import sqrt, isclose
import random
import re
from bootcamp import Basebootcamp

class Eacolourfulprospectbootcamp(Basebootcamp):
    class Vector:
        def __init__(self, x=0, y=0):
            self.x = x
            self.y = y
        def __add__(self, other):
            return self.__class__(self.x + other.x, self.y + other.y)
        def __sub__(self, other):
            return self.__class__(self.x - other.x, self.y - other.y)
        def __mul__(self, scalar):
            return self.__class__(self.x * scalar, self.y * scalar)
        def __mod__(self, other):
            return self.x * other.y - self.y * other.x
        def rot(self):
            return self.__class__(self.y, -self.x)
        def len_sq(self):
            return self.x**2 + self.y**2
        def normed(self):
            length = sqrt(self.len_sq())
            return self.__class__() if length == 0 else self * (1/length)
        def __eq__(self, other):
            return isclose(self.x, other.x, rel_tol=1e-9) and isclose(self.y, other.y, rel_tol=1e-9)
        def __hash__(self):
            return hash((round(self.x,9), round(self.y,9)))

    class Line:
        def __init__(self, a, b, c):
            self.a = a
            self.b = b
            self.c = c
        def intersect(self, other):
            det = self.a * other.b - self.b * other.a
            if isclose(det, 0, rel_tol=1e-9):
                return None
            x = (self.c * other.b - self.b * other.c) / det
            y = (self.a * other.c - self.c * other.a) / det
            return Eacolourfulprospectbootcamp.Vector(x, y)
        def fake(self, other):
            return self.a * other.b - self.b * other.a

    @classmethod
    def line_pt(cls, A, B):
        dir_vec = (A - B).rot()
        return cls.Line(dir_vec.x, dir_vec.y, dir_vec.x*A.x + dir_vec.y*A.y)

    class Circle:
        def __init__(self, O, r):
            self.O = O
            self.r = r
        def intersect(self, other):
            O1, O2 = self.O, other.O
            r1, r2 = self.r, other.r
            
            # Handle concentric circles
            if O1 == O2:
                return []
            
            d_sq = (O1 - O2).len_sq()
            sum_r = r1 + r2
            dif_r = abs(r1 - r2)
            
            if d_sq > sum_r**2 + 1e-9 or d_sq < dif_r**2 - 1e-9:
                return []
            
            line = Eacolourfulprospectbootcamp.Line(
                2*(O2.x - O1.x),
                2*(O2.y - O1.y),
                r1**2 - O1.len_sq() - r2**2 + O2.len_sq()
            )
            central = Eacolourfulprospectbootcamp.line_pt(O1, O2)
            M = line.intersect(central)
            
            if not M:
                return []
            
            direction = (O2 - O1).normed().rot()
            len_sq = r1**2 - (M - O1).len_sq()
            
            if len_sq < -1e-9:
                return []
            if len_sq < 1e-9:
                return [M]
            
            d = sqrt(len_sq)
            return [M + direction*d, M - direction*d]
        
        def fake(self, other):
            if self.O == other.O:
                return 1
            return Eacolourfulprospectbootcamp.line_pt(self.O, other.O).fake(
                Eacolourfulprospectbootcamp.Line(
                    2*(other.O.x - self.O.x),
                    2*(other.O.y - self.O.y),
                    self.r**2 - other.r**2 + (other.O - self.O).len_sq()
                )
            )

    def __init__(self, **params):
        self.min_n = params.get('min_n', 1)
        self.max_n = params.get('max_n', 3)
        self.min_x = params.get('min_x', -10)
        self.max_x = params.get('max_x', 10)
        self.min_y = params.get('min_y', -10)
        self.max_y = params.get('max_y', 10)
        self.min_r = params.get('min_r', 1)
        self.max_r = params.get('max_r', 10)

    def case_generator(self):
        MAX_ATTEMPTS = 100
        for _ in range(MAX_ATTEMPTS):
            try:
                n = random.randint(self.min_n, self.max_n)
                circles = []
                for _ in range(n):
                    x = random.randint(self.min_x, self.max_x)
                    y = random.randint(self.min_y, self.max_y)
                    r = random.randint(self.min_r, self.max_r)
                    circles.append({'x': x, 'y': y, 'r': r})
                
                # Validate case constraints
                circles = [self.Circle(self.Vector(c['x'], c['y']), c['r']) for c in circles]
                m = 1
                for i in range(n):
                    for j in range(i+1, n):
                        if circles[i].O == circles[j].O and circles[i].r == circles[j].r:
                            raise ValueError("Duplicate circles")
                        m *= circles[i].fake(circles[j])
                        if m == 0:
                            raise ValueError("Invalid configuration")
                
                return {
                    'n': n,
                    'circles': [
                        {'x': c.O.x, 'y': c.O.y, 'r': c.r}
                        for c in circles
                    ]
                }
            except (ValueError, ZeroDivisionError):
                continue
        raise RuntimeError(f"Failed to generate valid case after {MAX_ATTEMPTS} attempts")

    @staticmethod
    def prompt_func(question_case):
        circles = question_case['circles']
        circles_str = "\n".join(f"{c['x']} {c['y']} {c['r']}" for c in circles)
        return (
            "Calculate regions formed by intersecting circles.\n"
            f"Input:\n{question_case['n']}\n{circles_str}\n"
            "Output format: [answer]INTEGER[/answer]"
        )

    @staticmethod
    def extract_output(output):
        match = re.search(r'\[answer\]\s*(\d+)\s*\[/answer\]', output, re.IGNORECASE)
        return int(match.group(1)) if match else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return solution == cls._compute_regions(identity)
        except:
            return False

    @classmethod
    def _compute_regions(cls, identity):
        circles = [
            cls.Circle(
                cls.Vector(c['x'], c['y']),
                c['r']
            ) for c in identity['circles']
        ]
        n = identity['n']
        
        # Calculate intersection graph
        parent = list(range(n))
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        def union(u, v):
            parent[find(u)] = find(v)
        
        vertices = set()
        edges = 0
        
        for i in range(n):
            for j in range(i+1, n):
                points = circles[i].intersect(circles[j])
                if points:
                    union(i, j)
                    edges += len(points)
                    vertices.update(points)
        
        components = len({find(i) for i in range(n)})
        return edges - len(vertices) + components + 1
