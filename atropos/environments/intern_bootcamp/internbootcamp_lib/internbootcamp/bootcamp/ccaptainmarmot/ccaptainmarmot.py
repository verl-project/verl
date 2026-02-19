"""# 

### 谜题描述
Captain Marmot wants to prepare a huge and important battle against his enemy, Captain Snake. For this battle he has n regiments, each consisting of 4 moles.

Initially, each mole i (1 ≤ i ≤ 4n) is placed at some position (xi, yi) in the Cartesian plane. Captain Marmot wants to move some moles to make the regiments compact, if it's possible.

Each mole i has a home placed at the position (ai, bi). Moving this mole one time means rotating his position point (xi, yi) 90 degrees counter-clockwise around it's home point (ai, bi).

A regiment is compact only if the position points of the 4 moles form a square with non-zero area.

Help Captain Marmot to find out for each regiment the minimal number of moves required to make that regiment compact, if it's possible.

Input

The first line contains one integer n (1 ≤ n ≤ 100), the number of regiments.

The next 4n lines contain 4 integers xi, yi, ai, bi ( - 104 ≤ xi, yi, ai, bi ≤ 104).

Output

Print n lines to the standard output. If the regiment i can be made compact, the i-th line should contain one integer, the minimal number of required moves. Otherwise, on the i-th line print \"-1\" (without quotes).

Examples

Input

4
1 1 0 0
-1 1 0 0
-1 1 0 0
1 -1 0 0
1 1 0 0
-2 1 0 0
-1 1 0 0
1 -1 0 0
1 1 0 0
-1 1 0 0
-1 1 0 0
-1 1 0 0
2 2 0 1
-1 0 0 -2
3 0 0 -2
-1 1 -2 0


Output

1
-1
3
3

Note

In the first regiment we can move once the second or the third mole.

We can't make the second regiment compact.

In the third regiment, from the last 3 moles we can move once one and twice another one.

In the fourth regiment, we can move twice the first mole and once the third mole.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import math
def dis(a,b):
    x1,y1=a
    x2,y2=b
    return (x2-x1)**2+(y2-y1)**2

def rotate(origin, point, angle):
    \"\"\"
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    \"\"\"
    angle=math.radians(angle)
    ox, oy = origin
    px, py = point

    qx = round(ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy),2)
    qy = round(oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy),2)
    return [qx, qy]
#print rotate([0,0],[1,1],90)
n=input()
arr=[]
for i in range(4*n):
    arr.append(map(int,raw_input().split()))
for i in range(0,4*n,4):
    #print i
    a,b,c,d=arr[i]
    p1=[]
    angle=0
    for j in range(4):
        p1.append(rotate([c,d],[a,b],angle))
        angle+=90
    a, b, c, d = arr[i+1]
    p2 = []
    angle = 0
    for j in range(4):
        p2.append(rotate([c, d], [a, b], angle))
        angle += 90
    a, b, c, d = arr[i+2]
    p3 = []
    angle = 0
    for j in range(4):
        p3.append(rotate([c, d], [a, b], angle))
        angle += 90
    a, b, c, d = arr[i+3]
    p4 = []
    angle = 0
    for j in range(4):
        p4.append(rotate([c, d], [a, b], angle))
        angle += 90
    s=100
    #if i==12:
        #print p1,p2,p3,p4
    for t in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    a=dis(p1[t],p2[j])
                    b=dis(p2[j],p3[k])
                    c=dis(p3[k],p4[l])
                    d=dis(p4[l],p1[t])
                    e=dis(p1[t],p3[k])
                    f=dis(p2[j],p4[l])
                    #if i==12:
                        #print p1[t],p2[j],p3[k],p4[l],a,b,c,d
                    a,b,c,d,e,f=sorted([a,b,c,d,e,f])
                    if a==b==c==d and e==f and a>0:
                        #print a,b,c,d,t,j,k,l
                        s=min(t+j+k+l,s)
    if s==100:
        print -1
    else:
        print s
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import re
import random
from itertools import combinations
from bootcamp import Basebootcamp

class Ccaptainmarmotbootcamp(Basebootcamp):
    def __init__(self, n_regiments=1, same_origin=True, max_rotation=3, solvable_ratio=0.5, **kwargs):
        super().__init__()
        self.n_regiments = n_regiments
        self.same_origin = same_origin
        self.max_rotation = max_rotation
        self.solvable_ratio = solvable_ratio  # Probability to generate solvable cases

    def case_generator(self):
        case_data = {'regiments': [], 'n': self.n_regiments}
        
        for _ in range(self.n_regiments):
            # Generate regiment with configurable solvability
            regiment = self._generate_regiment(random.random() < self.solvable_ratio)
            case_data['regiments'].append(regiment)
        
        return case_data

    def _generate_regiment(self, solvable=True):
        """Generate a regiment that can be solvable or unsolvable"""
        regiment = []
        origin_map = []
        
        # 1. Generate base points configuration
        if solvable:
            # Generate valid square points
            d = random.randint(1, 5)
            square_points = [
                (d, 0), (0, d), (-d, 0), (0, -d)
            ]
            random.shuffle(square_points)
        else:
            # Generate invalid points (non-square)
            square_points = [
                (random.randint(-5, 5), random.randint(-5, 5))
                for _ in range(4)
            ]
            # Ensure at least 3 points are collinear
            square_points[-1] = self._create_collinear_point(square_points[:3])

        # 2. Generate origins for each mole
        if self.same_origin:
            common_origin = (random.randint(-10, 10), random.randint(-10, 10))
            origin_map = [common_origin]*4
        else:
            origin_map = [(random.randint(-10, 10), random.randint(-10, 10)) 
                         for _ in range(4)]

        # 3. Apply rotations and build moles
        for idx in range(4):
            x_base, y_base = square_points[idx]
            a, b = origin_map[idx]
            
            # Apply random rotations
            rotations = random.randint(0, self.max_rotation)
            cx, cy = x_base, y_base
            for _ in range(rotations):
                nx = a - (cy - b)
                ny = b + (cx - a)
                cx, cy = nx, ny
            
            regiment.append((cx, cy, a, b))
        
        return regiment

    def _create_collinear_point(self, points):
        """Create a collinear point to make square impossible"""
        x1, y1 = points[0]
        x2, y2 = points[1]
        x3, y3 = points[2]
        
        # Find vector for points 0->1 and 0->2
        dx1 = x2 - x1
        dy1 = y2 - y1
        dx2 = x3 - x1
        dy2 = y3 - y1
        
        # Ensure collinearity
        if dx1 * dy2 == dx2 * dy1:
            # Points are collinear, create another collinear point
            t = random.uniform(1.5, 3)
            return (x1 + t*dx1, y1 + t*dy1)
        else:
            # Force fourth point to be collinear
            t = random.uniform(0.5, 2)
            return (x2 + t*(x3 - x2), y2 + t*(y3 - y2))

    @staticmethod
    def prompt_func(question_case):
        input_lines = [str(question_case['n'])]
        for regiment in question_case['regiments']:
            for mole in regiment:
                input_lines.append(f"{mole[0]} {mole[1]} {mole[2]} {mole[3]}")
        
        problem_desc = (
            "Captain Marmot needs to rotate moles to form squares. Each mole can rotate 0-3 times around its home.\n"
            f"Input has {question_case['n']} regiments. Each regiment has 4 moles with format:\n"
            "x y a b (current position and home coordinates)\n"
            "Output the minimal total rotations per regiment, or -1 if impossible.\n"
            "Example format for 2 regiments:\n"
            "[answer]3[/answer]\n[answer]-1[/answer]\n"
            "Current input:\n" + "\n".join(input_lines)
        )
        return problem_desc

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(-?\d+)\s*\[/answer\]', output)
        return matches[-1] if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            user_answers = list(map(int, solution.strip().split()))
            if len(user_answers) != identity['n']:
                return False
        except:
            return False
        
        for i, regiment in enumerate(identity['regiments']):
            try:
                # Generate all possible rotation states per mole
                rotation_states = []
                for mole in regiment:
                    x, y, a, b = mole
                    states = []
                    current_x, current_y = x, y
                    states.append((current_x, current_y))
                    for _ in range(3):
                        current_x, current_y = a - (current_y - b), b + (current_x - a)
                        states.append((current_x, current_y))
                    rotation_states.append(states)
                
                # Find minimal rotations
                min_rotations = None
                for r0 in range(4):
                    for r1 in range(4):
                        for r2 in range(4):
                            for r3 in range(4):
                                points = [
                                    rotation_states[0][r0],
                                    rotation_states[1][r1],
                                    rotation_states[2][r2],
                                    rotation_states[3][r3]
                                ]
                                if cls._is_valid_square(points):
                                    total = r0 + r1 + r2 + r3
                                    if (min_rotations is None) or (total < min_rotations):
                                        min_rotations = total
                
                correct = min_rotations if min_rotations is not None else -1
                if user_answers[i] != correct:
                    return False
            except:
                return False
        
        return True

    @staticmethod
    def _is_valid_square(points):
        # Calculate all pairwise squared distances
        dists = []
        for (x1, y1), (x2, y2) in combinations(points, 2):
            dist_sq = (x2-x1)**2 + (y2-y1)**2
            dists.append(dist_sq)
        
        # Verify square properties: 2 distinct distances (sides and diagonals)
        dists.sort()
        return (
            len(dists) == 6 and
            dists[0] == dists[1] == dists[2] == dists[3] and  # 4 equal sides
            dists[4] == dists[5] and                          # 2 equal diagonals
            dists[4] == 2 * dists[0] and                      # Diagonal = side*sqrt(2)
            dists[0] > 0                                      # Non-degenerate
        )

    @classmethod
    def _verify_single_regiment(cls, answer, regiment):
        """Verify single regiment answer"""
        rotation_states = []
        for mole in regiment:
            x, y, a, b = mole
            states = []
            current_x, current_y = x, y
            states.append((current_x, current_y))
            for _ in range(3):
                current_x, current_y = a - (current_y - b), b + (current_x - a)
                states.append((current_x, current_y))
            rotation_states.append(states)
        
        min_rotations = None
        for r0 in range(4):
            for r1 in range(4):
                for r2 in range(4):
                    for r3 in range(4):
                        points = [
                            rotation_states[0][r0],
                            rotation_states[1][r1],
                            rotation_states[2][r2],
                            rotation_states[3][r3]
                        ]
                        if cls._is_valid_square(points):
                            total = r0 + r1 + r2 + r3
                            if min_rotations is None or total < min_rotations:
                                min_rotations = total
        
        correct = min_rotations if min_rotations is not None else -1
        return answer == correct
