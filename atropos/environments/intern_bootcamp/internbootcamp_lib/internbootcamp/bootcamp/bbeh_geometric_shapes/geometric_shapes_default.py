import random
import re
import random
import math
import json
import matplotlib.pyplot as plt
from bootcamp import Basebootcamp

def generate_rectangle():
    x = random.uniform(0, 50)
    y = random.uniform(0, 50)
    width = random.uniform(10, 30)
    height = random.uniform(10, 30)
    while abs(width - height) < 5:  # Ensure not square
        height = random.uniform(10, 30)
    return [
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height),
        (x, y)
    ]

def generate_square():
    x = random.uniform(0, 50)
    y = random.uniform(0, 50)
    size = random.uniform(10, 30)
    return [
        (x, y),
        (x + size, y),
        (x + size, y + size),
        (x, y + size),
        (x, y)
    ]

def generate_parallelogram():
    x = random.uniform(0, 50)
    y = random.uniform(0, 50)
    width = random.uniform(10, 30)
    height = random.uniform(10, 30)
    skew = random.uniform(5, 15)
    if random.random() < 0.5:
        skew = -skew
    if random.random() < 0.5:
        return [
            (x, y),
            (x + width + skew, y),
            (x + width, y + height),
            (x - skew, y + height),
            (x, y)
        ]
    else:
        return [
            (x, y),
            (x, y + width + skew),
            (x + height, y + width),
            (x + height, y - skew),
            (x, y)
        ]
    
def generate_triangle(non_right=True):
    if non_right:
        # Generate non-right triangle
        a = (random.uniform(0, 50), random.uniform(0, 50))
        b = (a[0] + random.uniform(10, 30), a[1])
        c = (a[0] + random.uniform(5, 15), a[1] + random.uniform(10, 30))
        # Check if right triangle
        v1 = (b[0]-a[0], b[1]-a[1])
        v2 = (c[0]-a[0], c[1]-a[1])
        if abs(v1[0]*v2[0] + v1[1]*v2[1]) < 1e-6:
            c = (c[0]+5, c[1])
        return [a, b, c, a]
    else:
        # Generate right triangle with random right angle vertex
        right_angle_at = random.choice(['a', 'b', 'c'])
        if right_angle_at == 'a':
            a = (random.uniform(-20, 70), random.uniform(-20, 70))
            dx = random.uniform(10, 30) * random.choice([1, -1])
            dy = random.uniform(10, 30) * random.choice([1, -1])
            b = (a[0] + dx, a[1])
            c = (a[0], a[1] + dy)
        elif right_angle_at == 'b':
            b = (random.uniform(-20, 70), random.uniform(-20, 70))
            dx = random.uniform(10, 30) * random.choice([1, -1])
            dy = random.uniform(10, 30) * random.choice([1, -1])
            a = (b[0] - dx, b[1])
            c = (b[0], b[1] + dy)
        else:  # right_angle_at == 'c'
            c = (random.uniform(-20, 70), random.uniform(-20, 70))
            dx = random.uniform(10, 30) * random.choice([1, -1])
            dy = random.uniform(10, 30) * random.choice([1, -1])
            a = (c[0] - dx, c[1])
            b = (c[0], c[1] - dy)
        return [a, b, c, a]

def generate_trapezoid():
    x = random.uniform(0, 50)
    y = random.uniform(0, 50)
    base1 = random.uniform(20, 40)
    base2 = base1 * random.uniform(0.3, 0.7)  # Ensure different lengths
    height = random.uniform(15, 30)
    offset = (base1 - base2) * random.uniform(0.4, 0.6)
    if random.random() < 0.5:
        x,y = y,x
    if random.random() < 0.5:
        return [
            (x, y),
            (x + base1, y),
            (x + base1 - offset, y + height),
            (x + offset, y + height),
            (x, y)
        ]
    else:
        return [
            (x, y),
            (x, y + base1),
            (x + height, y + base1 - offset),
            (x + height, y + offset),
            (x, y)
        ]

def generate_regular_polygon(sides, size=30):
    points = []
    # 固定半径（保证边长相等）
    radius = size * random.uniform(0.8, 1.2)  # 整体缩放
    start_angle = random.uniform(0, 2*math.pi)  # 随机旋转
    center_x = random.uniform(20, 80)  # 随机中心点
    center_y = random.uniform(20, 80)
    
    for i in range(sides):
        angle = 2 * math.pi * i / sides + start_angle
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        points.append((x, y))
    points.append(points[0])  # 闭合路径
    return points

def generate_irregular_convex_pentagon():
    angles = [i*72 + random.uniform(-15, 15) for i in range(5)]
    points = []
    for angle in angles:
        r = random.uniform(20, 35)
        x = 50 + r * math.cos(math.radians(angle))
        y = 50 + r * math.sin(math.radians(angle))
        points.append((x, y))
    points.append(points[0])
    return points

def generate_irregular_concave_pentagon():
    convex = generate_irregular_convex_pentagon()[:-1]
    # Create concave by moving one point inward
    idx = random.randint(0, 4)
    centroid_x = sum(p[0] for p in convex)/5
    centroid_y = sum(p[1] for p in convex)/5
    # Move towards centroid with overshoot
    new_x = convex[idx][0] + 1.5*(centroid_x - convex[idx][0])
    new_y = convex[idx][1] + 1.5*(centroid_y - convex[idx][1])
    convex[idx] = (new_x, new_y)
    convex.append(convex[0])
    return convex

def split_segment(a, b, num_splits=1):
    points = [a]
    for _ in range(num_splits):
        t = random.uniform(0.2, 0.8)
        x = a[0] + t*(b[0]-a[0])
        y = a[1] + t*(b[1]-a[1])
        points.append((x, y))
    points.append(b)
    return points


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]
        

class BbehGeometricShapesbootcamp(Basebootcamp):
    def __init__(self, distract_density=0.5, max_distractors=3):
        self.distract_density = distract_density
        self.max_distractors = max_distractors
    
    
    def case_generator(self):
        target = random.choice(list(range(1, 11)))
        if target == 1:
            points = generate_rectangle()
        elif target == 2:
            points = generate_square()
        elif target in (3,4):
            points = generate_triangle(non_right=(target==3))
        elif target == 5:
            points = generate_parallelogram()
        elif target == 6:
            points = generate_trapezoid()
        elif target == 7:
            points = generate_irregular_convex_pentagon()
        elif target == 8:
            points = generate_irregular_concave_pentagon()
        elif target == 9:
            points = generate_regular_polygon(5)
        elif target == 10:
            points = generate_regular_polygon(6)
        
        edges = []
        for i in range(len(points)-1):
            a, b = points[i], points[i+1]
            splits = random.randint(0, 2)
            split_pts = split_segment(a, b, splits)
            for j in range(len(split_pts)-1):
                edges.append((split_pts[j], split_pts[j+1]))
        
        # Add distractors
        if random.random() < 0.5:
            for _ in range(random.randint(1,3)):
                x1 = random.uniform(-10, 60)
                y1 = random.uniform(-10, 60)
                x2 = random.uniform(-10, 60)
                y2 = random.uniform(-10, 60)
                edges.append(((x1, y1), (x2, y2)))
        
        # Build SVG path
        path = []
        for edge in edges:
            path.append(f"M {edge[0][0]:.5f},{edge[0][1]:.5f} L {edge[1][0]:.5f},{edge[1][1]:.5f}")
        svg_path = " ".join(path)
        
        return {
            "Paths": svg_path,
            "ans": str(target),
        }
    
    
    @staticmethod
    def prompt_func(identity):
        svg_path = identity['Paths']
        intruction = f"Suppose we draw this SVG path element:{svg_path} ."
        instruction_following = """Out of the following shapes:\n1. rectangle that is not a square and with no diagonals drawn\n2. square with no diagonals drawn\n3. triangle that is not a right triangle\n4. right triangle\n5. parallelogram that is not a rectangle and with no diagonals drawn\n6. trapezoid with exactly one pair of parallel sides and with no diagonals drawn\n7. irregular convex pentagon with no diagonals drawn\n8. irregular concave pentagon with no diagonals drawn\n9. regular pentagon with no diagonals drawn\n10. regular hexagon with no diagonals drawn\nwhich one can be viewed when the lines in the SVG are visualized? Note that a shape with n sides should not necessarily be drawn by n lines; e.g., a triangle might be drawn by 4 lines, two of which collinear. \nCoordinates have been rounded to 5 decimal places so ignore slight differences. Solve this problem step by step and provide the final answer within \\boxed{}. For example: "Final Answer: \\boxed{3}." """
        prompt = intruction + instruction_following
        return prompt
        
        
    @staticmethod
    def extract_output(output):
        """
        Extract the output from the solution.
        
        Args:
            output: Model output to be processed.
        
        Returns:
            The processed output.
        """
        output = last_boxed_only_string(output)
        if output is None:
            return None
        return remove_boxed(output)
        
        
    @classmethod
    def _verify_correction(cls,solution,identity)->bool:
        """
        Verify the correction of the solution.
        """
        return solution.strip() == identity['ans'].strip()