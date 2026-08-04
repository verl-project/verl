"""# 

### 谜题描述
Peter had a cube with non-zero length of a side. He put the cube into three-dimensional space in such a way that its vertices lay at integer points (it is possible that the cube's sides are not parallel to the coordinate axes). Then he took a piece of paper and wrote down eight lines, each containing three integers — coordinates of cube's vertex (a single line contains coordinates of a single vertex, each vertex is written exactly once), put the paper on the table and left. While Peter was away, his little brother Nick decided to play with the numbers on the paper. In one operation Nick could swap some numbers inside a single line (Nick didn't swap numbers from distinct lines). Nick could have performed any number of such operations.

When Peter returned and found out about Nick's mischief, he started recollecting the original coordinates. Help Peter restore the original position of the points or else state that this is impossible and the numbers were initially recorded incorrectly.

Input

Each of the eight lines contains three space-separated integers — the numbers written on the piece of paper after Nick's mischief. All numbers do not exceed 106 in their absolute value.

Output

If there is a way to restore the cube, then print in the first line \"YES\". In each of the next eight lines print three integers — the restored coordinates of the points. The numbers in the i-th output line must be a permutation of the numbers in i-th input line. The numbers should represent the vertices of a cube with non-zero length of a side. If there are multiple possible ways, print any of them.

If there is no valid way, print \"NO\" (without the quotes) in the first line. Do not print anything else.

Examples

Input

0 0 0
0 0 1
0 0 1
0 0 1
0 1 1
0 1 1
0 1 1
1 1 1


Output

YES
0 0 0
0 0 1
0 1 0
1 0 0
0 1 1
1 0 1
1 1 0
1 1 1


Input

0 0 0
0 0 0
0 0 0
0 0 0
1 1 1
1 1 1
1 1 1
1 1 1


Output

NO

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys

sys.setrecursionlimit(10 ** 6)

def pyes_no(condition) :
  if condition :
    print (\"YES\")
  else :
    print (\"NO\")

def plist(a, s = ' ') :
  print (s.join(map(str, a)))

def rint() :
  return int(sys.stdin.readline())

def rints() :
  return map(int, sys.stdin.readline().split())

def rfield(n, m = None) :
  if m == None :
    m = n
  
  field = []
  for i in xrange(n) :
    chars = sys.stdin.readline().strip()
    assert(len(chars) == m)
    field.append(chars)
  return field

def pfield(field, separator = '') :
  print ('\n'.join(map(lambda x: separator.join(x), field)))

def check_field_equal(field, i, j, value) :
  if i >= 0 and i < len(field) and j >= 0 and j < len(field[i]) :
    return value == field[i][j]
  return None 

def digits(x, p) :
  digits = []
  while x > 0 :
    digits.append(x % p)
    x //= p
  return digits

def modpower(a, n, mod) :
  r = a ** (n % 2)
  if n > 1 :
    r *= modpower(a, n // 2, mod) ** 2
  return r % mod

def gcd(a, b) :
  if a > b :
    a, b = b, a
  
  while a > 0 :
    a, b = b % a, a

  return b

def vector_distance(a, b) :
  diff = vector_diff(a, b)
  
  return scalar_product(diff, diff) ** 0.5

def vector_inverse(v) :
  r = [-x for x in v]

  return tuple(r)

def vector_diff(a, b) :
  return vector_sum(a, vector_inverse(b))

def vector_sum(a, b) :
  r = [c1 + c2 for c1, c2 in zip(a, b)]
    
  return tuple(r)

def scalar_product(a, b) :
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def check_rectangle(points) :
  assert(len(points) == 4)

  A, B, C, D = points

  for A1, A2, A3, A4 in [
    (A, B, C, D),
    (A, C, B, D),
    (A, B, D, C),
    (A, C, D, B),
    (A, D, B, C),
    (A, D, C, B),
  ] :
    sides = (
      vector_diff(A1, A2),
      vector_diff(A2, A3),
      vector_diff(A3, A4),
      vector_diff(A4, A1),
    )
    if all(scalar_product(s1, s2) == 0 for s1, s2 in zip(sides, sides[1:])) :
       return True
  return False

def check_square(points) :
  if not check_rectangle(points) :
    return False
  A, B, C, D = points

  for A1, A2, A3, A4 in [
    (A, B, C, D),
    (A, C, B, D),
    (A, B, D, C),
    (A, C, D, B),
    (A, D, B, C),
    (A, D, C, B),
  ] :
    side_lengths = [
      (first[0] - next[0]) ** 2 + (first[1] - next[1]) ** 2 for first, next in zip([A1, A2, A3, A4], [A2, A3, A4, A1])
    ]
    if len(set(side_lengths)) == 1 :
      return True
    
  return False

def check_right(p) :
  # Check if there are same points
  for a, b in [
    (p[0], p[1]),
    (p[0], p[2]),
    (p[1], p[2]),
  ] :
    if a[0] == b[0] and a[1] == b[1] :
      return False

  a, b, c = p
  a, b, c = vector_diff(a, b), vector_diff(b, c), vector_diff(c, a)   

  return scalar_product(a, b) * scalar_product(a, c) * scalar_product(b, c) == 0

points = []

for i in range(8) :
  points.append(rints())

x0, y0, z0 = points[0]

from itertools import permutations, combinations
def unique_permutations(a) :
  return set(permutations(a))

sides = []

cube_scalars = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2]

def check_cube(sides) :
  sides_r = [a[0] * a[0] + a[1] * a[1] + a[2] * a[2] for a in sides]
  r2 = min(sides_r)

  if (r2 == 0 
    or not all([v % r2 == 0 for v in sides_r]) 
    or sorted(sides_r) != [r2, r2, r2, 2 * r2, 2 * r2, 2 * r2, 3 * r2]
  ) :
    return False

  products = [a[0] * b[0] + a[1] * b[1] + a[2] * b[2] for a, b in combinations(sides, 2)]
  
  if not all([v % r2 == 0 for v in products]) :
    return False
  
  products = sorted([v / r2 for v in products])
  # print products
  # print cube_scalars
  # print '---' 
  return products == cube_scalars
 

for x1, y1, z1 in unique_permutations(points[1]) :
  x1 -= x0
  y1 -= y0
  z1 -= z0
  sides.append((x1, y1, z1))
  for x2, y2, z2 in unique_permutations(points[2]) :
    x2 -= x0
    y2 -= y0
    z2 -= z0
    sides.append((x2, y2, z2))
    for x3, y3, z3 in unique_permutations(points[3]) :
      x3 -= x0
      y3 -= y0
      z3 -= z0
      sides.append((x3, y3, z3))
      for x4, y4, z4 in unique_permutations(points[4]) :
        x4 -= x0
        y4 -= y0
        z4 -= z0
        sides.append((x4, y4, z4))
        for x5, y5, z5 in unique_permutations(points[5]) :
          x5 -= x0
          y5 -= y0
          z5 -= z0
          sides.append((x5, y5, z5))
          for x6, y6, z6 in unique_permutations(points[6]) :
            x6 -= x0
            y6 -= y0
            z6 -= z0
            sides.append((x6, y6, z6))
            for x7, y7, z7 in unique_permutations(points[7]) :
              x7 -= x0
              y7 -= y0
              z7 -= z0
              sides.append((x7, y7, z7))
              if check_cube(sides) :
                print \"YES\"
                print x0, y0, z0
                for x, y, z in sides :
                  print x + x0, y + y0, z + z0
                exit(0)
              sides.pop()
            sides.pop()
          sides.pop()
        sides.pop()
      sides.pop()
    sides.pop()
  sides.pop()
              
print \"NO\"
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import defaultdict
import re
from itertools import product
from bootcamp import Basebootcamp

class Brestorecubebootcamp(Basebootcamp):
    BASE_VECTOR_GROUPS = [
        # Axis-aligned
        [(1,0,0), (0,1,0), (0,0,1)],
        # Non-axis aligned integer vectors
        [(2,1,2), (1,2,-2), (-2,2,1)],
        [(1,2,3), (2,-1,0), (3,0,-1)]
    ]
    
    def __init__(self, prob_solvable=0.5, **params):
        self.prob_solvable = prob_solvable
        self.origin_range = params.get('origin_range', (-5, 5))
        self.max_unsolvable_attempts = params.get('max_unsolvable_attempts', 5)
    
    def case_generator(self):
        if random.random() < self.prob_solvable:
            return self._generate_valid_cube()
        return self._generate_verified_unsolvable_case()
    
    def _generate_valid_cube(self):
        vectors = random.choice(self.BASE_VECTOR_GROUPS)
        u, v, w = vectors
        origin = [
            random.randint(*self.origin_range),
            random.randint(*self.origin_range),
            random.randint(*self.origin_range)
        ]
        
        vertices = []
        for a, b, c in product([0,1], repeat=3):
            vertex = (
                origin[0] + a*u[0] + b*v[0] + c*w[0],
                origin[1] + a*u[1] + b*v[1] + c*w[1],
                origin[2] + a*u[2] + b*v[2] + c*w[2]
            )
            vertices.append(vertex)
        
        scrambled = [random.sample(v,3) for v in vertices]
        return {
            "input": scrambled,
            "solvable": True,
            "origin": origin,
            "vectors": vectors
        }
    
    def _generate_verified_unsolvable_case(self):
        for _ in range(self.max_unsolvable_attempts):
            case = self._generate_unsolvable_candidate()
            if not self._quick_check_solvable(case['input']):
                return case
        return self._generate_forced_unsolvable()
    
    def _generate_unsolvable_candidate(self):
        strategies = [
            self._gen_duplicate_points,         # 包含重复顶点
            self._gen_invalid_edge_ratio,       # 边长比例错误
            self._gen_modified_valid_cube       # 修改有效立方体
        ]
        return random.choice(strategies)()
    
    def _gen_duplicate_points(self):
        points = [random.sample(range(-2,3),3) for _ in range(7)]
        points.append(points[0].copy())  # 添加重复点
        scrambled = [random.sample(p,3) for p in points]
        return {
            "input": scrambled,
            "solvable": False,
            "type": "duplicate"
        }
    
    def _gen_invalid_edge_ratio(self):
        vectors = [(3,0,0), (0,2,0), (0,0,2)]  # 无效边长比例
        return self._create_case_with_vectors(vectors, solvable=False)
    
    def _gen_modified_valid_cube(self):
        valid_case = self._generate_valid_cube()
        points = valid_case['input']
        # 修改一个点的两个坐标
        idx = random.randint(0,7)
        points[idx] = random.sample(
            [x + random.randint(1,2) for x in points[idx]], 3
        )
        return {
            "input": points,
            "solvable": False,
            "type": "modified"
        }
    
    def _create_case_with_vectors(self, vectors, solvable):
        origin = [random.randint(-3,3) for _ in range(3)]
        points = []
        for a, b, c in product([0,1], repeat=3):
            x = origin[0] + a*vectors[0][0] + b*vectors[1][0] + c*vectors[2][0]
            y = origin[1] + a*vectors[0][1] + b*vectors[1][1] + c*vectors[2][1]
            z = origin[2] + a*vectors[0][2] + b*vectors[1][2] + c*vectors[2][2]
            points.append([x,y,z])
        scrambled = [random.sample(p,3) for p in points]
        return {
            "input": scrambled,
            "solvable": solvable,
            "vectors": vectors
        }
    
    def _quick_check_solvable(self, scrambled):
        """快速排除明显可解的案例"""
        # 检查是否有重复点
        unique_points = set(tuple(sorted(p)) for p in scrambled)
        if len(unique_points) < 8: return False
        
        # 其他快速检查逻辑...
        return True
    
    def _generate_forced_unsolvable(self):
        """生成强制不可解案例（四行全0、四行全1）"""
        points = [[0]*3 for _ in range(4)] + [[1]*3 for _ in range(4)]
        scrambled = [random.sample(p,3) for p in points]
        return {
            "input": scrambled,
            "solvable": False,
            "type": "forced"
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        input_lines = [' '.join(map(str, row)) for row in question_case['input']]
        input_str = '\n'.join(input_lines)
        
        return f'''Determine if these scrambled cube vertices can form a valid cube:
{input_str}

Output format:
- Start with "YES" if possible, followed by 8 valid vertices
- Otherwise start with "NO"
- Enclose answer within [answer][/answer] tags'''

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            lines = solution.strip().split('\n')
            if not lines:
                return False
                
            first_line = lines[0].strip().upper()
            if first_line == 'NO':
                return not identity['solvable']
            
            if first_line != 'YES' or len(lines) != 9:
                return False
                
            # 验证每行是否输入的有效排列
            restored = [tuple(map(int, line.split())) for line in lines[1:9]]
            for i in range(8):
                if sorted(restored[i]) != sorted(identity['input'][i]):
                    return False
            
            # 几何验证
            points = restored
            dist_counts = defaultdict(int)
            
            # 计算所有点对的距离平方
            for i in range(8):
                for j in range(i+1, 8):
                    dx = points[i][0] - points[j][0]
                    dy = points[i][1] - points[j][1]
                    dz = points[i][2] - points[j][2]
                    dist_sq = dx*dx + dy*dy + dz*dz
                    dist_counts[dist_sq] += 1
                    if dist_sq == 0:
                        return False  # 存在重复点
                    
            # 验证立方体特征
            distances = sorted(dist_counts.keys())
            if len(distances) != 3:
                return False
                
            a, b, c = distances
            if not (b == 2*a and c == 3*a):
                return False
                
            # 验证边数分布：12边，12面对角线，4体对角线
            return (
                dist_counts[a] == 12 and
                dist_counts[b] == 12 and
                dist_counts[c] == 4
            )
            
        except Exception as e:
            return False
