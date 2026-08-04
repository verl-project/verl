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
#include <bits/stdc++.h>
using namespace std;
int vertices[8][3];
int edges[28][3];
void print_answer() {
  int i, j;
  for (i = 0; i < 8; i++) {
    for (j = 0; j < 3; j++) printf(\"%d \", vertices[i][j]);
    puts(\"\");
  }
}
unsigned long long sdist(int i, int j) {
  unsigned long long result = 0, tmp;
  int k;
  for (k = 0; k < 3; ++k) {
    tmp = (unsigned long long)(vertices[i][k] - vertices[j][k]);
    result += tmp * tmp;
  }
  return result;
}
unsigned long long sprod(int i, int j) {
  unsigned long long result = 0;
  int k;
  for (k = 0; k < 3; ++k)
    result += (unsigned long long)edges[i][k] * edges[j][k];
  return result;
}
int check() {
  int i, j, k, l, count = 0;
  unsigned long long sqr_l = ULLONG_MAX;
  for (i = 0; i < 8; ++i)
    for (j = i + 1; j < 8; ++j) {
      if (sdist(i, j) == sqr_l) {
        for (k = 0; k < 3; ++k)
          edges[count][k] = vertices[i][k] - vertices[j][k];
        ++count;
      } else if (sdist(i, j) < sqr_l) {
        for (k = 0; k < 3; ++k) edges[0][k] = vertices[i][k] - vertices[j][k];
        count = 1;
        sqr_l = sdist(i, j);
      }
    }
  if (sqr_l == 0 || count != 12) return 0;
  for (i = 0; i < count; ++i) {
    l = 0;
    for (j = 0; j < count; ++j)
      if (i != j && sprod(i, j) == 0) l++;
    if (l != 8) return 0;
  }
  return 1;
}
int main(int argc, char const *argv[]) {
  int i, j, i1, i2, i3, i4, i5, i6, i7;
  for (i = 0; i < 8; i++)
    for (j = 0; j < 3; j++) scanf(\"%d\", &vertices[i][j]);
  if (check()) {
    puts(\"YES\");
    print_answer();
    return 0;
  }
  for (i1 = 0; i1 < 6; ++i1) {
    swap(vertices[1][i1 % 3], vertices[1][(i1 + 1) % 3]);
    for (i2 = 0; i2 < 6; ++i2) {
      swap(vertices[2][i2 % 3], vertices[2][(i2 + 1) % 3]);
      for (i3 = 0; i3 < 6; ++i3) {
        swap(vertices[3][i3 % 3], vertices[3][(i3 + 1) % 3]);
        for (i4 = 0; i4 < 6; ++i4) {
          swap(vertices[4][i4 % 3], vertices[4][(i4 + 1) % 3]);
          for (i5 = 0; i5 < 6; ++i5) {
            swap(vertices[5][i5 % 3], vertices[5][(i5 + 1) % 3]);
            for (i6 = 0; i6 < 6; ++i6) {
              swap(vertices[6][i6 % 3], vertices[6][(i6 + 1) % 3]);
              for (i7 = 0; i7 < 6; ++i7) {
                swap(vertices[7][i7 % 3], vertices[7][(i7 + 1) % 3]);
                if (check()) {
                  puts(\"YES\");
                  print_answer();
                  return 0;
                }
              }
            }
          }
        }
      }
    }
  }
  puts(\"NO\");
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import math
from itertools import permutations, product, combinations

class Drestorecubebootcamp(Basebootcamp):
    def __init__(self, solvable=True, cube_size=1, max_shift=3):
        self.solvable = solvable
        self.cube_size = cube_size
        self.max_shift = max_shift  # 控制立方体位置偏移范围
    
    def case_generator(self):
        if self.solvable:
            # 生成随机偏移的轴对齐立方体
            offset = [random.randint(-self.max_shift, self.max_shift) for _ in range(3)]
            base = [0, self.cube_size]
            original = []
            for x in base:
                for y in base:
                    for z in base:
                        pt = [x + offset[0], y + offset[1], z + offset[2]]
                        original.append(pt)
            
            # 随机打乱每个顶点的坐标顺序
            shuffled = []
            for pt in original:
                shuffled.append(random.choice(list(permutations(pt))))
            
            return {
                'input_points': [list(p) for p in shuffled],
                'solvable': True,
                'original_cube': original,
                'offset': offset
            }
        else:
            # 生成无法构成立方体的案例（保证坐标唯一但几何结构错误）
            points = []
            while len(points) < 8:
                pt = [random.randint(-2, 5) for _ in range(3)]
                if pt not in points:
                    points.append(pt)
                if len(points) == 7:  # 强制最后一个点破坏立方体结构
                    invalid = True
                    while invalid:
                        last_pt = [random.randint(-2, 5) for _ in range(3)]
                        if last_pt not in points:
                            points.append(last_pt)
                            invalid = not self._makes_invalid_cube(points)
            return {
                'input_points': points,
                'solvable': False
            }
    
    def _makes_invalid_cube(self, points):
        """确保points不能组成有效立方体"""
        # 快速初步检查
        if len(points) != 8:
            return True
        
        # 检查坐标唯一性
        if len(set(map(tuple, points))) < 8:
            return True
        
        # 计算所有距离的平方
        dists = []
        for (x1, y1, z1), (x2, y2, z2) in combinations(points, 2):
            dists.append((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        
        # 有效立方体应有3种不同距离值（边、面对角线、体对角线）
        unique_dists = set(dists)
        if len(unique_dists) != 3:
            return True
        
        # 检查各距离的数量是否符合立方体特征
        min_dist = min(unique_dists)
        edge_count = dists.count(min_dist)
        return edge_count != 12
    
    @staticmethod
    def prompt_func(question_case) -> str:
        input_lines = [' '.join(map(str, p)) for p in question_case['input_points']]
        problem = "\n".join(input_lines)
        return (
            "Peter有一个各顶点坐标为整数的立方体，其弟Nick将每个顶点的三个数进行了排列交换。\n"
            "请判断能否恢复成立方体结构，若能，输出YES并按输入顺序给出正确坐标（每行为对应输入的排列），否则输出NO。\n"
            f"输入数据：\n{problem}\n"
            "答案要求：\n"
            "1. 第一行必须是YES或NO\n"
            "2. 如果YES，后续8行必须是对应输入的合法排列\n"
            "3. 坐标必须用空格分隔的三个整数\n"
            "请将最终答案放在[answer]和[/answer]标记之间，示例如下：\n"
            "[answer]\n"
            "YES\n"
            "0 0 0\n"
            "1 0 0\n"
            "...（其他6行）\n"
            "[/answer]"
        )
    
    @staticmethod
    def extract_output(output):
        import re
        # 查找所有匹配的答案块并取最后一个
        answers = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answers:
            return None
        answer_block = answers[-1].strip()
        
        lines = [ln.strip() for ln in answer_block.split('\n') if ln.strip()]
        if not lines:
            return None
        
        first_line = lines[0].upper()
        if first_line not in ('YES', 'NO'):
            return None
        
        result = {'result': first_line}
        if first_line == 'YES' and len(lines) != 9:
            return None
        
        if first_line == 'YES':
            points = []
            for ln in lines[1:]:
                parts = ln.split()
                if len(parts) != 3:
                    return None
                try:
                    points.append([int(p) for p in parts])
                except ValueError:
                    return None
            result['points'] = points
        
        return result
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 验证NO结果
        if solution['result'] == 'NO':
            return not identity.get('solvable', True) and not cls.check_cube_possible(identity['input_points'])
        
        # 验证YES结果
        points = solution.get('points', [])
        if len(points) != 8:
            return False
        
        # 检查排列合法性
        for sol_pt, inp_pt in zip(points, identity['input_points']):
            if sorted(sol_pt) != sorted(inp_pt):
                return False
        
        # 验证几何结构
        return cls.is_cube(points)
    
    @classmethod
    def check_cube_possible(cls, input_points):
        """实际检查输入是否可能恢复成立方体（用于不可解案例的验证）"""
        # 尝试所有点的排列组合（优化版，仅用于验证生成案例）
        from itertools import permutations
        
        # 预处理所有可能的顶点排列
        candidates = []
        for pt in input_points:
            candidates.append(set(permutations(pt)))
        
        # 快速排除明显无效的情况
        unique_points = len(set(map(tuple, input_points)))
        if unique_points < 8:
            return False
        
        # 随机采样部分排列组合进行验证
        MAX_TRIES = 1000
        for _ in range(MAX_TRIES):
            test_case = [random.choice(list(c)) for c in candidates]
            if cls.is_cube(test_case):
                return True
        return False
    
    @staticmethod
    def is_cube(points):
        """优化后的几何验证逻辑"""
        # 计算所有点对的平方距离
        dist_counter = {}
        vectors = {}
        for i, j in combinations(range(8), 2):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            dz = points[i][2] - points[j][2]
            dist_sq = dx*dx + dy*dy + dz*dz
            dist_counter[dist_sq] = dist_counter.get(dist_sq, 0) + 1
            vectors[(i,j)] = (dx, dy, dz)
        
        # 有效立方体应有3种距离：边长（min）、面对角线、体对角线
        if len(dist_counter) != 3:
            return False
        
        # 检查各距离的数量关系
        edges = sorted(dist_counter.keys())
        a2, b2, c2 = edges  # a < b < c
        if dist_counter[a2] != 12 or dist_counter[b2] != 12 or dist_counter[c2] != 4:
            return False
        
        # 验证平方关系：b^2 = 2a^2，c^2 = 3a^2
        if not (math.isclose(b2, 2*a2, rel_tol=1e-9) and math.isclose(c2, 3*a2, rel_tol=1e-9)):
            return False
        
        # 验证向量正交性
        edge_vectors = [vec for (i,j), vec in vectors.items() if (points[i][0]-points[j][0])**2 + 
                       (points[i][1]-points[j][1])**2 + (points[i][2]-points[j][2])**2 == a2]
        
        # 每个边应有三个正交边
        for vec in edge_vectors[:3]:  # 检查前三个边即可
            orthogonal = 0
            for other in edge_vectors:
                if sum(a*b for a, b in zip(vec, other)) == 0:
                    orthogonal += 1
            if orthogonal < 3:
                return False
        
        return True
