"""# 

### 谜题描述
Manao has invented a new mathematical term — a beautiful set of points. He calls a set of points on a plane beautiful if it meets the following conditions:

  1. The coordinates of each point in the set are integers. 
  2. For any two points from the set, the distance between them is a non-integer. 



Consider all points (x, y) which satisfy the inequations: 0 ≤ x ≤ n; 0 ≤ y ≤ m; x + y > 0. Choose their subset of maximum size such that it is also a beautiful set of points.

Input

The single line contains two space-separated integers n and m (1 ≤ n, m ≤ 100).

Output

In the first line print a single integer — the size k of the found beautiful set. In each of the next k lines print a pair of space-separated integers — the x- and y- coordinates, respectively, of a point from the set.

If there are several optimal solutions, you may print any of them.

Examples

Input

2 2


Output

3
0 1
1 2
2 0


Input

4 3


Output

4
0 3
2 1
3 0
4 2

Note

Consider the first sample. The distance between points (0, 1) and (1, 2) equals <image>, between (0, 1) and (2, 0) — <image>, between (1, 2) and (2, 0) — <image>. Thus, these points form a beautiful set. You cannot form a beautiful set with more than three points out of the given points. Note that this is not the only solution.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n , m = map(int , raw_input().split())
print min(n , m) + 1
if n <= m:
    for i in range(n):
        print i , n - i
    print n , 0
else:
    for i in range(m):
        print m - i , i
    print 0 , m
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import random
import re
from bootcamp import Basebootcamp

class Cbeautifulsetsofpointsbootcamp(Basebootcamp):
    def __init__(self, max_n=100, max_m=100, min_n=1, min_m=1):
        self.max_n = max_n
        self.max_m = max_m
        self.min_n = min_n
        self.min_m = min_m
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        m = random.randint(self.min_m, self.max_m)
        return {'n': n, 'm': m}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        m = question_case['m']
        prompt = f"""给定n={n}, m={m}，请构造最大美丽点集。规则：
1. 所有点坐标为整数且满足0≤x≤{n}、0≤y≤{m}、x+y>0
2. 任意两点距离必须为非整数

输出格式要求：
第一行为k（集合大小），随后k行每行为x y坐标，并将答案包裹在[answer]标签内。示例如下：

[answer]
3
0 1
1 2
2 0
[/answer]

请解决当前问题："""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        
        last_match = matches[-1].strip()
        lines = [line.strip() for line in last_match.split('\n') if line.strip()]
        if len(lines) < 1:
            return None
        
        try:
            k = int(lines[0])
            if len(lines) != k + 1:
                return None
        except ValueError:
            return None
        
        points = []
        for line in lines[1:]:
            coords = re.split(r'\s+', line.strip())
            if len(coords) != 2:
                return None
            try:
                x, y = map(int, coords)
                points.append((x, y))
            except ValueError:
                return None
        
        return points
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        m = identity['m']
        
        # 基本参数检查
        if not solution or len(solution) != min(n, m) + 1:
            return False
        
        # 坐标有效性检查
        points_set = set()
        for x, y in solution:
            if (x, y) in points_set:
                return False
            if x < 0 or x > n or y < 0 or y > m or (x + y <= 0):
                return False
            points_set.add((x, y))
        
        # 距离检查
        for i in range(len(solution)):
            for j in range(i+1, len(solution)):
                dx = solution[i][0] - solution[j][0]
                dy = solution[i][1] - solution[j][1]
                if math.isqrt(dx**2 + dy**2) ** 2 == dx**2 + dy**2:
                    return False
        
        return True
