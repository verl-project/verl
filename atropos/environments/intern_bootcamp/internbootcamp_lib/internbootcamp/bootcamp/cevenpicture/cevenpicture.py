"""# 

### 谜题描述
Leo Jr. draws pictures in his notebook with checkered sheets (that is, each sheet has a regular square grid printed on it). We can assume that the sheets are infinitely large in any direction.

To draw a picture, Leo Jr. colors some of the cells on a sheet gray. He considers the resulting picture beautiful if the following conditions are satisfied:

  * The picture is connected, that is, it is possible to get from any gray cell to any other by following a chain of gray cells, with each pair of adjacent cells in the path being neighbours (that is, sharing a side).
  * Each gray cell has an even number of gray neighbours.
  * There are exactly n gray cells with all gray neighbours. The number of other gray cells can be arbitrary (but reasonable, so that they can all be listed).



Leo Jr. is now struggling to draw a beautiful picture with a particular choice of n. Help him, and provide any example of a beautiful picture.

To output cell coordinates in your answer, assume that the sheet is provided with a Cartesian coordinate system such that one of the cells is chosen to be the origin (0, 0), axes 0x and 0y are orthogonal and parallel to grid lines, and a unit step along any axis in any direction takes you to a neighbouring cell.

Input

The only line contains a single integer n (1 ≤ n ≤ 500) — the number of gray cells with all gray neighbours in a beautiful picture.

Output

In the first line, print a single integer k — the number of gray cells in your picture. For technical reasons, k should not exceed 5 ⋅ 10^5.

Each of the following k lines should contain two integers — coordinates of a gray cell in your picture. All listed cells should be distinct, and the picture should satisdfy all the properties listed above. All coordinates should not exceed 10^9 by absolute value.

One can show that there exists an answer satisfying all requirements with a small enough k.

Example

Input


4


Output


12
1 0
2 0
0 1
1 1
2 1
3 1
0 2
1 2
2 2
3 2
1 3
2 3

Note

The answer for the sample is pictured below:

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import print_function, division
from sys import stdin, stdout
from fractions import gcd
from math import *
from collections import *
from operator import itemgetter, mul

rstr = lambda: stdin.readline().strip()
rstrs = lambda: [str(x) for x in stdin.readline().split()]
rint = lambda: int(stdin.readline())
rints = lambda: [int(x) for x in stdin.readline().split()]
arr_2d = lambda n: [rints() for i in range(n)]
out = [[0, 0], [1, 0], [0, 1], [1, 1]]
n = int(input())
r, c = 2, 1

for i in range(n):
    out.append([r, c])
    r -= 1
    c += 1
    out.append([r, c])
    r += 1
    out.append([r, c])
    r += 1

print(len(out))
for i in out:
    print(' '.join(map(str, i)))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from collections import deque
from bootcamp import Basebootcamp

class Cevenpicturebootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=500):
        self.min_n = min_n
        self.max_n = max_n
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        return {"n": n}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case["n"]
        return f"""You need to construct a beautiful grid pattern for Leo Jr. with exactly {n} core cells (cells where all four neighbors are gray). 

Conditions:
1. All gray cells are connected through adjacent edges.
2. Each gray cell has an even number of gray neighbors.
3. Exactly {n} gray cells have all four neighboring cells also gray.

Output format:
- First line: k (total gray cells ≤500,000)
- Next k lines: coordinates of gray cells (each between -1e9 and 1e9)

Put your answer between [answer] and [/answer], like:
[answer]
12
1 0
2 0
... (other coordinates)
[/answer]"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        content = matches[-1].strip()
        lines = [ln.strip() for ln in content.split('\n') if ln.strip()]
        if not lines:
            return None
        
        try:
            k = int(lines[0])
            coords = []
            for ln in lines[1:k+1]:
                x, y = map(int, ln.split())
                coords.append([x, y])
            if len(coords) != k:
                return None
            return coords
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        
        cells = set()
        for coord in solution:
            if len(coord) != 2:
                return False
            x, y = coord
            if abs(x) > 1e9 or abs(y) > 1e9:
                return False
            if (x, y) in cells:
                return False
            cells.add((x, y))
        
        k = len(cells)
        if k > 5 * 10**5 or k == 0:
            return False
        
        # Check connectivity
        directions = [(0,1), (0,-1), (1,0), (-1,0)]
        start = next(iter(cells))
        visited = {start}
        q = deque([start])
        while q:
            x, y = q.popleft()
            for dx, dy in directions:
                neighbor = (x+dx, y+dy)
                if neighbor in cells and neighbor not in visited:
                    visited.add(neighbor)
                    q.append(neighbor)
        if len(visited) != k:
            return False
        
        # Check neighbors
        full_gray = 0
        for (x, y) in cells:
            neighbors = sum(1 for dx, dy in directions if (x+dx, y+dy) in cells)
            if neighbors % 2 != 0:
                return False
            if neighbors == 4:
                full_gray += 1
        
        return full_gray == identity["n"]
