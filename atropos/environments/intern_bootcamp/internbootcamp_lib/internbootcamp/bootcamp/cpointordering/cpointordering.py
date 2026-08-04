"""# 

### 谜题描述
This is an interactive problem.

Khanh has n points on the Cartesian plane, denoted by a_1, a_2, …, a_n. All points' coordinates are integers between -10^9 and 10^9, inclusive. No three points are collinear. He says that these points are vertices of a convex polygon; in other words, there exists a permutation p_1, p_2, …, p_n of integers from 1 to n such that the polygon a_{p_1} a_{p_2} … a_{p_n} is convex and vertices are listed in counter-clockwise order.

Khanh gives you the number n, but hides the coordinates of his points. Your task is to guess the above permutation by asking multiple queries. In each query, you give Khanh 4 integers t, i, j, k; where either t = 1 or t = 2; and i, j, k are three distinct indices from 1 to n, inclusive. In response, Khanh tells you:

  * if t = 1, the area of the triangle a_ia_ja_k multiplied by 2. 
  * if t = 2, the sign of the cross product of two vectors \overrightarrow{a_ia_j} and \overrightarrow{a_ia_k}. 



Recall that the cross product of vector \overrightarrow{a} = (x_a, y_a) and vector \overrightarrow{b} = (x_b, y_b) is the integer x_a ⋅ y_b - x_b ⋅ y_a. The sign of a number is 1 it it is positive, and -1 otherwise. It can be proven that the cross product obtained in the above queries can not be 0.

You can ask at most 3 ⋅ n queries.

Please note that Khanh fixes the coordinates of his points and does not change it while answering your queries. You do not need to guess the coordinates. In your permutation a_{p_1}a_{p_2}… a_{p_n}, p_1 should be equal to 1 and the indices of vertices should be listed in counter-clockwise order.

Interaction

You start the interaction by reading n (3 ≤ n ≤ 1 000) — the number of vertices.

To ask a query, write 4 integers t, i, j, k (1 ≤ t ≤ 2, 1 ≤ i, j, k ≤ n) in a separate line. i, j and k should be distinct.

Then read a single integer to get the answer to this query, as explained above. It can be proven that the answer of a query is always an integer.

When you find the permutation, write a number 0. Then write n integers p_1, p_2, …, p_n in the same line.

After printing a query do not forget to output end of line and flush the output. Otherwise, you will get Idleness limit exceeded. To do this, use:

  * fflush(stdout) or cout.flush() in C++;
  * System.out.flush() in Java;
  * flush(output) in Pascal;
  * stdout.flush() in Python;
  * see documentation for other languages.



Hack format

To hack, use the following format:

The first line contains an integer n (3 ≤ n ≤ 1 000) — the number of vertices.

The i-th of the next n lines contains two integers x_i and y_i (-10^9 ≤ x_i, y_i ≤ 10^9) — the coordinate of the point a_i.

Example

Input


6

15

-1

1

Output


1 1 4 6

2 1 5 6

2 2 1 4

0 1 3 4 2 6 5

Note

The image below shows the hidden polygon in the example:

<image>

The interaction in the example goes as below: 

  * Contestant reads n = 6. 
  * Contestant asks a query with t = 1, i = 1, j = 4, k = 6. 
  * Jury answers 15. The area of the triangle A_1A_4A_6 is 7.5. Note that the answer is two times the area of the triangle. 
  * Contestant asks a query with t = 2, i = 1, j = 5, k = 6. 
  * Jury answers -1. The cross product of \overrightarrow{A_1A_5} = (2, 2) and \overrightarrow{A_1A_6} = (4, 1) is -2. The sign of -2 is -1. 
  * Contestant asks a query with t = 2, i = 2, j = 1, k = 4. 
  * Jury answers 1. The cross product of \overrightarrow{A_2A_1} = (-5, 2) and \overrightarrow{A_2A_4} = (-2, -1) is 1. The sign of 1 is 1. 
  * Contestant says that the permutation is (1, 3, 4, 2, 6, 5). 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
#pragma GCC optimize(\"O3\")
#pragma GCC target(\"sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx\")
using namespace std;
int a, b, c, d, n, m, k;
long long sq[1002];
bool uu[1002];
inline void sr(vector<int>& a) {
  sort(a.begin(), a.end(),
       [](const int& a, const int& b) { return sq[a] < sq[b]; });
}
int main() {
  srand(29756111 + time(0));
  scanf(\"%d\", &n);
  int lf = 0, rt = rand() % n;
  while (!rt) rt = rand() % n;
  vector<int> up, down;
  for (int _n((n)-1), i(0); i <= _n; i++) {
    if (i == lf || i == rt) continue;
    printf(\"%d %d %d %d\n\", 2, lf + 1, rt + 1, i + 1);
    fflush(stdout);
    scanf(\"%d\", &a);
    if (a == 1) {
      up.push_back(i);
      uu[i] = 1;
    } else {
      down.push_back(i);
    }
    printf(\"%d %d %d %d\n\", 1, lf + 1, rt + 1, i + 1);
    fflush(stdout);
    scanf(\"%I64d\", &sq[i]);
  }
  int uv = -1, dv = -1;
  for (int _n(((int)((up).size())) - 1), i(0); i <= _n; i++) {
    int v = up[i];
    if (uv == -1 || sq[v] > sq[uv]) {
      uv = v;
    }
  }
  for (int _n(((int)((down).size())) - 1), i(0); i <= _n; i++) {
    int v = down[i];
    if (dv == -1 || sq[v] > sq[dv]) {
      dv = v;
    }
  }
  vector<int> lu, ru, ld, rd;
  for (int _n((n)-1), i(0); i <= _n; i++) {
    if (i == lf || i == rt || i == uv || i == dv) continue;
    if (uu[i]) {
      printf(\"%d %d %d %d\n\", 2, lf + 1, uv + 1, i + 1);
      fflush(stdout);
      scanf(\"%d\", &a);
      if (a == 1) {
        lu.push_back(i);
      } else {
        ru.push_back(i);
      }
    } else {
      printf(\"%d %d %d %d\n\", 2, lf + 1, dv + 1, i + 1);
      fflush(stdout);
      scanf(\"%d\", &a);
      if (a == 1) {
        rd.push_back(i);
      } else {
        ld.push_back(i);
      }
    }
  }
  sr(lu);
  sr(ru);
  sr(ld);
  sr(rd);
  vector<int> res;
  res.push_back(lf);
  for (int _n(((int)((ld).size())) - 1), i(0); i <= _n; i++)
    res.push_back(ld[i]);
  if (dv != -1) res.push_back(dv);
  for (int i = ((int)((rd).size()) - 1), _b = (0); i >= _b; i--)
    res.push_back(rd[i]);
  res.push_back(rt);
  for (int _n(((int)((ru).size())) - 1), i(0); i <= _n; i++)
    res.push_back(ru[i]);
  if (uv != -1) res.push_back(uv);
  for (int i = ((int)((lu).size()) - 1), _b = (0); i >= _b; i--)
    res.push_back(lu[i]);
  printf(\"0\");
  for (int _n(((int)((res).size())) - 1), i(0); i <= _n; i++)
    printf(\" %d\", res[i] + 1);
  printf(\"\n\");
  fflush(stdout);
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import random
import re
from bootcamp import Basebootcamp

class Cpointorderingbootcamp(Basebootcamp):
    def __init__(self, n=6):
        if n < 3 or n > 1000:
            raise ValueError("n must be between 3 and 1000")
        self.n = n

    def case_generator(self):
        points = self.generate_convex_polygon(self.n)
        return {
            'n': self.n,
            'points': {i+1: p for i, p in enumerate(points)},
            'permutation': list(range(1, self.n+1))
        }

    @staticmethod
    def ccw(a, b, c):
        """Counter-clockwise test."""
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

    def generate_convex_polygon(self, n):
        """Generate convex polygon with guaranteed non-collinear points."""
        # Generate random star-shaped polygon
        def is_valid(points):
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    for k in range(j+1, len(points)):
                        if self.ccw(points[i], points[j], points[k]) == 0:
                            return False
            return True
        
        points = []
        attempts = 0
        while len(points) < n or not is_valid(points):
            points = []
            # Generate center point
            cx, cy = random.randint(-10**8, 10**8), random.randint(-10**8, 10**8)
            
            # Generate random angles and radii
            angles = sorted([random.uniform(0, 2*math.pi) for _ in range(n)])
            radii = [random.randint(10**8, 10**9) for _ in range(n)]
            
            # Convert to coordinates with random offsets
            points = [
                (
                    int(cx + r * math.cos(a)) + random.randint(-1000, 1000),
                    int(cy + r * math.sin(a)) + random.randint(-1000, 1000)
                ) for a, r in zip(angles, radii)
            ]
            
            # Verify convexity and sort
            hull = self.convex_hull(points)
            if len(hull) != n:
                continue  # Regenerate if not full convex hull
            
            # Sort hull points by polar angle
            cx = sum(x for x, y in hull) / n
            cy = sum(y for x, y in hull) / n
            hull.sort(key=lambda p: math.atan2(p[1]-cy, p[0]-cx))
            points = hull
            
            attempts += 1
            if attempts > 100:
                raise RuntimeError("Failed to generate valid convex polygon")
        
        return points

    @staticmethod
    def convex_hull(points):
        """Andrew's monotone chain convex hull algorithm."""
        points = sorted(points)
        lower = []
        for p in points:
            while len(lower) >= 2 and Cpointorderingbootcamp.ccw(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and Cpointorderingbootcamp.ccw(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        return lower[:-1] + upper[:-1]

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        return f"""Given a convex polygon with {n} vertices, determine the counter-clockwise vertex order starting with 1. 

Query options:
1. "1 i j k" → 2×area of triangle formed by points i,j,k
2. "2 i j k" → sign of (a_j - a_i) × (a_k - a_i)

Make up to {3*n} queries. Provide the final permutation as a line starting with 0 followed by space-separated indices, enclosed in [answer] tags.

Example:
[answer]
0 1 3 4 2 6 5
[/answer]"""

    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\][\s]*((?:.|\n)*?)[\s]*\[/answer\]', output)
        if not answer_blocks:
            return None
            
        for block in reversed(answer_blocks):
            lines = block.strip().split('\n')
            for line in reversed(lines):
                line = line.strip()
                if line.startswith('0'):
                    parts = line.split()
                    if len(parts) != parts[0].count('0') + len(parts[1:]):
                        continue
                    try:
                        permutation = list(map(int, parts[1:]))
                        if sorted(permutation) == list(range(1, len(permutation)+1)):
                            return permutation
                    except:
                        continue
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # Basic validation
        if not solution or solution[0] != 1 or len(solution) != identity['n']:
            return False
        if set(solution) != set(range(1, identity['n']+1)):
            return False
        
        points = [identity['points'][i] for i in solution]
        points.append(points[0])  # Close polygon
        
        # Check convexity and orientation
        sign = None
        for i in range(len(points)-2):
            a, b, c = points[i], points[i+1], points[i+2]
            cross = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
            if cross == 0:
                return False
            current_sign = 1 if cross > 0 else -1
            if sign is None:
                sign = current_sign
            elif current_sign != sign:
                return False
        
        # Verify all points are used (no duplicates)
        return True
