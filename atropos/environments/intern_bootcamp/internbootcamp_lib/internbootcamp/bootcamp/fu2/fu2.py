"""# 

### 谜题描述
Recently Vasya learned that, given two points with different x coordinates, you can draw through them exactly one parabola with equation of type y = x^2 + bx + c, where b and c are reals. Let's call such a parabola an U-shaped one.

Vasya drew several distinct points with integer coordinates on a plane and then drew an U-shaped parabola through each pair of the points that have different x coordinates. The picture became somewhat messy, but Vasya still wants to count how many of the parabolas drawn don't have any drawn point inside their internal area. Help Vasya.

The internal area of an U-shaped parabola is the part of the plane that lies strictly above the parabola when the y axis is directed upwards.

Input

The first line contains a single integer n (1 ≤ n ≤ 100 000) — the number of points.

The next n lines describe the points, the i-th of them contains two integers x_i and y_i — the coordinates of the i-th point. It is guaranteed that all points are distinct and that the coordinates do not exceed 10^6 by absolute value.

Output

In the only line print a single integer — the number of U-shaped parabolas that pass through at least two of the given points and do not contain any of the given points inside their internal area (excluding the parabola itself).

Examples

Input


3
-1 0
0 2
1 0


Output


2


Input


5
1 0
1 -1
0 -1
-1 0
-1 -1


Output


1

Note

On the pictures below all U-shaped parabolas that pass through at least two given points are drawn for each of the examples. The U-shaped parabolas that do not have any given point inside their internal area are drawn in red. 

<image> The first example.  <image> The second example. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int MAXN = 1e5 + 7;
pair<long long, long long> p[MAXN];
int n, res = 0;
bool cw(pair<long long, long long> a, pair<long long, long long> b,
        pair<long long, long long> c) {
  return (b.first - a.first) * (c.second - b.second) -
             (c.first - b.first) * (b.second - a.second) <
         0;
}
int main() {
  cin >> n;
  for (int i = 1; i <= n; ++i) {
    cin >> p[i].first >> p[i].second;
    p[i].second -= p[i].first * p[i].first;
  }
  sort(p + 1, p + 1 + n);
  vector<pair<long long, long long> > hull;
  hull.push_back(p[1]);
  for (int i = 2; i <= n; ++i)
    if (cw(p[1], p[i], p[n]) || i == n) {
      while ((int)hull.size() > 1 &&
             !cw(hull[(int)hull.size() - 2], hull.back(), p[i]))
        hull.pop_back();
      hull.push_back(p[i]);
    }
  for (int i = 1; i < (int)hull.size(); ++i)
    if (hull[i].first != hull[i - 1].first) ++res;
  cout << res;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

def cw(a, b, c):
    return (b[0] - a[0]) * (c[1] - b[1]) - (c[0] - b[0]) * (b[1] - a[1]) < 0

def compute_expected(transformed_points):
    if not transformed_points:
        return 0
    transformed_points.sort()
    n = len(transformed_points)
    hull = [transformed_points[0]]
    for i in range(1, n):
        current_point = transformed_points[i]
        if cw(transformed_points[0], current_point, transformed_points[-1]) or (i == n - 1):
            while len(hull) >= 2 and not cw(hull[-2], hull[-1], current_point):
                hull.pop()
            hull.append(current_point)
    res = 0
    for i in range(1, len(hull)):
        if hull[i][0] != hull[i-1][0]:
            res += 1
    return res

class Fu2bootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_x=1000):
        self.max_n = max_n
        self.max_x = max_x
    
    def case_generator(self):
        while True:
            n = random.randint(1, self.max_n)
            points = set()
            tried = 0
            max_attempts = 1000
            valid = True
            
            while len(points) < n and tried < max_attempts:
                tried += 1
                x = random.randint(-self.max_x, self.max_x)
                x_sq = x * x
                y_min = -10**6 - x_sq
                y_max = 10**6 - x_sq
                if y_min > y_max:
                    valid = False
                    break
                y_ = random.randint(y_min, y_max)
                points.add((x, y_ + x_sq))  # Store original (x, y)
            
            if not valid or len(points) < n:
                continue

            # Convert to transformed points for convex hull calculation
            transformed = [(x, y - x**2) for (x, y) in points]
            expected = compute_expected(transformed)
            
            case = {
                "n": n,
                "points": [[x, y] for (x, y) in points],
                "expected": expected
            }
            return case
    
    @staticmethod
    def prompt_func(question_case):
        points = question_case['points']
        points_str = '\n'.join([f"{x} {y}" for x, y in points])
        prompt = f"""Given {question_case['n']} distinct points, determine how many U-shaped parabolas (y=x²+bx+c) through at least two points have no other points strictly above them.

Input Format:
n
x₁ y₁
...
xₙ yₙ

Current Input:
{question_case['n']}
{points_str}

Output the count as [answer]integer[/answer]. Example: [answer]3[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        import re
        answers = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(answers[-1]) if answers else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity["expected"]
