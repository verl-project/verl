"""# 

### 谜题描述
You are given n points on the plane. The polygon formed from all the n points is strictly convex, that is, the polygon is convex, and there are no three collinear points (i.e. lying in the same straight line). The points are numbered from 1 to n, in clockwise order.

We define the distance between two points p_1 = (x_1, y_1) and p_2 = (x_2, y_2) as their Manhattan distance: $$$d(p_1, p_2) = |x_1 - x_2| + |y_1 - y_2|.$$$

Furthermore, we define the perimeter of a polygon, as the sum of Manhattan distances between all adjacent pairs of points on it; if the points on the polygon are ordered as p_1, p_2, …, p_k (k ≥ 3), then the perimeter of the polygon is d(p_1, p_2) + d(p_2, p_3) + … + d(p_k, p_1).

For some parameter k, let's consider all the polygons that can be formed from the given set of points, having any k vertices, such that the polygon is not self-intersecting. For each such polygon, let's consider its perimeter. Over all such perimeters, we define f(k) to be the maximal perimeter.

Please note, when checking whether a polygon is self-intersecting, that the edges of a polygon are still drawn as straight lines. For instance, in the following pictures:

<image>

In the middle polygon, the order of points (p_1, p_3, p_2, p_4) is not valid, since it is a self-intersecting polygon. The right polygon (whose edges resemble the Manhattan distance) has the same order and is not self-intersecting, but we consider edges as straight lines. The correct way to draw this polygon is (p_1, p_2, p_3, p_4), which is the left polygon.

Your task is to compute f(3), f(4), …, f(n). In other words, find the maximum possible perimeter for each possible number of points (i.e. 3 to n).

Input

The first line contains a single integer n (3 ≤ n ≤ 3⋅ 10^5) — the number of points. 

Each of the next n lines contains two integers x_i and y_i (-10^8 ≤ x_i, y_i ≤ 10^8) — the coordinates of point p_i.

The set of points is guaranteed to be convex, all points are distinct, the points are ordered in clockwise order, and there will be no three collinear points.

Output

For each i (3≤ i≤ n), output f(i).

Examples

Input

4
2 4
4 3
3 0
1 3


Output

12 14 

Input

3
0 0
0 2
2 0


Output

8 

Note

In the first example, for f(3), we consider four possible polygons: 

  * (p_1, p_2, p_3), with perimeter 12. 
  * (p_1, p_2, p_4), with perimeter 8. 
  * (p_1, p_3, p_4), with perimeter 12. 
  * (p_2, p_3, p_4), with perimeter 12. 



For f(4), there is only one option, taking all the given points. Its perimeter 14.

In the second example, there is only one possible polygon. Its perimeter is 8.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int n;
int ans[300003];
pair<int, int> A[300003];
vector<pair<int, int> > vec;
int state(int first) { return (first < 0) ? -1 : +1; }
int main() {
  scanf(\"%d\", &n);
  for (int i = 0; i < n; i++) scanf(\"%d %d\", &A[i].first, &A[i].second);
  for (int i = 0; i < n; i++) {
    int nxt = (i + 1) % n;
    int bck = (i - 1 + n) % n;
    if (state(A[i].first - A[bck].first) != state(A[nxt].first - A[i].first) ||
        state(A[i].second - A[bck].second) !=
            state(A[nxt].second - A[i].second))
      vec.push_back(A[i]);
  }
  int nn = vec.size();
  if (vec.size() <= 3) {
    for (int i = 0; i < nn; i++)
      ans[3] += abs(vec[i].first - vec[(i + 1) % nn].first) +
                abs(vec[i].second - vec[(i + 1) % nn].second);
    ans[4] = ans[3];
  } else {
    for (int i = 0; i < nn; i++)
      ans[4] += abs(vec[i].first - vec[(i + 1) % nn].first) +
                abs(vec[i].second - vec[(i + 1) % nn].second);
    for (int i = 0; i < nn; i++) {
      for (int j = i + 1; j < nn; j++) {
        for (int k = 0; k < n; k++) {
          if (A[k] == vec[i] || A[k] == vec[j])
            continue;
          else {
            int b1 = max({A[k].first, vec[i].first, vec[j].first}) -
                     min({A[k].first, vec[i].first, vec[j].first});
            int b2 = max({A[k].second, vec[i].second, vec[j].second}) -
                     min({A[k].second, vec[i].second, vec[j].second});
            ans[3] = max(ans[3], 2 * (b1 + b2));
          }
        }
      }
    }
  }
  for (int i = 5; i <= n; i++) ans[i] = ans[4];
  for (int i = 3; i <= n; i++) printf(\"%d%c\", ans[i], (i == n) ? '\n' : ' ');
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import re
import random
import math

class Eoptimalpolygonperimeterbootcamp(Basebootcamp):
    def __init__(self, max_n=6):
        self.max_n = max_n
        random.seed(42)
    
    def generate_convex_polygon(self, n):
        """生成严格凸多边形，确保无三点共线"""
        while True:
            # 生成随机点并计算凸包
            points = []
            for _ in range(n*2):  # 生成足够多的点以提高找到严格凸包的概率
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if (x, y) not in points:
                    points.append((x, y))
            
            # 计算凸包
            points = sorted(points)
            if len(points) < n:
                continue
            
            lower = []
            for p in points:
                while len(lower) >= 2 and self.cross(lower[-2], lower[-1], p) <= 0:
                    lower.pop()
                lower.append(p)
            upper = []
            for p in reversed(points):
                while len(upper) >= 2 and self.cross(upper[-2], upper[-1], p) <= 0:
                    upper.pop()
                upper.append(p)
            convex = lower[:-1] + upper[:-1]
            
            # 严格凸检查
            if len(convex) >= n and self.is_strictly_convex(convex):
                convex = convex[:n]
                # 顺时针排序
                center = (sum(x for x, y in convex)/n, sum(y for x, y in convex)/n)
                convex.sort(key=lambda p: (-math.atan2(p[1]-center[1], p[0]-center[0]), p))
                return convex
    
    def is_strictly_convex(self, points):
        """检查多边形是否严格凸"""
        n = len(points)
        for i in range(n):
            a, b, c = points[i], points[(i+1)%n], points[(i+2)%n]
            if self.cross(a, b, c) == 0:
                return False
        return True
    
    def cross(self, o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    
    def case_generator(self):
        n = random.randint(3, self.max_n)
        while True:
            points = self.generate_convex_polygon(n)
            if points and len(points) == n:
                break
        
        # 计算vec
        vec = []
        for i in range(n):
            prev = (i-1) % n
            next_ = (i+1) % n
            dx_prev = points[i][0] - points[prev][0]
            dx_next = points[next_][0] - points[i][0]
            dy_prev = points[i][1] - points[prev][1]
            dy_next = points[next_][1] - points[i][1]
            
            sign = lambda x: -1 if x < 0 else 1 if x > 0 else 0
            if sign(dx_prev) != sign(dx_next) or sign(dy_prev) != sign(dy_next):
                vec.append(points[i])
        
        answers = [0]*(n+1)
        
        # 计算正确答案
        if len(vec) <= 3:
            perimeter = 0
            m = len(vec)
            for i in range(m):
                x1, y1 = vec[i]
                x2, y2 = vec[(i+1)%m]
                perimeter += abs(x1-x2) + abs(y1-y2)
            answers[3] = perimeter
            for k in range(4, n+1):
                answers[k] = perimeter
        else:
            # 计算k=4的情况
            perimeter = 0
            m = len(vec)
            for i in range(m):
                x1, y1 = vec[i]
                x2, y2 = vec[(i+1)%m]
                perimeter += abs(x1-x2) + abs(y1-y2)
            answers[4] = perimeter
            for k in range(5, n+1):
                answers[k] = perimeter
            
            # 正确计算k=3的情况（遍历所有原始点）
            max3 = 0
            for i in range(len(vec)):
                for j in range(i+1, len(vec)):
                    p1 = vec[i]
                    p2 = vec[j]
                    # 遍历所有原始点中的第三个点
                    for p3 in points:
                        if p3 == p1 or p3 == p2:
                            continue
                        min_x = min(p1[0], p2[0], p3[0])
                        max_x = max(p1[0], p2[0], p3[0])
                        min_y = min(p1[1], p2[1], p3[1])
                        max_y = max(p1[1], p2[1], p3[1])
                        candidate = 2 * (max_x - min_x + max_y - min_y)
                        if candidate > max3:
                            max3 = candidate
            answers[3] = max3
        
        return {
            "n": n,
            "points": points,
            "answers": answers[3:n+1]
        }
    
    @staticmethod
    def prompt_func(question_case):
        points = question_case["points"]
        n = question_case["n"]
        points_str = '\n'.join([f"{x} {y}" for x, y in points])
        return f"""Given a strictly convex polygon with {n} vertices in clockwise order:
{points_str}

Compute the maximum possible perimeter for each k from 3 to {n}. The polygon must be non-self-intersecting and use Manhattan distance.

Output format: Space-separated integers (k=3 to k={n}) within [answer] and [/answer].

Example format: [answer]12 14 16[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            parts = list(map(int, matches[-1].strip().split()))
            return parts
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity["answers"]
        return solution == expected
