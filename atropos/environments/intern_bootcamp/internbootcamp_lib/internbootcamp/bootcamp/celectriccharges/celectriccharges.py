"""# 

### 谜题描述
Programmer Sasha is a student at MIPT (Moscow Institute of Physics and Technology) and he needs to make a laboratory work to pass his finals.

A laboratory unit is a plane with standard coordinate axes marked on it. Physicists from Moscow Institute of Physics and Technology charged the axes by large electric charges: axis X is positive and axis Y is negative.

Experienced laboratory worker marked n points with integer coordinates (xi, yi) on the plane and stopped the time. Sasha should use \"atomic tweezers\" to place elementary particles in these points. He has an unlimited number of electrons (negatively charged elementary particles) and protons (positively charged elementary particles). He can put either an electron or a proton at each marked point. As soon as all marked points are filled with particles, laboratory worker will turn on the time again and the particles will come in motion and after some time they will stabilize in equilibrium. The objective of the laboratory work is to arrange the particles in such a way, that the diameter of the resulting state (the maximum distance between the pairs of points of the set) is as small as possible.

Since Sasha is a programmer, he naively thinks that all the particles will simply \"fall\" into their projections on the corresponding axes: electrons will fall on axis X, while protons will fall on axis Y. As we are programmers too, we will consider the same model as Sasha. That is, a particle gets from point (x, y) to point (x, 0) if it is an electron and to point (0, y) if it is a proton.

As the laboratory has high background radiation and Sasha takes care of his laptop, he did not take it with him, and now he can't write a program that computes the minimum possible diameter of the resulting set. Therefore, you will have to do it for him.

Print a square of the minimum possible diameter of the set.

Input

The first line of the input contains a single integer n (1 ≤ n ≤ 100 000) — the number of points marked on the plane.

Each of the next n lines contains two integers xi and yi ( - 108 ≤ xi, yi ≤ 108) — the coordinates of the i-th point. It is guaranteed that no two points coincide.

Output

Print a single integer — the square of the minimum possible diameter of the set.

Examples

Input

3
1 10
1 20
1 30


Output

0


Input

2
1 10
10 1


Output

2

Note

In the first sample Sasha puts electrons at all points, all particles eventually fall at a single point (1, 0).

In the second sample Sasha puts an electron at point (1, 10), and a proton at point (10, 1). The result is a set of two points (1, 0) and (0, 1), which has a diameter of <image>.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
const int N = 100000 + 10;
int n;
struct Point {
  int x, y;
  inline bool operator<(const Point &rhs) const {
    return x != rhs.x ? x < rhs.x : y < rhs.y;
  }
} point[N];
struct Info {
  int min, max;
  Info() : min(INT_MAX), max(INT_MIN) {}
  Info(int _x) : min(_x), max(_x) {}
  Info(int _min, int _max) : min(_min), max(_max) {}
  inline Info &operator+=(const Info &rhs) {
    min = std::min(min, rhs.min);
    max = std::max(max, rhs.max);
    return *this;
  }
} pre[N], suf[N];
inline Info operator+(Info lhs, const Info &rhs) { return lhs += rhs; }
inline long long sqr(long long x) { return x * x; }
long long calc(int l, int r) {
  Info horz(point[l].x, point[r].x);
  long long res = sqr(horz.max - horz.min);
  Info vert = pre[l - 1] + suf[r + 1];
  if (vert.min == INT_MAX && vert.max == INT_MIN) return res;
  res = std::max(res, sqr(vert.max - vert.min));
  res = std::max(res, sqr(vert.min) + sqr(horz.min));
  res = std::max(res, sqr(vert.min) + sqr(horz.max));
  res = std::max(res, sqr(vert.max) + sqr(horz.min));
  res = std::max(res, sqr(vert.max) + sqr(horz.max));
  return res;
}
bool solve(long long th) {
  for (int i = 1; i <= n; ++i) pre[i] = pre[i - 1] + point[i].y;
  if (sqr(pre[n].max - pre[n].min) <= th) return true;
  for (int i = n; i > 0; --i) suf[i] = suf[i + 1] + point[i].y;
  for (int i = 1, j = 1; i <= n; ++i) {
    while (j < n && abs(point[j + 1].x) <= abs(point[i].x) &&
           sqr(point[j + 1].x - point[i].x) <= th)
      ++j;
    while (j > i && abs(point[j].x) > abs(point[i].x)) --j;
    if (calc(i, j) <= th) return true;
  }
  return false;
}
inline bool check(long long th) {
  if (solve(th)) return true;
  std::reverse(point + 1, point + n + 1);
  return solve(th);
}
int main() {
  scanf(\"%d\", &n);
  for (int i = 1; i <= n; ++i) scanf(\"%d%d\", &point[i].x, &point[i].y);
  std::sort(point + 1, point + n + 1);
  long long l = 0, r = sqr(point[n].x - point[1].x);
  while (l < r) {
    long long mid = (l + r) / 2;
    if (check(mid))
      r = mid;
    else
      l = mid + 1;
  }
  printf(\"%I64d\n\", l);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Celectricchargesbootcamp(Basebootcamp):
    def __init__(self, min_points=1, max_points=10, x_range=(-100, 100), y_range=(-100, 100)):
        self.min_points = min_points
        self.max_points = max_points
        self.x_range = x_range
        self.y_range = y_range
    
    def case_generator(self):
        points = set()
        n = random.randint(self.min_points, self.max_points)
        while len(points) < n:
            x = random.randint(*self.x_range)
            y = random.randint(*self.y_range)
            points.add((x, y))
        # 按x升序排序，x相同按y升序
        sorted_points = sorted(points, key=lambda p: (p[0], p[1]))
        return {
            "n": n,
            "points": [list(p) for p in sorted_points]
        }
    
    @staticmethod
    def prompt_func(question_case):
        points = question_case["points"]
        points_str = "\n".join(f"{x} {y}" for x, y in points)
        return f"""Programmer Celectriccharges needs to place electrons or protons at {len(points)} distinct points. Electrons move to (x, 0), protons to (0, y). Find the square of the minimal possible diameter after movement.

Points:
{points_str}

Put your answer within [answer] and [/answer]. Example: [answer]42[/answer]"""

    @staticmethod
    def extract_output(output):
        answers = re.findall(r'\[answer\](.*?)\[/answer\]', output, flags=re.DOTALL)
        if not answers:
            return None
        try:
            return int(answers[-1].strip())
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        points = [tuple(p) for p in identity["points"]]
        return solution == cls._compute_min_diameter(points)
    
    @staticmethod
    def _compute_min_diameter(points):
        # 转换为按x排序的列表，确保与case_generator中的排序一致
        points_sorted = sorted(points, key=lambda p: (p[0], p[1]))
        n = len(points_sorted)
        if n == 0:
            return 0
        if n == 1:
            return 0

        # 预处理前缀和后缀的y的min和max
        pre_min = [0] * n
        pre_max = [0] * n
        pre_min[0] = points_sorted[0][1]
        pre_max[0] = points_sorted[0][1]
        for i in range(1, n):
            pre_min[i] = min(pre_min[i-1], points_sorted[i][1])
            pre_max[i] = max(pre_max[i-1], points_sorted[i][1])
        
        suf_min = [0] * n
        suf_max = [0] * n
        suf_min[-1] = points_sorted[-1][1]
        suf_max[-1] = points_sorted[-1][1]
        for i in range(n-2, -1, -1):
            suf_min[i] = min(suf_min[i+1], points_sorted[i][1])
            suf_max[i] = max(suf_max[i+1], points_sorted[i][1])

        # 辅助函数计算最大平方距离
        def max_sq_distance(electrons, protons):
            max_sq = 0
            # 电子移动到 (x,0)
            e_points = [(x, 0) for x in electrons]
            # 质子移动到 (0,y)
            p_points = [(0, y) for y in protons]
            all_points = e_points + p_points
            for i in range(len(all_points)):
                for j in range(i, len(all_points)):
                    dx = all_points[i][0] - all_points[j][0]
                    dy = all_points[i][1] - all_points[j][1]
                    sq = dx*dx + dy*dy
                    if sq > max_sq:
                        max_sq = sq
            return max_sq

        # 穷举所有可能的电子和质子的选择组合
        min_sq = float('inf')
        # 优化：对每个点，可以选择电子或质子，但n较大时穷举不适用，但此处假设n较小
        from itertools import product
        for choices in product([0, 1], repeat=n):
            electrons_x = []
            protons_y = []
            for i in range(n):
                if choices[i] == 0:
                    electrons_x.append(points_sorted[i][0])
                else:
                    protons_y.append(points_sorted[i][1])
            current_sq = max_sq_distance(electrons_x, protons_y)
            if current_sq < min_sq:
                min_sq = current_sq
        return min_sq
