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
using namespace std;
const long long INF = 9223372036854775807LL, SQ_MAX = 40000000000000000LL;
void getmin(long long &a, const long long b) {
  if (b < a) a = b;
}
void getmax(long long &a, const long long b) {
  if (b > a) a = b;
}
long long Sq(const long long a) { return a * a; }
struct Point {
  long long x, y;
  Point() {}
  Point(const long long _x, const long long _y) : x(_x), y(_y) {}
};
int N;
Point P[100000];
bool SolveTilt(const long long tsq) {
  long long ymn = 0, ymx = 0;
  int l, r = 0;
  for (l = 0; l < N && P[l].x < 0 && Sq(P[l].x) < tsq; l++)
    getmax(ymx, P[l].y), getmin(ymn, P[l].y);
  for (;;) {
    if (l == N || P[l].x >= 0) return false;
    for (; r < N && Sq(P[r].x - P[l].x) <= tsq; r++)
      ;
    const long long y2 = sqrt(tsq - Sq(P[l].x));
    if (y2 > -P[l].x) return false;
    const long long y1 = max(-y2, (long long)(y2 - sqrt(tsq)));
    if (r == N && ymn <= y1 && y2 <= ymx) return true;
    getmax(ymx, P[l].y), getmin(ymn, P[l].y);
    l++;
  }
}
bool SolveLine(const long long lsq) {
  static long long lymx[100000], lymn[100000], rymx[100000], rymn[100000];
  lymx[0] = lymn[0] = P[0].y;
  for (int i = 1; i < N; i++)
    lymx[i] = max(lymx[i - 1], P[i].y), lymn[i] = min(lymn[i - 1], P[i].y);
  rymx[N - 1] = rymn[N - 1] = P[N - 1].y;
  for (int i = N - 1; i >= 0; i--)
    rymx[i] = max(rymx[i + 1], P[i].y), rymn[i] = min(rymn[i + 1], P[i].y);
  for (int r = 0, l = 0; r < N; r++) {
    for (; Sq(P[r].x - P[l].x) > lsq; l++)
      ;
    if (l == 0 && r == N - 1) return true;
    const long long x1 = P[l].x, x2 = P[r].x,
                    y1 = min(l > 0 ? lymn[l - 1] : INF,
                             r + 1 < N ? rymn[r + 1] : INF),
                    y2 = max(l > 0 ? lymx[l - 1] : -INF,
                             r + 1 < N ? rymx[r + 1] : -INF);
    if (Sq(y2 - y1) <= lsq && max(Sq(y1), Sq(y2)) + max(Sq(x1), Sq(x2)) <= lsq)
      return true;
  }
  return false;
}
bool CmpX(const Point &a, const Point &b) { return a.x < b.x; }
long long TiltMin() {
  sort(P, P + N, CmpX);
  long long l = 0, r = SQ_MAX;
  while (l < r) {
    const long long mid = (l + r) / 2;
    if (SolveTilt(mid))
      r = mid;
    else
      l = mid + 1LL;
  }
  return r;
}
long long LineMin() {
  sort(P, P + N, CmpX);
  long long l = 0, r = SQ_MAX;
  while (l < r) {
    const long long mid = (l + r) / 2;
    if (SolveLine(mid))
      r = mid;
    else
      l = mid + 1LL;
  }
  return r;
}
void Flip1(bool ish) {
  for (int i = 0; i < N; i++) (ish ? P[i].x : P[i].y) *= -1;
}
void Flip2() {
  for (int i = 0; i < N; i++) swap(P[i].x, P[i].y);
}
void Flip3() {
  Flip2();
  Flip1(true), Flip1(false);
}
int main() {
  while (scanf(\"%d\", &N) == 1) {
    for (int i = 0; i < N; i++) scanf(\"%lld%lld\", &P[i].x, &P[i].y);
    long long ans = INF;
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 2; j++) getmin(ans, TiltMin()), Flip3();
      Flip1(i & 1);
    }
    for (int i = 0; i < 2; i++) getmin(ans, LineMin()), Flip2();
    printf(\"%lld\n\", ans);
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import random
import re
from bootcamp import Basebootcamp

class Eelectricchargesbootcamp(Basebootcamp):
    def __init__(self, max_n=5, coord_range=10):
        self.max_n = min(max_n, 5)
        self.coord_range = coord_range
    
    def case_generator(self):
        n = random.randint(1, self.max_n)
        points = set()
        # 保证x或y至少有一个非零（避免全投影到原点）
        while len(points) < n:
            x, y = random.randint(-self.coord_range, self.coord_range), random.randint(-self.coord_range, self.coord_range)
            if x != 0 or y != 0:
                points.add((x, y))
        points = list(points)
        return {
            'points': sorted(points),  # 标准化输出顺序
            'expected': self._compute_min_sq(points)
        }

    def _compute_min_sq(self, original_points):
        INF = 1 << 60
        min_sq = INF

        # 修正后的变换逻辑：覆盖所有8种对称情况
        for flip_x in [False, True]:
            for flip_y in [False, True]:
                for swap_xy in [False, True]:
                    points = self._transform_points(original_points, flip_x, flip_y, swap_xy)
                    min_sq = min(min_sq, self._tilt_min(points.copy()))
                    min_sq = min(min_sq, self._line_min(points.copy()))
        return min_sq

    def _transform_points(self, points, flip_x, flip_y, swap_xy):
        """更易理解的坐标变换逻辑"""
        transformed = []
        for x, y in points:
            # 坐标翻转
            if flip_x: x = -x
            if flip_y: y = -y
            # 坐标交换
            if swap_xy: x, y = y, x
            transformed.append((x, y))
        return transformed

    def _tilt_min(self, points):
        points.sort(key=lambda p: p[0])  # 确保排序
        left, right = 0, 4 * (10**8)**2
        ans = right
    
        while left <= right:
            mid = (left + right) // 2
            if self._solve_tilt(points, mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        return ans

    def _solve_tilt(self, points, tsq):
        n = len(points)
        if n == 0: return True
        
        ymn = 0
        ymx = 0
        l = 0
        # 预处理左侧点集
        while l < n and points[l][0] < 0 and points[l][0]**2 < tsq:
            ymn = min(ymn, points[l][1])
            ymx = max(ymx, points[l][1])
            l += 1
        
        while l < n and points[l][0] < 0:
            # 扩展右边界
            r = l
            while r < n and (points[r][0] - points[l][0])**2 <= tsq:
                r += 1
            
            # 计算y约束范围
            remaining = tsq - points[l][0]**2
            if remaining < 0:
                l += 1
                continue
            
            max_y = int(math.isqrt(remaining))
            if max_y > -points[l][0]:
                l += 1
                continue
            
            min_y = max(-max_y, max_y - int(math.isqrt(tsq)))
            
            if r == n and ymn <= min_y and max_y <= ymx:
                return True
            
            # 更新当前点状态
            ymn = min(ymn, points[l][1])
            ymx = max(ymx, points[l][1])
            l += 1
        
        return False

    def _line_min(self, points):
        points.sort(key=lambda p: p[0])  # 确保排序
        left, right = 0, 4 * (10**8)**2
        ans = right
    
        while left <= right:
            mid = (left + right) // 2
            if self._solve_line(points, mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        return ans

    def _solve_line(self, points, lsq):
        n = len(points)
        if n == 0: return True
        
        # 预计算前缀极值
        l_min = [0] * n
        l_max = [0] * n
        l_min[0] = l_max[0] = points[0][1]
        for i in range(1, n):
            l_min[i] = min(l_min[i-1], points[i][1])
            l_max[i] = max(l_max[i-1], points[i][1])
        
        # 预计算后缀极值
        r_min = [0] * n
        r_max = [0] * n
        r_min[-1] = r_max[-1] = points[-1][1]
        for i in range(n-2, -1, -1):
            r_min[i] = min(r_min[i+1], points[i][1])
            r_max[i] = max(r_max[i+1], points[i][1])
        
        l_ptr = 0
        for r_ptr in range(n):
            # 移动左指针保持区间有效性
            while l_ptr <= r_ptr and (points[r_ptr][0] - points[l_ptr][0])**2 > lsq:
                l_ptr += 1
            
            if l_ptr > r_ptr: continue
            
            # 获取左右区间的极值
            left_min = l_min[l_ptr-1] if l_ptr > 0 else math.inf
            left_max = l_max[l_ptr-1] if l_ptr > 0 else -math.inf
            right_min = r_min[r_ptr+1] if r_ptr+1 < n else math.inf
            right_max = r_max[r_ptr+1] if r_ptr+1 < n else -math.inf
            
            curr_min = min(left_min, right_min)
            curr_max = max(left_max, right_max)
            
            if (curr_max - curr_min)**2 > lsq:
                continue
            
            # 计算边界x值
            x_min = points[l_ptr][0]
            x_max = points[r_ptr][0]
            
            # 检查四个角点组合
            max_sq = max(
                x_min**2 + curr_min**2,
                x_min**2 + curr_max**2,
                x_max**2 + curr_min**2,
                x_max**2 + curr_max**2
            )
            
            if max_sq <= lsq:
                return True
        
        return False

    @staticmethod
    def prompt_func(case):
        points = case['points']
        return f"""## 带电坐标轴粒子排列问题

实验室工作需求：
- 每个点选择放置电子（移动到X轴）或质子（移动到Y轴）
- 目标：使最终所有点之间的最大距离平方最小

输入点坐标（共{len(points)}个）：
{len(points)}
""" + '\n'.join(f"{x} {y}" for x, y in points) + """

请输出最小直径平方值并用[answer]标签包裹答案，如：[answer]25[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected']
