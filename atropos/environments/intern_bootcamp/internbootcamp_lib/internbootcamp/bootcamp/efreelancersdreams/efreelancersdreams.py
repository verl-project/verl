"""# 

### 谜题描述
Mikhail the Freelancer dreams of two things: to become a cool programmer and to buy a flat in Moscow. To become a cool programmer, he needs at least p experience points, and a desired flat in Moscow costs q dollars. Mikhail is determined to follow his dreams and registered at a freelance site.

He has suggestions to work on n distinct projects. Mikhail has already evaluated that the participation in the i-th project will increase his experience by ai per day and bring bi dollars per day. As freelance work implies flexible working hours, Mikhail is free to stop working on one project at any time and start working on another project. Doing so, he receives the respective share of experience and money. Mikhail is only trying to become a cool programmer, so he is able to work only on one project at any moment of time.

Find the real value, equal to the minimum number of days Mikhail needs to make his dream come true.

For example, suppose Mikhail is suggested to work on three projects and a1 = 6, b1 = 2, a2 = 1, b2 = 3, a3 = 2, b3 = 6. Also, p = 20 and q = 20. In order to achieve his aims Mikhail has to work for 2.5 days on both first and third projects. Indeed, a1·2.5 + a2·0 + a3·2.5 = 6·2.5 + 1·0 + 2·2.5 = 20 and b1·2.5 + b2·0 + b3·2.5 = 2·2.5 + 3·0 + 6·2.5 = 20.

Input

The first line of the input contains three integers n, p and q (1 ≤ n ≤ 100 000, 1 ≤ p, q ≤ 1 000 000) — the number of projects and the required number of experience and money.

Each of the next n lines contains two integers ai and bi (1 ≤ ai, bi ≤ 1 000 000) — the daily increase in experience and daily income for working on the i-th project.

Output

Print a real value — the minimum number of days Mikhail needs to get the required amount of experience and money. Your answer will be considered correct if its absolute or relative error does not exceed 10 - 6. 

Namely: let's assume that your answer is a, and the answer of the jury is b. The checker program will consider your answer correct, if <image>.

Examples

Input

3 20 20
6 2
1 3
2 6


Output

5.000000000000000


Input

4 1 1
2 3
3 2
2 3
3 2


Output

0.400000000000000

Note

First sample corresponds to the example in the problem statement.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const double eps = 1e-9;
int dcmp(const double &x) {
  if (x < -eps) return -1;
  if (x > eps) return 1;
  return 0;
}
struct Point {
  double x, y;
  Point(double x = 0.0, double y = 0.0) : x(x), y(y) {}
};
Point operator-(Point a, Point b) { return Point(a.x - b.x, a.y - b.y); }
Point operator+(Point a, Point b) { return Point(a.x + b.x, a.y + b.y); }
Point operator*(Point a, double b) { return Point(a.x * b, a.y * b); }
bool operator<(Point a, Point b) {
  return dcmp(a.x - b.x) < 0 || (dcmp(a.x - b.x) == 0 && dcmp(a.y - b.y) < 0);
}
inline double norm(Point &a) { return hypot(a.x, a.y); }
inline double cross(Point a, Point b) { return a.x * b.y - a.y * b.x; }
int ConvexHull(vector<Point> &p, Point ch[]) {
  int n = p.size();
  sort(p.begin(), p.end());
  int m = 0;
  for (int i = 0; i < n; ++i) {
    while (m > 1 && cross(ch[m - 1] - ch[m - 2], p[i] - ch[m - 2]) <= 0) m--;
    ch[m++] = p[i];
  }
  int k = m;
  for (int i = n - 2; i >= 0; --i) {
    while (m > k && cross(ch[m - 1] - ch[m - 2], p[i] - ch[m - 2]) <= 0) m--;
    ch[m++] = p[i];
  }
  return m > 1 ? m - 1 : m;
}
struct Seg {
  Point o, e, v;
  Seg(Point o = Point(), Point e = Point()) : o(o), e(e) { v = e - o; }
};
inline bool intersect(Seg &s1, Seg s2) {
  return cross(s2.o - s1.o, s1.v) * cross(s2.e - s1.o, s1.v) <= 0;
}
Point intersection(Seg &s1, Seg s2) {
  double t = cross(s2.v, s1.o - s2.o) / cross(s1.v, s2.v);
  return s1.o + s1.v * t;
}
double a[100050], b[100050];
Point ch[100050];
int main(void) {
  int n;
  double p, q;
  cin >> n >> p >> q;
  for (int i = 0; i < n; ++i) cin >> a[i] >> b[i];
  vector<Point> vec;
  for (int i = 0; i < n; ++i) vec.push_back(Point(a[i], b[i]));
  vec.push_back(Point(*max_element(a, a + n), 0));
  vec.push_back(Point(0, *max_element(b, b + n)));
  vec.push_back(Point());
  int m = ConvexHull(vec, ch);
  Point goal = Point(p, q);
  Seg ss = Seg(Point(), goal);
  Point inter;
  for (int i = 1; i < m - 1; ++i)
    if (intersect(ss, Seg(ch[i], ch[i + 1]))) {
      inter = intersection(ss, Seg(ch[i], ch[i + 1]));
      break;
    }
  double ans = norm(goal) / norm(inter);
  printf(\"%.9f\n\", ans);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from math import sqrt

class Efreelancersdreamsbootcamp(Basebootcamp):
    def __init__(self, n_range=(1, 10), a_range=(1, 100), b_range=(1, 100), p_range=(1, 100), q_range=(1, 100)):
        self.n_range = n_range
        self.a_range = a_range
        self.b_range = b_range
        self.p_range = p_range
        self.q_range = q_range

    def case_generator(self):
        while True:
            n = random.randint(*self.n_range)
            projects = [
                (random.randint(*self.a_range), random.randint(*self.b_range))
                for _ in range(n)
            ]
            p = random.randint(*self.p_range)
            q = random.randint(*self.q_range)
            answer = self.compute_min_days(n, p, q, projects)
            if answer < float('inf'):
                break
        return {
            'n': n,
            'p': p,
            'q': q,
            'projects': projects,
            'answer': answer
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        projects_desc = "\n".join(
            [f"- Project {i+1}: {a} experience per day, {b} dollars per day" 
             for i, (a, b) in enumerate(question_case['projects'])]
        )
        return f"""Efreelancersdreams the Freelancer needs at least {question_case['p']} experience points and {question_case['q']} dollars. He can work on one project at a time, switching any time. 

Available projects:
{projects_desc}

Find the minimum number of days required (as a real number). Format your answer with 12 decimal places within [answer] and [/answer]."""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return float(matches[-1].strip())
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            user_ans = float(solution)
            correct_ans = identity['answer']
        except:
            return False
        
        abs_err = abs(user_ans - correct_ans)
        if abs_err <= 1e-6:
            return True
        
        if correct_ans == 0:
            return user_ans == 0
        
        rel_err = abs_err / correct_ans
        return rel_err <= 1e-6

    # Helper methods for computing the correct answer
    @staticmethod
    def compute_min_days(n, p, q, projects):
        class Point:
            def __init__(self, x=0.0, y=0.0):
                self.x = x
                self.y = y

        def cross(a, b):
            return a.x * b.y - a.y * b.x

        def convex_hull(points):
            if not points:
                return []
            points = sorted(points, key=lambda p: (p.x, p.y))
            lower = []
            for p in points:
                while len(lower) >= 2:
                    a, b = lower[-2], lower[-1]
                    if cross(Point(b.x - a.x, b.y - a.y), Point(p.x - a.x, p.y - a.y)) <= 0:
                        lower.pop()
                    else:
                        break
                lower.append(p)
            upper = []
            for p in reversed(points):
                while len(upper) >= 2:
                    a, b = upper[-2], upper[-1]
                    if cross(Point(b.x - a.x, b.y - a.y), Point(p.x - a.x, p.y - a.y)) <= 0:
                        upper.pop()
                    else:
                        break
                upper.append(p)
            full = lower[:-1] + upper[:-1]
            return full

        points = [Point(a, b) for a, b in projects]
        if not points:
            return float('inf')
        max_a = max(a for a, b in projects)
        max_b = max(b for a, b in projects)
        points.extend([Point(max_a, 0), Point(0, max_b), Point(0, 0)])
        hull = convex_hull(points)
        goal = Point(p, q)
        o = Point(0, 0)
        inter = None

        for i in range(len(hull)-1):
            a, b = hull[i], hull[i+1]
            a_to_b = Point(b.x - a.x, b.y - a.y)
            seg_v = Point(goal.x - o.x, goal.y - o.y)
            area1 = cross(Point(a.x - o.x, a.y - o.y), seg_v)
            area2 = cross(Point(b.x - o.x, b.y - o.y), seg_v)
            if area1 * area2 < 0:
                t = cross(Point(a.x - o.x, a.y - o.y), a_to_b) / cross(seg_v, a_to_b)
                inter_x = o.x + seg_v.x * t
                inter_y = o.y + seg_v.y * t
                inter = Point(inter_x, inter_y)
                break
            elif area1 == 0 and (a.x * seg_v.y != a.y * seg_v.x or seg_v.x == 0 and seg_v.y == 0):
                inter = a
                break
            elif area2 == 0 and (b.x * seg_v.y != b.y * seg_v.x or seg_v.x == 0 and seg_v.y == 0):
                inter = b
                break

        if not inter:
            return float('inf')

        norm_goal = sqrt(p**2 + q**2)
        norm_inter = sqrt(inter.x**2 + inter.y**2)
        if norm_inter == 0:
            return float('inf')
        return norm_goal / norm_inter
