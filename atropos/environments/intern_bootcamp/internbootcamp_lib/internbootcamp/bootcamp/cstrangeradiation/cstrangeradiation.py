"""# 

### 谜题描述
n people are standing on a coordinate axis in points with positive integer coordinates strictly less than 106. For each person we know in which direction (left or right) he is facing, and his maximum speed.

You can put a bomb in some point with non-negative integer coordinate, and blow it up. At this moment all people will start running with their maximum speed in the direction they are facing. Also, two strange rays will start propagating from the bomb with speed s: one to the right, and one to the left. Of course, the speed s is strictly greater than people's maximum speed.

The rays are strange because if at any moment the position and the direction of movement of some ray and some person coincide, then the speed of the person immediately increases by the speed of the ray.

You need to place the bomb is such a point that the minimum time moment in which there is a person that has run through point 0, and there is a person that has run through point 106, is as small as possible. In other words, find the minimum time moment t such that there is a point you can place the bomb to so that at time moment t some person has run through 0, and some person has run through point 106.

Input

The first line contains two integers n and s (2 ≤ n ≤ 105, 2 ≤ s ≤ 106) — the number of people and the rays' speed.

The next n lines contain the description of people. The i-th of these lines contains three integers xi, vi and ti (0 < xi < 106, 1 ≤ vi < s, 1 ≤ ti ≤ 2) — the coordinate of the i-th person on the line, his maximum speed and the direction he will run to (1 is to the left, i.e. in the direction of coordinate decrease, 2 is to the right, i.e. in the direction of coordinate increase), respectively.

It is guaranteed that the points 0 and 106 will be reached independently of the bomb's position.

Output

Print the minimum time needed for both points 0 and 106 to be reached.

Your answer is considered correct if its absolute or relative error doesn't exceed 10 - 6. Namely, if your answer is a, and the jury's answer is b, then your answer is accepted, if <image>.

Examples

Input

2 999
400000 1 2
500000 1 1


Output

500000.000000000000000000000000000000


Input

2 1000
400000 500 1
600000 500 2


Output

400.000000000000000000000000000000

Note

In the first example, it is optimal to place the bomb at a point with a coordinate of 400000. Then at time 0, the speed of the first person becomes 1000 and he reaches the point 106 at the time 600. The bomb will not affect on the second person, and he will reach the 0 point at the time 500000.

In the second example, it is optimal to place the bomb at the point 500000. The rays will catch up with both people at the time 200. At this time moment, the first is at the point with a coordinate of 300000, and the second is at the point with a coordinate of 700000. Their speed will become 1500 and at the time 400 they will simultaneously run through points 0 and 106.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
using ll = long long;
using ld = long double;
using ii = pair<ll, ll>;
using vi = vector<ll>;
using vb = vector<bool>;
using vvi = vector<vi>;
using vii = vector<ii>;
using vvii = vector<vii>;
constexpr int INF = 2000000000;
constexpr ll LLINF = 9000000000000000000;
struct ConvexHullSet {
  struct Line {
    ld a, b;
    mutable ld x;
    bool type;
    bool operator<(const Line &rhs) const {
      return type || rhs.type ? x < rhs.x : a < rhs.a;
    }
    ld intersect(const Line &rhs) const {
      return ld(b - rhs.b) / ld(rhs.a - a);
    }
  };
  set<Line> lines;
  static constexpr ld MAX = std::numeric_limits<ld>::max();
  ld query(ld x) {
    auto it = lines.lower_bound(Line{0.0, 0.0, x, true});
    return (it != lines.end() ? it->a * x + it->b : -1e300);
  }
  void adjust(set<Line>::iterator it) {
    if (it != lines.end())
      it->x = next(it) != lines.end() ? it->intersect(*next(it)) : MAX;
    if (it != lines.begin())
      prev(it)->x = it != lines.end() ? it->intersect(*prev(it)) : MAX;
  }
  void insert(ld a, ld b) {
    Line ln = Line{a, b, 0.0, false};
    auto it1 = lines.lower_bound(ln);
    if (it1 != lines.end() && it1->a == a) {
      if (it1->b >= b) return;
      it1 = lines.erase(it1);
      adjust(it1);
    }
    ln.x = it1 != lines.end() ? ln.intersect(*it1) : MAX;
    while (it1 != lines.end() && ln.x >= it1->x) {
      it1 = lines.erase(it1);
      ln.x = it1 != lines.end() ? it1->intersect(ln) : MAX;
      adjust(it1);
    }
    while (it1 != lines.begin()) {
      --it1;
      ld nx = it1->intersect(ln);
      if (nx >= it1->x) return;
      if (it1 != lines.begin() && prev(it1)->x >= nx) {
        it1 = lines.erase(it1);
        adjust(it1);
      } else
        break;
    }
    it1 = lines.insert(ln).first;
    adjust(it1);
  }
};
constexpr int L = 0, R = 1000000;
vector<ld> solve(vii &pairs, int s) {
  vector<ld> d(R - L + 1, 1e300);
  sort(pairs.begin(), pairs.end());
  ConvexHullSet chs;
  for (int i = 0, p = L; p <= R; ++p) {
    while (i < (int)pairs.size() && pairs[i].first == p) {
      int x = pairs[i].first, v = pairs[i].second;
      ld a = 1.0 / ld(s - v) - ld(v) / (ld(s - v) * ld(s + v));
      ld b = -ld(x) / ld(s - v) + ld(x) / ld(s + v) +
             ld(v) * ld(x) / (ld(s - v) * ld(s + v));
      chs.insert(-a, -b);
      ++i;
    }
    d[p] = -chs.query(p);
  }
  return d;
}
int main() {
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  int n, s;
  cin >> n >> s;
  vi x(n, 0), v(n, 0), t(n, 0);
  ld lr = 1e300, rr = 1e300;
  int minl = R + 1, maxr = L - 1;
  for (int i = 0; i < n; ++i) {
    cin >> x[i] >> v[i] >> t[i];
    if (t[i] == 1) {
      lr = min(lr, ld(x[i]) / ld(v[i]));
      if (x[i] < minl) minl = x[i];
    } else {
      rr = min(rr, ld(R - x[i]) / ld(v[i]));
      if (x[i] > maxr) maxr = x[i];
    }
  }
  cerr << \"lr = \" << lr << \", rr = \" << rr << endl;
  vector<ld> lp, rp;
  {
    vii pairs[2];
    for (int i = 0; i < n; ++i) pairs[t[i] - 1].push_back({x[i], v[i]});
    for (ii &pr : pairs[1]) pr.first = R - pr.first;
    lp = solve(pairs[0], s);
    rp = solve(pairs[1], s);
    reverse(rp.begin(), rp.end());
  }
  ld ans = max(lr, rr);
  for (int p = L; p <= R; ++p)
    ans = min(ans, max(min(lr, lp[p]), min(rr, rp[p])));
  printf(\"%.10lf\n\", double(ans));
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from math import inf

class Cstrangeradiationbootcamp(Basebootcamp):
    def __init__(self, n_min=2, n_max=10, s_min=2, s_max=1000, **kwargs):
        self.n_min = n_min
        self.n_max = n_max
        self.s_min = s_min
        self.s_max = s_max

    def case_generator(self):
        while True:
            s = random.randint(self.s_min, self.s_max)
            n = random.randint(self.n_min, self.n_max)
            people = []
            has_left = False
            has_right = False
            for _ in range(n):
                x = random.randint(1, 10**6 - 1)
                v = random.randint(1, s - 1)
                if len(people) < 2:
                    t = 1 if not has_left else 2
                else:
                    t = random.choice([1, 2])
                if t == 1:
                    has_left = True
                else:
                    has_right = True
                people.append({'x': x, 'v': v, 't': t})
            if has_left and has_right:
                break
        case = {
            'n': n,
            's': s,
            'people': people
        }
        case['expected_time'] = self.calculate_correct_answer(case)
        return case

    @staticmethod
    def calculate_correct_answer(case):
        def get_left_time(p, x, v, s):
            if p >= x:
                denom = s - v
                if denom <= 0:
                    return x / v
                t0 = (p - x) / denom
                pos_after_t0 = x - v * t0
                if pos_after_t0 <= 0:
                    return x / v
                arrival = t0 + pos_after_t0 / (v + s)
                return min(arrival, x / v)
            else:
                return x / v

        def get_right_time(p, x, v, s):
            if p <= x:
                denom = s - v
                if denom <= 0:
                    return (1e6 - x) / v
                t0 = (x - p) / denom
                pos_after_t0 = x + v * t0
                if pos_after_t0 >= 1e6:
                    return (1e6 - x) / v
                arrival = t0 + (1e6 - pos_after_t0) / (v + s)
                return min(arrival, (1e6 - x) / v)
            else:
                return (1e6 - x) / v

        left_people = [p for p in case['people'] if p['t'] == 1]
        right_people = [p for p in case['people'] if p['t'] == 2]
        s_val = case['s']
        candidates = set()

        for p in left_people:
            candidates.add(p['x'])
        for p in right_people:
            candidates.add(p['x'])
        candidate_list = [0, 1e6] + list(candidates)
        for i in range(len(left_people)):
            for j in range(i + 1, len(left_people)):
                avg = (left_people[i]['x'] + left_people[j]['x']) / 2
                candidate_list.append(avg)
        for i in range(len(right_people)):
            for j in range(i + 1, len(right_people)):
                avg = (right_people[i]['x'] + right_people[j]['x']) / 2
                candidate_list.append(avg)
        for l_p in left_people:
            for r_p in right_people:
                avg = (l_p['x'] + r_p['x']) / 2
                candidate_list.append(avg)

        min_total = inf
        for p in candidate_list:
            left_times = [get_left_time(p, person['x'], person['v'], s_val) for person in left_people]
            right_times = [get_right_time(p, person['x'], person['v'], s_val) for person in right_people]
            if not left_times or not right_times:
                continue
            current_max = max(min(left_times), min(right_times))
            if current_max < min_total:
                min_total = current_max
        return min_total

    @staticmethod
    def prompt_func(question_case):
        prompt = [
            "You are tasked with solving a bomb placement optimization problem to minimize the time when both ends of the coordinate axis (0 and 1,000,000) are reached. The problem is as follows:",
            f"- Number of people (n): {question_case['n']}",
            f"- Speed of the rays (s): {question_case['s']}",
            "Each person has the following attributes:",
            "- Position (x): An integer where 0 < x < 1,000,000",
            "- Speed (v): An integer less than s",
            "- Direction (1 for left, 2 for right)"
        ]
        for i, person in enumerate(question_case['people']):
            prompt.append(f"  Person {i+1}: x={person['x']}, v={person['v']}, direction={person['t']}")
        prompt.append("\nYour task is to determine the earliest possible time t such that there exists a bomb position where at least one person reaches 0 and another reaches 1,000,000. The answer must be precise to at least six decimal places and formatted within [answer] tags.")
        prompt.append("Example answer format: [answer]500.000000[/answer]")
        return '\n'.join(prompt)

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return float(last_match)
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False
        expected = identity.get('expected_time')
        if expected is None:
            return False
        if expected == 0:
            return solution == 0
        abs_error = abs(solution - expected)
        rel_error = abs_error / expected if expected != 0 else inf
        return abs_error < 1e-6 or rel_error < 1e-6
