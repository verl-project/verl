"""# 

### 谜题描述
Johnny drives a truck and must deliver a package from his hometown to the district center. His hometown is located at point 0 on a number line, and the district center is located at the point d.

Johnny's truck has a gas tank that holds exactly n liters, and his tank is initially full. As he drives, the truck consumes exactly one liter per unit distance traveled. Moreover, there are m gas stations located at various points along the way to the district center. The i-th station is located at the point xi on the number line and sells an unlimited amount of fuel at a price of pi dollars per liter. Find the minimum cost Johnny must pay for fuel to successfully complete the delivery.

Input

The first line of input contains three space separated integers d, n, and m (1 ≤ n ≤ d ≤ 109, 1 ≤ m ≤ 200 000) — the total distance to the district center, the volume of the gas tank, and the number of gas stations, respectively.

Each of the next m lines contains two integers xi, pi (1 ≤ xi ≤ d - 1, 1 ≤ pi ≤ 106) — the position and cost of gas at the i-th gas station. It is guaranteed that the positions of the gas stations are distinct.

Output

Print a single integer — the minimum cost to complete the delivery. If there is no way to complete the delivery, print -1.

Examples

Input

10 4 4
3 5
5 8
6 3
8 4


Output

22


Input

16 5 2
8 2
5 1


Output

-1

Note

In the first sample, Johnny's truck holds 4 liters. He can drive 3 units to the first gas station, buy 2 liters of gas there (bringing the tank to 3 liters total), drive 3 more units to the third gas station, buy 4 liters there to fill up his tank, and then drive straight to the district center. His total cost is 2·5 + 4·3 = 22 dollars.

In the second sample, there is no way for Johnny to make it to the district center, as his tank cannot hold enough gas to take him from the latest gas station to the district center.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int N = 2e5 + 10;
const long long mod = 1000000007;
int A[N], rmq[N][25], getL[N];
void go(int n) {
  for (int i = 0; i < n; i++) rmq[i][0] = i;
  for (int i = 1; i < N; i++) getL[i] = log2(i);
  for (int j = 1; (1 << j) <= n; j++) {
    for (int i = 0; i + (1 << j) - 1 < n; i++) {
      if (A[rmq[i][j - 1]] < A[rmq[i + (1 << (j - 1))][j - 1]]) {
        rmq[i][j] = rmq[i][j - 1];
      } else {
        rmq[i][j] = rmq[i + (1 << (j - 1))][j - 1];
      }
    }
  }
}
int getmin(int i, int j) {
  int k = getL[j - i + 1];
  if (A[rmq[i][k]] < A[rmq[j - (1 << k) + 1][k]]) {
    return A[rmq[i][k]];
  }
  return A[rmq[j - (1 << k) + 1][k]];
}
vector<pair<int, int> > v;
int main() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  int n, d, m;
  cin >> d >> n >> m;
  for (int i = 0; i < m; i++) {
    int x, y;
    cin >> x >> y;
    v.push_back(make_pair(x, y));
  }
  v.push_back(make_pair(d, 0));
  sort(v.begin(), v.end());
  int st = 0, fuel = n;
  for (int i = 0; i < v.size(); i++) A[i] = v[i].second;
  go(m + 1);
  long long cst = 0;
  for (int i = 0; i < v.size(); i++) {
    if (v[i].first - st > n) {
      cst = -1;
      break;
    }
    fuel -= (v[i].first - st);
    int l = i + 1, r = v.size() - 1, res = i;
    while (l <= r) {
      int mid = (l + r) >> 1;
      if (v[mid].first - v[i].first <= fuel) {
        l = mid + 1;
        res = mid;
      } else
        r = mid - 1;
    }
    int nxtmin = i;
    l = i + 1, r = v.size() - 1;
    while (l <= r) {
      int mid = (l + r) >> 1;
      if (getmin(i + 1, mid) <= v[i].second) {
        r = mid - 1;
        nxtmin = mid;
      } else
        l = mid + 1;
    }
    st = v[i].first;
    if (res >= nxtmin) {
      continue;
    }
    cst += (min(n, v[nxtmin].first - v[i].first) - fuel) * 1LL * v[i].second;
    fuel = min(n, v[nxtmin].first - v[i].first);
  }
  cout << cst;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def compute_min_cost(d, n, m, stations):
    sorted_stations = sorted(stations + [(d, 0)], key=lambda x: x[0])
    prev_pos = 0
    for x, _ in sorted_stations:
        if x - prev_pos > n:
            return -1
        prev_pos = x
    
    stack = []
    next_lower = [None] * len(sorted_stations)
    
    # Preprocess next_lower using monotonic stack
    for i in reversed(range(len(sorted_stations))):
        while stack and sorted_stations[stack[-1]][1] >= sorted_stations[i][1]:
            stack.pop()
        if stack:
            next_lower[i] = stack[-1]
        else:
            next_lower[i] = None
        stack.append(i)
    
    current_pos = 0
    current_fuel = n
    total_cost = 0
    
    for i, (x, p) in enumerate(sorted_stations):
        distance = x - current_pos
        current_fuel -= distance
        if current_fuel < 0:
            return -1
        current_pos = x
        
        if x == d:
            break
        
        j = next_lower[i]
        if j is None:
            max_reach = min(current_pos + n, d)
            buy = min(n - current_fuel, max_reach - x)
            if buy < 0:
                continue
            total_cost += buy * p
            current_fuel += buy
        else:
            max_reach = sorted_stations[j][0]
            required = max(0, (max_reach - x) - current_fuel)
            buy = min(required, n - current_fuel)
            total_cost += buy * p
            current_fuel += buy
        
        if current_fuel < 0:
            return -1
    
    return total_cost if current_pos == d else -1

class Cpackagedeliverybootcamp(Basebootcamp):
    def __init__(self, d_range=(5, 100), n_range=(2, 10), m_range=(1, 10)):
        self.d_range = d_range
        self.n_range = n_range
        self.m_range = m_range
        
        min_d, max_d = self.d_range
        if min_d < 1 or max_d < 1:
            raise ValueError("d must be at least 1")

    def case_generator(self):
        d = random.randint(*self.d_range)
        n = random.randint(max(1, self.n_range[0]), min(d, self.n_range[1]))
        
        possible_xi = list(range(1, d))
        max_valid_m = len(possible_xi)
        min_m = max(self.m_range[0], 0)
        max_m = min(self.m_range[1], max_valid_m)
        m = random.randint(min_m, max_m) if max_m >= min_m else 0
        
        stations_xi = []
        if m > 0:
            stations_xi = random.sample(possible_xi, m)
            stations_xi.sort()
        stations_pi = [random.randint(1, 100) for _ in range(m)]
        
        return {
            'd': d,
            'n': n,
            'm': m,
            'stations': list(zip(stations_xi, stations_pi))
        }

    @staticmethod
    def prompt_func(question_case):
        d = question_case['d']
        n = question_case['n']
        m = question_case['m']
        stations = question_case['stations']
        stations_list = "\n".join([f"- 位置 {x}，价格 {p} 美元/升" for x, p in stations])
        return f"""Johnny需要驾驶卡车从0点前往{d}点，油箱容量{n}升。当前有{m}个加油站：
{stations_list}

请计算最小油费（无法到达时输出-1）。答案置于[answer]标签内，如：[answer]42[/answer]。"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            d = identity['d']
            n = identity['n']
            stations = identity['stations']
            correct = compute_min_cost(d, n, len(stations), stations)
            return solution == correct
        except:
            return False
