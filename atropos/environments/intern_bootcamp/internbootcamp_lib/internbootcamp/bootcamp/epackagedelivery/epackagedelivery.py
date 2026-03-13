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
const int MAXM = 2e5 + 7;
int d, n, m;
int x[MAXM], p[MAXM], id[MAXM];
priority_queue<pair<int, int>, vector<pair<int, int> >,
               greater<pair<int, int> > >
    f;
priority_queue<pair<int, int>, vector<pair<int, int> >,
               greater<pair<int, int> > >
    dis;
bool cmp(int a, int b) { return x[a] < x[b]; }
int main() {
  scanf(\"%d%d%d\", &d, &n, &m);
  for (int i = 1; i <= m; i++) {
    scanf(\"%d%d\", &x[i], &p[i]);
    id[i] = i;
  }
  sort(id + 1, id + 1 + m, cmp);
  for (int i = 1; i <= m; i++) {
    int ii = id[i];
    dis.push(make_pair(x[ii], ii));
  }
  dis.push(make_pair(n, 0));
  dis.push(make_pair(d, 0));
  f.push(make_pair(0, 0));
  int gx = 0;
  long long gc = 0;
  while (!dis.empty()) {
    pair<int, int> td = dis.top();
    dis.pop();
    int tx = td.first;
    int tid = td.second;
    while (!f.empty() && tx > x[f.top().second] + n) {
      f.pop();
    }
    if (f.empty()) {
      printf(\"-1\");
      return 0;
    }
    gc += ((long long)(tx - gx)) * ((long long)(f.top().first));
    if (tx == d) {
      break;
    }
    gx = tx;
    if (tid) {
      dis.push(make_pair(x[tid] + n, 0));
      f.push(make_pair(p[tid], tid));
    }
  }
  printf(\"%lld\", gc);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import heapq
import random
import re

def compute_min_cost(d, n, stations):
    stations = sorted(stations, key=lambda x: x[0])
    # Add start and end points
    stations = [(0, 0)] + stations + [(d, 0)]
    heap = []
    total_cost = 0
    current_fuel = n  # 初始满油
    prev_pos = 0
    
    for i in range(1, len(stations)):
        current_pos, price = stations[i]
        distance = current_pos - prev_pos
        
        # Consume fuel for this distance
        current_fuel -= distance
        
        # Need to refuel if current_fuel < 0
        while current_fuel < 0:
            if not heap:
                return -1
            # Get cheapest fuel station
            p, pos = heapq.heappop(heap)
            # Calculate maximum fuel can be taken from this station
            max_refuel = min(-current_fuel, n - (prev_pos - pos))
            total_cost += max_refuel * p
            current_fuel += max_refuel
        
        if current_fuel < 0:
            return -1
        
        # Add current station to heap
        if i < len(stations)-1:  # 终点不加入堆
            heapq.heappush(heap, (price, current_pos))
        prev_pos = current_pos
    
    return total_cost

class Epackagedeliverybootcamp(Basebootcamp):
    def __init__(self, max_d=1000, max_n=100, max_m=100):
        self.max_d = max_d
        self.max_n = max_n
        self.max_m = max_m
    
    def case_generator(self):
        while True:
            d = random.randint(1, self.max_d)
            n = random.randint(1, d)
            m = random.randint(1, min(self.max_m, d-1))
            
            # Generate valid stations
            valid = False
            stations = []
            for _ in range(3):  # 最多尝试三次生成有效用例
                xi = []
                while len(xi) < m:
                    x = random.randint(1, d-1)
                    if x not in xi:
                        xi.append(x)
                xi.sort()
                pi = [random.randint(1, 1000) for _ in range(m)]
                stations = list(zip(xi, pi))
                
                # 检查有效性
                valid = True
                prev = 0
                for x, _ in stations:
                    if x - prev > n:
                        valid = False
                        break
                    prev = x
                if d - prev > n:
                    valid = False
                if valid:
                    break
            
            # 计算正确答案
            try:
                if not valid:
                    correct_output = -1
                else:
                    correct_output = compute_min_cost(d, n, stations)
                    # 交叉验证
                    sorted_stations = sorted(stations, key=lambda x:x[0])
                    prev = 0
                    for x, _ in sorted_stations:
                        if x - prev > n:
                            correct_output = -1
                            break
                        prev = x
                    if d - prev > n:
                        correct_output = -1
            except:
                correct_output = -1
            
            return {
                'd': d,
                'n': n,
                'm': m,
                'stations': stations,
                'correct_output': correct_output
            }
    
    @staticmethod
    def prompt_func(question_case):
        stations = sorted(question_case['stations'], key=lambda x: x[0])
        stations_str = '\n'.join([f"{x} {p}" for x, p in stations])
        return f"""Johnny需要从位置0驾驶到{question_case['d']}。卡车油箱容量为{question_case['n']}升，初始满油，每单位距离消耗1升。沿途的加油站坐标为：
{stations_str}

请计算完成运输的最低燃料成本，无法到达时输出-1。答案请包含在[answer]和[/answer]标记中。"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](-?\d+)\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_output']
