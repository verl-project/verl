"""# 

### 谜题描述
Edo has got a collection of n refrigerator magnets!

He decided to buy a refrigerator and hang the magnets on the door. The shop can make the refrigerator with any size of the door that meets the following restrictions: the refrigerator door must be rectangle, and both the length and the width of the door must be positive integers.

Edo figured out how he wants to place the magnets on the refrigerator. He introduced a system of coordinates on the plane, where each magnet is represented as a rectangle with sides parallel to the coordinate axes.

Now he wants to remove no more than k magnets (he may choose to keep all of them) and attach all remaining magnets to the refrigerator door, and the area of ​​the door should be as small as possible. A magnet is considered to be attached to the refrigerator door if its center lies on the door or on its boundary. The relative positions of all the remaining magnets must correspond to the plan.

Let us explain the last two sentences. Let's suppose we want to hang two magnets on the refrigerator. If the magnet in the plan has coordinates of the lower left corner (x1, y1) and the upper right corner (x2, y2), then its center is located at (<image>, <image>) (may not be integers). By saying the relative position should correspond to the plan we mean that the only available operation is translation, i.e. the vector connecting the centers of two magnets in the original plan, must be equal to the vector connecting the centers of these two magnets on the refrigerator.

The sides of the refrigerator door must also be parallel to coordinate axes.

Input

The first line contains two integers n and k (1 ≤ n ≤ 100 000, 0 ≤ k ≤ min(10, n - 1)) — the number of magnets that Edo has and the maximum number of magnets Edo may not place on the refrigerator.

Next n lines describe the initial plan of placing magnets. Each line contains four integers x1, y1, x2, y2 (1 ≤ x1 < x2 ≤ 109, 1 ≤ y1 < y2 ≤ 109) — the coordinates of the lower left and upper right corners of the current magnet. The magnets can partially overlap or even fully coincide.

Output

Print a single integer — the minimum area of the door of refrigerator, which can be used to place at least n - k magnets, preserving the relative positions. 

Examples

Input

3 1
1 1 2 2
2 2 3 3
3 3 4 4


Output

1


Input

4 1
1 1 2 2
1 9 2 10
9 9 10 10
9 1 10 2


Output

64


Input

3 0
1 1 2 2
1 1 1000000000 1000000000
1 3 8 12


Output

249999999000000001

Note

In the first test sample it is optimal to remove either the first or the third magnet. If we remove the first magnet, the centers of two others will lie at points (2.5, 2.5) and (3.5, 3.5). Thus, it is enough to buy a fridge with door width 1 and door height 1, the area of the door also equals one, correspondingly.

In the second test sample it doesn't matter which magnet to remove, the answer will not change — we need a fridge with door width 8 and door height 8.

In the third sample you cannot remove anything as k = 0.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
using ll = long long;
using pll = pair<ll, ll>;
int main() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  int n, k;
  cin >> n >> k;
  vector<pll> xs, ys;
  for (int i = 0; i < n; ++i) {
    ll a, b, c, d;
    cin >> a >> b >> c >> d;
    xs.emplace_back(a + c, i);
    ys.emplace_back(b + d, i);
  }
  sort((xs).begin(), (xs).end());
  sort((ys).begin(), (ys).end());
  long long ans = -1;
  for (int a = 0; a <= k; ++a) {
    for (int b = 0; b <= k; ++b) {
      for (int c = 0; c <= k; ++c) {
        for (int d = 0; d <= k; ++d) {
          set<int> rem;
          for (int i = 0; i < a; ++i) rem.insert(xs[i].second);
          for (int i = 0; i < b; ++i) rem.insert(xs[n - 1 - i].second);
          for (int i = 0; i < c; ++i) rem.insert(ys[i].second);
          for (int i = 0; i < d; ++i) rem.insert(ys[n - 1 - i].second);
          if ((int)rem.size() > k) continue;
          ll w = (xs[n - 1 - b].first - xs[a].first + 1LL) / 2LL;
          ll h = (ys[n - 1 - d].first - ys[c].first + 1LL) / 2LL;
          ans = min(ans, max(1ll, w) * max(1ll, h));
          if (ans == -1) ans = max(1ll, w) * max(1ll, h);
        }
      }
    }
  }
  cout << ans << '\n';
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from typing import Dict, List, Optional, Tuple

class Cedoandmagnetsbootcamp(Basebootcamp):
    def __init__(self, max_n: int = 15, max_k: int = 5, coord_range: Tuple[int, int] = (1, 1e4)):
        self.max_n = max_n
        self.max_k = max_k
        self.coord_range = coord_range

    def case_generator(self) -> Dict:
        n = random.randint(2, self.max_n)  # 保证至少两个磁铁以便移除操作有意义
        k = random.randint(0, min(self.max_k, n-1))
        
        # 生成核心簇和干扰磁铁
        core_magnets = self._generate_core_cluster(n - k)
        noise_magnets = self._generate_noise_cluster(k)
        magnets = core_magnets + noise_magnets
        random.shuffle(magnets)
        
        # 计算正确答案
        try:
            correct_answer = self.compute_min_area(n, k, magnets)
        except Exception as e:
            raise RuntimeError("Error computing minimal area") from e
        
        return {
            "n": n,
            "k": k,
            "magnets": magnets,
            "correct_answer": correct_answer
        }

    def _generate_core_cluster(self, size: int) -> List[Tuple[int, int, int, int]]:
        """生成中心点紧密分布的磁铁"""
        base_x = random.randint(*self.coord_range)
        base_y = random.randint(*self.coord_range)
        cluster = []
        for _ in range(size):
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            x_center = base_x + dx
            y_center = base_y + dy
            # 生成有效坐标对
            x1 = x_center - random.randint(1, 3)
            x2 = x_center + random.randint(1, 3)
            y1 = y_center - random.randint(1, 3)
            y2 = y_center + random.randint(1, 3)
            cluster.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
        return cluster

    def _generate_noise_cluster(self, size: int) -> List[Tuple[int, int, int, int]]:
        """生成干扰磁铁"""
        noise = []
        for _ in range(size):
            x1 = random.randint(*self.coord_range)
            x2 = x1 + random.randint(1, 100)
            y1 = random.randint(*self.coord_range)
            y2 = y1 + random.randint(1, 100)
            noise.append((x1, y1, x2, y2))
        return noise

    @staticmethod
    def prompt_func(question_case: Dict) -> str:
        problem = (
            "Arrange refrigerator magnets by removing up to k magnets while preserving relative positions.\n"
            "Rules:\n"
            "1. The refrigerator door must be a rectangle with integer side lengths\n"
            "2. Magnet centers must lie on the door\n"
            "3. Relative positions must be preserved\n\n"
            f"Input:\n{question_case['n']} {question_case['k']}\n"
        )
        for mag in question_case["magnets"]:
            problem += f"{mag[0]} {mag[1]} {mag[2]} {mag[3]}\n"
        problem += "\nOutput the minimal area as an integer within [answer]...[/answer]."
        return problem

    @staticmethod
    def extract_output(output: str) -> Optional[int]:
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except (ValueError, TypeError):
            return None

    @classmethod
    def _verify_correction(cls, solution: int, identity: Dict) -> bool:
        return solution == identity["correct_answer"]

    @staticmethod
    def compute_min_area(n: int, k: int, magnets: List[Tuple[int, int, int, int]]) -> int:
        xs = [(x1 + x2, i) for i, (x1, y1, x2, y2) in enumerate(magnets)]
        ys = [(y1 + y2, i) for i, (x1, y1, x2, y2) in enumerate(magnets)]
        xs.sort()
        ys.sort()
        min_area = float('inf')

        for a in range(k+1):
            for b in range(k+1 - a):
                for c in range(k+1 - a - b):
                    d = k - a - b - c
                    if d < 0: continue

                    removed = set()
                    # Collect removed indices
                    for i in range(a): removed.add(xs[i][1])
                    for i in range(b): removed.add(xs[-1-i][1])
                    for i in range(c): removed.add(ys[i][1])
                    for i in range(d): removed.add(ys[-1-i][1])
                    
                    if len(removed) > k: continue
                    
                    # Calculate x range
                    x_values = [xi for xi, idx in xs if idx not in removed]
                    if not x_values: continue
                    x_min = x_values[0]
                    x_max = x_values[-1]
                    
                    # Calculate y range
                    y_values = [yi for yi, idx in ys if idx not in removed]
                    if not y_values: continue
                    y_min = y_values[0]
                    y_max = y_values[-1]
                    
                    width = max(1, (x_max - x_min + 1) // 2)
                    height = max(1, (y_max - y_min + 1) // 2)
                    min_area = min(min_area, width * height)
        return min_area if min_area != float('inf') else 0
