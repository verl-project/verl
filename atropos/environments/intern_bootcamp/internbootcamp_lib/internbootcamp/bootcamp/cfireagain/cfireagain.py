"""# 

### 谜题描述
After a terrifying forest fire in Berland a forest rebirth program was carried out. Due to it N rows with M trees each were planted and the rows were so neat that one could map it on a system of coordinates so that the j-th tree in the i-th row would have the coordinates of (i, j). However a terrible thing happened and the young forest caught fire. Now we must find the coordinates of the tree that will catch fire last to plan evacuation.

The burning began in K points simultaneously, which means that initially K trees started to burn. Every minute the fire gets from the burning trees to the ones that aren’t burning and that the distance from them to the nearest burning tree equals to 1.

Find the tree that will be the last to start burning. If there are several such trees, output any.

Input

The first input line contains two integers N, M (1 ≤ N, M ≤ 2000) — the size of the forest. The trees were planted in all points of the (x, y) (1 ≤ x ≤ N, 1 ≤ y ≤ M) type, x and y are integers.

The second line contains an integer K (1 ≤ K ≤ 10) — amount of trees, burning in the beginning. 

The third line contains K pairs of integers: x1, y1, x2, y2, ..., xk, yk (1 ≤ xi ≤ N, 1 ≤ yi ≤ M) — coordinates of the points from which the fire started. It is guaranteed that no two points coincide.

Output

Output a line with two space-separated integers x and y — coordinates of the tree that will be the last one to start burning. If there are several such trees, output any.

Examples

Input

3 3
1
2 2


Output

1 1


Input

3 3
1
1 1


Output

3 3


Input

3 3
2
1 1 3 3


Output

2 2

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
inline long long max_(long long a, long long b) { return (a > b) ? a : b; }
inline long long min_(long long a, long long b) { return (a < b) ? a : b; }
int main() {
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  if (fopen(\"input.txt\", \"r\")) {
    freopen(\"input.txt\", \"r\", stdin);
    freopen(\"output.txt\", \"w\", stdout);
  }
  long long n, m;
  cin >> n >> m;
  long long k, i, x, y;
  cin >> k;
  pair<long long, long long> a[k];
  for (i = 0; i < k; i++) {
    cin >> a[i].first >> a[i].second;
  }
  long long ans = -1, j, l;
  for (i = 1; i <= n; i++)
    for (j = 1; j <= m; j++) {
      long long tmp = 4000;
      for (l = 0; l < k; l++)
        tmp = min_(tmp, abs(i - a[l].first) + abs(j - a[l].second));
      if (ans < tmp) {
        ans = tmp;
        x = i, y = j;
      }
    }
  cout << x << \" \" << y;
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cfireagainbootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=2000, m_min=1, m_max=2000, k_min=1, k_max=10):
        self.n_min = n_min
        self.n_max = n_max
        self.m_min = m_min
        self.m_max = m_max
        self.k_min = k_min
        self.k_max = k_max
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        m = random.randint(self.m_min, self.m_max)
        max_possible_k = n * m
        k = random.randint(self.k_min, min(self.k_max, max_possible_k))
        
        initial_points = []
        existing = set()
        while len(initial_points) < k:
            x = random.randint(1, n)
            y = random.randint(1, m)
            if (x, y) not in existing:
                existing.add((x, y))
                initial_points.append((x, y))
        
        # 计算理论最大距离点
        candidate_points = []
        for x in [1, n]:
            for y in [1, m]:
                candidate_points.append((x, y))
        
        # 寻找候选点中的最优解
        d_max = 0
        optimal_points = []
        for i, j in candidate_points:
            min_dist = min(abs(i - x) + abs(j - y) for x, y in initial_points)
            if min_dist > d_max:
                d_max = min_dist
                optimal_points = [(i, j)]
            elif min_dist == d_max:
                optimal_points.append((i, j))
        
        # 确保至少有一个候选点
        if not optimal_points:
            optimal_points = [(random.randint(1,n), random.randint(1,m))]
        
        expected = random.choice(optimal_points)
        
        return {
            'n': n,
            'm': m,
            'k': k,
            'initial_points': initial_points,
            'expected': expected
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        coords = ' '.join(f"{x} {y}" for x, y in question_case['initial_points'])
        
        problem = f"""在一次森林火灾中，{n}行{m}列的树木网格（坐标从(1,1)到({n},{m})）有{len(question_case['initial_points'])}个初始火源，坐标分别为：{coords}。火势每分钟向曼哈顿距离为1的相邻树木扩散。请找出最后被点燃的树木坐标，如果有多个答案可任意选择一个。

答案格式要求：将两个整数用空格分隔放在[answer]标签内，例如：[answer]3 4[/answer]"""
        return problem
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            x, y = map(int, last_match.split())
            return (x, y)
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not isinstance(solution, tuple) or len(solution) != 2:
            return False
        return solution == identity['expected']
