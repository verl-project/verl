"""# 

### 谜题描述
Ilya is sitting in a waiting area of Metropolis airport and is bored of looking at time table that shows again and again that his plane is delayed. So he took out a sheet of paper and decided to solve some problems.

First Ilya has drawn a grid of size n × n and marked n squares on it, such that no two marked squares share the same row or the same column. He calls a rectangle on a grid with sides parallel to grid sides beautiful if exactly two of its corner squares are marked. There are exactly n·(n - 1) / 2 beautiful rectangles.

Ilya has chosen q query rectangles on a grid with sides parallel to grid sides (not necessarily beautiful ones), and for each of those rectangles he wants to find its beauty degree. Beauty degree of a rectangle is the number of beautiful rectangles that share at least one square with the given one.

Now Ilya thinks that he might not have enough time to solve the problem till the departure of his flight. You are given the description of marked cells and the query rectangles, help Ilya find the beauty degree of each of the query rectangles.

Input

The first line of input contains two integers n and q (2 ≤ n ≤ 200 000, 1 ≤ q ≤ 200 000) — the size of the grid and the number of query rectangles.

The second line contains n integers p1, p2, ..., pn, separated by spaces (1 ≤ pi ≤ n, all pi are different), they specify grid squares marked by Ilya: in column i he has marked a square at row pi, rows are numbered from 1 to n, bottom to top, columns are numbered from 1 to n, left to right.

The following q lines describe query rectangles. Each rectangle is described by four integers: l, d, r, u (1 ≤ l ≤ r ≤ n, 1 ≤ d ≤ u ≤ n), here l and r are the leftmost and the rightmost columns of the rectangle, d and u the bottommost and the topmost rows of the rectangle.

Output

For each query rectangle output its beauty degree on a separate line.

Examples

Input

2 3
1 2
1 1 1 1
1 1 1 2
1 1 2 2


Output

1
1
1


Input

4 2
1 3 2 4
4 1 4 4
1 1 2 3


Output

3
5

Note

The first sample test has one beautiful rectangle that occupies the whole grid, therefore the answer to any query is 1.

In the second sample test the first query rectangle intersects 3 beautiful rectangles, as shown on the picture below:

<image> <image> <image>

There are 5 beautiful rectangles that intersect the second query rectangle, as shown on the following picture:

<image> <image> <image> <image> <image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int MAXN = int(2e5) + 10;
const int MOD = int(1e9) + 7;
const int oo = INT_MAX;
struct Query {
  int d, u, id;
};
int n, q;
int p[MAXN];
long long ans[MAXN][10];
vector<Query> open[MAXN], close[MAXN];
int bit[MAXN];
void update(int i) {
  for (i = MAXN - i; i < MAXN; i += i & -i) bit[i]++;
}
int query(int i) {
  int sum = 0;
  for (i = MAXN - i; i > 0; i -= i & -i) sum += bit[i];
  return sum;
}
long long gauss(long long x) {
  if (x == 0) return 0;
  return x * (x - 1) / 2LL;
}
int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  while (cin >> n >> q) {
    for (int i = 1; i <= n; ++i) {
      cin >> p[i];
      open[i].clear();
      close[i].clear();
    }
    for (int i = 0; i < q; ++i) {
      int l, d, r, u;
      cin >> l >> d >> r >> u;
      open[l].push_back({d, u, i});
      close[r].push_back({d, u, i});
    }
    memset(ans, 0, sizeof(ans));
    memset(bit, 0, sizeof(bit));
    for (int i = n; i > 0; --i) {
      for (auto& e : close[i]) {
        ans[e.id][3] = query(e.u + 1);
        ans[e.id][6] = query(e.d) - ans[e.id][3];
        ans[e.id][9] = query(1) - ans[e.id][6] - ans[e.id][3];
      }
      update(p[i]);
      for (auto& e : open[i]) {
        ans[e.id][2] = query(e.u + 1) - ans[e.id][3];
        ans[e.id][5] = query(e.d) - ans[e.id][2] - ans[e.id][3] - ans[e.id][6];
        ans[e.id][8] = query(1) - ans[e.id][5] - ans[e.id][6] - ans[e.id][2] -
                       ans[e.id][3] - ans[e.id][9];
      }
    }
    memset(bit, 0, sizeof(bit));
    for (int i = 1; i <= n; ++i) {
      for (auto& e : open[i]) {
        ans[e.id][1] = query(e.u + 1);
        ans[e.id][4] = query(e.d) - ans[e.id][1];
        ans[e.id][7] = query(1) - ans[e.id][4] - ans[e.id][1];
      }
      update(p[i]);
    }
    for (int i = 0; i < q; ++i) {
      long long r = 0;
      r += ans[i][1] * (ans[i][6] + ans[i][8] + ans[i][9]);
      r += ans[i][2] *
           (ans[i][4] + ans[i][6] + ans[i][7] + ans[i][8] + ans[i][9]);
      r += ans[i][3] * (ans[i][4] + ans[i][7] + ans[i][8]);
      r += ans[i][4] * (ans[i][6] + ans[i][8] + ans[i][9]);
      r += ans[i][6] * (ans[i][7] + ans[i][8]);
      r += ans[i][5] * (n - ans[i][5]);
      r += gauss(ans[i][5]);
      cout << r << '\n';
    }
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import defaultdict
from bisect import bisect_left, bisect_right
from bootcamp import Basebootcamp

class Eboredombootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', 4)
        self.q = params.get('q', 2)
        self.max_limit = min(params.get('max_limit', 200000), 200000)  # 强制约束题目限制
        
        # 参数有效性验证
        if self.n < 2:
            self.n = 2
        if self.q < 1:
            self.q = 1

    def case_generator(self):
        """生成完全符合题目约束的有效案例"""
        n = min(max(self.n, 2), self.max_limit)  # 确保n≥2
        q = min(max(self.q, 1), self.max_limit)  # 确保q≥1
        
        # 生成合法排列
        p = list(range(1, n+1))
        random.shuffle(p)
        
        # 生成合法查询（保证1 ≤ l ≤ r ≤ n，1 ≤ d ≤ u ≤ n）
        queries = []
        for _ in range(q):
            l = random.randint(1, n)
            r = random.randint(l, n)
            d = random.randint(1, n)
            u = random.randint(d, n)
            queries.append((l, d, r, u))
        
        # 使用优化后的正确算法计算答案
        try:
            correct_answers = self._calculate_correct_answers(n, p, queries)
        except Exception as e:
            raise RuntimeError(f"Answer calculation failed: {str(e)}")
        
        return {
            'n': n,
            'q': q,
            'p': p.copy(),
            'queries': queries.copy(),
            'correct_answers': correct_answers.copy()
        }

    def _calculate_correct_answers(self, n, p, queries):
        """精确实现官方题解的扫描线算法"""
        # 初始化数据结构
        open_events = defaultdict(list)
        close_events = defaultdict(list)
        ans = defaultdict(lambda: [0]*10)
        
        # 处理查询事件
        for idx, (l, d, r, u) in enumerate(queries):
            open_events[l].append((d, u, idx))
            close_events[r].append((d, u, idx))

        # 实现扫描线处理函数
        def process_side(events, reverse=False):
            bit = [0] * (n + 2)
            
            def update(x):
                x = n + 1 - x
                while x <= n + 1:
                    bit[x] += 1
                    x += x & -x
                    
            def query(x):
                x = n + 1 - x
                res = 0
                while x > 0:
                    res += bit[x]
                    x -= x & -x
                return res
            
            # 处理列的顺序
            cols = sorted(events.keys(), reverse=reverse)
            for col in cols:
                # 处理关闭事件
                if not reverse:
                    for d, u, idx in events[col]:
                        ans[idx][3] = query(u + 1)
                        ans[idx][6] = query(d) - ans[idx][3]
                        ans[idx][9] = query(1) - ans[idx][6] - ans[idx][3]
                
                # 更新当前列的值
                update(p[col-1])  # p是1-based索引
                
                # 处理打开事件
                if reverse:
                    for d, u, idx in events[col]:
                        ans[idx][2] = query(u + 1) - ans[idx][3]
                        ans[idx][5] = query(d) - ans[idx][2] - ans[idx][3] - ans[idx][6]
                        ans[idx][8] = query(1) - ans[idx][5] - ans[idx][6] - ans[idx][2] - ans[idx][3] - ans[idx][9]

        # 先处理右边界（反向扫描）
        process_side(close_events, reverse=True)
        
        # 再处理左边界（正向扫描）
        process_side(open_events)
        
        # 计算最终结果
        results = []
        for idx in range(len(queries)):
            a = ans[idx]
            total = 0
            total += a[1] * (a[6] + a[8] + a[9])
            total += a[2] * (a[4] + a[6] + a[7] + a[8] + a[9])
            total += a[3] * (a[4] + a[7] + a[8])
            total += a[4] * (a[6] + a[8] + a[9])
            total += a[6] * (a[7] + a[8])
            total += a[5] * (n - a[5])
            total += a[5] * (a[5] - 1) // 2
            results.append(total)
        
        return results

    @staticmethod
    def prompt_func(question_case) -> str:
        p_str = ' '.join(map(str, question_case['p']))
        queries = []
        for q in question_case['queries']:
            queries.append(f"{q[0]} {q[1]} {q[2]} {q[3]}")
        queries_str = '\n'.join(queries)
        
        prompt = f"""Ilya needs help calculating beauty degrees for query rectangles on a {question_case['n']}x{question_case['n']} grid. The marked positions (by column) are: {p_str}

Each query rectangle is defined by (l, d, r, u):
{queries_str}

A beautiful rectangle has exactly two marked corners in opposite positions. Calculate how many beautiful rectangles intersect each query rectangle. 

Provide exactly {question_case['q']} space-separated integers in the format [answer]numbers[/answer]. Example: [answer]3 5 2[/answer]"""
        return prompt

    @staticmethod
    def extract_output(output):
        # 匹配最后一个答案块，处理可能的换行和空格
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            numbers = list(map(int, matches[-1].replace('\n', ' ').split()))
            return numbers
        except (ValueError, TypeError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return solution == identity['correct_answers']
        except KeyError:
            return False
