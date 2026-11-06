"""# 

### 谜题描述
Let a_1, …, a_n be an array of n positive integers. In one operation, you can choose an index i such that a_i = i, and remove a_i from the array (after the removal, the remaining parts are concatenated).

The weight of a is defined as the maximum number of elements you can remove.

You must answer q independent queries (x, y): after replacing the x first elements of a and the y last elements of a by n+1 (making them impossible to remove), what would be the weight of a?

Input

The first line contains two integers n and q (1 ≤ n, q ≤ 3 ⋅ 10^5) — the length of the array and the number of queries.

The second line contains n integers a_1, a_2, ..., a_n (1 ≤ a_i ≤ n) — elements of the array.

The i-th of the next q lines contains two integers x and y (x, y ≥ 0 and x+y < n).

Output

Print q lines, i-th line should contain a single integer — the answer to the i-th query.

Examples

Input


13 5
2 2 3 9 5 4 6 5 7 8 3 11 13
3 1
0 0
2 4
5 0
0 12


Output


5
11
6
1
0


Input


5 2
1 4 1 2 4
0 0
1 0


Output


2
0

Note

Explanation of the first query:

After making first x = 3 and last y = 1 elements impossible to remove, a becomes [×, ×, ×, 9, 5, 4, 6, 5, 7, 8, 3, 11, ×] (we represent 14 as × for clarity).

Here is a strategy that removes 5 elements (the element removed is colored in red):

  * [×, ×, ×, 9, \color{red}{5}, 4, 6, 5, 7, 8, 3, 11, ×] 
  * [×, ×, ×, 9, 4, 6, 5, 7, 8, 3, \color{red}{11}, ×] 
  * [×, ×, ×, 9, 4, \color{red}{6}, 5, 7, 8, 3, ×] 
  * [×, ×, ×, 9, 4, 5, 7, \color{red}{8}, 3, ×] 
  * [×, ×, ×, 9, 4, 5, \color{red}{7}, 3, ×] 
  * [×, ×, ×, 9, 4, 5, 3, ×] (final state) 



It is impossible to remove more than 5 elements, hence the weight is 5.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int N = 3e5 + 1;
int n, q, a[N], ans[N], it[4 * N], b[N];
vector<pair<pair<int, int>, int>> v;
bool cmp(pair<pair<int, int>, int> x, pair<pair<int, int>, int> y) {
  return (x.first.second < y.first.second);
}
void upd(int id, int l, int r, int pos, int val) {
  if (pos < l || r < pos) return;
  if (l == r) {
    it[id] += val;
    return;
  }
  int mid = (l + r) / 2;
  upd(2 * id, l, mid, pos, val);
  upd(2 * id + 1, mid + 1, r, pos, val);
  it[id] = it[2 * id] + it[2 * id + 1];
}
int get_pos(int id, int l, int r, int pos) {
  if (l == r) return l;
  int mid = (l + r) / 2;
  if (it[2 * id] >= pos)
    return get_pos(2 * id, l, mid, pos);
  else
    return get_pos(2 * id + 1, mid + 1, r, pos - it[2 * id]);
}
int get_sum(int id, int l, int r, int l1, int r1) {
  if (r1 < l || r < l1) return 0;
  if (l1 <= l && r <= r1) return it[id];
  int mid = (l + r) / 2;
  return get_sum(2 * id, l, mid, l1, r1) +
         get_sum(2 * id + 1, mid + 1, r, l1, r1);
}
int main() {
  ios_base::sync_with_stdio(false);
  cin.tie();
  cin >> n >> q;
  for (int i = 1; i <= n; i++) {
    cin >> a[i];
  }
  for (int i = 1; i <= q; i++) {
    int l, r;
    cin >> l >> r;
    v.push_back({{l, r}, i});
  }
  sort(v.begin(), v.end(), cmp);
  int cnt = 0;
  for (int i = 1; i <= n; i++) {
    if (a[i] > i) continue;
    int tmp = i - a[i];
    if (tmp <= cnt) {
      if (tmp == 0) {
        b[i] = i;
        upd(1, 1, n, i, 1);
      } else {
        tmp = cnt - tmp + 1;
        b[i] = get_pos(1, 1, n, tmp);
        upd(1, 1, n, b[i], 1);
      }
      cnt++;
    }
  }
  int cur = n + 1;
  for (int i = 0; i < q; i++) {
    int tmp = n - v[i].first.second + 1;
    for (int j = tmp; j < cur; j++) {
      if (b[j] == 0) continue;
      upd(1, 1, n, b[j], -1);
    }
    ans[v[i].second] = get_sum(1, 1, n, v[i].first.first + 1, n);
    cur = tmp;
  }
  for (int i = 1; i <= q; i++) cout << ans[i] << endl;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
from bisect import bisect_right

class Cfixedpointremovalbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n_min = params.get('n_min', 5)
        self.n_max = params.get('n_max', 15)
        self.q_max = params.get('q_max', 5)
        self.seed = params.get('seed', None)
        if self.seed is not None:
            random.seed(self.seed)
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        a = [random.randint(1, n) for _ in range(n)]
        
        # 预计算有效元素位置（改进算法）
        valid = []
        for i in range(n):
            pos = i + 1  # 1-based index
            if a[i] == pos:
                valid.append(pos)
            elif a[i] < pos:
                required = pos - a[i]
                if required <= len(valid):
                    start = len(valid) - required
                    if all(valid[j] >= a[i] for j in range(start, len(valid))):
                        valid = valid[:start] + [pos]

        # 生成查询并计算答案
        q = random.randint(1, self.q_max)
        queries = []
        answers = []
        for _ in range(q):
            while True:
                x = random.randint(0, n//2)
                y = random.randint(0, (n-1)-x)
                if x + y < n:
                    break
            
            l = x + 1
            r = n - y
            # 二分查找有效范围内元素数量
            left = bisect_right(valid, l-1)
            right_idx = bisect_right(valid, r)
            answers.append(right_idx - left)
            queries.append((x, y))

        return {
            'n': n,
            'q': q,
            'a': a,
            'valid': valid,
            'queries': queries,
            'answers': answers
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        q = question_case['q']
        a = question_case['a']
        queries = question_case['queries']
        return f"""## 数组权重谜题

给定长度为{n}的数组：{a}

**规则**：
1. 每次移除一个元素a_i（满足a_i = 当前索引）
2. 移除后数组重新索引
3. 权重是最大可移除元素数量

**查询操作**：
对于每个查询(x,y):
- 前x个元素设置为{n+1}
- 后y个元素设置为{n+1}
- 计算修改后的数组权重

**输入格式**：
{q}个查询，每个查询x和y满足x+y<{n}

**输出格式**：
将每个查询的答案按顺序放在[answer]标签内，每个答案占一行：

示例：
[answer]
3
0
5
[/answer]"""
    
    @staticmethod
    def extract_output(output):
        import re
        answer_block = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not answer_block:
            return None
        answers = []
        for line in answer_block[-1].strip().split('\n'):
            line = line.strip()
            if line and line.isdigit():
                answers.append(int(line))
        return answers if answers else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity['answers']
        return isinstance(solution, list) and solution == expected
