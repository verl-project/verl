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
inline int re() {
  int x = 0, f = 1;
  char ch = getchar();
  while (ch < '0' || ch > '9') {
    if (ch == '-') f = -1;
    ch = getchar();
  }
  while (ch >= '0' && ch <= '9') {
    x = (x << 3) + (x << 1) + (ch ^ 48);
    ch = getchar();
  }
  return x * f;
}
long long gcd(long long a, long long b) {
  long long r;
  while (b) {
    r = a % b;
    a = b;
    b = r;
  }
  return a;
}
long long kk(long long a, long long b) {
  long long sum = 1;
  while (b) {
    if (b) sum = sum * a;
    a = a * a;
    b >>= 1;
  }
  return sum;
}
int ly[300030 << 2], tree[300030 << 2], a[300030];
void pdd(int l, int r, int tr) {
  if (ly[tr]) {
    ly[2 * tr] += ly[tr];
    ly[2 * tr + 1] += ly[tr];
    int mid = (l + r) >> 1;
    tree[2 * tr] += (mid - l + 1) * ly[tr];
    tree[2 * tr + 1] += (r - mid) * ly[tr];
    ly[tr] = 0;
  }
}
void pup(int tr) { tree[tr] = tree[2 * tr] + tree[2 * tr + 1]; }
void upd(int l, int r, int L, int R, int tr) {
  if (L <= l && r <= R) {
    ly[tr] += 1;
    tree[tr] += (r - l + 1);
    return;
  }
  pdd(l, r, tr);
  int mid = (l + r) >> 1;
  if (L <= mid) upd(l, mid, L, R, 2 * tr);
  if (R > mid) upd(mid + 1, r, L, R, 2 * tr + 1);
  pup(tr);
}
int qry(int l, int r, int pos, int tr) {
  if (l == r) return tree[tr];
  pdd(l, r, tr);
  int mid = (l + r) >> 1;
  if (pos <= mid)
    return qry(l, mid, pos, 2 * tr);
  else
    return qry(mid + 1, r, pos, 2 * tr + 1);
}
vector<pair<int, int>> v[300030];
int e[300030];
int main() {
  int n = re(), q = re();
  for (int i = 1; i <= n; i++) {
    a[i] = re();
    a[i] = i - a[i];
    if (a[i] < 0) a[i] = 0x3f3f3f3f;
  }
  for (int i = 1; i <= q; i++) {
    int u = re(), r = re();
    r = n - r;
    v[r].push_back({u, i});
  }
  for (int i = 1; i <= n; i++) {
    int l = 1, r = i;
    int ans = -1;
    while (l <= r) {
      int mid = (l + r) >> 1;
      if (qry(1, n, mid, 1) >= a[i]) {
        ans = mid;
        l = mid + 1;
      } else
        r = mid - 1;
    }
    if (ans != -1) {
      upd(1, n, 1, ans, 1);
    }
    for (int j = 0; j < v[i].size(); j++) {
      int u = v[i][j].first;
      e[v[i][j].second] = qry(1, n, u + 1, 1);
    }
  }
  for (int i = 1; i <= q; i++) cout << e[i] << endl;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Efixedpointremovalbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = params
        self.n = params.get('n', 5)
        self.q = params.get('q', 2)
        self.seed = params.get('seed', None)
        if self.seed is not None:
            random.seed(self.seed)

    def case_generator(self):
        n = self.n
        q = self.q

        a = [random.randint(1, n) for _ in range(n)]
        
        queries = []
        for _ in range(q):
            while True:
                x = random.randint(0, n - 1)
                max_y = n - 1 - x
                if max_y < 0:
                    continue
                y = random.randint(0, max_y)
                if x + y < n:
                    queries.append((x, y))
                    break
        
        answers = self.solve_case(n, q, a, queries)
        
        return {
            'n': n,
            'q': q,
            'a': a,
            'queries': queries,
            'answers': answers
        }

    @staticmethod
    def solve_case(n, q, a_list, queries):
        a = [0] * (n + 2)
        for i in range(1, n + 1):
            ai = a_list[i - 1]
            a_i_val = i - ai
            a[i] = a_i_val if a_i_val >= 0 else float('inf')
        
        v = [[] for _ in range(n + 2)]
        for idx, (x, y) in enumerate(queries):
            r = n - y
            v[r].append((x, idx))
        
        st = Efixedpointremovalbootcamp.SegmentTree(n)
        answers = [0] * len(queries)
        
        for i in range(1, n + 1):
            current_a = a[i]
            if current_a == float('inf'):
                ans = -1
            else:
                l, r, ans = 1, i, -1
                while l <= r:
                    mid = (l + r) // 2
                    val = st.point_query(mid)
                    if val >= current_a:
                        ans = mid
                        l = mid + 1
                    else:
                        r = mid - 1
            if ans != -1:
                st.range_add(1, ans, 1)
            
            for (x, query_idx) in v[i]:
                pos = x + 1
                val = st.point_query(pos) if pos <= n else 0
                answers[query_idx] = val
        
        return answers

    class SegmentTree:
        def __init__(self, size):
            self.n = size
            self.tree = [0] * (4 * size)
            self.lazy = [0] * (4 * size)

        def _push_down(self, node, l, r):
            if self.lazy[node]:
                mid = (l + r) // 2
                left, right = 2 * node, 2 * node + 1
                self.tree[left] += self.lazy[node] * (mid - l + 1)
                self.lazy[left] += self.lazy[node]
                self.tree[right] += self.lazy[node] * (r - mid)
                self.lazy[right] += self.lazy[node]
                self.lazy[node] = 0

        def _range_add(self, node, l, r, L, R, val):
            if R < l or L > r:
                return
            if L <= l and r <= R:
                self.tree[node] += val * (r - l + 1)
                self.lazy[node] += val
                return
            self._push_down(node, l, r)
            mid = (l + r) // 2
            self._range_add(2*node, l, mid, L, R, val)
            self._range_add(2*node+1, mid+1, r, L, R, val)
            self.tree[node] = self.tree[2*node] + self.tree[2*node+1]

        def _point_query(self, node, l, r, pos):
            if l == r:
                return self.tree[node]
            self._push_down(node, l, r)
            mid = (l + r) // 2
            if pos <= mid:
                return self._point_query(2*node, l, mid, pos)
            else:
                return self._point_query(2*node+1, mid+1, r, pos)

        def range_add(self, L, R, val):
            self._range_add(1, 1, self.n, L, R, val)

        def point_query(self, pos):
            return self._point_query(1, 1, self.n, pos)

    @staticmethod
    def prompt_func(question_case) -> str:
        problem_desc = """\
给定一个由n个正整数构成的数组a。每个查询要求将前x个元素和后y个元素替换为n+1（无法移除），计算处理后数组的最大可移除元素数目。

规则：
1. 每次操作可以选择一个索引i，使得当前a_i == i，并移除该元素。
2. 移除后数组的后续元素索引依次前移。
3. 权重是最多能移除的元素数量。

输入格式：
- 第1行：n 和 q（数组长度，查询数量）
- 第2行：a_1 a_2 ... a_n
- 接下来q行：每行两个整数x和y

输出格式：
- 每个查询的答案占一行，置于[answer]和[/answer]标签内。

示例输入：
5 2
1 4 1 2 4
0 0
1 0

示例输出：
2
0
"""
        current_input = f"{question_case['n']} {question_case['q']}\n"
        current_input += ' '.join(map(str, question_case['a'])) + '\n'
        current_input += '\n'.join(f"{x} {y}" for x, y in question_case['queries'])
        prompt = f"{problem_desc}\n当前输入数据为：\n{current_input}\n请输出每个查询的结果，每行一个，放在[answer]标签内。"
        return prompt

    @staticmethod
    def extract_output(output):
        import re
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        last_block = answer_blocks[-1].strip()
        answers = []
        for line in last_block.split('\n'):
            line = line.strip()
            if line:
                try:
                    answers.append(int(line))
                except ValueError:
                    continue
        return answers if answers else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['answers']
