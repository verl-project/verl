"""# 

### 谜题描述
I won't feel lonely, nor will I be sorrowful... not before everything is buried.

A string of n beads is left as the message of leaving. The beads are numbered from 1 to n from left to right, each having a shape numbered by integers between 1 and n inclusive. Some beads may have the same shapes.

The memory of a shape x in a certain subsegment of beads, is defined to be the difference between the last position and the first position that shape x appears in the segment. The memory of a subsegment is the sum of memories over all shapes that occur in it.

From time to time, shapes of beads change as well as the memories. Sometimes, the past secreted in subsegments are being recalled, and you are to find the memory for each of them.

Input

The first line of input contains two space-separated integers n and m (1 ≤ n, m ≤ 100 000) — the number of beads in the string, and the total number of changes and queries, respectively.

The second line contains n integers a1, a2, ..., an (1 ≤ ai ≤ n) — the initial shapes of beads 1, 2, ..., n, respectively.

The following m lines each describes either a change in the beads or a query of subsegment. A line has one of the following formats: 

  * 1 p x (1 ≤ p ≤ n, 1 ≤ x ≤ n), meaning that the shape of the p-th bead is changed into x; 
  * 2 l r (1 ≤ l ≤ r ≤ n), denoting a query of memory of the subsegment from l to r, inclusive. 

Output

For each query, print one line with an integer — the memory of the recalled subsegment.

Examples

Input

7 6
1 2 3 1 3 2 1
2 3 7
2 1 3
1 7 2
1 3 2
2 1 6
2 5 7


Output

5
0
7
1


Input

7 5
1 3 2 1 4 2 3
1 1 4
2 2 3
1 1 7
2 4 5
1 1 7


Output

0
0

Note

The initial string of beads has shapes (1, 2, 3, 1, 3, 2, 1).

Consider the changes and queries in their order: 

  1. 2 3 7: the memory of the subsegment [3, 7] is (7 - 4) + (6 - 6) + (5 - 3) = 5; 
  2. 2 1 3: the memory of the subsegment [1, 3] is (1 - 1) + (2 - 2) + (3 - 3) = 0; 
  3. 1 7 2: the shape of the 7-th bead changes into 2. Beads now have shapes (1, 2, 3, 1, 3, 2, 2) respectively; 
  4. 1 3 2: the shape of the 3-rd bead changes into 2. Beads now have shapes (1, 2, 2, 1, 3, 2, 2) respectively; 
  5. 2 1 6: the memory of the subsegment [1, 6] is (4 - 1) + (6 - 2) + (5 - 5) = 7; 
  6. 2 5 7: the memory of the subsegment [5, 7] is (7 - 6) + (5 - 5) = 1. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
struct fenwick_tree_1D {
  long long n, log_n;
  vector<long long> tree;
  fenwick_tree_1D(long long _n)
      : n(_n), log_n((long long)(log2(_n) + 1.0L)), tree(_n + 5) {}
  void update(long long x, long long v) {
    while (x <= n) {
      tree[x] += v;
      x += x & (-x);
    }
  }
  long long prefix_sum(long long x) {
    long long r = 0;
    while (x) {
      r += tree[x];
      x -= x & (-x);
    }
    return r;
  }
  long long range_sum(long long l, long long r) {
    return prefix_sum(r) - prefix_sum(l - 1);
  }
};
const long long bs = 300;
fenwick_tree_1D bit(100005);
long long a[100005], nxt[100005], last[100005] = {0}, dl[2 * bs + 5],
                                  id[100005], dc = 0, result[100005] = {0};
set<long long> s[100005];
long long qc;
struct query {
  long long t, l, r, qno;
} q[bs + 5];
long long q2c;
struct query2 {
  long long l, r, qno;
} q2[bs + 5];
vector<long long> ans;
int32_t main() {
  ios::sync_with_stdio(false);
  cin.tie(0);
  long long n, m;
  cin >> n >> m;
  for (long long i = 1; i <= n; i++) {
    cin >> a[i];
    s[a[i]].insert(i);
  }
  for (long long h = 0; h < m; h += bs) {
    qc = min(bs, m - h);
    for (long long j = 0; j < qc; j++) {
      cin >> q[j].t >> q[j].l >> q[j].r;
      q[j].qno = h + j;
      if (q[j].t == 1) {
        if (!id[a[q[j].l]]) dl[dc] = a[q[j].l], id[a[q[j].l]] = 1, dc++;
        if (!id[q[j].r]) dl[dc] = q[j].r, id[q[j].r] = 1, dc++;
      } else
        q2[q2c].l = q[j].l, q2[q2c].r = q[j].r, q2[q2c].qno = q[j].qno,
        ans.push_back(q[j].qno), q2c++;
    }
    for (long long i = n; i >= 1; i--) {
      if (id[a[i]]) {
        nxt[i] = 0;
        continue;
      }
      nxt[i] = last[a[i]];
      if (nxt[i]) bit.update(nxt[i], nxt[i] - i);
      last[a[i]] = i;
    }
    sort(q2, q2 + q2c,
         [](const query2 &l, const query2 &r) { return l.l < r.l; });
    long long pos = 1;
    for (long long i = 0; i < q2c; i++) {
      while (pos < q2[i].l) {
        if (id[a[pos]] || !nxt[pos]) {
          pos++;
          continue;
        }
        bit.update(nxt[pos], pos - nxt[pos]);
        pos++;
      }
      result[q2[i].qno] = bit.prefix_sum(q2[i].r);
    }
    for (long long i = 0; i < qc; i++) {
      if (q[i].t == 1) {
        s[a[q[i].l]].erase(q[i].l);
        a[q[i].l] = q[i].r;
        s[a[q[i].l]].insert(q[i].l);
      } else {
        for (long long j = 0; j < dc; j++) {
          auto it1 = s[dl[j]].lower_bound(q[i].l);
          auto it2 = s[dl[j]].upper_bound(q[i].r);
          if (it1 != s[dl[j]].end() && it2 != s[dl[j]].begin()) {
            it2--;
            if (*it2 > *it1) result[q[i].qno] += *it2 - *it1;
          }
        }
      }
    }
    for (long long i = 0; i < dc; i++) id[dl[i]] = 0;
    dc = 0;
    q2c = 0;
    for (long long i = 1; i <= n; i++) bit.tree[i] = 0, last[i] = 0;
  }
  for (auto &k : ans) cout << result[k] << \"\n\";
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

def compute_memory(a, l, r):
    positions = {}
    for i in range(l-1, r):
        x = a[i]
        if x not in positions:
            positions[x] = {'first': i+1, 'last': i+1}
        else:
            if i+1 < positions[x]['first']:
                positions[x]['first'] = i+1
            if i+1 > positions[x]['last']:
                positions[x]['last'] = i+1
    total = 0
    for x in positions:
        total += positions[x]['last'] - positions[x]['first']
    return total

class Egoodbyesouvenirbootcamp(Basebootcamp):
    def __init__(self, max_n=7, max_m=6, random_seed=None, **params):
        super().__init__(**params)
        self.max_n = max_n
        self.max_m = max_m
        self.rng = random.Random(random_seed)
    
    def case_generator(self):
        n = self.rng.randint(1, self.max_n)
        m = self.rng.randint(1, self.max_m)
        initial = [self.rng.randint(1, n) for _ in range(n)]
        current_a = initial.copy()
        operations = []
        answers = []
        
        for _ in range(m):
            if self.rng.random() < 0.3 and n > 0:
                p = self.rng.randint(1, n)
                x = self.rng.randint(1, n)
                operations.append(('1', p, x))
                current_a[p-1] = x
            else:
                l = self.rng.randint(1, n)
                r = self.rng.randint(l, n)
                operations.append(('2', l, r))
                mem = compute_memory(current_a, l, r)
                answers.append(mem)
        
        return {
            'n': n,
            'm': m,
            'initial': initial,
            'operations': operations,
            'answers': answers
        }
    
    @staticmethod
    def prompt_func(question_case):
        input_lines = [
            f"{question_case['n']} {question_case['m']}",
            ' '.join(map(str, question_case['initial']))
        ]
        for op in question_case['operations']:
            if op[0] == '1':
                input_lines.append(f"1 {op[1]} {op[2]}")
            else:
                input_lines.append(f"2 {op[1]} {op[2]}")
        input_example = '\n'.join(input_lines)
        
        prompt = f"""你是一名编程竞赛选手，需要解决一个关于珠子序列memory计算的问题。请仔细阅读问题描述并正确处理所有操作，输出所有查询的结果。

问题描述：
有一个由n个珠子组成的序列，每个珠子有一个形状编号（1到n之间的整数）。处理m次操作，操作分为两种类型：
1. 修改操作：将某个位置的珠子形状改为新的值。
2. 查询操作：计算某个子段的memory值。memory定义为该子段中每个形状的最后出现位置减去首次出现位置的总和。

输入格式：
第一行是两个整数n和m，表示珠子数量和操作数量。
第二行包含n个整数，表示初始形状。
接下来的m行每行描述一个操作，格式如下：
- 1 p x：将第p个珠子的形状改为x。
- 2 l r：查询从第l到第r个珠子的子段的memory值。

输出要求：
对于每个查询操作，输出其对应的memory值，每个结果占一行。请将所有答案按查询出现的顺序依次放在[answer]和[/answer]之间，每个结果占一行。

输入示例：
{input_example}

请根据上述输入示例格式处理所有操作，并将所有查询结果按顺序放在答案块中。例如：
[answer]
5
0
7
1
[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        last_block = answer_blocks[-1].strip()
        answers = []
        for line in last_block.splitlines():
            line = line.strip()
            if line:
                try:
                    answers.append(int(line))
                except ValueError:
                    continue
        return answers if answers else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity.get('answers', [])
