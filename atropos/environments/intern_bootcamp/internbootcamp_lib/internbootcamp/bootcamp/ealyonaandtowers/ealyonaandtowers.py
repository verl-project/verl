"""# 

### 谜题描述
Alyona has built n towers by putting small cubes some on the top of others. Each cube has size 1 × 1 × 1. A tower is a non-zero amount of cubes standing on the top of each other. The towers are next to each other, forming a row.

Sometimes Alyona chooses some segment towers, and put on the top of each tower several cubes. Formally, Alyouna chooses some segment of towers from li to ri and adds di cubes on the top of them.

Let the sequence a1, a2, ..., an be the heights of the towers from left to right. Let's call as a segment of towers al, al + 1, ..., ar a hill if the following condition holds: there is integer k (l ≤ k ≤ r) such that al < al + 1 < al + 2 < ... < ak > ak + 1 > ak + 2 > ... > ar.

After each addition of di cubes on the top of the towers from li to ri, Alyona wants to know the maximum width among all hills. The width of a hill is the number of towers in it.

Input

The first line contain single integer n (1 ≤ n ≤ 3·105) — the number of towers.

The second line contain n integers a1, a2, ..., an (1 ≤ ai ≤ 109) — the number of cubes in each tower. 

The third line contain single integer m (1 ≤ m ≤ 3·105) — the number of additions.

The next m lines contain 3 integers each. The i-th of these lines contains integers li, ri and di (1 ≤ l ≤ r ≤ n, 1 ≤ di ≤ 109), that mean that Alyona puts di cubes on the tio of each of the towers from li to ri.

Output

Print m lines. In i-th line print the maximum width of the hills after the i-th addition.

Example

Input

5
5 5 5 5 5
3
1 3 2
2 2 1
4 4 1


Output

2
4
5

Note

The first sample is as follows:

After addition of 2 cubes on the top of each towers from the first to the third, the number of cubes in the towers become equal to [7, 7, 7, 5, 5]. The hill with maximum width is [7, 5], thus the maximum width is 2.

After addition of 1 cube on the second tower, the number of cubes in the towers become equal to [7, 8, 7, 5, 5]. The hill with maximum width is now [7, 8, 7, 5], thus the maximum width is 4.

After addition of 1 cube on the fourth tower, the number of cubes in the towers become equal to [7, 8, 7, 6, 5]. The hill with maximum width is now [7, 8, 7, 6, 5], thus the maximum width is 5.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
struct four {
  int a, b, c, d;
  four(int a_ = 0, int b_ = 0, int c_ = 0, int d_ = 0) {
    a = a_;
    b = b_;
    c = c_;
    d = d_;
  }
  int f() { return d - a; }
};
bool operator<(const four& f, const four& g) { return (f.a < g.a); }
int n, m;
vector<long long> a, b, l, r, d;
map<int, int> ma;
set<four> se;
four el, er;
four take_seg(int i) {
  four f = *(--se.upper_bound(four(i)));
  se.erase(f);
  if (!(--ma[f.f()])) ma.erase(f.f());
  return f;
}
void push_seg(four f) {
  if (f.f()) {
    se.insert(f);
    ++ma[f.f()];
  }
}
void neg_zer(int i) {
  el = er = take_seg(i);
  el.d = i;
  er.a = er.b = er.c = i + 1;
  push_seg(el);
  push_seg(er);
}
void pos_zer(int i) {
  el = er = take_seg(i);
  el.b = el.c = el.d = i;
  er.a = i + 1;
  push_seg(el);
  push_seg(er);
}
void zer_neg(int i) {
  el = four(i, i, i, i + 1);
  if (i > 0 && b[i - 1] != 0) {
    el = take_seg(i - 1);
    ++el.d;
  }
  if (i < n - 2 && b[i + 1] < 0) el.d = (take_seg(i + 1)).d;
  push_seg(el);
}
void zer_pos(int i) {
  el = four(i, i + 1, i + 1, i + 1);
  if (i < n - 2 && b[i + 1] != 0) {
    el = take_seg(i + 1);
    --el.a;
  }
  if (i > 0 && b[i - 1] > 0) el.a = (take_seg(i - 1)).a;
  push_seg(el);
}
void add(int i, long long x) {
  if (b[i] == 0) zer_pos(i);
  if (b[i] < 0 && b[i] + x >= 0) neg_zer(i);
  if (b[i] < 0 && b[i] + x > 0) zer_pos(i);
  b[i] += x;
}
void rem(int i, long long x) {
  if (b[i] == 0) zer_neg(i);
  if (b[i] > 0 && b[i] - x <= 0) pos_zer(i);
  if (b[i] > 0 && b[i] - x < 0) zer_neg(i);
  b[i] -= x;
}
int main() {
  ios_base::sync_with_stdio(false);
  cin >> n;
  a.resize(n);
  for (int i = 0; i < n; ++i) cin >> a[i];
  cin >> m;
  l.resize(m);
  r.resize(m);
  d.resize(m);
  for (int i = 0; i < m; ++i) {
    cin >> l[i] >> r[i] >> d[i];
    --l[i];
    --r[i];
  }
  b.assign(n - 1, 0);
  for (int i = 0; i < n - 1; ++i) {
    long long dif = a[i + 1] - a[i];
    if (dif > 0) add(i, dif);
    if (dif < 0) rem(i, -dif);
  }
  ++ma[0];
  for (int i = 0; i < m; ++i) {
    if (l[i] > 0) add(--l[i], d[i]);
    if (r[i] < n - 1) rem(r[i], d[i]);
    cout << (--ma.end())->first + 1 << \"\n\";
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Ealyonaandtowersbootcamp(Basebootcamp):
    def __init__(self, **params):
        # 扩展参数范围，支持更大规模测试案例生成
        self.n_max = params.get('n_max', 10)  # 测试时可适当增大，但保持验证可行性
        self.d_max = params.get('d_max', 10)
        self.a_max = params.get('a_max', 10)
        # 保持其他参数的默认范围限制以确保暴力验证可行
        
        # 继承其他参数设置...
        self.n_min = params.get('n_min', 1)
        self.m_min = params.get('m_min', 1)
        self.m_max = params.get('m_max', 5)
        self.d_min = params.get('d_min', 1)
        self.a_min = params.get('a_min', 1)

    def case_generator(self):
        # 保证生成的案例有独特结构以形成有效山脉
        n = random.randint(self.n_min, self.n_max)
        
        # 生成初始数组时增加峰形结构的概率
        if random.random() < 0.7 and n > 2:
            # 生成峰形结构
            peak = random.randint(1, n-2)
            a = [random.randint(1, 5) for _ in range(n)]
            for i in range(peak):
                a[i+1] = a[i] + random.randint(1, 3)
            for i in range(peak, n-1):
                a[i+1] = a[i] - random.randint(1, 3)
                if a[i+1] <= 0: a[i+1] = 1
        else:
            a = [random.randint(self.a_min, self.a_max) for _ in range(n)]

        m = random.randint(self.m_min, self.m_max)
        operations = []
        expected_outputs = []
        current_a = a.copy()
        
        for _ in range(m):
            # 生成有效区间操作
            l = random.randint(1, n)
            r = random.randint(l, n)
            d = random.randint(self.d_min, self.d_max)
            operations.append({'l': l, 'r': r, 'd': d})
            
            # 更新当前数组状态
            for i in range(l-1, r):
                current_a[i] += d
                
            # 计算当前最大山脉宽度
            max_width = self.compute_max_hill_width(current_a)
            expected_outputs.append(max_width)

        return {
            'n': n,
            'a': a,
            'm': m,
            'operations': operations,
            'expected_outputs': expected_outputs
        }

    @staticmethod
    def compute_max_hill_width(arr):
        n = len(arr)
        max_width = 1  # 单个塔也算宽1的山脉
        left = [1] * n
        right = [1] * n
        
        # 预处理递增序列
        for i in range(1, n):
            if arr[i] > arr[i-1]:
                left[i] = left[i-1] + 1
                
        # 预处理递减序列
        for i in range(n-2, -1, -1):
            if arr[i] > arr[i+1]:
                right[i] = right[i+1] + 1
                
        # 计算最大宽度
        for i in range(n):
            current = left[i] + right[i] - 1
            if current > max_width:
                max_width = current
                
        return max_width

    @staticmethod
    def prompt_func(question_case):
        # ...保持原有prompt结构，优化规则描述...
        return f"""Ealyonaandtowers's Towers Problem:
        
Input format:
n
a₁ a₂ ... aₙ
m
l₁ r₁ d₁
...
lₘ rₘ dₘ

After each operation, output the maximum hill width. A hill is a sequence of towers where:
1. There exists a peak position k
2. Towers strictly increase from left to the peak
3. Towers strictly decrease from the peak to right

Current Problem:
n = {question_case['n']}
Initial cubes: {' '.join(map(str, question_case['a']))}
m = {question_case['m']}
Operations:
{chr(10).join(f"{op['l']} {op['r']} {op['d']}" for op in question_case['operations'])}

Output the m results in order, each in [answer] tags:
[answer]
3
1
4
[/answer]"""

    @staticmethod
    def extract_output(output):
        # 增强格式抽取的容错性
        matches = re.findall(r'\[answer\][\s]*((?:\d+\s*)+)[\s]*\[\/answer\]', output, re.IGNORECASE)
        if not matches:
            return None
        try:
            return [int(x) for x in matches[-1].split()]
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 增加类型检查和安全访问
        expected = identity.get('expected_outputs', [])
        if not isinstance(solution, list) or len(solution) != len(expected):
            return False
        return all(s == e for s, e in zip(solution, expected))
