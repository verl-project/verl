"""# 

### 谜题描述
Serge came to the school dining room and discovered that there is a big queue here. There are m pupils in the queue. He's not sure now if he wants to wait until the queue will clear, so he wants to know which dish he will receive if he does. As Serge is very tired, he asks you to compute it instead of him.

Initially there are n dishes with costs a_1, a_2, …, a_n. As you already know, there are the queue of m pupils who have b_1, …, b_m togrogs respectively (pupils are enumerated by queue order, i.e the first pupil in the queue has b_1 togrogs and the last one has b_m togrogs)

Pupils think that the most expensive dish is the most delicious one, so every pupil just buys the most expensive dish for which he has money (every dish has a single copy, so when a pupil has bought it nobody can buy it later), and if a pupil doesn't have money for any dish, he just leaves the queue (so brutal capitalism...)

But money isn't a problem at all for Serge, so Serge is buying the most expensive dish if there is at least one remaining.

Moreover, Serge's school has a very unstable economic situation and the costs of some dishes or number of togrogs of some pupils can change. More formally, you must process q queries:

  * change a_i to x. It means that the price of the i-th dish becomes x togrogs. 
  * change b_i to x. It means that the i-th pupil in the queue has x togrogs now. 



Nobody leaves the queue during those queries because a saleswoman is late.

After every query, you must tell Serge price of the dish which he will buy if he has waited until the queue is clear, or -1 if there are no dishes at this point, according to rules described above.

Input

The first line contains integers n and m (1 ≤ n, m ≤ 300\ 000) — number of dishes and pupils respectively. The second line contains n integers a_1, a_2, …, a_n (1 ≤ a_i ≤ 10^{6}) — elements of array a. The third line contains m integers b_1, b_2, …, b_{m} (1 ≤ b_i ≤ 10^{6}) — elements of array b. The fourth line conatins integer q (1 ≤ q ≤ 300\ 000) — number of queries.

Each of the following q lines contains as follows: 

  * if a query changes price of some dish, it contains 1, and two integers i and x (1 ≤ i ≤ n, 1 ≤ x ≤ 10^{6}), what means a_i becomes x. 
  * if a query changes number of togrogs of some pupil, it contains 2, and two integers i and x (1 ≤ i ≤ m, 1 ≤ x ≤ 10^{6}), what means b_i becomes x. 

Output

For each of q queries prints the answer as the statement describes, the answer of the i-th query in the i-th line (the price of the dish which Serge will buy or -1 if nothing remains)

Examples

Input


1 1
1
1
1
1 1 100


Output


100


Input


1 1
1
1
1
2 1 100


Output


-1


Input


4 6
1 8 2 4
3 3 6 1 5 2
3
1 1 1
2 5 10
1 1 6


Output


8
-1
4

Note

In the first sample after the first query, there is one dish with price 100 togrogs and one pupil with one togrog, so Serge will buy the dish with price 100 togrogs.

In the second sample after the first query, there is one dish with price one togrog and one pupil with 100 togrogs, so Serge will get nothing.

In the third sample after the first query, nobody can buy the dish with price 8, so Serge will take it. After the second query, all dishes will be bought, after the third one the third and fifth pupils will by the first and the second dishes respectively and nobody will by the fourth one.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const long long maxn = 1e6, big = 1e18;
long long val[maxn], t[4 * maxn + 5], sum[4 * maxn + 5];
void build(long long v, long long l, long long r) {
  if (l == r - 1) {
    sum[v] = 0;
    t[v] = val[l];
    return;
  }
  long long m = (l + r) / 2;
  build(2 * v, l, m);
  build(2 * v + 1, m, r);
  t[v] = max(t[2 * v], t[2 * v + 1]);
  sum[v] = 0;
}
void upd(long long v, long long l, long long r, long long tl, long long tr,
         long long c) {
  if (tr <= l || tl >= r) return;
  if (tl >= l && tr <= r) {
    sum[v] += c;
    return;
  }
  long long tm = (tl + tr) / 2;
  upd(2 * v, l, r, tl, tm, c);
  upd(2 * v + 1, l, r, tm, tr, c);
  t[v] = max(t[2 * v] + sum[2 * v], t[2 * v + 1] + sum[2 * v + 1]);
}
long long get(long long v, long long l, long long r, long long tl,
              long long tr) {
  if (tr <= l || tl >= r) return -big;
  if (tl >= l && tr <= r) return t[v] + sum[v];
  long long tm = (tl + tr) / 2;
  return max(get(2 * v, l, r, tl, tm), get(2 * v + 1, l, r, tm, tr)) + sum[v];
}
void solve() {
  long long n, m;
  cin >> n >> m;
  long long a[n], b[m];
  for (long long i = 0; i < n; ++i) cin >> a[i];
  for (long long i = 0; i < m; ++i) cin >> b[i];
  for (long long i = 0; i < maxn; ++i) val[i] = 0;
  build(1, 0, 1e6);
  for (long long i = 0; i < n; ++i) upd(1, 0, a[i], 0, 1e6, 1);
  for (long long i = 0; i < m; ++i) upd(1, 0, b[i], 0, 1e6, -1);
  long long q;
  cin >> q;
  while (q) {
    --q;
    long long type, v, id;
    cin >> type >> id >> v;
    --id;
    if (type == 1) {
      upd(1, 0, a[id], 0, 1e6, -1);
      a[id] = v;
      upd(1, 0, a[id], 0, 1e6, 1);
    } else {
      upd(1, 0, b[id], 0, 1e6, 1);
      b[id] = v;
      upd(1, 0, b[id], 0, 1e6, -1);
    }
    long long l = -1, r = 1e6;
    while (l < r - 1) {
      long long mid = (l + r) / 2;
      if (get(1, mid, 1e6, 0, 1e6) > 0)
        l = mid;
      else
        r = mid;
    }
    if (l == -1)
      cout << \"-1\n\";
    else
      cout << l + 1 << '\n';
  }
}
signed main() {
  ios_base::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);
  solve();
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bisect import bisect_right
from bootcamp import Basebootcamp

class Esergeanddiningroombootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = {
            'max_n': params.get('max_n', 5),
            'max_m': params.get('max_m', 5),
            'max_q': params.get('max_q', 3),
            'max_value': params.get('max_value', 100),
        }
        super().__init__()

    def case_generator(self):
        n = random.randint(1, self.params['max_n'])
        m = random.randint(1, self.params['max_m'])
        a_initial = [random.randint(1, self.params['max_value']) for _ in range(n)]
        b_initial = [random.randint(1, self.params['max_value']) for _ in range(m)]
        q = random.randint(1, self.params['max_q'])
        queries = []
        for _ in range(q):
            query_type = random.choice([1, 2])
            if query_type == 1:
                i = random.randint(1, n)
                x = random.randint(1, self.params['max_value'])
            else:
                i = random.randint(1, m)
                x = random.randint(1, self.params['max_value'])
            queries.append((query_type, i, x))
        
        current_a = a_initial.copy()
        current_b = b_initial.copy()
        correct_answers = []
        for query in queries:
            type_, i, x = query
            if type_ == 1:
                current_a[i-1] = x
            else:
                current_b[i-1] = x
            serge_answer = self.simulate_after_query(current_a, current_b)
            correct_answers.append(serge_answer)
        
        return {
            'n': n,
            'm': m,
            'a_initial': a_initial,
            'b_initial': b_initial,
            'queries': queries,
            'correct_answers': correct_answers
        }

    @staticmethod
    def simulate_after_query(a, b):
        # 关键修正：维护降序排序，使用bisect_right的正确方式
        remaining_dishes = sorted(a, reverse=True)
        for money in b:
            # 使用bisect_left在降序列表中找到插入点
            idx = bisect_right(remaining_dishes, money, key=lambda x: -x)
            if idx < len(remaining_dishes) and remaining_dishes[idx] <= money:
                del remaining_dishes[idx]
        return max(remaining_dishes) if remaining_dishes else -1

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        m = question_case['m']
        a = question_case['a_initial']
        b = question_case['b_initial']
        q = len(question_case['queries'])
        queries = question_case['queries']
        
        input_lines = [
            f"{n} {m}",
            " ".join(map(str, a)),
            " ".join(map(str, b)),
            str(q)
        ]
        for query in queries:
            type_, i, x = query
            input_lines.append(f"{type_} {i} {x}")
        input_block = "\n".join(input_lines)
        
        prompt = f"""Serge来到学校餐厅，需要计算每次修改后他能买到的菜肴价格。初始时有{n}个菜肴和{m}个学生。每个学生依次购买能买的最贵菜肴。每次修改后，请输出Serge能买到的菜肴价格（没有则为-1）。

输入数据：
{input_block}

请输出每次查询后的结果，每个结果占一行，并包裹在[answer]和[/answer]标签中。"""
        return prompt

    @staticmethod
    def extract_output(output):
        start_tag = '[answer]'
        end_tag = '[/answer]'
        start_indices = [i for i in range(len(output)) if output.startswith(start_tag, i)]
        end_indices = [i for i in range(len(output)) if output.startswith(end_tag, i)]
        
        if not start_indices or not end_indices:
            return None
        
        # 找到最后一个完整匹配的answer块
        valid_pairs = []
        for s_idx in reversed(start_indices):
            for e_idx in reversed(end_indices):
                if e_idx > s_idx:
                    valid_pairs.append((s_idx, e_idx))
                    break
        if not valid_pairs:
            return None
        
        s_idx, e_idx = valid_pairs[0]
        answer_block = output[s_idx+len(start_tag):e_idx].strip()
        answers = []
        for line in answer_block.splitlines():
            line = line.strip()
            if line == '-1':
                answers.append(-1)
            else:
                try:
                    answers.append(int(line))
                except:
                    pass
        return answers if answers else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity['correct_answers']
        if not solution or len(solution) != len(expected):
            return False
        return solution == expected
