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
using ll = long long int;
const int N = 1e6 + 5;
ll seg[6 * N];
ll lazy[6 * N];
ll qu(int node, int l, int r) {
  if (lazy[node]) {
    seg[node] += lazy[node];
    if (l < r) {
      lazy[2 * node] += lazy[node];
      lazy[2 * node + 1] += lazy[node];
    }
    lazy[node] = 0;
  }
  if (seg[node] <= 0) return -1;
  if (l == r) return l;
  int m = (l + r) / 2;
  auto ret = qu(2 * node + 1, m + 1, r);
  if (ret == -1) ret = qu(2 * node, l, m);
  return ret;
}
void upd_rn(int node, int l, int r, int x, int y, int val) {
  if (lazy[node]) {
    seg[node] += lazy[node];
    if (l < r) {
      lazy[2 * node] += lazy[node];
      lazy[2 * node + 1] += lazy[node];
    }
    lazy[node] = 0;
  }
  if (l > y or r < x) return;
  if (l >= x and r <= y) {
    seg[node] += val;
    lazy[2 * node] += val;
    lazy[2 * node + 1] += val;
    return;
  }
  int m = (l + r) / 2;
  upd_rn(2 * node, l, m, x, y, val);
  upd_rn(2 * node + 1, m + 1, r, x, y, val);
  seg[node] = max(seg[2 * node], seg[2 * node + 1]);
}
void solve() {
  int n, m;
  cin >> n >> m;
  int a[n], b[m];
  for (int i = 0; i < n; ++i) {
    cin >> a[i];
    upd_rn(1, 1, N, 1, a[i], 1);
  }
  for (int i = 0; i < m; ++i) {
    cin >> b[i];
    upd_rn(1, 1, N, 1, b[i], -1);
  }
  int q;
  cin >> q;
  while (q--) {
    int type;
    cin >> type;
    if (type == 1) {
      int pos, val;
      cin >> pos >> val;
      --pos;
      if (a[pos] < val)
        upd_rn(1, 1, N, a[pos] + 1, val, 1);
      else if (a[pos] > val)
        upd_rn(1, 1, N, val + 1, a[pos], -1);
      a[pos] = val;
    } else {
      int pos, val;
      cin >> pos >> val;
      --pos;
      if (b[pos] < val)
        upd_rn(1, 1, N, b[pos] + 1, val, -1);
      else if (b[pos] > val)
        upd_rn(1, 1, N, val + 1, b[pos], 1);
      b[pos] = val;
    }
    cout << qu(1, 1, N) << '\n';
  }
}
void InputSetup() {
  ios::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);
}
int main(void) {
  auto start = chrono::high_resolution_clock::now();
  InputSetup();
  solve();
  auto finish = chrono::high_resolution_clock::now();
  cerr << \"Time elapsed: \"
       << (chrono::duration<long double>(finish - start)).count() << \"s\n\";
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import bisect
import random
import re
from bootcamp import Basebootcamp

class Csergeanddiningroombootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_m=10, max_q=10, seed=None):
        self.max_n = max_n
        self.max_m = max_m
        self.max_q = max_q
        self.seed = seed

    def case_generator(self):
        if self.seed is not None:
            random.seed(self.seed)
        n = random.randint(1, self.max_n)
        m = random.randint(1, self.max_m)
        q = random.randint(1, self.max_q)
        
        a_initial = [random.randint(1, 1000) for _ in range(n)]
        b_initial = [random.randint(1, 1000) for _ in range(m)]
        
        queries = []
        for _ in range(q):
            query_type = random.choice([1, 2])
            if query_type == 1:
                i = random.randint(1, n)
                x = random.randint(1, 1000)
                queries.append({'type': query_type, 'i': i, 'x': x})
            else:
                i = random.randint(1, m)
                x = random.randint(1, 1000)
                queries.append({'type': query_type, 'i': i, 'x': x})
        
        current_a = a_initial.copy()
        current_b = b_initial.copy()
        expected_answers = []
        for query in queries:
            if query['type'] == 1:
                current_a[query['i'] - 1] = query['x']
            else:
                current_b[query['i'] - 1] = query['x']
            
            # 修正后的核心逻辑：升序排序+bisect_right
            available = sorted(current_a)
            for money in current_b:
                idx = bisect.bisect_right(available, money)
                if idx > 0:
                    del available[idx-1]
                    if not available:
                        break
            
            result = available[-1] if available else -1
            expected_answers.append(result)
        
        return {
            'n': n,
            'm': m,
            'a_initial': a_initial,
            'b_initial': b_initial,
            'queries': queries,
            'expected_answers': expected_answers
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        input_lines = [
            f"{question_case['n']} {question_case['m']}",
            ' '.join(map(str, question_case['a_initial'])),
            ' '.join(map(str, question_case['b_initial'])),
            str(len(question_case['queries']))
        ]
        for query in question_case['queries']:
            input_lines.append(f"{query['type']} {query['i']} {query['x']}")
        input_str = '\n'.join(input_lines)
        
        prompt = (
            "Serge想知道每次查询后他能买到的菜的价格。每个查询会修改菜的价格或学生的钱，处理所有查询并输出每次查询后的结果。\n\n"
            "规则说明：\n"
            "1. 初始时n道菜（价格可重复）和m个学生（按顺序排队）\n"
            "2. 每个学生购买自己能支付的最贵菜品（如有多个相同价格按剩余菜单顺序）\n"
            "3. 每次查询修改菜品价格或学生金额后，计算所有学生按序购买后的剩余菜品\n"
            "4. 输出剩余菜品的最贵价格，若无则输出-1\n\n"
            "输入格式：\n"
            f"{input_str}\n\n"
            "请将q个查询结果按顺序写在[answer]标签内，每行一个数字。"
            "示例：\n[answer]\n8\n-1\n4\n[/answer]"
        )
        return prompt

    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        answers = []
        for line in answer_blocks[-1].strip().split('\n'):
            line = line.strip()
            if line == '-1':
                answers.append(-1)
            elif line.isdigit() or (line.startswith('-') and line[1:].isdigit()):
                answers.append(int(line))
        return answers if answers else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected_answers']
