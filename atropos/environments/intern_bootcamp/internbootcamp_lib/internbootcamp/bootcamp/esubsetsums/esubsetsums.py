"""# 

### 谜题描述
You are given an array a1, a2, ..., an and m sets S1, S2, ..., Sm of indices of elements of this array. Let's denote Sk = {Sk, i} (1 ≤ i ≤ |Sk|). In other words, Sk, i is some element from set Sk.

In this problem you have to answer q queries of the two types:

  1. Find the sum of elements with indices from set Sk: <image>. The query format is \"? k\". 
  2. Add number x to all elements at indices from set Sk: aSk, i is replaced by aSk, i + x for all i (1 ≤ i ≤ |Sk|). The query format is \"+ k x\". 



After each first type query print the required sum.

Input

The first line contains integers n, m, q (1 ≤ n, m, q ≤ 105). The second line contains n integers a1, a2, ..., an (|ai| ≤ 108) — elements of array a. 

Each of the following m lines describes one set of indices. The k-th line first contains a positive integer, representing the number of elements in set (|Sk|), then follow |Sk| distinct integers Sk, 1, Sk, 2, ..., Sk, |Sk| (1 ≤ Sk, i ≤ n) — elements of set Sk.

The next q lines contain queries. Each query looks like either \"? k\" or \"+ k x\" and sits on a single line. For all queries the following limits are held: 1 ≤ k ≤ m, |x| ≤ 108. The queries are given in order they need to be answered.

It is guaranteed that the sum of sizes of all sets Sk doesn't exceed 105.

Output

After each first type query print the required sum on a single line.

Please, do not write the %lld specifier to read or write 64-bit integers in С++. It is preferred to use the cin, cout streams or the %I64d specifier.

Examples

Input

5 3 5
5 -5 5 1 -4
2 1 2
4 2 1 4 5
2 2 5
? 2
+ 3 4
? 1
+ 2 1
? 2


Output

-3
4
9

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int B = 300;
int main() {
  cin.tie(nullptr);
  ios::sync_with_stdio(false);
  int n, m, q;
  cin >> n >> m >> q;
  vector<int> a(n);
  for (auto &ai : a) cin >> ai;
  vector<vector<int>> s(m);
  vector<long long> init_sum(m);
  for (int i = 0; i < m; i++) {
    int k;
    cin >> k;
    s[i].resize(k);
    for (int &sij : s[i]) cin >> sij, sij--, init_sum[i] += a[sij];
  }
  auto is_large = [&](int i) { return s[i].size() >= B; };
  vector<vector<int>> ist_sz(m);
  vector<int> id;
  int large_num = 0;
  for (int i = 0; i < m; i++)
    if (is_large(i)) {
      vector<bool> b(n);
      for (int si : s[i]) b[si] = true;
      for (int j = 0; j < m; j++) {
        int cnt = 0;
        for (int sj : s[j]) cnt += b[sj];
        ist_sz[j].emplace_back(cnt);
      }
      id.emplace_back(i), large_num++;
    }
  vector<long long> large_sum(m), large_add(m), small_add(n);
  while (q--) {
    char op;
    cin >> op;
    int k;
    cin >> k;
    k--;
    if (op == '?') {
      if (is_large(k)) {
        cout << init_sum[k] + large_sum[k] << endl;
      } else {
        long long ans = init_sum[k];
        for (int sk : s[k]) ans += small_add[sk];
        for (int i = 0; i < large_num; i++)
          ans += large_add[id[i]] * ist_sz[k][i];
        cout << ans << endl;
      }
    }
    if (op == '+') {
      long long x;
      cin >> x;
      for (int i = 0; i < large_num; i++) large_sum[id[i]] += x * ist_sz[k][i];
      if (is_large(k))
        large_add[k] += x;
      else
        for (int sk : s[k]) small_add[sk] += x;
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
from bootcamp import Basebootcamp

class Esubsetsumsbootcamp(Basebootcamp):
    def __init__(self, n_max=5, m_max=3, q_max=5, max_element=10, min_element=-10, set_size_min=1, set_size_max=5, query_prob=0.3):
        self.n_max = n_max
        self.m_max = m_max
        self.q_max = q_max
        self.max_element = max_element
        self.min_element = min_element
        self.set_size_min = set_size_min
        self.set_size_max = set_size_max
        self.query_prob = query_prob
    
    def case_generator(self):
        while True:  # 新增循环确保至少一个查询输出
            n = random.randint(1, self.n_max)
            m = random.randint(1, self.m_max)
            q = random.randint(1, self.q_max)
            
            a = [random.randint(self.min_element, self.max_element) for _ in range(n)]
            
            sets = []
            for _ in range(m):
                max_possible = min(n, self.set_size_max)
                k = random.randint(max(1, self.set_size_min), max_possible)
                available = list(range(1, n+1))
                random.shuffle(available)
                sets.append(sorted(available[:k]))
            
            queries = []
            current_a = a.copy()
            correct_outputs = []
            has_query = False  # 新增验证标记
            
            for _ in range(q):
                if not has_query:  # 保证至少一个查询是?
                    op_type = '?'
                else:
                    op_type = '?' if random.random() < self.query_prob else '+'
                
                k = random.randint(1, m)
                if op_type == '?':
                    s = sum(current_a[i-1] for i in sets[k-1])
                    correct_outputs.append(s)
                    queries.append(('?', k))
                    has_query = True
                else:
                    x = random.randint(-self.max_element, self.max_element)
                    for idx in sets[k-1]:
                        current_a[idx-1] += x
                    queries.append(('+', k, x))
            
            if has_query:  # 确保存在需要输出的查询
                return {
                    'n': n,
                    'm': m,
                    'q': q,
                    'a': a,
                    'sets': sets,
                    'queries': queries,
                    'correct_outputs': correct_outputs
                }

    @staticmethod
    def prompt_func(question_case) -> str:
        input_lines = [
            f"{question_case['n']} {question_case['m']} {question_case['q']}",
            ' '.join(map(str, question_case['a']))
        ]
        for s in question_case['sets']:
            input_lines.append(f"{len(s)} " + ' '.join(map(str, s)))
        for q in question_case['queries']:
            if q[0] == '?':
                input_lines.append(f"? {q[1]}")
            else:
                input_lines.append(f"+ {q[1]} {q[2]}")
        
        prompt = (
            "请解决数组处理问题：\n"
            "给定初始数组和多个索引集合，处理增查操作。\n"
            "规则说明：\n"
            "1. 初始数组元素为第二行数字\n"
            "2. 每个集合描述格式：大小+索引（1-based）\n"
            "3. 查询类型：\n"
            "   ? k → 输出第k个集合元素和\n"
            "   + k x → 给第k个集合元素加x\n"
            "\n输入数据：\n" + '\n'.join(input_lines) + 
            "\n\n请将每个?查询的结果按顺序放在[answer]内，如：\n[answer]\n12\n-5\n[/answer]"
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        
        last_answer = matches[-1].strip()
        answers = []
        for line in last_answer.split('\n'):
            clean_line = line.strip()
            if clean_line:
                if re.match(r'^-?\d+$', clean_line):
                    answers.append(int(clean_line))
                elif re.match(r'^-?\d+\.\d+$', clean_line):
                    answers.append(int(float(clean_line)))
        return answers if answers else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity.get('correct_outputs', [])

