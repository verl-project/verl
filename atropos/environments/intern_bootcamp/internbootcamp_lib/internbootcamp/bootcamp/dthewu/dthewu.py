"""# 

### 谜题描述
Childan is making up a legendary story and trying to sell his forgery — a necklace with a strong sense of \"Wu\" to the Kasouras. But Mr. Kasoura is challenging the truth of Childan's story. So he is going to ask a few questions about Childan's so-called \"personal treasure\" necklace.

This \"personal treasure\" is a multiset S of m \"01-strings\".

A \"01-string\" is a string that contains n characters \"0\" and \"1\". For example, if n=4, strings \"0110\", \"0000\", and \"1110\" are \"01-strings\", but \"00110\" (there are 5 characters, not 4) and \"zero\" (unallowed characters) are not.

Note that the multiset S can contain equal elements.

Frequently, Mr. Kasoura will provide a \"01-string\" t and ask Childan how many strings s are in the multiset S such that the \"Wu\" value of the pair (s, t) is not greater than k. 

Mrs. Kasoura and Mr. Kasoura think that if s_i = t_i (1≤ i≤ n) then the \"Wu\" value of the character pair equals to w_i, otherwise 0. The \"Wu\" value of the \"01-string\" pair is the sum of the \"Wu\" values of every character pair. Note that the length of every \"01-string\" is equal to n.

For example, if w=[4, 5, 3, 6], \"Wu\" of (\"1001\", \"1100\") is 7 because these strings have equal characters only on the first and third positions, so w_1+w_3=4+3=7.

You need to help Childan to answer Mr. Kasoura's queries. That is to find the number of strings in the multiset S such that the \"Wu\" value of the pair is not greater than k.

Input

The first line contains three integers n, m, and q (1≤ n≤ 12, 1≤ q, m≤ 5⋅ 10^5) — the length of the \"01-strings\", the size of the multiset S, and the number of queries.

The second line contains n integers w_1, w_2, …, w_n (0 ≤ w_i ≤ 100) — the value of the i-th caracter.

Each of the next m lines contains the \"01-string\" s of length n — the string in the multiset S.

Each of the next q lines contains the \"01-string\" t of length n and integer k (0≤ k≤ 100) — the query.

Output

For each query, print the answer for this query.

Examples

Input

2 4 5
40 20
01
01
10
11
00 20
00 40
11 20
11 40
11 60


Output

2
4
2
3
4


Input

1 2 4
100
0
1
0 0
0 100
1 0
1 100


Output

1
2
1
2

Note

In the first example, we can get:

\"Wu\" of (\"01\", \"00\") is 40.

\"Wu\" of (\"10\", \"00\") is 20.

\"Wu\" of (\"11\", \"00\") is 0.

\"Wu\" of (\"01\", \"11\") is 20.

\"Wu\" of (\"10\", \"11\") is 40.

\"Wu\" of (\"11\", \"11\") is 60.

In the first query, pairs (\"11\", \"00\") and (\"10\", \"00\") satisfy the condition since their \"Wu\" is not greater than 20.

In the second query, all strings satisfy the condition.

In the third query, pairs (\"01\", \"11\") and (\"01\", \"11\") satisfy the condition. Note that since there are two \"01\" strings in the multiset, the answer is 2, not 1.

In the fourth query, since k was increased, pair (\"10\", \"11\") satisfies the condition too.

In the fifth query, since k was increased, pair (\"11\", \"11\") satisfies the condition too.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int maxn = 5e5 + 5;
int w[15];
int a[maxn];
int n;
int change(char *s) {
  int res = 0, tmp = 1;
  for (int i = n - 1; ~i; i--) {
    if (s[i] == '1') res += tmp;
    tmp *= 2;
  }
  return res;
}
int sum[maxn / 5][105];
int cnt[maxn / 5];
int get2(int x, int t) {
  if (x & (1 << t))
    return 1;
  else
    return 0;
}
int main() {
  int m, q;
  char s[15];
  cin >> n >> m >> q;
  for (int i = 0; i < n; i++) cin >> w[i];
  for (int i = 0; i < m; i++) {
    scanf(\"%s\", s);
    a[i] = change(s);
    cnt[a[i]]++;
  }
  sort(a, a + m);
  int siz = unique(a, a + m) - a;
  int mx = (1 << n) - 1;
  int tmp;
  for (int i = 0; i <= mx; i++) {
    for (int j = 0; j < siz; j++) {
      tmp = 0;
      for (int t = 0; t < n; t++) {
        if (get2(i, t) ^ get2(a[j], t) == 0) {
          tmp += w[n - t - 1];
        }
      }
      if (tmp <= 100) sum[i][tmp] += cnt[a[j]];
    }
    for (int j = 1; j <= 100; j++) sum[i][j] += sum[i][j - 1];
  }
  int k, id;
  while (q--) {
    scanf(\"%s%d\", s, &k);
    id = change(s);
    printf(\"%d\n\", sum[id][k]);
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
from bootcamp import Basebootcamp

class Dthewubootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', random.randint(2, 4))
        self.m = params.get('m', random.randint(5, 20))
        self.n = min(max(self.n, 1), 12)  # 确保n在合法范围
        self.m = min(max(self.m, 1), 10000)

    def case_generator(self):
        MAX_ATTEMPTS = 20
        n = self.n  # 定义局部变量避免self引用错误
        
        w = [random.randint(0, 100) for _ in range(n)]
        
        s_multiset = defaultdict(int)
        for _ in range(self.m):
            s = ''.join(random.choices('01', k=n))
            s_multiset[s] += 1
        
        valid_k = None
        for _ in range(MAX_ATTEMPTS):
            t = ''.join(random.choices('01', k=n))
            
            wu_counts = defaultdict(int)
            for s, cnt in s_multiset.items():
                xor = int(s, 2) ^ int(t, 2)
                # 修正变量作用域错误（原错误行）
                wu = sum(w[n - i - 1] for i in range(n) if (xor & (1 << i)) == 0)
                wu_counts[wu] += cnt
            
            sorted_wu = sorted(wu_counts.keys())
            if not sorted_wu:
                valid_k = 0
                break
            
            for candidate in [sorted_wu[len(sorted_wu)//2], 
                             sorted_wu[0] + (sorted_wu[-1]-sorted_wu[0])//3,
                             sorted_wu[-1]]:
                total = sum(cnt for wu, cnt in wu_counts.items() if wu <= candidate)
                if 0 < total < self.m:
                    valid_k = candidate
                    break
            if valid_k is not None:
                break
        
        if valid_k is None:
            valid_k = random.choice(sorted_wu) if sorted_wu else 0
        
        return {
            'n': n,
            'w': w,
            's_multiset': dict(s_multiset),
            't': t,
            'k': valid_k
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        problem_desc = [
            "## 谜题背景",
            "你需要帮助Childan验证'Dthewu'值阈值条件下的项链数量。每个01字符串的长度为n，每个位置的权重不同。",
            "\n## 参数说明",
            f"字符串长度 n = {question_case['n']}",
            f"权重数组 w = {question_case['w']}",
            "\n## 多集合S内容（字符串:出现次数）",
            *[f"- {s} × {cnt}" for s, cnt in question_case['s_multiset'].items()],
            "\n## 当前查询",
            f"目标字符串 t = {question_case['t']}",
            f"阈值 k = {question_case['k']}",
            "\n请计算满足条件的字符串总数，并将最终数值用[answer]标签包裹。"
        ]
        return '\n'.join(problem_desc)

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except (ValueError, TypeError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        w = identity['w']
        s_multiset = identity['s_multiset']
        t_str = identity['t']
        k = identity['k']
        
        t_int = int(t_str, 2)
        correct = 0
        
        for s_str, count in s_multiset.items():
            s_int = int(s_str, 2)
            xor = s_int ^ t_int
            wu = sum(w[n - i - 1] for i in range(n) if (xor & (1 << i)) == 0)
            
            if wu <= k:
                correct += count
        
        return solution == correct
