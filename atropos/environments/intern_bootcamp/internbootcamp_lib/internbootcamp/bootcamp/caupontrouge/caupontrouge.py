"""# 

### 谜题描述
VK just opened its second HQ in St. Petersburg! Side of its office building has a huge string s written on its side. This part of the office is supposed to be split into m meeting rooms in such way that meeting room walls are strictly between letters on the building. Obviously, meeting rooms should not be of size 0, but can be as small as one letter wide. Each meeting room will be named after the substring of s written on its side.

<image>

For each possible arrangement of m meeting rooms we ordered a test meeting room label for the meeting room with lexicographically minimal name. When delivered, those labels got sorted backward lexicographically.

What is printed on kth label of the delivery?

Input

In the first line, you are given three integer numbers n, m, k — length of string s, number of planned meeting rooms to split s into and number of the interesting label (2 ≤ n ≤ 1 000; 1 ≤ m ≤ 1 000; 1 ≤ k ≤ 10^{18}).

Second input line has string s, consisting of n lowercase english letters.

For given n, m, k there are at least k ways to split s into m substrings.

Output

Output single string – name of meeting room printed on k-th label of the delivery.

Examples

Input


4 2 1
abac


Output


aba


Input


19 5 1821
aupontrougevkoffice


Output


au

Note

In the first example, delivery consists of the labels \"aba\", \"ab\", \"a\".

In the second example, delivery consists of 3060 labels. The first label is \"aupontrougevkof\" and the last one is \"a\".

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int N = 1005;
string a, s[N * N];
int n, m, tot;
long long k;
long long f[N][N], g[N][N];
long long Rank(string ss) {
  int len = ss.size();
  memset(f, 0, sizeof(f)), memset(g, 0, sizeof(g));
  f[n][0] = 1, g[n][0] = 1;
  for (register int i = n - 1; i >= 0; i--) {
    int tail = 0;
    while (i + tail < n && tail < len && a[i + tail] == ss[tail]) tail++;
    int tag = 0;
    if (i + tail >= n)
      tag = 1;
    else if (i + tail < n && tail >= len)
      tail++;
    else if (i + tail < n && tail < len) {
      if (a[i + tail] < ss[tail])
        tag = 1;
      else
        tail++;
    }
    for (register int j = 0; j <= m; j++) {
      if (j) f[i][j] = tag ? 0 : g[i + tail][j - 1];
      g[i][j] = g[i + 1][j] + f[i][j];
      if (g[i][j] > k) g[i][j] = k + 1;
    }
  }
  return f[0][m];
}
int main() {
  ios::sync_with_stdio(false), cin.tie(0);
  cin >> n >> m >> k;
  cin >> a;
  for (register int i = 0; i < n; i++) {
    for (register int j = i; j < n; j++) {
      s[tot++] = a.substr(i, j - i + 1);
    }
  }
  sort(s, s + tot, greater<string>());
  int l = 0, r = tot, ans = 0;
  while (l < r) {
    int mid = l + r >> 1;
    if (Rank(s[mid]) < k)
      l = mid + 1, ans = mid;
    else
      r = mid;
  }
  cout << s[ans] << '\n';
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
from itertools import combinations
from bootcamp import Basebootcamp

class Caupontrougebootcamp(Basebootcamp):
    def __init__(self, max_length=10, max_m=5):
        """
        优化参数校验和字符多样性保证:
        1. 限制max_length >=3保证有效分割
        2. 强制字符串至少包含两个不同字符
        """
        self.max_length = max(max_length, 3)
        self.max_m = max_m

    def case_generator(self):
        """增强案例生成鲁棒性"""
        import random
        from random import randint
        
        while True:
            # 生成符合要求的字符串
            n = randint(3, self.max_length)
            m = randint(1, min(n-1, self.max_m))
            
            # 生成包含至少两个不同字符的字符串
            while True:
                s = ''.join([chr(randint(97, 99)) for _ in range(n)])
                if len(set(s)) >= 2: break
            
            # 获取有效候选集
            candidates = self._get_all_candidates(s, m)
            if len(candidates) >= max(3, m):  # 保证足够测试意义
                k = randint(1, len(candidates))
                return {
                    'n': n, 'm': m, 'k': k, 's': s,
                    'candidates': sorted(candidates, reverse=True)
                }

    def _get_all_candidates(self, s, m):
        """优化分割点生成算法"""
        candidates = set()
        n = len(s)
        
        # 使用迭代器避免内存爆炸
        for splits in combinations(range(1, n), m-1):
            parts = []
            prev = 0
            for pos in sorted(splits):
                parts.append(s[prev:pos])
                prev = pos
            parts.append(s[prev:])
            candidates.add(min(parts))
        
        return sorted(candidates)

    @staticmethod
    def prompt_func(question_case) -> str:
        return f"""给定字符串{question_case['s']!r}，请分割为{question_case['m']}个非空子串。
所有可能分割方案中的最小标签按逆字典序排列后，求第{question_case['k']}个标签。
答案请用[answer]标签包裹，如：[answer]答案[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return identity['candidates'][identity['k']-1] == solution
        except (IndexError, KeyError):
            return False
