"""# 

### 谜题描述
You are given an infinite periodic array a0, a1, ..., an - 1, ... with the period of length n. Formally, <image>. A periodic subarray (l, s) (0 ≤ l < n, 1 ≤ s < n) of array a is an infinite periodic array with a period of length s that is a subsegment of array a, starting with position l.

A periodic subarray (l, s) is superior, if when attaching it to the array a, starting from index l, any element of the subarray is larger than or equal to the corresponding element of array a. An example of attaching is given on the figure (top — infinite array a, bottom — its periodic subarray (l, s)):

<image>

Find the number of distinct pairs (l, s), corresponding to the superior periodic arrays.

Input

The first line contains number n (1 ≤ n ≤ 2·105). The second line contains n numbers a0, a1, ..., an - 1 (1 ≤ ai ≤ 106), separated by a space.

Output

Print a single integer — the sought number of pairs.

Examples

Input

4
7 1 2 3


Output

2


Input

2
2 1


Output

1


Input

3
1 1 1


Output

6

Note

In the first sample the superior subarrays are (0, 1) and (3, 2).

Subarray (0, 1) is superior, as a0 ≥ a0, a0 ≥ a1, a0 ≥ a2, a0 ≥ a3, a0 ≥ a0, ...

Subarray (3, 2) is superior a3 ≥ a3, a0 ≥ a0, a3 ≥ a1, a0 ≥ a2, a3 ≥ a3, ...

In the third sample any pair of (l, s) corresponds to a superior subarray as all the elements of an array are distinct.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int MAX_N = 1000000;
int A[MAX_N], N, C[MAX_N], G[MAX_N];
bool U[MAX_N];
long long ans;
inline int inc(int v) { return (v + 1 == N) ? 0 : (v + 1); }
inline int gcd(int a, int b) { return b ? gcd(b, a % b) : a; }
int main() {
  int i, j, g, l, r, mx, len;
  long long ans = 0;
  scanf(\"%d\", &N);
  for (i = 0; i < N; i++) scanf(\"%d\", A + i);
  for (i = 1; i <= N; i++) G[i] = gcd(i, N);
  for (g = 1; g < N; g++) {
    if (N % g != 0) continue;
    for (i = 0; i < N; i++) U[i] = false, C[i] = 0;
    for (i = 0; i < g; i++) {
      mx = -1;
      for (j = i; j < N; j += g) mx = max(mx, A[j]);
      for (j = i; j < N; j += g)
        if (A[j] == mx) U[j] = true;
    }
    bool any = false;
    for (l = 0; l < N;) {
      r = inc(l);
      if (U[l]) {
        l++;
        continue;
      }
      any = true;
      len = 0;
      while (U[r]) len++, r = inc(r);
      for (i = 1; i <= len; i++) C[i] += len - i + 1;
      if (r <= l) break;
      l = r;
    }
    if (!any)
      for (i = 1; i <= N; i++) C[i] += N;
    for (i = 1; i <= N; i++)
      if (G[i] == g) ans += C[i];
  }
  printf(\"%I64d\", ans);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
from typing import Dict, List, Optional
import re

class Csuperiorperiodicsubarraysbootcamp(Basebootcamp):
    def __init__(self, n_min: int = 2, n_max: int = 10, max_value: int = 10):
        self.n_min = max(n_min, 1)
        self.n_max = min(n_max, 20)  # 限制n范围保证性能
        self.max_value = max_value

    def case_generator(self) -> Dict:
        """生成小规模测试用例"""
        n = random.randint(self.n_min, self.n_max)
        a = [random.randint(1, self.max_value) for _ in range(n)]
        return {
            'n': n,
            'a': a,
            'correct_answer': self._solve(n, a)
        }

    @staticmethod
    def prompt_func(question_case: Dict) -> str:
        a_str = ' '.join(map(str, question_case['a']))
        return f"""给定周期为{question_case['n']}的数组：
{a_str}

请计算满足条件的(l, s)对的数量。答案放于[answer][/answer]中。"""

    @staticmethod
    def extract_output(output: str) -> Optional[int]:
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if matches:
            try:
                return int(matches[-1].strip())
            except ValueError:
                pass
        return None

    @classmethod
    def _verify_correction(cls, solution: int, identity: Dict) -> bool:
        return solution == identity['correct_answer']

    @staticmethod
    def _solve(n: int, a: List[int]) -> int:
        """优化后的暴力解法"""
        ans = 0
        for s in range(1, n):
            for l in range(n):
                valid = True
                for i in range(n):
                    original_pos = (l + i) % n
                    subarray_pos = (l + (i % s)) % n
                    if a[subarray_pos] < a[original_pos]:
                        valid = False
                        break
                if valid:
                    ans += 1
        return ans
