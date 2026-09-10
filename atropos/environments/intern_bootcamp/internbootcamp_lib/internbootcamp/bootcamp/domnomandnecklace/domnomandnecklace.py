"""# 

### 谜题描述
One day Om Nom found a thread with n beads of different colors. He decided to cut the first several beads from this thread to make a bead necklace and present it to his girlfriend Om Nelly.

<image>

Om Nom knows that his girlfriend loves beautiful patterns. That's why he wants the beads on the necklace to form a regular pattern. A sequence of beads S is regular if it can be represented as S = A + B + A + B + A + ... + A + B + A, where A and B are some bead sequences, \" + \" is the concatenation of sequences, there are exactly 2k + 1 summands in this sum, among which there are k + 1 \"A\" summands and k \"B\" summands that follow in alternating order. Om Nelly knows that her friend is an eager mathematician, so she doesn't mind if A or B is an empty sequence.

Help Om Nom determine in which ways he can cut off the first several beads from the found thread (at least one; probably, all) so that they form a regular pattern. When Om Nom cuts off the beads, he doesn't change their order.

Input

The first line contains two integers n, k (1 ≤ n, k ≤ 1 000 000) — the number of beads on the thread that Om Nom found and number k from the definition of the regular sequence above.

The second line contains the sequence of n lowercase Latin letters that represent the colors of the beads. Each color corresponds to a single letter.

Output

Print a string consisting of n zeroes and ones. Position i (1 ≤ i ≤ n) must contain either number one if the first i beads on the thread form a regular sequence, or a zero otherwise.

Examples

Input

7 2
bcabcab


Output

0000011

Input

21 2
ababaababaababaababaa


Output

000110000111111000011

Note

In the first sample test a regular sequence is both a sequence of the first 6 beads (we can take A = \"\", B = \"bca\"), and a sequence of the first 7 beads (we can take A = \"b\", B = \"ca\").

In the second sample test, for example, a sequence of the first 13 beads is regular, if we take A = \"aba\", B = \"ba\".

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int INF = INT_MAX;
const long long INFL = LLONG_MAX;
const long double pi = acos(-1);
int N, K;
string s;
int ans[1001001];
int z[1001001];
int main() {
  ios_base::sync_with_stdio(0);
  cout.precision(15);
  cout << fixed;
  cout.tie(0);
  cin.tie(0);
  cin >> N >> K >> s;
  int L = 0, R = 0;
  for (int i = 1; i < N; i++) {
    if (i > R) {
      L = R = i;
      while (R < N && s[R - L] == s[R]) R++;
      z[i] = R - L;
      R--;
    } else {
      int k = i - L;
      if (z[k] < R - i + 1) {
        z[i] = z[k];
      } else {
        L = i;
        while (R < N && s[R - L] == s[R]) R++;
        z[i] = R - L;
        R--;
      }
    }
  }
  for (int lenAB = 1; lenAB <= N; lenAB++) {
    int all = 1;
    int cur = lenAB;
    for (int(i) = 0, j123 = K - 1; (i) < j123; (i)++) {
      if (cur >= N) {
        all = 0;
        break;
      }
      if (cur + lenAB - 1 < N) {
        all &= (z[cur] >= lenAB);
      } else {
        all &= (z[cur] == N - cur);
      }
      cur += lenAB;
    }
    if (lenAB == 2) {
    }
    if (all) {
      int l = lenAB * K - 1;
      int r = l + min(lenAB, z[K * lenAB]);
      if (l < N) {
        ans[l]++;
        ans[min(N, r) + 1]--;
      }
    }
  }
  for (int(i) = 0, j123 = N; (i) < j123; (i)++)
    if (i >= 1) ans[i] += ans[i - 1];
  for (int(i) = 0, j123 = N; (i) < j123; (i)++) cout << (ans[i] > 0);
  cout << '\n';
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
from bootcamp import Basebootcamp
import random

class Domnomandnecklacebootcamp(Basebootcamp):
    def __init__(self, max_n=20, max_k=5):
        self.max_n = max_n
        self.max_k = max_k
    
    def case_generator(self):
        # 确保k不会超过n的合理范围
        n = random.randint(1, self.max_n)
        max_possible_k = min((n + 1) // 2, self.max_k)  # k最大不超过(n+1)/2
        k = random.randint(1, max_possible_k) if max_possible_k >= 1 else 1
        
        # 生成非空字符串时避免全相同字符导致误判
        chars = [chr(ord('a') + random.randint(0, 25)) for _ in range(n)]
        s = ''.join(chars)
        correct_output = self.solve(n, k, s)
        return {
            'n': n,
            'k': k,
            's': s,
            'correct_output': correct_output
        }
    
    @staticmethod
    def compute_z(s):
        # 保持与C++完全一致的Z算法实现
        n = len(s)
        z = [0] * n
        z[0] = n  # 空字符匹配整个字符串
        l, r = 0, 0
        for i in range(1, n):
            if i > r:
                l = r = i
                while r < n and s[r - l] == s[r]:
                    r += 1
                z[i] = r - l
                r -= 1
            else:
                k = i - l
                if z[k] < r - i + 1:
                    z[i] = z[k]
                else:
                    l = i
                    while r < n and s[r - l] == s[r]:
                        r += 1
                    z[i] = r - l
                    r -= 1
        return z
    
    def solve(self, n, k, s):
        # 移除k=0处理分支
        if k == 0:
            return '0' * n
        z = self.compute_z(s)
        ans = [0] * (n + 2)  # 增加缓冲空间
        
        for lenAB in range(1, n + 1):
            # 检查前k个B是否满足条件
            valid = True
            current_pos = lenAB
            for _ in range(k - 1):
                if current_pos >= n:
                    valid = False
                    break
                required = lenAB
                if current_pos + required > n:
                    if z[current_pos] < n - current_pos:
                        valid = False
                        break
                else:
                    if z[current_pos] < required:
                        valid = False
                        break
                current_pos += lenAB
            
            if not valid:
                continue
            
            # 计算可选A的长度范围
            l = lenAB * k - 1
            if l >= n:
                continue
            
            a_start = lenAB * k
            if a_start >= n:
                max_a = 0
            else:
                max_a = z[a_start]
            
            possible_a = min(lenAB, max_a)
            r = l + possible_a
            
            # 修正差分数组标记
            end = min(r, n)
            ans[l] += 1
            if end < n:
                ans[end + 1] -= 1
            else:
                ans[n] -= 1
        
        # 重建结果数组
        res = []
        current = 0
        for i in range(n):
            current += ans[i]
            res.append('1' if current > 0 else '0')
        return ''.join(res)
    
    @staticmethod
    def prompt_func(question_case):
        return (
            "Om Nom needs to cut a bead necklace following specific pattern rules.\n"
            f"Given a string of {question_case['n']} beads: {question_case['s']}\n"
            f"and k = {question_case['k']}, determine for each prefix length (1-{question_case['n']}) "
            "if it forms a regular pattern (A+B+A+B+...+A).\n\n"
            "Output should be a string of '0's and '1's where '1' indicates valid at that position.\n"
            "Put your answer between [answer] and [/answer], e.g.:\n"
            "[answer]010111[/answer]"
        )
    
    @staticmethod
    def extract_output(output):
        # 严格匹配答案格式，允许前后有空格
        matches = re.findall(r'\[answer\s*\]\s*([01]+)\s*\[/?answer\s*\]', output, re.IGNORECASE)
        return matches[-1] if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 添加长度校验
        if len(solution) != identity['n']:
            return False
        return solution == identity['correct_output']
