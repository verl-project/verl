"""# 

### 谜题描述
Jeff loves regular bracket sequences.

Today Jeff is going to take a piece of paper and write out the regular bracket sequence, consisting of nm brackets. Let's number all brackets of this sequence from 0 to nm - 1 from left to right. Jeff knows that he is going to spend ai mod n liters of ink on the i-th bracket of the sequence if he paints it opened and bi mod n liters if he paints it closed.

You've got sequences a, b and numbers n, m. What minimum amount of ink will Jeff need to paint a regular bracket sequence of length nm?

Operation x mod y means taking the remainder after dividing number x by number y.

Input

The first line contains two integers n and m (1 ≤ n ≤ 20; 1 ≤ m ≤ 107; m is even). The next line contains n integers: a0, a1, ..., an - 1 (1 ≤ ai ≤ 10). The next line contains n integers: b0, b1, ..., bn - 1 (1 ≤ bi ≤ 10). The numbers are separated by spaces.

Output

In a single line print the answer to the problem — the minimum required amount of ink in liters.

Examples

Input

2 6
1 2
2 1


Output

12


Input

1 10000000
2
3


Output

25000000

Note

In the first test the optimal sequence is: ()()()()()(), the required number of ink liters is 12.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int N = 2e5 + 10;
int n, m, a[N], b[N];
int dp[22][44];
struct uzi {
  long long A[44][44];
  uzi() { memset(A, 0x3f3f3f, sizeof A); };
} G;
uzi operator*(const uzi& a, const uzi& b) {
  uzi c;
  for (int i = 0; i <= 40; i++) {
    for (int j = 0; j <= 40; j++) {
      for (int k = 0; k <= 40; k++) {
        c.A[i][j] = min(a.A[i][k] + b.A[k][j], c.A[i][j]);
      }
    }
  }
  return c;
}
uzi pm() {
  uzi c;
  for (int i = 0; i <= 40; i++) c.A[i][i] = 0;
  while (m) {
    if (m & 1) c = c * G;
    G = G * G;
    m >>= 1;
  }
  return c;
}
int main() {
  ios::sync_with_stdio(false);
  cin >> n >> m;
  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }
  for (int i = 0; i < n; i++) {
    cin >> b[i];
  }
  for (int i = 0; i <= 40; i++) {
    for (int j = 0; j <= n; j++) {
      for (int k = 0; k <= 40; k++) {
        dp[j][k] = 1e9;
        if (!j) {
          if (k == i) dp[j][k] = 0;
        } else {
          if (k) {
            dp[j][k] = min(dp[j][k], dp[j - 1][k - 1] + a[j - 1]);
          }
          if (k + 1 <= 40) {
            dp[j][k] = min(dp[j][k], dp[j - 1][k + 1] + b[j - 1]);
          }
        }
      }
      for (int k = 0; k <= 40; k++) {
        G.A[i][k] = dp[n][k];
      }
    }
  }
  cout << pm().A[0][0];
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Ejeffandbracketsbootcamp(Basebootcamp):
    def __init__(self, max_n=5, max_m=100):
        self.max_n = max_n
        self.max_m = max_m
    
    def case_generator(self):
        n = random.randint(1, self.max_n)
        m = random.randint(1, self.max_m // 2) * 2
        a = [random.randint(1, 10) for _ in range(n)]
        b = [random.randint(1, 10) for _ in range(n)]
        return {
            'n': n,
            'm': m,
            'a': a,
            'b': b
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        m = question_case['m']
        a = ' '.join(map(str, question_case['a']))
        b = ' '.join(map(str, question_case['b']))
        prompt = f"""Jeff需要绘制一个长度为{n * m}的合法括号序列。每个括号的开闭状态决定消耗的墨水量：

规则：
1. 序列必须合法（正确嵌套）
2. 第i个括号（0索引）若为开括号，消耗a[i%n]升墨水；若为闭，消耗b[i%n]升
3. 参数：n={n}, m={m}（保证总长度是偶数）

输入格式：
{n} {m}
{a}
{b}

请计算最小墨水用量。将最终答案放入[answer]标签内，例如：[answer]42[/answer]"""

        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            n = identity['n']
            m = identity['m']
            a = identity['a']
            b = identity['b']
            correct = cls.compute_min_ink(n, m, a, b)
            return solution == correct
        except:
            return False
    
    @staticmethod
    def compute_min_ink(n, m, a, b):
        class Uzi:
            def __init__(self):
                self.A = [[float('inf')] * 41 for _ in range(41)]
        
        def multiply(a_mat, b_mat):
            res = Uzi()
            for i in range(41):
                for j in range(41):
                    min_val = float('inf')
                    for k in range(41):
                        if a_mat.A[i][k] + b_mat.A[k][j] < min_val:
                            min_val = a_mat.A[i][k] + b_mat.A[k][j]
                    res.A[i][j] = min_val
            return res
        
        G = Uzi()
        for i in range(41):
            dp = [[float('inf')] * 41 for _ in range(n+1)]
            dp[0][i] = 0
            for j in range(1, n+1):
                for k in range(41):
                    if dp[j-1][k] == float('inf'):
                        continue
                    # Open bracket
                    if k < 40:
                        new_k = k + 1
                        cost = a[(j-1) % n]  # Fixed modulo position
                        if dp[j][new_k] > dp[j-1][k] + cost:
                            dp[j][new_k] = dp[j-1][k] + cost
                    # Close bracket
                    if k > 0:
                        new_k = k - 1
                        cost = b[(j-1) % n]  # Fixed modulo position
                        if dp[j][new_k] > dp[j-1][k] + cost:
                            dp[j][new_k] = dp[j-1][k] + cost
            for k in range(41):
                G.A[i][k] = dp[n][k]
        
        # Matrix exponentiation
        result = Uzi()
        for i in range(41):
            result.A[i][i] = 0
        exponent = m
        current = G
        while exponent > 0:
            if exponent % 2 == 1:
                result = multiply(result, current)
            current = multiply(current, current)
            exponent = exponent // 2
        return result.A[0][0]
