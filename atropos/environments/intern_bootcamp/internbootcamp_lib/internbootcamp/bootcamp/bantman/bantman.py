"""# 

### 谜题描述
Scott Lang is at war with Darren Cross. There are n chairs in a hall where they are, numbered with 1, 2, ..., n from left to right. The i-th chair is located at coordinate xi. Scott is on chair number s and Cross is on chair number e. Scott can jump to all other chairs (not only neighboring chairs). He wants to start at his position (chair number s), visit each chair exactly once and end up on chair number e with Cross. 

As we all know, Scott can shrink or grow big (grow big only to his normal size), so at any moment of time he can be either small or large (normal). The thing is, he can only shrink or grow big while being on a chair (not in the air while jumping to another chair). Jumping takes time, but shrinking and growing big takes no time. Jumping from chair number i to chair number j takes |xi - xj| seconds. Also, jumping off a chair and landing on a chair takes extra amount of time. 

If Scott wants to jump to a chair on his left, he can only be small, and if he wants to jump to a chair on his right he should be large.

Jumping off the i-th chair takes:

  * ci extra seconds if he's small. 
  * di extra seconds otherwise (he's large). 



Also, landing on i-th chair takes:

  * bi extra seconds if he's small. 
  * ai extra seconds otherwise (he's large). 



In simpler words, jumping from i-th chair to j-th chair takes exactly:

  * |xi - xj| + ci + bj seconds if j < i. 
  * |xi - xj| + di + aj seconds otherwise (j > i). 



Given values of x, a, b, c, d find the minimum time Scott can get to Cross, assuming he wants to visit each chair exactly once.

Input

The first line of the input contains three integers n, s and e (2 ≤ n ≤ 5000, 1 ≤ s, e ≤ n, s ≠ e) — the total number of chairs, starting and ending positions of Scott.

The second line contains n integers x1, x2, ..., xn (1 ≤ x1 < x2 < ... < xn ≤ 109).

The third line contains n integers a1, a2, ..., an (1 ≤ a1, a2, ..., an ≤ 109).

The fourth line contains n integers b1, b2, ..., bn (1 ≤ b1, b2, ..., bn ≤ 109).

The fifth line contains n integers c1, c2, ..., cn (1 ≤ c1, c2, ..., cn ≤ 109).

The sixth line contains n integers d1, d2, ..., dn (1 ≤ d1, d2, ..., dn ≤ 109).

Output

Print the minimum amount of time Scott needs to get to the Cross while visiting each chair exactly once.

Example

Input

7 4 3
8 11 12 16 17 18 20
17 16 20 2 20 5 13
17 8 8 16 12 15 13
12 4 16 4 15 7 6
8 14 2 11 17 12 8


Output

139

Note

In the sample testcase, an optimal solution would be <image>. Spent time would be 17 + 24 + 23 + 20 + 33 + 22 = 139.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
long long f[5010][5010];
int a[5010], b[5010], c[5010], d[5010], x[5010];
int main() {
  int n, s, e;
  scanf(\"%d%d%d\", &n, &s, &e);
  for (int i = 1; i <= n; i++) scanf(\"%d\", &x[i]);
  for (int i = 1; i <= n; i++) scanf(\"%d\", &a[i]), a[i] += x[i];
  for (int i = 1; i <= n; i++) scanf(\"%d\", &b[i]), b[i] -= x[i];
  for (int i = 1; i <= n; i++) scanf(\"%d\", &c[i]), c[i] += x[i];
  for (int i = 1; i <= n; i++) scanf(\"%d\", &d[i]), d[i] -= x[i];
  memset(f, 0x3f, sizeof(f));
  f[0][0] = 0;
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= i; j++) {
      if (s >= i || e >= i || j > 2)
        f[i][j] = std::min(f[i][j], f[i - 1][j - 1] + (i != s ? b[i] : 0) +
                                        (i != e ? d[i] : 0));
      if (i != e && (s >= i || j > 1))
        f[i][j] = std::min(f[i][j], f[i - 1][j] + c[i] + (i != s ? b[i] : 0));
      if (i != s && (e >= i || j > 1))
        f[i][j] = std::min(f[i][j], f[i - 1][j] + a[i] + (i != e ? d[i] : 0));
      if (i != e && i != s)
        f[i][j] = std::min(f[i][j], f[i - 1][j + 1] + a[i] + c[i]);
    }
  printf(\"%lld\n\", f[n][1]);
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def calculate_min_time(n, s, e, x_list, a_list, b_list, c_list, d_list):
    x = [0] * (n + 2)  # 1-based index
    a = [0] * (n + 2)
    b = [0] * (n + 2)
    c = [0] * (n + 2)
    d = [0] * (n + 2)
    
    for i in range(1, n+1):
        x[i] = x_list[i-1]
        a[i] = a_list[i-1] + x[i]
        b[i] = b_list[i-1] - x[i]
        c[i] = c_list[i-1] + x[i]
        d[i] = d_list[i-1] - x[i]
    
    INF = float('inf')
    f = [[INF] * (n + 2) for _ in range(n + 2)]
    f[0][0] = 0
    
    for i in range(1, n+1):
        for j in range(1, i+1):
            # Case 1: Start new chain
            if (s >= i or e >= i or j > 2):
                cost = 0
                if i != s:
                    cost += b[i]
                if i != e:
                    cost += d[i]
                if f[i-1][j-1] + cost < f[i][j]:
                    f[i][j] = f[i-1][j-1] + cost
            
            # Case 2: Extend left end
            if i != e:
                if (s >= i or j > 1):
                    cost = c[i]
                    if i != s:
                        cost += b[i]
                    if f[i-1][j] + cost < f[i][j]:
                        f[i][j] = f[i-1][j] + cost
            
            # Case 3: Extend right end
            if i != s:
                if (e >= i or j > 1):
                    cost = a[i]
                    if i != e:
                        cost += d[i]
                    if f[i-1][j] + cost < f[i][j]:
                        f[i][j] = f[i-1][j] + cost
            
            # Case 4: Merge two chains
            if i != s and i != e:
                if j+1 <= i-1:
                    cost = a[i] + c[i]
                    if f[i-1][j+1] + cost < f[i][j]:
                        f[i][j] = f[i-1][j+1] + cost
    
    return f[n][1]

class Bantmanbootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=10):
        self.min_n = min_n
        self.max_n = max_n
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        chairs = list(range(1, n+1))
        s, e = random.sample(chairs, 2)
        
        # Generate strictly increasing x
        x = []
        current = random.randint(1, 100)
        for _ in range(n):
            current += random.randint(1, 100)
            x.append(current)
        
        # Generate parameters with valid ranges
        def gen_params():
            return [random.randint(1, 10**9) for _ in range(n)]
        
        return {
            'n': n,
            's': s,
            'e': e,
            'x': x,
            'a': gen_params(),
            'b': gen_params(),
            'c': gen_params(),
            'd': gen_params()
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        s = question_case['s']
        e = question_case['e']
        x = question_case['x']
        a = question_case['a']
        b = question_case['b']
        c = question_case['c']
        d = question_case['d']
        
        problem = f"""Scott Lang needs to navigate between chairs to reach Darren Cross with minimal time. 

Rules:
1. Chairs are numbered 1-{n} left to right at coordinates {x}
2. Start at chair {s}, end at chair {e}, visit each chair exactly once
3. State changes (small/large) only allowed on chairs:
   - Small state for jumps left (chair j < current i)
   - Large state for jumps right (chair j > current i)
4. Time calculation:
   - Jump left: |x_i - x_j| + c_i (small jump cost) + b_j (small landing cost)
   - Jump right: |x_i - x_j| + d_i (large jump cost) + a_j (large landing cost)

Input format:
{n} {s} {e}
{' '.join(map(str, x))}
{' '.join(map(str, a))}
{' '.join(map(str, b))}
{' '.join(map(str, c))}
{' '.join(map(str, d))}

Calculate the minimum time required. Put your final numerical answer within [answer] tags like: [answer]139[/answer]"""
        return problem
    
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
            s = identity['s']
            e = identity['e']
            x = identity['x']
            a = identity['a']
            b = identity['b']
            c = identity['c']
            d = identity['d']
            
            expected = calculate_min_time(n, s, e, x, a, b, c, d)
            return solution == expected
        except:
            return False
