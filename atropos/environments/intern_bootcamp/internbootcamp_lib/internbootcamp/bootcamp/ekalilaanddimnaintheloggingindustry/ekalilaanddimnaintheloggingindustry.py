"""# 

### 谜题描述
Kalila and Dimna are two jackals living in a huge jungle. One day they decided to join a logging factory in order to make money. 

The manager of logging factory wants them to go to the jungle and cut n trees with heights a1, a2, ..., an. They bought a chain saw from a shop. Each time they use the chain saw on the tree number i, they can decrease the height of this tree by one unit. Each time that Kalila and Dimna use the chain saw, they need to recharge it. Cost of charging depends on the id of the trees which have been cut completely (a tree is cut completely if its height equal to 0). If the maximum id of a tree which has been cut completely is i (the tree that have height ai in the beginning), then the cost of charging the chain saw would be bi. If no tree is cut completely, Kalila and Dimna cannot charge the chain saw. The chainsaw is charged in the beginning. We know that for each i < j, ai < aj and bi > bj and also bn = 0 and a1 = 1. Kalila and Dimna want to cut all the trees completely, with minimum cost. 

They want you to help them! Will you?

Input

The first line of input contains an integer n (1 ≤ n ≤ 105). The second line of input contains n integers a1, a2, ..., an (1 ≤ ai ≤ 109). The third line of input contains n integers b1, b2, ..., bn (0 ≤ bi ≤ 109).

It's guaranteed that a1 = 1, bn = 0, a1 < a2 < ... < an and b1 > b2 > ... > bn.

Output

The only line of output must contain the minimum cost of cutting all the trees completely.

Please, do not write the %lld specifier to read or write 64-bit integers in С++. It is preferred to use the cin, cout streams or the %I64d specifier.

Examples

Input

5
1 2 3 4 5
5 4 3 2 0


Output

25


Input

6
1 2 3 10 20 30
6 5 4 3 2 0


Output

138

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const long long maxL = 20;
long long mod = 1e9 + 7;
long long mod1 = 1e9 + 7;
long long inf = 1e12;
long long linf = 1e18;
const long long maxn = 1e5 + 100;
const long long maxk = 210;
const long long baze = 330;
vector<pair<long long, long long> > vc;
long long n, a[maxn], b[maxn], dp[maxn];
long long saghf(long long x, long long y) {
  if (y < 0) y = -y, x = -x;
  return (x + y - 1) / y;
}
long long when(long long i, long long j) {
  return saghf(dp[i] - dp[j], b[j] - b[i]);
}
void add(long long x) {
  while (((long long)((vc).size())) and
         when(vc.back().second, x) <= vc.back().first)
    vc.pop_back();
  if (((long long)((vc).size())))
    vc.push_back(pair<long long, long long>(when(vc.back().second, x), x));
  else
    vc.push_back(pair<long long, long long>(0, x));
}
long long get(long long x) {
  return (--lower_bound((vc).begin(), (vc).end(),
                        pair<long long, long long>(x + 1, 0)))
      ->second;
}
int32_t main() {
  cin >> n;
  for (long long i = 0; i < n; i++) cin >> a[i];
  for (long long i = 0; i < n; i++) cin >> b[i];
  add(0);
  for (long long i = 1; i < n; i++) {
    long long j = get(a[i]);
    dp[i] = dp[j] + a[i] * b[j];
    add(i);
  }
  cout << dp[n - 1] << endl;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import bisect
from bootcamp import Basebootcamp

class Ekalilaanddimnaintheloggingindustrybootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10):
        self.min_n = min_n
        self.max_n = max_n
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        # Generate strictly increasing a starting with 1
        a = [1]
        for _ in range(n-1):
            delta = random.randint(1, 5)
            a.append(a[-1] + delta)
        # Generate strictly decreasing b ending with 0
        b = [0]
        for _ in range(n-1):
            delta = random.randint(1, 10)
            new_val = b[0] + delta
            b.insert(0, new_val)  # Ensure strictly decreasing
        # Compute correct answer
        correct_answer = self.__class__.compute_min_cost(n, a, b)
        return {
            "n": n,
            "a": a,
            "b": b,
            "correct_answer": correct_answer
        }
    
    @staticmethod
    def compute_min_cost(n, a, b):
        if n == 0:
            return 0
        dp = [0] * n
        vc = []  # Convex hull trick structure (time, index)

        def saghf(x, y):
            if y < 0:
                x, y = -x, -y
            if y == 0:
                return float('inf') if x > 0 else -float('inf')
            return (x + y - 1) // y

        def when(i, j):
            return saghf(dp[i] - dp[j], b[j] - b[i])

        def add(x):
            while vc and when(vc[-1][1], x) <= vc[-1][0]:
                vc.pop()
            if not vc:
                vc.append((0, x))
            else:
                t = when(vc[-1][1], x)
                vc.append((t, x))

        def get_current(x_val):
            pos = bisect.bisect_left(vc, (x_val+1, )) - 1
            return vc[pos][1] if vc else 0

        add(0)
        for i in range(1, n):
            j = get_current(a[i])
            dp[i] = dp[j] + a[i] * b[j]
            add(i)
        return dp[-1]
    
    @staticmethod
    def prompt_func(question_case) -> str:
        return f"""Cut trees {question_case['a']} with costs {question_case['b']}. Minimum cost? Put answer in [answer]...[/answer]."""
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](\d+)\[/answer\]', output)
        return int(matches[-1]) if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
