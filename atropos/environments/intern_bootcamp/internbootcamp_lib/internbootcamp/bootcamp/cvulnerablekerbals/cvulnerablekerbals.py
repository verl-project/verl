"""# 

### 谜题描述
You are given an integer m, and a list of n distinct integers between 0 and m - 1.

You would like to construct a sequence satisfying the properties:

  * Each element is an integer between 0 and m - 1, inclusive. 
  * All prefix products of the sequence modulo m are distinct. 
  * No prefix product modulo m appears as an element of the input list. 
  * The length of the sequence is maximized. 



Construct any sequence satisfying the properties above.

Input

The first line of input contains two integers n and m (0 ≤ n < m ≤ 200 000) — the number of forbidden prefix products and the modulus.

If n is non-zero, the next line of input contains n distinct integers between 0 and m - 1, the forbidden prefix products. If n is zero, this line doesn't exist.

Output

On the first line, print the number k, denoting the length of your sequence.

On the second line, print k space separated integers, denoting your sequence.

Examples

Input

0 5


Output

5
1 2 4 3 0


Input

3 10
2 9 1


Output

6
3 9 2 9 8 0

Note

For the first case, the prefix products of this sequence modulo m are [1, 2, 3, 4, 0].

For the second case, the prefix products of this sequence modulo m are [3, 7, 4, 6, 8, 0].

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int N = 2e5 + 5;
int n, m, w[N], vis[N], dp[N], pre[N], cnt;
vector<int> g[N];
long long exgcd(long long &x, long long &y, long long a, long long b) {
  if (!b) {
    x = 1, y = 0;
    return a;
  }
  long long g = exgcd(x, y, b, a % b);
  long long t = x;
  x = y;
  y = t - a / b * y;
  return g;
}
long long getans(long long a, long long b) {
  long long x, y;
  long long c = exgcd(x, y, a, m);
  x = (x * b / c % m + m) % m;
  return x;
}
int main() {
  long long x, y;
  scanf(\"%d%d\", &n, &m);
  for (int i = 1; i <= n; i++) {
    scanf(\"%lld\", &x);
    vis[x] = 1;
  }
  for (int i = 0; i < m; i++)
    if (!vis[i]) g[exgcd(x, y, (long long)i, (long long)m)].push_back(i);
  for (int i = 1; i <= m; i++) {
    if (m % i) continue;
    dp[i] += g[i].size();
    for (int j = i + i; j <= m; j += i) {
      if (m % j) continue;
      if (dp[j] <= dp[i]) {
        dp[j] = dp[i];
        pre[j] = i;
      }
    }
  }
  int now = m;
  while (1) {
    for (int i = 0; i < (int)g[now].size(); i++) w[cnt++] = g[now][i];
    if (now == 1) break;
    now = pre[now];
  }
  printf(\"%d\n\", cnt);
  printf(\"%d \", w[cnt - 1]);
  for (int i = cnt - 1; i; i--) printf(\"%lld \", getans(w[i], w[i - 1]));
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import math
import random
from collections import defaultdict
from bootcamp import Basebootcamp

def exgcd(a, b):
    if b == 0:
        return (a, 1, 0)
    else:
        g, x, y = exgcd(b, a % b)
        return (g, y, x - (a // b) * y)

def generate_solution_for_m(m):
    vis = set()
    g = defaultdict(list)
    for i in range(m):
        if i not in vis:
            g_val = math.gcd(i, m)
            g[g_val].append(i)
    
    divisors = [d for d in range(1, m + 1) if m % d == 0]
    divisors.sort()
    
    dp = {d: 0 for d in divisors}
    pre = {d: None for d in divisors}
    
    for d in divisors:
        dp[d] = len(g.get(d, []))
        j = 2 * d
        while j <= m:
            if j not in divisors:
                j += d
                continue
            if dp[j] < dp[d]:
                dp[j] = dp[d]
                pre[j] = d
            elif dp[j] == dp[d]:
                if pre[j] is None or pre[j] < d:
                    pre[j] = d
            j += d
    
    current_d = m
    w = []
    while True:
        w.extend(g.get(current_d, []))
        if current_d == 1:
            break
        current_d = pre.get(current_d)
        if current_d is None:
            break
    
    if not w:
        return 0, []
    
    sequence = []
    sequence.append(w[-1])
    for i in range(len(w)-1, 0, -1):
        a = w[i]
        b = w[i-1]
        g_val, x, y = exgcd(a, m)
        assert b % g_val == 0, "No solution"
        x0 = (x * (b // g_val)) % (m // g_val)
        sequence.append(x0)
    
    current = 1
    prefix_products = []
    for num in sequence:
        current = (current * num) % m
        prefix_products.append(current)
    
    return len(sequence), prefix_products

class Cvulnerablekerbalsbootcamp(Basebootcamp):
    def __init__(self, m_min=2, m_max=20):
        self.m_min = m_min
        self.m_max = m_max
    
    def case_generator(self):
        m = random.randint(self.m_min, self.m_max)
        k_max, P = generate_solution_for_m(m)
        all_numbers = set(range(m))
        P_set = set(P)
        C = list(all_numbers - P_set)
        max_n = min(len(C), m-1)
        n = random.randint(0, max_n) if max_n >= 0 else 0
        L = random.sample(C, n) if n > 0 else []
        L.sort()
        return {
            'm': m,
            'n': n,
            'forbidden': L,
            'k_max': k_max
        }
    
    @staticmethod
    def prompt_func(question_case):
        m = question_case['m']
        n = question_case['n']
        forbidden = question_case['forbidden']
        problem = f"You are given an integer m and a list of n distinct integers between 0 and m-1.\n\n"
        problem += "Your task is to construct a sequence satisfying the following properties:\n"
        problem += "1. Each element is an integer between 0 and m-1, inclusive.\n"
        problem += "2. All prefix products of the sequence modulo m are distinct.\n"
        problem += "3. No prefix product modulo m appears in the forbidden list.\n"
        problem += "4. The length of the sequence is maximized.\n\n"
        problem += "Input:\n"
        problem += f"{n} {m}\n"
        if n > 0:
            problem += " ".join(map(str, forbidden)) + "\n"
        problem += "\nOutput your answer in the following format:\n"
        problem += "[answer]\n<length>\n<sequence elements space-separated>\n[/answer]"
        return problem
    
    @staticmethod
    def extract_output(output):
        pattern = r'\[answer\](.*?)\[/answer\]'
        matches = re.findall(pattern, output, re.DOTALL)
        if not matches:
            return None
        answer = matches[-1].strip()
        lines = [l.strip() for l in answer.split('\n') if l.strip()]
        if len(lines) < 2:
            return None
        try:
            k = int(lines[0])
            elements = list(map(int, lines[1].split()))
            if len(elements) != k:
                return None
            return elements
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False
        m = identity['m']
        forbidden = set(identity['forbidden'])
        k_max = identity['k_max']
        
        if len(solution) != k_max:
            return False
        
        for num in solution:
            if not (0 <= num < m):
                return False
        
        current = 1
        prefix_products = []
        for num in solution:
            current = (current * num) % m
            if current in forbidden or current in prefix_products:
                return False
            prefix_products.append(current)
        
        return True
