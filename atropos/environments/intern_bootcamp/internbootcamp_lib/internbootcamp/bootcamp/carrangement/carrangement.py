"""# 

### 谜题描述
In the year 2500 the annual graduation ceremony in the German University in Cairo (GUC) has run smoothly for almost 500 years so far.

The most important part of the ceremony is related to the arrangement of the professors in the ceremonial hall.

Traditionally GUC has n professors. Each professor has his seniority level. All seniorities are different. Let's enumerate the professors from 1 to n, with 1 being the most senior professor and n being the most junior professor.

The ceremonial hall has n seats, one seat for each professor. Some places in this hall are meant for more senior professors than the others. More specifically, m pairs of seats are in \"senior-junior\" relation, and the tradition requires that for all m pairs of seats (ai, bi) the professor seated in \"senior\" position ai should be more senior than the professor seated in \"junior\" position bi.

GUC is very strict about its traditions, which have been carefully observed starting from year 2001. The tradition requires that: 

  * The seating of the professors changes every year. 
  * Year 2001 ceremony was using lexicographically first arrangement of professors in the ceremonial hall. 
  * Each consecutive year lexicographically next arrangement of the professors is used. 



The arrangement of the professors is the list of n integers, where the first integer is the seniority of the professor seated in position number one, the second integer is the seniority of the professor seated in position number two, etc.

Given n, the number of professors, y, the current year and m pairs of restrictions, output the arrangement of the professors for this year.

Input

The first line contains three integers n, y and m (1 ≤ n ≤ 16, 2001 ≤ y ≤ 1018, 0 ≤ m ≤ 100) — the number of professors, the year for which the arrangement should be computed, and the number of pairs of seats for which the seniority relation should be kept, respectively.

The next m lines contain one pair of integers each, \"ai bi\", indicating that professor on the ai-th seat is more senior than professor on the bi-th seat (1 ≤ ai, bi ≤ n, ai ≠ bi). Some pair may be listed more than once.

Please, do not use the %lld specificator to read or write 64-bit integers in С++. It is preferred to use the cin stream (you may also use the %I64d specificator).

Output

Print the order in which the professors should be seated in the requested year.

If by this year the GUC would have ran out of arrangements, or the given \"senior-junior\" relation are contradictory, print \"The times have changed\" (without quotes).

Examples

Input

3 2001 2
1 2
2 3


Output

1 2 3


Input

7 2020 6
1 2
1 3
2 4
2 5
3 6
3 7


Output

1 2 3 7 4 6 5


Input

10 3630801 0


Output

The times have changed


Input

3 2001 3
1 2
2 3
3 1


Output

The times have changed

Note

In the first example the lexicographically first order of seating is 1 2 3.

In the third example the GUC will run out of arrangements after the year 3630800.

In the fourth example there are no valid arrangements for the seating.

The lexicographical comparison of arrangements is performed by the < operator in modern programming languages. The arrangement a is lexicographically less that the arrangement b, if there exists such i (1 ≤ i ≤ n), that ai < bi, and for any j (1 ≤ j < i) aj = bj.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
long long f[1 << 16], y;
int g[16], ans[16], i, k, z, Z, n, m, a, b;
bool mark[16 + 1];
int main() {
  scanf(\"%d%I64d%d\", &n, &y, &m);
  y -= 2001;
  while (m--) {
    scanf(\"%d%d\", &a, &b);
    g[--b] |= (1 << (--a));
  }
  Z = (1 << (n)) - 1;
  memset(ans, -1, sizeof(ans));
  for (i = 0; i < n; ++i) {
    for (ans[i] = 0;; ++ans[i]) {
      if (!mark[ans[i]]) {
        if (ans[i] >= n) return puts(\"The times have changed\");
        memset(f, 0, sizeof(f));
        f[0] = 1;
        for (z = 0; z <= Z; ++z) {
          if (f[z]) {
            b = 0;
            for (int x = 0; x < n; x++) {
              if (z & (1 << x)) b++;
            }
            for (k = 0; k < n; ++k) {
              if (!(z & (1 << (k)))) {
                if ((ans[k] == -1 || ans[k] == b) && ((z & g[k]) == g[k])) {
                  f[z | (1 << (k))] += f[z];
                }
              }
            }
          }
        }
        if (y >= f[Z]) {
          y -= f[Z];
        } else {
          mark[ans[i]] = true;
          break;
        }
      }
    }
  }
  for (i = 0; i < n; ++i) printf(\"%d%c\", ans[i] + 1, i + 1 == n ? '\n' : ' ');
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

def topological_sort(g, n):
    in_degree = [0] * n
    for u in range(n):
        for v in g[u]:
            in_degree[v] += 1
    queue = [u for u in range(n) if in_degree[u] == 0]
    order = []
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in g[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    return len(order) == n

def compute_answer(n, y_original, m, constraints):
    if n == 0:
        return "The times have changed"
    
    y = y_original - 2001
    if y < 0:
        return "The times have changed"
    
    adjacency = [[] for _ in range(n)]
    for a, b in constraints:
        adjacency[a-1].append(b-1)
    
    # Check if constraints form a DAG
    if not topological_sort(adjacency, n):
        return "The times have changed"
    
    # Dynamic programming for valid permutations
    g = [0] * n
    for a, b in constraints:
        g[b-1] |= 1 << (a-1)
    
    ans = [-1] * n
    used = [False] * n
    
    for pos in range(n):
        ans[pos] = 0
        while True:
            if ans[pos] >= n:
                return "The times have changed"
            
            if not used[ans[pos]]:
                dp = [0] * (1 << n)
                dp[0] = 1
                for mask in range(1 << n):
                    if not dp[mask]:
                        continue
                    cnt = bin(mask).count('1')
                    for seat in range(n):
                        if mask & (1 << seat):
                            continue
                        if ans[seat] != -1 and ans[seat] != cnt:
                            continue
                        if (mask & g[seat]) == g[seat]:
                            new_mask = mask | (1 << seat)
                            dp[new_mask] += dp[mask]
                
                total = dp[(1 << n) - 1]
                if total <= y:
                    y -= total
                    ans[pos] += 1
                else:
                    used[ans[pos]] = True
                    break
            else:
                ans[pos] += 1
    
    return ' '.join(str(x+1 for x in ans)) if -1 not in ans else "The times have changed"

class Carrangementbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n_range = params.get('n_range', (1, 6))
        self.m_max = params.get('m_max', 15)
        self.seed = params.get('seed', None)
        if self.seed is not None:
            random.seed(self.seed)
    
    def case_generator(self):
        n = random.randint(*self.n_range)
        
        # Generate constraints
        m = 0
        constraints = []
        if n > 1:
            m = random.randint(0, min(self.m_max, n*(n-1)))
            for _ in range(m):
                a, b = random.sample(range(1, n+1), k=2)
                constraints.append((a, b))
        
        # Generate year with 30% chance to create impossible cases
        if random.random() < 0.3:
            y = random.randint(2001 + 10**16, 2001 + 10**18)
        else:
            y = random.randint(2001, 2001 + 10000)
        
        return {
            'n': n,
            'y': y,
            'm': m,
            'constraints': constraints
        }
    
    @staticmethod
    def prompt_func(case):
        problem_desc = [
            f"In the year {case['y']}, the German University in Cairo must arrange {case['n']} professors.",
            "Rules:",
            "1. Professors are ranked 1 (most senior) to n (most junior)",
            f"2. Must satisfy {case['m']} seniority constraints:",
        ]
        for a, b in case['constraints']:
            problem_desc.append(f"   - Seat {a} must be more senior than seat {b}")
        problem_desc.extend([
            "3. Arrangements follow lex order starting from 2001",
            "4. Output 'The times have changed' if impossible",
            "Format: Place your answer within [answer][/answer] tags."
        ])
        return '\n'.join(problem_desc)
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            expected = compute_answer(
                identity['n'],
                identity['y'],
                identity['m'],
                identity['constraints']
            )
            return solution == expected
        except:
            return False
