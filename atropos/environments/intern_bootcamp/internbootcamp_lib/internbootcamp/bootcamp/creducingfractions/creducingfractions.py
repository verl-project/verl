"""# 

### 谜题描述
To confuse the opponents, the Galactic Empire represents fractions in an unusual format. The fractions are represented as two sets of integers. The product of numbers from the first set gives the fraction numerator, the product of numbers from the second set gives the fraction denominator. However, it turned out that the programs that work with fractions in this representations aren't complete, they lack supporting the operation of reducing fractions. Implement this operation and the Empire won't forget you.

Input

The first input line contains two space-separated integers n, m (1 ≤ n, m ≤ 105) that show how many numbers the first set (the numerator) and the second set (the denominator) contain, correspondingly.

The second line contains n space-separated integers: a1, a2, ..., an (1 ≤ ai ≤ 107) — the numbers that are multiplied to produce the numerator.

The third line contains m space-separated integers: b1, b2, ..., bm (1 ≤ bi ≤ 107) — the numbers that are multiplied to produce the denominator.

Output

Print the answer to the problem in the form, similar to the form of the input data. The number of values in the sets you print nout, mout must satisfy the inequality 1 ≤ nout, mout ≤ 105, and the actual values in the sets aout, i and bout, i must satisfy the inequality 1 ≤ aout, i, bout, i ≤ 107. 

Separate the values in the lines by spaces. The printed fraction must be reduced, that is, there mustn't be such integer x (x > 1), that the numerator and the denominator of the printed fraction are divisible by x. If there are several matching answers, print any of them.

Examples

Input

3 2
100 5 2
50 10


Output

2 3
2 1
1 1 1


Input

4 3
2 5 10 20
100 1 3


Output

1 1
20
3

Note

In the first test sample the numerator equals 1000, the denominator equals 500. If we reduce fraction 1000/500 by the greatest common divisor of the numerator and the denominator (by 500), we obtain fraction 2/1.

In the second test sample the numerator equals 2000, the denominator equals 300. If we reduce fraction 2000/300 by the greatest common divisor of the numerator and the denominator (by 100), we obtain fraction 20/3.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int N = (int)1e7;
int n, m, a[200000], b[200000], pn, p[N];
bool u[N + 2];
vector<int> e[664600], g[664600];
int main() {
  p[pn++] = 2;
  for (int i = 3; i < N; i += 2)
    if (!u[i]) {
      p[pn++] = i;
      if (1LL * i * i > N) continue;
      for (int j = i * i; j < N; j += (i << 1)) u[j] = true;
    }
  scanf(\"%d%d\", &n, &m);
  for (int i = 1, x; i <= n; ++i) {
    scanf(\"%d\", &x);
    a[i] = x;
    for (int j = 0; j < 447; ++j)
      if (x % p[j] == 0) {
        e[j].push_back(i);
        while (x % p[j] == 0) {
          x /= p[j];
        }
      }
    if (x > 1) e[lower_bound(p, p + pn, x) - p].push_back(i);
  }
  for (int i = 1, x; i <= m; ++i) {
    scanf(\"%d\", &x);
    b[i] = x;
    for (int j = 0; j < 447; ++j)
      if (x % p[j] == 0) {
        while (x % p[j] == 0) {
          x /= p[j];
        }
        g[j].push_back(i);
      }
    if (x > 1) g[lower_bound(p, p + pn, x) - p].push_back(i);
  }
  for (int i = 0; i < pn; ++i) {
    int an = 0, bn = 0;
    while (an < e[i].size() && bn < g[i].size()) {
      int c1 = 0, c2 = 0, x = a[e[i][an]], y = b[g[i][bn]];
      while (x % p[i] == 0) {
        x /= p[i];
        c1++;
      }
      while (y % p[i] == 0) {
        y /= p[i];
        c2++;
      }
      if (c1 > c2) {
        c1 -= c2;
        for (int j = 0; j < c1; ++j) x *= p[i];
        a[e[i][an]] = x;
        b[g[i][bn++]] = y;
      } else if (c2 > c1) {
        c2 -= c1;
        for (int j = 0; j < c2; ++j) y *= p[i];
        b[g[i][bn]] = y;
        a[e[i][an++]] = x;
      } else
        a[e[i][an++]] = x, b[g[i][bn++]] = y;
    }
  }
  printf(\"%d %d\n\", n, m);
  for (int i = 1; i <= n; ++i) printf(\"%d \", a[i]);
  puts(\"\");
  for (int j = 1; j <= m; ++j) printf(\"%d \", b[j]);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from math import gcd
from collections import defaultdict
from bootcamp import Basebootcamp
import re

class Creducingfractionsbootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_m=10, max_A_B=1e4, max_D=1e4):
        self.max_n = int(max_n)
        self.max_m = int(max_m)
        self.max_A_B = int(max_A_B)
        self.max_D = int(max_D)
    
    def case_generator(self):
        for _ in range(1000):  # Fallback to prevent infinite loops
            A, B = self._generate_coprime_pair()
            D = random.randint(1, self.max_D)
            
            numerator = A * D
            denominator = B * D
            
            if numerator > 1e7 or denominator > 1e7:
                continue
            if numerator == 0 or denominator == 0:
                continue
                
            n = random.randint(1, self.max_n)
            m = random.randint(1, self.max_m)
            
            try:
                a_list = self._safe_split(numerator, n)
                b_list = self._safe_split(denominator, m)
                
                return {
                    'n_input': n,
                    'm_input': m,
                    'a_list': a_list,
                    'b_list': b_list,
                    'A': A,
                    'B': B
                }
            except ValueError:
                continue
        raise RuntimeError("Failed to generate valid case after 1000 attempts")

    def _safe_split(self, num, count):
        for _ in range(100):
            factors = self.prime_factors(num)
            elements = [1] * count
            current = 0
            
            for prime, exp in factors.items():
                remaining = exp
                while remaining > 0:
                    idx = random.randint(0, count-1)
                    take = min(remaining, random.randint(0, remaining))
                    elements[idx] *= (prime ** take)
                    if elements[idx] > 1e7:
                        break  # Retry splitting
                    remaining -= take
                else:
                    continue
                break  # Break if exceeded
            else:
                # Verify all elements are valid
                if all(1 <= x <= 1e7 for x in elements):
                    return elements
        raise ValueError("Cannot split factors within constraints")

    @staticmethod
    def _generate_coprime_pair(max_value=1e4):
        for _ in range(100):
            a = random.randint(1, max_value)
            b = random.randint(1, max_value)
            if gcd(a, b) == 1:
                return a, b
        return (2, 3)  # Fallback
    
    @staticmethod
    def prime_factors(n):
        factors = defaultdict(int)
        while n % 2 == 0:
            factors[2] += 1
            n //= 2
        i = 3
        while i*i <= n:
            while n % i == 0:
                factors[i] += 1
                n //= i
            i += 2
        if n > 2:
            factors[n] += 1
        return factors

    @staticmethod
    def prompt_func(question_case) -> str:
        input_data = (
            f"{question_case['n_input']} {question_case['m_input']}\n"
            f"{' '.join(map(str, question_case['a_list']))}\n"
            f"{' '.join(map(str, question_case['b_list']))}"
        )
        return f"""Simplify the Galactic Empire fraction representation:

Input:
{input_data}

Format your answer EXACTLY like:
[answer]
<num_numerator> <num_denominator>
<numerator_elements>
<denominator_elements>
[/answer]

Rules:
1. Reduced fraction must be coprime
2. Each element ∈ [1, 10000000]
3. Element counts must satisfy 1 ≤ n, m ≤ 100000"""

    @staticmethod
    def extract_output(output):
        try:
            matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
            if not matches:
                return None
            content = matches[-1].strip().split('\n')
            header = list(map(int, content[0].strip().split()))
            if len(header) != 2:
                return None
            n_out, m_out = header
            a_list = list(map(int, content[1].strip().split()))
            b_list = list(map(int, content[2].strip().split()))
            if len(a_list) != n_out or len(b_list) != m_out:
                return None
            return {'n_out': n_out, 'm_out': m_out, 'a_list': a_list, 'b_list': b_list}
        except Exception:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        
        try:
            a = solution['a_list']
            b = solution['b_list']
            n_out = solution['n_out']
            m_out = solution['m_out']
        except KeyError:
            return False
        
        # Validate output constraints
        if not (1 <= n_out <= 1e5 and 1 <= m_out <= 1e5):
            return False
        if any(not (1 <= x <= 1e7) for x in a + b):
            return False
        
        # Verify products and coprimality
        prod_a = 1
        for x in a:
            prod_a *= x
        prod_b = 1
        for x in b:
            prod_b *= x
            
        return (
            prod_a == identity['A'] and
            prod_b == identity['B'] and
            gcd(prod_a, prod_b) == 1
        )
