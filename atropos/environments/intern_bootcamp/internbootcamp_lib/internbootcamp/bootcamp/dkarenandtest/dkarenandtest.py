"""# 

### 谜题描述
Karen has just arrived at school, and she has a math test today!

<image>

The test is about basic addition and subtraction. Unfortunately, the teachers were too busy writing tasks for Codeforces rounds, and had no time to make an actual test. So, they just put one question in the test that is worth all the points.

There are n integers written on a row. Karen must alternately add and subtract each pair of adjacent integers, and write down the sums or differences on the next row. She must repeat this process on the values on the next row, and so on, until only one integer remains. The first operation should be addition.

Note that, if she ended the previous row by adding the integers, she should start the next row by subtracting, and vice versa.

The teachers will simply look at the last integer, and then if it is correct, Karen gets a perfect score, otherwise, she gets a zero for the test.

Karen has studied well for this test, but she is scared that she might make a mistake somewhere and it will cause her final answer to be wrong. If the process is followed, what number can she expect to be written on the last row?

Since this number can be quite large, output only the non-negative remainder after dividing it by 109 + 7.

Input

The first line of input contains a single integer n (1 ≤ n ≤ 200000), the number of numbers written on the first row.

The next line contains n integers. Specifically, the i-th one among these is ai (1 ≤ ai ≤ 109), the i-th number on the first row.

Output

Output a single integer on a line by itself, the number on the final row after performing the process above.

Since this number can be quite large, print only the non-negative remainder after dividing it by 109 + 7.

Examples

Input

5
3 6 9 12 15


Output

36


Input

4
3 7 5 2


Output

1000000006

Note

In the first test case, the numbers written on the first row are 3, 6, 9, 12 and 15.

Karen performs the operations as follows:

<image>

The non-negative remainder after dividing the final number by 109 + 7 is still 36, so this is the correct output.

In the second test case, the numbers written on the first row are 3, 7, 5 and 2.

Karen performs the operations as follows:

<image>

The non-negative remainder after dividing the final number by 109 + 7 is 109 + 6, so this is the correct output.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
vector<long long> fact;
void precalc(long long n) {
  fact.resize(n / 2 + 1);
  fact[0] = 1;
  for (long long i = 1; i < fact.size(); ++i) {
    fact[i] = fact[i - 1] * i % 1000000007;
  }
}
long long binpow(long long a, long long b) {
  long long res = 1;
  while (b) {
    if (b & 1) {
      res = res * a % 1000000007;
    }
    a = a * a % 1000000007;
    b >>= 1;
  }
  return res;
}
long long divis(long long a, long long b) {
  return a * binpow(b, 1000000007 - 2) % 1000000007;
}
long long fun(long long a, long long b) {
  return divis(divis(fact[a], fact[b]), fact[a - b]);
}
void solve() {
  long long Q;
  Q = 1;
  while (Q--) {
    long long n;
    cin >> n;
    vector<long long> a(n);
    for (long long i = 0; i < n; ++i) cin >> a[i];
    ;
    precalc(n);
    vector<long long> b(n);
    if (n % 4 == 0) {
      for (long long i = 0, cnt = 1; i < n; ++i, cnt = -cnt) {
        b[i] = fun((n - 2) / 2, i / 2) * cnt;
      }
    } else if (n % 4 == 1) {
      for (long long i = 0; i < n; ++i) {
        if (i % 2 == 0) {
          b[i] = fun((n - 1) / 2, i / 2);
        } else {
          b[i] = 0;
        }
      }
    } else if (n % 4 == 2) {
      for (long long i = 0; i < n; ++i) {
        b[i] = fun((n - 2) / 2, i / 2);
      }
    } else {
      b[0] = 1;
      b[n - 1] = 1000000007 - 1;
      for (long long i = 1; i < n - 1; ++i) {
        if (i % 2 == 0) {
          b[i] = (fun((n - 3) / 2, i / 2) - fun((n - 3) / 2, i / 2 - 1) +
                  1000000007) %
                 1000000007;
        } else {
          b[i] = 2 * fun((n - 3) / 2, i / 2) % 1000000007;
        }
      }
    }
    long long res = 0;
    for (long long i = 0; i < n; ++i) {
      res = (res + a[i] * b[i]) % 1000000007;
    }
    cout << (res + 1000000007) % 1000000007;
  }
}
signed main(signed argc, char **argv) {
  ios_base::sync_with_stdio(false);
  cin.tie(nullptr);
  if (argc > 1 && (string)argv[1] == \"local\") {
    freopen(\"input.txt\", \"r\", stdin);
    freopen(\"output.txt\", \"w\", stdout);
    solve();
    while (cin.peek() != EOF) {
      if (isspace(cin.peek()))
        cin.get();
      else {
        cout << '\n';
        solve();
      }
    }
  } else {
    solve();
  }
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp

MOD = 10**9 + 7

class Dkarenandtestbootcamp(Basebootcamp):
    def __init__(self, max_n=10, min_n=1, max_a=10, **kwargs):
        self.max_n = max_n
        self.min_n = min_n
        self.max_a = max_a
    
    def case_generator(self):
        import random
        n = random.randint(self.min_n, self.max_n)
        a = [random.randint(1, self.max_a) for _ in range(n)]
        return {
            'n': n,
            'a': a
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = question_case['a']
        a_str = ' '.join(map(str, a))
        problem = f"""Karen has a math test involving alternating addition and subtraction operations on a sequence of numbers. Here's the problem:

Input:
- The first line contains an integer n ({n} in this case).
- The second line contains n integers: {a_str}.

Process:
1. Alternately add and subtract adjacent numbers starting with addition.
2. Repeat the process on each new row, swapping the starting operation (add/subtract) from the previous row's end.
3. Continue until one number remains.

Output the non-negative remainder of the final number modulo 1e9+7. 

Put your final answer within [answer] and [/answer] tags, like [answer]42[/answer]."""
        return problem
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            solution = int(solution)
        except:
            return False
        
        n = identity['n']
        a = identity['a']
        correct = cls.calculate_correct_answer(n, a)
        return (solution % MOD) == (correct % MOD)
    
    @classmethod
    def calculate_correct_answer(cls, n, a):
        def precalc(max_n):
            size = (max_n // 2) + 1
            fact = [1] * size
            for i in range(1, size):
                fact[i] = (fact[i-1] * i) % MOD
            return fact
        
        def combination(fact, a, b):
            if b < 0 or b > a:
                return 0
            numerator = fact[a]
            denominator = (fact[b] * fact[a - b]) % MOD
            inv_denominator = pow(denominator, MOD-2, MOD)
            return (numerator * inv_denominator) % MOD
        
        fact = precalc(n)
        b = [0] * n
        
        if n % 4 == 0:
            k = (n - 2) // 2
            for i in range(n):
                bi = combination(fact, k, i//2)
                cnt = (-1) ** i
                b[i] = (bi * cnt) % MOD
        elif n %4 == 1:
            k = (n -1) //2
            for i in range(n):
                if i %2 ==0:
                    bi = combination(fact, k, i//2)
                    b[i] = bi % MOD
                else:
                    b[i] =0
        elif n %4 ==2:
            k = (n-2) //2
            for i in range(n):
                bi = combination(fact, k, i//2)
                b[i] = bi % MOD
        else:
            b[0] =1
            if n >1:
                b[-1] = MOD-1
            k = (n-3) //2
            for i in range(1, n-1):
                if i %2 ==0:
                    comb1 = combination(fact, k, i//2)
                    comb2 = combination(fact, k, i//2 -1)
                    bi = (comb1 - comb2) % MOD
                else:
                    comb = combination(fact, k, i//2)
                    bi = (2 * comb) % MOD
                b[i] = bi
        
        total =0
        for ai, bi in zip(a, b):
            total = (total + ai * bi) % MOD
        return total % MOD
