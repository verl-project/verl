"""# 

### 谜题描述
This is the easy version of the problem. The only difference between easy and hard versions is the constraint of m. You can make hacks only if both versions are solved.

Chiori loves dolls and now she is going to decorate her bedroom!

<image>

As a doll collector, Chiori has got n dolls. The i-th doll has a non-negative integer value a_i (a_i < 2^m, m is given). Chiori wants to pick some (maybe zero) dolls for the decoration, so there are 2^n different picking ways.

Let x be the bitwise-xor-sum of values of dolls Chiori picks (in case Chiori picks no dolls x = 0). The value of this picking way is equal to the number of 1-bits in the binary representation of x. More formally, it is also equal to the number of indices 0 ≤ i < m, such that \left⌊ (x)/(2^i) \right⌋ is odd.

Tell her the number of picking ways with value i for each integer i from 0 to m. Due to the answers can be very huge, print them by modulo 998 244 353.

Input

The first line contains two integers n and m (1 ≤ n ≤ 2 ⋅ 10^5, 0 ≤ m ≤ 35) — the number of dolls and the maximum value of the picking way.

The second line contains n integers a_1, a_2, …, a_n (0 ≤ a_i < 2^m) — the values of dolls.

Output

Print m+1 integers p_0, p_1, …, p_m — p_i is equal to the number of picking ways with value i by modulo 998 244 353.

Examples

Input


4 4
3 5 8 14


Output


2 2 6 6 0 

Input


6 7
11 45 14 9 19 81


Output


1 2 11 20 15 10 5 0 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
#pragma GCC optimize(\"Ofast\", \"unroll-loops\", \"omit-frame-pointer\", \"inline\")
#pragma GCC option(\"arch=native\", \"tune=native\", \"no-zero-upper\")
#pragma GCC target(\"avx2\")
using namespace std;
template <typename T>
void maxtt(T &t1, T t2) {
  t1 = max(t1, t2);
}
template <typename T>
void mintt(T &t1, T t2) {
  t1 = min(t1, t2);
}
bool debug = 0;
int n, m, k;
string direc = \"RDLU\";
const long long MOD2 = (long long)998244353 * (long long)998244353;
long long ln, lk, lm;
void etp(bool f = 0) {
  puts(f ? \"YES\" : \"NO\");
  exit(0);
}
void addmod(int &x, int y, int mod = 998244353) {
  x += y;
  if (x >= mod) x -= mod;
  if (x < 0) x += mod;
  assert(x >= 0 && x < mod);
}
void et(int x = -1) {
  printf(\"%d\n\", x);
  exit(0);
}
long long fastPow(long long x, long long y, int mod = 998244353) {
  long long ans = 1;
  while (y > 0) {
    if (y & 1) ans = (x * ans) % mod;
    x = x * x % mod;
    y >>= 1;
  }
  return ans;
}
long long gcd1(long long x, long long y) { return y ? gcd1(y, x % y) : x; }
long long a[200135];
struct lsp {
  long long a[60] = {0};
  const int maxBit = 54;
  bool insert(long long x) {
    for (int i = maxBit; ~i; i--)
      if (x & (1LL << i)) {
        if (a[i] != 0)
          x ^= a[i];
        else {
          for (int(j) = 0; (j) < (int)(i); (j)++)
            if (x & (1LL << j)) x ^= a[j];
          for (int j = i + 1; j <= maxBit; j++)
            if (a[j] & (1LL << i)) a[j] ^= x;
          a[i] = x;
          return 1;
        }
      }
    return 0;
  }
  lsp getOrthogonal(int m) {
    lsp res;
    vector<int> vp;
    for (int j = m - 1; j >= 0; j--)
      if (!a[j]) {
        vp.push_back(j);
        res.a[j] |= 1LL << j;
      }
    for (int j = m - 1; j >= 0; j--)
      if (a[j]) {
        int cc = 0;
        for (int z = m - 1; z >= 0; z--)
          if (!a[z]) {
            long long w = (a[j] >> z) & 1;
            res.a[vp[cc]] |= w << j;
            cc++;
          }
      }
    return res;
  }
} sp;
int p[66], q[66];
void ppt() {
  for (int i = 0; i <= m; i++) {
    printf(\"%lld \", (long long)p[i] * fastPow(2, n - k) % 998244353);
  }
  exit(0);
}
vector<long long> bs;
inline void dfs(int *p, int k, int i, long long x) {
  if (i == k) {
    p[__builtin_popcountll(x)]++;
    return;
  }
  dfs(p, k, i + 1, x);
  dfs(p, k, i + 1, x ^ bs[i]);
}
void calsm(lsp sp, int *p, int k) {
  bs.clear();
  for (int(j) = 0; (j) < (int)(sp.maxBit); (j)++)
    if (sp.a[j]) bs.push_back(sp.a[j]);
  assert(bs.size() == k);
  dfs(p, k, 0, 0);
}
void calbg() {
  vector<vector<int>> C(60, vector<int>(60, 0));
  C[0][0] = 1;
  for (int i = 1; i <= 55; i++) {
    C[i][0] = C[i][i] = 1;
    for (int j = 1; j < i; j++) {
      C[i][j] = (C[i - 1][j - 1] + C[i - 1][j]) % 998244353;
    }
  }
  vector<vector<int>> w(m + 1, vector<int>(m + 1, 0));
  for (int c = 0; c <= m; c++) {
    for (int d = 0; d <= m; d++) {
      for (int j = 0; j <= c; j++) {
        int val = (long long)C[d][j] * C[m - d][c - j] % 998244353;
        if (j % 2 == 0)
          addmod(w[c][d], val);
        else
          addmod(w[c][d], 998244353 - val);
      }
    }
  }
  lsp B = sp.getOrthogonal(m);
  calsm(B, q, m - k);
  for (int(c) = 0; (c) < (int)(m + 1); (c)++) {
    for (int d = 0; d <= m; d++) {
      addmod(p[c], (long long)q[d] * w[c][d] % 998244353);
    }
  }
  int tmp = fastPow(2, m - k);
  tmp = fastPow(tmp, 998244353 - 2);
  for (int(c) = 0; (c) < (int)(m + 1); (c)++)
    p[c] = (long long)p[c] * tmp % 998244353;
  ppt();
}
void fmain(int tid) {
  scanf(\"%d%d\", &n, &m);
  for (int(i) = 1; (i) <= (int)(n); (i)++) scanf(\"%lld\", a + i);
  for (int(i) = 1; (i) <= (int)(n); (i)++) k += sp.insert(a[i]);
  if (k <= m / 2) {
    calsm(sp, p, k);
    ppt();
  }
  calbg();
}
int main() {
  int t = 1;
  for (int(i) = 1; (i) <= (int)(t); (i)++) {
    fmain(i);
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

MOD = 998244353

class E1chiorianddollpickingeasyversionbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10, min_m=0, max_m=10, min_k=0, max_k=5):
        self.min_n = min_n
        self.max_n = max_n
        self.min_m = min_m
        self.max_m = max_m
        self.min_k = min_k
        # Ensure max_k doesn't exceed m//2 during case generation
        self.max_k = max_k
    
    def case_generator(self):
        m = random.randint(self.min_m, self.max_m)
        # Calculate maximum allowed k based on m//2 to ensure correctness
        max_k_allowed = m // 2 if m > 0 else 0
        max_k = min(self.max_k, max_k_allowed)
        k = random.randint(self.min_k, max_k) if max_k >= self.min_k else 0
        
        if m == 0:
            n = random.randint(self.min_n, self.max_n)
            a = [0] * n
            return {'n': n, 'm': m, 'a': a, 'correct_output': [pow(2, n, MOD)]}
        
        n = random.randint(max(k, self.min_n), self.max_n)
        basis = []
        for i in range(k):
            basis.append(1 << i)
        
        a = basis.copy()
        for _ in range(n - k):
            xor = 0
            selected = random.choices(basis, k=random.randint(0, k)) if k > 0 else []
            for num in selected:
                xor ^= num
            a.append(xor)
        random.shuffle(a)
        
        xor_sums = {0}
        for b in basis:
            new_xors = set()
            for x in xor_sums:
                new_xors.add(x ^ b)
            xor_sums.update(new_xors)
        
        bit_counts = {}
        for x in xor_sums:
            cnt = bin(x).count('1')
            bit_counts[cnt] = bit_counts.get(cnt, 0) + 1
        
        multiplier = pow(2, n - k, MOD)
        correct_output = [0] * (m + 1)
        for bits, count in bit_counts.items():
            if bits <= m:
                correct_output[bits] = (count * multiplier) % MOD
        
        return {'n': n, 'm': m, 'a': a, 'correct_output': correct_output}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        example_input = "4 4\n3 5 8 14"
        example_output = "2 2 6 6 0"
        prompt = f"""Chiori loves decorating her bedroom with dolls. As a doll collector, she has n dolls with certain values and wants to determine the number of ways to pick dolls such that the bitwise XOR sum of their values has a specific number of 1s in its binary form. 

Problem Statement:
You are given {question_case['n']} dolls with values listed below. Each value is a non-negative integer less than 2^{question_case['m']}. Calculate the number of ways to pick a subset (including picking none) such that the XOR sum of the selected dolls' values has exactly i 1s in its binary representation, for each i from 0 to {question_case['m']}. Output each count modulo 998244353.

Input:
- The first line contains two integers n and m: {question_case['n']} {question_case['m']}
- The second line contains the doll values: {' '.join(map(str, question_case['a']))}

Output:
Print {question_case['m']+1} integers p_0, p_1, ..., p_{question_case['m']} where each p_i is the count of subsets with exactly i 1s in the XOR sum's binary form, modulo 998244353.

Example Input:
{example_input}
Example Output:
{example_output}

Your Task:
Enclose your answer within [answer] and [/answer]. For example: [answer]0 1 2 3 4[/answer]
Ensure your answer includes all {question_case['m']+1} integers separated by spaces, even if some are zero.
"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            extracted = list(map(int, last_match.split()))
            return extracted
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity['correct_output']
        return solution == expected
