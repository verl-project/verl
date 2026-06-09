"""# 

### 谜题描述
Grigory loves strings. Recently he found a metal strip on a loft. The strip had length n and consisted of letters \"V\" and \"K\". Unfortunately, rust has eaten some of the letters so that it's now impossible to understand which letter was written.

Grigory couldn't understand for a long time what these letters remind him of, so he became interested in the following question: if we put a letter \"V\" or \"K\" on each unreadable position, which values can the period of the resulting string be equal to?

A period of a string is such an integer d from 1 to the length of the string that if we put the string shifted by d positions to the right on itself, then all overlapping letters coincide. For example, 3 and 5 are periods of \"VKKVK\".

Input

There are several (at least one) test cases in the input. The first line contains single integer — the number of test cases.

There is an empty line before each test case. Each test case is described in two lines: the first line contains single integer n (1 ≤ n ≤ 5·105) — the length of the string, the second line contains the string of length n, consisting of letters \"V\", \"K\" and characters \"?\". The latter means the letter on its position is unreadable.

It is guaranteed that the sum of lengths among all test cases doesn't exceed 5·105.

For hacks you can only use tests with one test case.

Output

For each test case print two lines. In the first line print the number of possible periods after we replace each unreadable letter with \"V\" or \"K\". In the next line print all these values in increasing order.

Example

Input

3
 
5
V??VK
 
6
??????
 
4
?VK?


Output

2
3 5
6
1 2 3 4 5 6
3
2 3 4

Note

In the first test case from example we can obtain, for example, \"VKKVK\", which has periods 3 and 5.

In the second test case we can obtain \"VVVVVV\" which has all periods from 1 to 6.

In the third test case string \"KVKV\" has periods 2 and 4, and string \"KVKK\" has periods 3 and 4.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int N = (1 << 20);
const double pi = acos(-1);
inline int read() {
  int x = 0, f = 1;
  char ch = getchar();
  while (ch < '0' || ch > '9') {
    if (ch == '-') f = -1;
    ch = getchar();
  }
  while ('0' <= ch && ch <= '9') x = x * 10 + (ch ^ '0'), ch = getchar();
  return x * f;
}
struct CP {
  double x, y;
  CP(double xx = 0, double yy = 0) { x = xx, y = yy; }
};
CP operator+(CP aa, CP bb) { return CP(aa.x + bb.x, aa.y + bb.y); }
CP operator-(CP aa, CP bb) { return CP(aa.x - bb.x, aa.y - bb.y); }
CP operator*(CP aa, CP bb) {
  return CP(aa.x * bb.x - aa.y * bb.y, aa.x * bb.y + aa.y * bb.x);
}
CP operator*(CP aa, double bb) { return CP(aa.x * bb, aa.y * bb); }
int pp[N], lim, Lim;
CP Pow[N], iPow[N];
void up(int x) {
  for (lim = 1; lim <= x; lim <<= 1)
    ;
}
void revlim() {
  for (int i = 0, iE = lim - 1; i <= iE; i++)
    pp[i] = ((pp[i >> 1] >> 1) | ((i & 1) * (lim >> 1)));
}
void init(int x) {
  up(x), Lim = lim;
  Pow[0] = iPow[0] = CP(1, 0);
  for (int i = 1, iE = Lim - 1; i <= iE; i++)
    Pow[i] = iPow[Lim - i] = CP(cos(2 * pi / Lim * i), sin(2 * pi / Lim * i));
}
void FFT(CP *f, int flag) {
  for (int i = 0, iE = lim - 1; i <= iE; i++)
    if (pp[i] < i) swap(f[pp[i]], f[i]);
  for (int i = 2; i <= lim; i <<= 1)
    for (int j = 0, l = (i >> 1), ch = Lim / i; j < lim; j += i) {
      for (int k = j, now = 0; k < j + l; k++) {
        CP pa = f[k], pb = f[k + l] * (flag == 1 ? Pow[now] : iPow[now]);
        f[k] = pa + pb, f[k + l] = pa - pb, now += ch;
      }
    }
  if (flag == -1)
    for (int i = 0, iE = lim - 1; i <= iE; i++) f[i].x /= lim, f[i].y /= lim;
}
int n, m, a[N], f[N], tot, g[N];
CP A[N], AA[N], AAA[N];
CP B[N], BB[N], BBB[N];
char s[N];
void Main() {
  n = read(), scanf(\"%s\", s);
  up(n << 1), revlim();
  for (int i = 0, iE = lim - 1; i <= iE; i++)
    A[i] = B[i] = AA[i] = BB[i] = AAA[i] = BBB[i] = CP(0, 0);
  for (int i = 0, iE = n - 1; i <= iE; i++) {
    int o;
    if (s[i] == '?') o = 0;
    if (s[i] == 'V') o = 1;
    if (s[i] == 'K') o = 2;
    B[n - 1 - i].x = A[i].x = o;
    BB[n - 1 - i].x = AA[i].x = o * o;
    BBB[n - 1 - i].x = AAA[i].x = o * o * o;
    ;
  }
  FFT(A, 1), FFT(AA, 1), FFT(AAA, 1);
  FFT(B, 1), FFT(BB, 1), FFT(BBB, 1);
  for (int i = 0, iE = lim - 1; i <= iE; i++)
    A[i] = A[i] * BBB[i] + AAA[i] * B[i] - AA[i] * BB[i] * 2;
  FFT(A, -1);
  for (int i = 0, iE = n - 1; i <= iE; i++) f[i] = A[n - 1 - i].x;
  tot = 0;
  for (int i = 1, iE = n; i <= iE; i++) {
    bool ok = 1;
    for (int j = 1, jE = (n - 1) / i; j <= jE; j++)
      if (f[i * j]) ok = 0;
    if (ok) g[++tot] = i;
  }
  printf(\"%d\n\", tot);
  for (int i = 1, iE = tot; i <= iE; i++) printf(\"%d \", g[i]);
  puts(\"\");
}
int main() {
  init(1e6);
  int T = read();
  while (T--) Main();
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Erustystringbootcamp(Basebootcamp):
    def __init__(self, max_length=5, max_questions=3):
        self.max_length = max_length
        self.max_questions = max_questions
    
    def case_generator(self):
        n = random.randint(1, self.max_length)
        pattern_type = random.choice(['all_v', 'all_k', 'alternating', 'random'])
        original = []
        if pattern_type == 'all_v':
            original = ['V'] * n
        elif pattern_type == 'all_k':
            original = ['K'] * n
        elif pattern_type == 'alternating':
            original = ['V' if i%2 == 0 else 'K' for i in range(n)]
        else:
            original = [random.choice(['V', 'K']) for _ in range(n)]
        
        num_q = random.randint(0, min(self.max_questions, n))
        q_indices = random.sample(range(n), k=num_q)
        s = original.copy()
        for i in q_indices:
            s[i] = '?'
        s_str = ''.join(s)
        
        correct_periods = self.calculate_possible_periods(s_str)
        correct_periods.sort()
        return {
            'n': n,
            's': s_str,
            'correct_periods': correct_periods
        }

    def calculate_possible_periods(self, s):
        n = len(s)
        valid_periods = []
        
        for d in range(1, n+1):
            valid = True
            for r in range(d):  # Check each residue group
                has_v = False
                has_k = False
                # Check all positions in this residue group
                for pos in range(r, n, d):
                    char = s[pos]
                    if char == 'V':
                        has_v = True
                    elif char == 'K':
                        has_k = True
                    # Conflict detected in this group
                    if has_v and has_k:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                valid_periods.append(d)
                
        return valid_periods
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        s = question_case['s']
        prompt = f"""You are a programming competition participant. Solve the following string period puzzle.

Problem Description:
A string's period is an integer d (1 ≤ d ≤ n) where for all positions i, characters at i and i+d are equal (where i+d < n). Determine all possible periods after replacing '?' with 'V' or 'K'.

Input:
- A string of length {n}: {s}

Output Format:
Two lines:
1. Number of valid periods
2. Sorted valid periods separated by spaces

Enclose your answer between [answer] and [/answer]. Example:
[answer]
3
2 3 4
[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        start = output.rfind('[answer]')
        end = output.rfind('[/answer]')
        if start == -1 or end == -1 or start >= end:
            return None
        
        content = output[start+8:end].strip().split('\n')
        try:
            if len(content) < 2:
                return None
            periods = list(map(int, content[1].strip().split()))
            return periods
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_periods']
