"""# 

### 谜题描述
During the archaeological research in the Middle East you found the traces of three ancient religions: First religion, Second religion and Third religion. You compiled the information on the evolution of each of these beliefs, and you now wonder if the followers of each religion could coexist in peace.

The Word of Universe is a long word containing the lowercase English characters only. At each moment of time, each of the religion beliefs could be described by a word consisting of lowercase English characters.

The three religions can coexist in peace if their descriptions form disjoint subsequences of the Word of Universe. More formally, one can paint some of the characters of the Word of Universe in three colors: 1, 2, 3, so that each character is painted in at most one color, and the description of the i-th religion can be constructed from the Word of Universe by removing all characters that aren't painted in color i.

The religions however evolve. In the beginning, each religion description is empty. Every once in a while, either a character is appended to the end of the description of a single religion, or the last character is dropped from the description. After each change, determine if the religions could coexist in peace.

Input

The first line of the input contains two integers n, q (1 ≤ n ≤ 100 000, 1 ≤ q ≤ 1000) — the length of the Word of Universe and the number of religion evolutions, respectively. The following line contains the Word of Universe — a string of length n consisting of lowercase English characters.

Each of the following line describes a single evolution and is in one of the following formats: 

  * + i c (i ∈ \{1, 2, 3\}, c ∈ \{a, b, ..., z\}: append the character c to the end of i-th religion description. 
  * - i (i ∈ \{1, 2, 3\}) – remove the last character from the i-th religion description. You can assume that the pattern is non-empty. 



You can assume that no religion will have description longer than 250 characters.

Output

Write q lines. The i-th of them should be YES if the religions could coexist in peace after the i-th evolution, or NO otherwise.

You can print each character in any case (either upper or lower).

Examples

Input


6 8
abdabc
+ 1 a
+ 1 d
+ 2 b
+ 2 c
+ 3 a
+ 3 b
+ 1 c
- 2


Output


YES
YES
YES
YES
YES
YES
NO
YES


Input


6 8
abbaab
+ 1 a
+ 2 a
+ 3 a
+ 1 b
+ 2 b
+ 3 b
- 1
+ 2 z


Output


YES
YES
YES
YES
YES
NO
YES
NO

Note

In the first example, after the 6th evolution the religion descriptions are: ad, bc, and ab. The following figure shows how these descriptions form three disjoint subsequences of the Word of Universe:

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
int n, q;
char cc[100001];
int nxt[100001][26], nx[26];
int dp[251][251][251];
int st1[251], st2[251], st3[251];
int trans(int k, int d) { return k == -1 ? -1 : nxt[k][d]; }
int better(int a, int b) {
  if (a == -1) {
    return b;
  }
  if (b == -1) {
    return a;
  }
  return min(a, b);
}
void solve() {
  scanf(\"%d%d%s\", &n, &q, cc);
  memset(nx, -1, sizeof(nx));
  int ll = strlen(cc);
  for (int i = ll; i >= 0; i--) {
    if (i < ll) {
      nx[cc[i] - 'a'] = i + 1;
    }
    for (int j = 0; j < 26; j++) {
      nxt[i][j] = nx[j];
    }
  }
  int c1 = 0, c2 = 0, c3 = 0;
  dp[0][0][0] = 0;
  while (q--) {
    char cmd[2], dd[2];
    int id;
    scanf(\"%s%d\", cmd, &id);
    if (cmd[0] == '+') {
      scanf(\"%s\", dd);
      int d = dd[0] - 'a';
      if (id == 1) {
        for (int i = 0; i <= c2; i++) {
          for (int j = 0; j <= c3; j++) {
            dp[c1 + 1][i][j] = trans(dp[c1][i][j], d);
            if (i > 0) {
              int di = st2[i - 1];
              dp[c1 + 1][i][j] =
                  better(dp[c1 + 1][i][j], trans(dp[c1 + 1][i - 1][j], di));
            }
            if (j > 0) {
              int dj = st3[j - 1];
              dp[c1 + 1][i][j] =
                  better(dp[c1 + 1][i][j], trans(dp[c1 + 1][i][j - 1], dj));
            }
          }
        }
        st1[c1++] = d;
      } else if (id == 2) {
        for (int i = 0; i <= c1; i++) {
          for (int j = 0; j <= c3; j++) {
            dp[i][c2 + 1][j] = trans(dp[i][c2][j], d);
            if (i > 0) {
              int di = st1[i - 1];
              dp[i][c2 + 1][j] =
                  better(dp[i][c2 + 1][j], trans(dp[i - 1][c2 + 1][j], di));
            }
            if (j > 0) {
              int dj = st3[j - 1];
              dp[i][c2 + 1][j] =
                  better(dp[i][c2 + 1][j], trans(dp[i][c2 + 1][j - 1], dj));
            }
          }
        }
        st2[c2++] = d;
      } else if (id == 3) {
        for (int i = 0; i <= c1; i++) {
          for (int j = 0; j <= c2; j++) {
            dp[i][j][c3 + 1] = trans(dp[i][j][c3], d);
            if (i > 0) {
              int di = st1[i - 1];
              dp[i][j][c3 + 1] =
                  better(dp[i][j][c3 + 1], trans(dp[i - 1][j][c3 + 1], di));
            }
            if (j > 0) {
              int dj = st2[j - 1];
              dp[i][j][c3 + 1] =
                  better(dp[i][j][c3 + 1], trans(dp[i][j - 1][c3 + 1], dj));
            }
          }
        }
        st3[c3++] = d;
      }
    } else {
      if (id == 1) {
        c1--;
      } else if (id == 2) {
        c2--;
      } else if (id == 3) {
        c3--;
      }
    }
    printf(\"%s\n\", dp[c1][c2][c3] == -1 ? \"NO\" : \"YES\");
  }
}
int main() { solve(); }
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from string import ascii_lowercase
import re

def preprocess_nxt(s):
    n = len(s)
    nx = [-1] * 26
    nxt = [[-1] * 26 for _ in range(n + 1)]
    for i in range(n, -1, -1):
        if i < n:
            c = ord(s[i]) - ord('a')
            nx[c] = i + 1
        for j in range(26):
            nxt[i][j] = nx[j]
    return nxt

def trans(nxt, k, d):
    return -1 if k == -1 else nxt[k][d]

def better(a, b):
    if a == -1:
        return b
    if b == -1:
        return a
    return min(a, b)

def simulate_operations(s, operations):
    nxt = preprocess_nxt(s)
    dp = [[[-1 for _ in range(251)] for _ in range(251)] for __ in range(251)]
    dp[0][0][0] = 0
    st1, st2, st3 = [], [], []
    c1, c2, c3 = 0, 0, 0
    expected_outputs = []
    for op in operations:
        parts = op.split()
        cmd, id = parts[0], int(parts[1])
        if cmd == '+':
            d = ord(parts[2]) - ord('a')
            if id == 1:
                st1.append(d)
                new_c1 = c1 + 1
                for i in range(c2 + 1):
                    for j in range(c3 + 1):
                        val = trans(nxt, dp[c1][i][j], d)
                        if i > 0:
                            di = st2[i-1]
                            val = better(val, trans(nxt, dp[new_c1][i-1][j], di))
                        if j > 0:
                            dj = st3[j-1]
                            val = better(val, trans(nxt, dp[new_c1][i][j-1], dj))
                        dp[new_c1][i][j] = val
                c1 += 1
            elif id == 2:
                st2.append(d)
                new_c2 = c2 + 1
                for i in range(c1 + 1):
                    for j in range(c3 + 1):
                        val = trans(nxt, dp[i][c2][j], d)
                        if i > 0:
                            di = st1[i-1]
                            val = better(val, trans(nxt, dp[i-1][new_c2][j], di))
                        if j > 0:
                            dj = st3[j-1]
                            val = better(val, trans(nxt, dp[i][new_c2][j-1], dj))
                        dp[i][new_c2][j] = val
                c2 += 1
            else:
                st3.append(d)
                new_c3 = c3 + 1
                for i in range(c1 + 1):
                    for j in range(c2 + 1):
                        val = trans(nxt, dp[i][j][c3], d)
                        if i > 0:
                            di = st1[i-1]
                            val = better(val, trans(nxt, dp[i-1][j][new_c3], di))
                        if j > 0:
                            dj = st2[j-1]
                            val = better(val, trans(nxt, dp[i][j-1][new_c3], dj))
                        dp[i][j][new_c3] = val
                c3 += 1
        else:
            if id == 1:
                st1.pop()
                c1 -= 1
            elif id == 2:
                st2.pop()
                c2 -= 1
            else:
                st3.pop()
                c3 -= 1
        current_dp = dp[c1][c2][c3]
        expected_outputs.append('YES' if current_dp != -1 else 'NO')
    return expected_outputs

class Dthreereligionsbootcamp(Basebootcamp):
    def __init__(self, max_n=100000, max_q=1000, max_op_len=250):
        self.max_n = max_n
        self.max_q = max_q
        self.max_op_len = max_op_len

    def case_generator(self):
        n = random.randint(5, 10)
        q = random.randint(5, 10)
        s = ''.join(random.choice(ascii_lowercase) for _ in range(n))
        operations = []
        len1, len2, len3 = 0, 0, 0
        for _ in range(q):
            can_remove = []
            for r in [1, 2, 3]:
                if (r == 1 and len1 > 0) or (r == 2 and len2 > 0) or (r == 3 and len3 > 0):
                    can_remove.append(r)
            can_add = []
            for r in [1, 2, 3]:
                if (r == 1 and len1 < self.max_op_len) or (r == 2 and len2 < self.max_op_len) or (r == 3 and len3 < self.max_op_len):
                    can_add.append(r)
            if not can_add and not can_remove:
                break  # Should not happen with small q
            if can_remove and (random.random() < 0.5 or not can_add):
                r = random.choice(can_remove)
                operations.append(f"- {r}")
                if r == 1:
                    len1 -= 1
                elif r == 2:
                    len2 -= 1
                else:
                    len3 -= 1
            else:
                r = random.choice(can_add)
                c = random.choice(ascii_lowercase)
                operations.append(f"+ {r} {c}")
                if r == 1:
                    len1 += 1
                elif r == 2:
                    len2 += 1
                else:
                    len3 += 1
        expected_outputs = simulate_operations(s, operations)
        return {
            's': s,
            'operations': operations,
            'expected_outputs': expected_outputs
        }

    @staticmethod
    def prompt_func(question_case):
        s = question_case['s']
        operations = question_case['operations']
        q = len(operations)
        n = len(s)
        problem = (
            "During the archaeological research in the Middle East, you found the traces of three ancient religions. "
            "Each's description evolves through a series of operations. The Word of Universe is a string, and after each evolution, "
            "you need to determine if the three descriptions can form disjoint subsequences of this string.\n\n"
            f"The Word of Universe is '{s}' (length {n}). There are {q} evolutions:\n"
        )
        for op in operations:
            problem += f"{op}\n"
        problem += (
            "\nAfter each evolution, output 'YES' if the three descriptions can coexist peacefully as disjoint subsequences, otherwise 'NO'. "
            "Provide your answers in order, each on a new line. Enclose the entire answer within [answer] and [/answer] tags. "
            "For example:\n"
            "[answer]\n"
            "YES\n"
            "NO\n"
            "...\n"
            "[/answer]"
        )
        return problem

    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        last_answer = answer_blocks[-1].strip()
        lines = [line.strip().upper() for line in last_answer.split('\n') if line.strip()]
        valid = all(line in {'YES', 'NO'} for line in lines)
        return lines if valid else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity['expected_outputs']
        if not solution or len(solution) != len(expected):
            return False
        return all(s.upper() == e.upper() for s, e in zip(solution, expected))
