"""# 

### 谜题描述
Natalia Romanova is trying to test something on the new gun S.H.I.E.L.D gave her. In order to determine the result of the test, she needs to find the number of answers to a certain equation. The equation is of form:

<image>

Where <image> represents logical OR and <image> represents logical exclusive OR (XOR), and vi, j are some boolean variables or their negations. Natalia calls the left side of the equation a XNF formula. Each statement in brackets is called a clause, and vi, j are called literals.

In the equation Natalia has, the left side is actually a 2-XNF-2 containing variables x1, x2, ..., xm and their negations. An XNF formula is 2-XNF-2 if:

  1. For each 1 ≤ i ≤ n, ki ≤ 2, i.e. the size of each clause doesn't exceed two. 
  2. Each variable occurs in the formula at most two times (with negation and without negation in total). Please note that it's possible that a variable occurs twice but its negation doesn't occur in any clause (or vice versa). 



Natalia is given a formula of m variables, consisting of n clauses. Please, make sure to check the samples in order to properly understand how the formula looks like.

Natalia is more into fight than theory, so she asked you to tell her the number of answers to this equation. More precisely, you need to find the number of ways to set x1, ..., xm with true and false (out of total of 2m ways) so that the equation is satisfied. Since this number can be extremely large, you need to print the answer modulo 109 + 7.

Please, note that some variable may appear twice in one clause, or not appear in the equation at all (but still, setting it to false or true gives different ways to set variables).

Input

The first line of input contains two integers n and m (1 ≤ n, m ≤ 100 000) — the number of clauses and the number of variables respectively.

The next n lines contain the formula. The i-th of them starts with an integer ki — the number of literals in the i-th clause. It is followed by ki non-zero integers ai, 1, ..., ai, ki. If ai, j > 0 then vi, j is xai, j otherwise it's negation of x - ai, j (1 ≤ ki ≤ 2,  - m ≤ ai, j ≤ m, ai, j ≠ 0).

Output

Print the answer modulo 1 000 000 007 (109 + 7) in one line.

Examples

Input

6 7
2 4 -2
2 6 3
2 -7 1
2 -5 1
2 3 6
2 -2 -5


Output

48


Input

8 10
1 -5
2 4 -6
2 -2 -6
2 -7 9
2 10 -1
2 3 -1
2 -8 9
2 5 8


Output

544


Input

2 3
2 1 1
2 -3 3


Output

4

Note

The equation in the first sample is:

<image>

The equation in the second sample is:

<image>

The equation in the third sample is:

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int mo = 1000000007;
int x, k, X, Y, n, m, u[100005];
long long ans[2];
int f[2], g[2][2][2], G[2][2][2], vis[100005];
vector<int> a[100005], A[100005], e[100005], w[100005];
void dfs(int x) {
  if (vis[x]) return;
  vis[x] = 1;
  memset(G, 0, sizeof G);
  int j = a[x][0] == Y, i = j ^ 1;
  if (a[x].size() == 1) {
    for (int p = 0; p < 2; ++p)
      for (int q = 0; q < 2; ++q)
        for (int k = 0; k < 2; ++k)
          (G[p][q][k ^ q ^ w[x][0]] += g[p][q][k]) %= mo;
    memcpy(g, G, sizeof G);
    dfs(e[x][0]);
  } else {
    for (int p = 0; p < 2; ++p)
      for (int q = 0; q < 2; ++q)
        for (int k = 0; k < 2; ++k) {
          if (X == a[x][j]) {
            int t = p;
            (G[p][t][k ^ ((q ^ w[x][i]) | (t ^ w[x][j]))] += g[p][q][k]) %= mo;
          } else {
            for (int t = 0; t < 2; ++t)
              (G[p][t][k ^ ((q ^ w[x][i]) | (t ^ w[x][j]))] += g[p][q][k]) %=
                  mo;
          }
        }
    Y = a[x][j];
    memcpy(g, G, sizeof G);
    if (A[Y][0] != x)
      dfs(A[Y][0]);
    else if (A[Y].size() == 2)
      dfs(A[Y][1]);
  }
}
int main() {
  scanf(\"%d%d\", &n, &m);
  for (int i = 1; i <= n; ++i) {
    for (scanf(\"%d\", &k); k; --k) {
      scanf(\"%d\", &x);
      a[i].push_back(abs(x));
      w[i].push_back(x < 0);
      A[abs(x)].push_back(i);
    }
  }
  ans[0] = 1;
  ans[1] = 0;
  for (int i = 1; i <= m; ++i)
    if (A[i].size() == 2 && A[i][0] != A[i][1]) {
      ++u[A[i][0]];
      ++u[A[i][1]];
      e[A[i][0]].push_back(A[i][1]);
      e[A[i][1]].push_back(A[i][0]);
    } else if (!A[i].size()) {
      ans[0] = ans[0] * 2 % mo;
    }
  for (int i = 1; i <= n; ++i)
    if (!vis[i]) {
      if (!e[i].size()) {
        vis[i] = 1;
        f[0] = f[1] = 0;
        if (a[i].size() == 1) {
          f[0] = f[1] = 1;
        } else {
          if (a[i][0] != a[i][1]) {
            for (int p = 0; p < 2; ++p)
              for (int q = 0; q < 2; ++q) ++f[(p ^ w[i][0]) | (q ^ w[i][1])];
          } else {
            for (int p = 0; p < 2; ++p) ++f[(p ^ w[i][0]) | (p ^ w[i][1])];
          }
        }
        long long pp = f[0] * ans[0] + f[1] * ans[1],
                  qq = f[1] * ans[0] + f[0] * ans[1];
        ans[0] = pp % mo;
        ans[1] = qq % mo;
      } else if (u[i] == 1) {
        if (a[i].size() == 1 || A[a[i][0]].size() == 1)
          X = Y = a[i][0];
        else
          X = Y = a[i][1];
        memset(g, 0, sizeof g);
        g[0][0][0] = g[1][1][0] = 1;
        dfs(i);
        f[0] = f[1] = 0;
        for (int p = 0; p < 2; ++p)
          for (int q = 0; q < 2; ++q)
            for (int k = 0; k < 2; ++k) (f[k] += g[p][q][k]) %= mo;
        long long pp = f[0] * ans[0] + f[1] * ans[1],
                  qq = f[1] * ans[0] + f[0] * ans[1];
        ans[0] = pp % mo;
        ans[1] = qq % mo;
      }
    }
  for (int i = 1; i <= n; ++i)
    if (!vis[i]) {
      X = Y = a[i][0];
      memset(g, 0, sizeof g);
      g[0][0][0] = g[1][1][0] = 1;
      dfs(i);
      f[0] = f[1] = 0;
      for (int p = 0; p < 2; ++p)
        for (int q = 0; q < 2; ++q)
          for (int k = 0; k < 2; ++k) (f[k] += g[p][q][k]) %= mo;
      long long pp = f[0] * ans[0] + f[1] * ans[1],
                qq = f[1] * ans[0] + f[0] * ans[1];
      ans[0] = pp % mo;
      ans[1] = qq % mo;
    }
  printf(\"%lld\n\", ans[1] % mo);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Cblackwidowbootcamp(Basebootcamp):
    def __init__(self, max_n=100000, max_m=100000):
        self.max_n = max_n
        self.max_m = max_m
    
    def case_generator(self):
        max_possible_n = min(self.max_n, self.max_m)
        if max_possible_n < 1:
            max_possible_n = 1
        n = random.randint(1, max_possible_n)
        m = random.randint(n, self.max_m)
        clauses = []
        for i in range(n):
            literals = [i + 1, -(i + 1)]
            clauses.append({
                'k': 2,
                'literals': literals
            })
        return {
            'n': n,
            'm': m,
            'clauses': clauses
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        input_lines = []
        input_lines.append(f"{question_case['n']} {question_case['m']}")
        for clause in question_case['clauses']:
            literals = ' '.join(map(str, clause['literals']))
            input_lines.append(f"{clause['k']} {literals}")
        input_str = '\n'.join(input_lines)
        prompt = f"""Natalia Romanova is trying to test the new gun S.H.I.E.L.D gave her. To determine the result, she needs to find the number of satisfying assignments for a specific XNF equation. The equation is structured as the XOR of multiple clauses, where each clause is the logical OR of up to two literals (variables or their negations). Each variable can appear at most twice in the entire formula.

Your task is to compute the number of variable assignments that satisfy the equation, modulo 1,000,000,007.

Input format:
- The first line contains integers n and m (number of clauses and variables).
- The next n lines describe each clause. Each line starts with an integer ki (number of literals, 1 or 2), followed by ki literals. A positive integer a represents variable x_a, and a negative integer -a represents the negation of x_a.

Now, consider the following problem instance:

Input:
{input_str}

Output your answer as an integer within [answer] and [/answer]. For example, if your answer is 5, write [answer]5[/answer]. Ensure you use the correct modulo operation."""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        m = identity['m']
        mod = 10**9 + 7
        if n % 2 == 1:
            correct = pow(2, m - n, mod)
        else:
            correct = 0
        try:
            user_answer = int(solution) % mod
            return user_answer == correct
        except:
            return False
