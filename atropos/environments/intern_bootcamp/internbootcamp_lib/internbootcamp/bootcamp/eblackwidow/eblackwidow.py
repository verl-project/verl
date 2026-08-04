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
template <typename T, typename U>
inline void smin(T &a, U b) {
  if (a > b) a = b;
}
template <typename T, typename U>
inline void smax(T &a, U b) {
  if (a < b) a = b;
}
template <typename T>
inline void gn(T &first) {
  char c, sg = 0;
  while (c = getchar(), (c > '9' || c < '0') && c != '-')
    ;
  for ((c == '-' ? sg = 1, c = getchar() : 0), first = 0; c >= '0' && c <= '9';
       c = getchar())
    first = (first << 1) + (first << 3) + c - '0';
  if (sg) first = -first;
}
template <class T, class T1>
inline void gn(T &first, T1 &second) {
  gn(first);
  gn(second);
}
template <class T, class T1, class T2>
inline void gn(T &first, T1 &second, T2 &z) {
  gn(first);
  gn(second);
  gn(z);
}
template <typename T>
inline void print(T first) {
  if (first < 0) {
    putchar('-');
    return print(-first);
  }
  if (first < 10) {
    putchar('0' + first);
    return;
  }
  print(first / 10);
  putchar(first % 10 + '0');
}
template <typename T>
inline void println(T first) {
  print(first), putchar('\n');
}
template <typename T>
inline void printsp(T first) {
  print(first), putchar(' ');
}
template <class T, class T1>
inline void print(T first, T1 second) {
  printsp(first), println(second);
}
template <class T, class T1, class T2>
inline void print(T first, T1 second, T2 z) {
  printsp(first), printsp(second), println(z);
}
int power(int a, int b, int m, int ans = 1) {
  for (; b; b >>= 1, a = 1LL * a * a % m)
    if (b & 1) ans = 1LL * ans * a % m;
  return ans;
}
vector<int> val[100100];
vector<int> num[100100];
vector<int> adj[100100];
int vst[100100];
vector<int> vec;
void dfs(int u) {
  vec.push_back(u);
  vst[u] = 1;
  for (int i = 0; i < adj[u].size(); i++) {
    int v = adj[u][i];
    if (vst[v]) continue;
    dfs(v);
  }
}
vector<pair<int, int> > P;
int dp[100100][2][2];
int ddp[100100][2][2][2];
int DP[100100][2];
inline void add(int &a, int b) {
  a += b;
  if (a >= 1000000007) a -= 1000000007;
}
int main() {
  int n, m;
  gn(n, m);
  for (int i = 1; i <= n; i++) {
    int k;
    gn(k);
    int u;
    while (k--) {
      gn(u);
      val[i].push_back(u);
      if (u > 0)
        num[u].push_back(i);
      else
        num[-u].push_back(-i);
    }
  }
  int flag = 0;
  int tar = 1;
  int ans = 1;
  for (int i = 1; i <= m; i++) {
    if (num[i].size() > 1) {
      adj[abs(num[i][0])].push_back(abs(num[i][1]));
      adj[abs(num[i][1])].push_back(abs(num[i][0]));
    } else if (num[i].empty()) {
      ans = ans * 2 % 1000000007;
    }
  }
  if (flag) {
    println(power(2, m - 1, 1000000007));
    return 0;
  }
  for (int i = 1; i <= n; i++) {
    if (adj[i].empty()) {
      vst[i] = 1;
      P.push_back(pair<int, int>(1, (1 << val[i].size()) - 1));
    } else if (!vst[i] && adj[i].size() == 1) {
      vec.clear();
      dfs(i);
      for (int j = 0; j < vec.size(); j++) memset(dp[j], 0, sizeof(dp[j]));
      if (val[vec[0]].size() == 2)
        dp[0][1][1] = 1, dp[0][0][0] = 1;
      else
        dp[0][0][0] = 1;
      for (int j = 1; j < vec.size(); j++) {
        int fl = 0;
        for (int a = 0; a < val[vec[j]].size(); a++) {
          for (int b = 0; b < val[vec[j - 1]].size(); b++)
            if (val[vec[j]][a] == val[vec[j - 1]][b]) fl = 1;
        }
        if (fl) {
          for (int a = 0; a < 2; a++) {
            for (int b = 0; b < 2; b++) {
              add(dp[j][a][0], dp[j - 1][a][b]);
              add(dp[j][a ^ (b == 0) ^ 1][1], dp[j - 1][a][b]);
            }
          }
        } else {
          for (int a = 0; a < 2; a++) {
            for (int b = 0; b < 2; b++) {
              add(dp[j][a ^ (b == 0)][0], dp[j - 1][a][b]);
              add(dp[j][a ^ 1][1], dp[j - 1][a][b]);
            }
          }
        }
      }
      int v[2] = {0, 0};
      for (int k = 0; k < 1 + (val[vec.back()].size() == 2); k++)
        for (int a = 0; a < 2; a++)
          for (int b = 0; b < 2; b++) {
            add(v[a ^ (k && b == 0)], dp[vec.size() - 1][a][b]);
          }
      P.push_back(pair<int, int>(v[0], v[1]));
    }
  }
  for (int i = 1; i <= n; i++)
    if (!vst[i]) {
      vec.clear();
      dfs(i);
      for (int j = 0; j < vec.size(); j++) memset(ddp[j], 0, sizeof(ddp[j]));
      ddp[0][1][1][1] = 1;
      ddp[0][0][0][0] = 1;
      for (int j = 1; j < vec.size(); j++) {
        int fl = 0;
        for (int a = 0; a < val[vec[j]].size(); a++) {
          for (int b = 0; b < val[vec[j - 1]].size(); b++)
            if (val[vec[j]][a] == val[vec[j - 1]][b]) fl = 1;
        }
        if (fl) {
          for (int k = 0; k < 2; k++) {
            for (int a = 0; a < 2; a++) {
              for (int b = 0; b < 2; b++) {
                add(ddp[j][k][a][0], ddp[j - 1][k][a][b]);
                add(ddp[j][k][a ^ (b == 0) ^ 1][1], ddp[j - 1][k][a][b]);
              }
            }
          }
        } else {
          for (int k = 0; k < 2; k++) {
            for (int a = 0; a < 2; a++) {
              for (int b = 0; b < 2; b++) {
                add(ddp[j][k][a ^ (b == 0)][0], ddp[j - 1][k][a][b]);
                add(ddp[j][k][a ^ 1][1], ddp[j - 1][k][a][b]);
              }
            }
          }
        }
      }
      int ok = 0;
      if (vec.size() == 1)
        ok = val[vec[0]][0] == val[vec[0]][1];
      else
        for (int a = 0; a < val[vec[0]].size(); a++) {
          for (int b = 0; b < val[vec.back()].size(); b++)
            if (val[vec[0]][a] == val[vec.back()][b]) ok = 1;
        }
      int v[] = {0, 0};
      for (int k = 0; k < 2; k++)
        for (int a = 0; a < 2; a++)
          for (int b = 0; b < 2; b++) {
            int kk = k ^ (!ok);
            add(v[a ^ (b == 0 && kk)], ddp[vec.size() - 1][k][a][b]);
          }
      P.push_back(pair<int, int>(v[0], v[1]));
    }
  DP[0][0] = 1;
  DP[0][1] = 0;
  for (int i = 0; i < P.size(); i++) {
    DP[i + 1][0] =
        ((long long)DP[i][0] * P[i].first + (long long)DP[i][1] * P[i].second) %
        1000000007;
    DP[i + 1][1] =
        ((long long)DP[i][1] * P[i].first + (long long)DP[i][0] * P[i].second) %
        1000000007;
  }
  println((long long)DP[P.size()][tar] * ans % 1000000007);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Eblackwidowbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=5, min_m=1, max_m=5):
        self.min_n = min(min_n, max_n)
        self.max_n = max(min_n, max_n)
        self.min_m = min(min_m, max_m)
        self.max_m = max(min_m, max_m)
        
    def case_generator(self):
        MOD = 10**9 +7
        while True:
            m = random.randint(self.min_m, self.max_m)
            max_allowed = m * 2
            n = random.randint(self.min_n, min(self.max_n, max_allowed))
            var_usage = {x:0 for x in range(1, m+1)}
            clauses = []
            success = True
            
            for _ in range(n):
                available = [x for x in var_usage if var_usage[x] < 2]
                if not available:
                    success = False
                    break
                
                k = random.choice([1,2])
                temp_usage = var_usage.copy()
                literals = []
                
                # 生成子句的多次尝试
                for attempt in range(3):
                    current = available.copy()
                    temp = temp_usage.copy()
                    lits = []
                    valid = True
                    
                    for _ in range(k):
                        if not current:
                            valid = False
                            break
                        var = random.choice(current)
                        sign = random.choice([1,-1])
                        lits.append(sign * var)
                        temp[var] +=1
                        current = [x for x in current if temp[x] <2]
                        
                    if valid and len(lits) ==k:
                        literals = lits
                        temp_usage = temp
                        break
                    
                if literals and len(literals)==k: 
                    clauses.append({'k':k, 'literals':literals})
                    var_usage = temp_usage
                else:
                    success = False
                    break
            
            if success and len(clauses)==n:
                return {'n':n, 'm':m, 'clauses':clauses}

    @staticmethod
    def prompt_func(case):
        varset = {abs(l) for clause in case['clauses'] for l in clause['literals']}
        unused = case['m'] - len(varset)
        case_desc = "\n".join(
            f"Clause {i+1}: ({' OR '.join(f'x{abs(l)}' if l>0 else '¬x{abs(l)}' for l in c['literals'])})" 
            for i,c in enumerate(case['clauses'])
        )
        return f"""Natalia Romanova's 2-Eblackwidow-2 Equation

**Equation Specification**
- Total variables: {case['m']} (x1~x{case['m']})
- Unused variables: {unused}
- Total clauses: {case['n']}

**Clauses Structure**
{case_desc}

**Constraints**
1. Each clause contains 1-2 literals 
2. Variables appear ≤2 times total

**Task**: Calculate the number of satisfying assignments modulo 1e9+7

**Answer Format**  
Place your final numeric answer within [answer] tags, e.g.:  
[answer]1234[/answer]"""

    @staticmethod 
    def extract_output(text):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', text, re.DOTALL)
        if not matches: return None
        try: return int(matches[-1].strip().split()[0])
        except: return None

    @classmethod
    def _verify_correction(cls, solution, case):
        try:
            expected = cls._compute_answer(case)
            return solution % (10**9+7) == expected
        except:
            return False

    @staticmethod
    def _compute_answer(case):
        MOD = 10**9+7
        clauses = case['clauses']
        used = {abs(l) for c in clauses for l in c['literals']}
        free = case['m'] - len(used)
        var_list = sorted(used)
        total = 0
        
        # 暴力解法仅适用于小规模案例
        for bits in range(2**len(var_list)):
            assign = {v:(bits>>i)&1 for i,v in enumerate(var_list)}
            xor_sum = 0
            for c in clauses:
                clause_val = 0
                for lit in c['literals']:
                    v = abs(lit)
                    val = assign[v] if lit>0 else 1-assign[v]
                    if val:
                        clause_val = 1
                        break
                xor_sum ^= clause_val
            total += (xor_sum ==1)
        
        return (total * pow(2, free, MOD)) % MOD
