"""# 

### 谜题描述
'In Boolean logic, a formula is in conjunctive normal form (CNF) or clausal normal form if it is a conjunction of clauses, where a clause is a disjunction of literals' (cited from https://en.wikipedia.org/wiki/Conjunctive_normal_form)

In the other words, CNF is a formula of type <image>, where & represents a logical \"AND\" (conjunction), <image> represents a logical \"OR\" (disjunction), and vij are some boolean variables or their negations. Each statement in brackets is called a clause, and vij are called literals.

You are given a CNF containing variables x1, ..., xm and their negations. We know that each variable occurs in at most two clauses (with negation and without negation in total). Your task is to determine whether this CNF is satisfiable, that is, whether there are such values of variables where the CNF value is true. If CNF is satisfiable, then you also need to determine the values of the variables at which the CNF is true. 

It is guaranteed that each variable occurs at most once in each clause.

Input

The first line contains integers n and m (1 ≤ n, m ≤ 2·105) — the number of clauses and the number variables, correspondingly.

Next n lines contain the descriptions of each clause. The i-th line first contains first number ki (ki ≥ 1) — the number of literals in the i-th clauses. Then follow space-separated literals vij (1 ≤ |vij| ≤ m). A literal that corresponds to vij is x|vij| either with negation, if vij is negative, or without negation otherwise.

Output

If CNF is not satisfiable, print a single line \"NO\" (without the quotes), otherwise print two strings: string \"YES\" (without the quotes), and then a string of m numbers zero or one — the values of variables in satisfying assignment in the order from x1 to xm.

Examples

Input

2 2
2 1 -2
2 2 -1


Output

YES
11


Input

4 3
1 1
1 2
3 -1 -2 3
1 -3


Output

NO


Input

5 6
2 1 2
3 1 -2 3
4 -3 5 4 6
2 -6 -4
1 5


Output

YES
100010

Note

In the first sample test formula is <image>. One of possible answer is x1 = TRUE, x2 = TRUE.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const long long mod = 1000000007;
long long powmod(long long a, long long b) {
  long long res = 1;
  a %= mod;
  for (; b; b >>= 1) {
    if (b & 1) res = res * a % mod;
    a = a * a % mod;
  }
  return res;
}
int n, m, l, a;
int idx[400005][2];
vector<int> model;
vector<vector<int> > clauses;
vector<bool> satisfied;
int cntDetermined;
stack<pair<vector<int>, vector<int> > > st;
bool isDeterministic(int i) {
  int k = idx[i][0];
  if (i > m) i -= m;
  for (int l = 0; l < (int)clauses[k].size(); l++) {
    int ii = clauses[k][l];
    if (abs(ii) == i) continue;
    if (model[abs(ii)] == -1) return 0;
  }
  return 1;
}
void findModel(int i);
void updateSatisfied(int i) {
  satisfied[i] = 1;
  for (int j = 0; j < (int)clauses[i].size(); j++) {
    int next = clauses[i][j];
    if (model[abs(next)] == -1) {
      if (next < 0) next = abs(next) + m;
      if (idx[next][0] == i)
        idx[next][0] = idx[next][1];
      else
        idx[next][1] = -1;
      if (idx[next][0] == -1) {
        if (next > m) next -= m;
        findModel(next);
      }
    }
  }
}
void findModel(int i) {
  if (idx[i][0] == -1 && idx[i + m][0] == -1) {
    model[i] = 1;
    cntDetermined++;
    return;
  } else if (idx[i][0] == -1) {
    model[i] = 0;
    cntDetermined++;
    if (satisfied[idx[i + m][0]] == 0) updateSatisfied(idx[i + m][0]);
    if (idx[i + m][1] != -1 && satisfied[idx[i + m][1]] == 0)
      updateSatisfied(idx[i + m][1]);
  } else if (idx[i + m][0] == -1) {
    model[i] = 1;
    cntDetermined++;
    if (satisfied[idx[i][0]] == 0) updateSatisfied(idx[i][0]);
    if (idx[i][1] != -1 && satisfied[idx[i][1]] == 0)
      updateSatisfied(idx[i][1]);
  } else {
    if (satisfied[idx[i][0]] && satisfied[idx[i + m][0]]) {
      model[i] = 1;
      cntDetermined++;
      return;
    } else if (satisfied[idx[i][0]]) {
      model[i] = 0;
      cntDetermined++;
      updateSatisfied(idx[i + m][0]);
      return;
    } else if (satisfied[idx[i + m][0]]) {
      model[i] = 1;
      cntDetermined++;
      updateSatisfied(idx[i][0]);
      return;
    }
    bool determined = isDeterministic(i);
    if (determined) {
      model[i] = 1;
      cntDetermined++;
      if (satisfied[idx[i][0]] == 0) updateSatisfied(idx[i][0]);
      return;
    }
    determined = isDeterministic(i + m);
    if (determined) {
      model[i] = 0;
      cntDetermined++;
      if (satisfied[idx[i + m][0]] == 0) updateSatisfied(idx[i + m][0]);
      return;
    }
  }
}
void findModel(int i, vector<int>& idx1, vector<int>& idx2);
void updateSatisfied(int i, vector<int>& idx1, vector<int>& idx2) {
  satisfied[i] = 1;
  idx2.push_back(i);
  for (int j = 0; j < (int)clauses[i].size(); j++) {
    int next = clauses[i][j];
    if (model[abs(next)] == -1) {
      if (next < 0) next = abs(next) + m;
      if (idx[next][0] == i)
        idx[next][0] = idx[next][1];
      else
        idx[next][1] = -1;
      if (idx[next][0] == -1) {
        if (next > m) next -= m;
        findModel(next, idx1, idx2);
      }
    }
  }
}
void findModel(int i, vector<int>& idx1, vector<int>& idx2) {
  if (idx[i][0] == -1 && idx[i + m][0] == -1) {
    model[i] = 1;
    idx1.push_back(i);
    cntDetermined++;
    return;
  } else if (idx[i][0] == -1) {
    model[i] = 0;
    idx1.push_back(i);
    cntDetermined++;
    if (satisfied[idx[i + m][0]] == 0)
      updateSatisfied(idx[i + m][0], idx1, idx2);
    if (idx[i + m][1] != -1 && satisfied[idx[i + m][1]] == 0)
      updateSatisfied(idx[i + m][1], idx1, idx2);
  } else if (idx[i + m][0] == -1) {
    model[i] = 1;
    idx1.push_back(i);
    cntDetermined++;
    if (satisfied[idx[i][0]] == 0) updateSatisfied(idx[i][0], idx1, idx2);
    if (idx[i][1] != -1 && satisfied[idx[i][1]] == 0)
      updateSatisfied(idx[i][1], idx1, idx2);
  } else {
    if (satisfied[idx[i][0]] && satisfied[idx[i + m][0]]) {
      model[i] = 1;
      idx1.push_back(i);
      cntDetermined++;
      return;
    } else if (satisfied[idx[i][0]]) {
      model[i] = 0;
      idx1.push_back(i);
      cntDetermined++;
      updateSatisfied(idx[i + m][0], idx1, idx2);
      return;
    } else if (satisfied[idx[i + m][0]]) {
      model[i] = 1;
      idx1.push_back(i);
      cntDetermined++;
      updateSatisfied(idx[i][0], idx1, idx2);
      return;
    }
    bool determined = isDeterministic(i);
    if (determined) {
      model[i] = 1;
      idx1.push_back(i);
      cntDetermined++;
      if (satisfied[idx[i][0]] == 0) updateSatisfied(idx[i][0], idx1, idx2);
      return;
    }
    determined = isDeterministic(i + m);
    if (determined) {
      model[i] = 0;
      idx1.push_back(i);
      cntDetermined++;
      if (satisfied[idx[i + m][0]] == 0)
        updateSatisfied(idx[i + m][0], idx1, idx2);
      return;
    }
  }
}
bool isCorrectModel() {
  for (int i = 0; i < (int)n; i++) {
    bool possible = false;
    for (int j = 0; j < (int)clauses[i].size(); j++) {
      int ii = clauses[i][j];
      if (ii > 0 && model[ii] == 1) {
        possible = true;
        break;
      } else if (ii < 0 && model[-ii] == 0) {
        possible = true;
        break;
      }
    }
    if (!possible) return false;
  }
  return true;
}
bool fullSearch(int lim) {
  if (cntDetermined == m) {
    if (isCorrectModel()) {
      return true;
    }
    return false;
  }
  for (int j = lim; j < (int)m + 1; j++) {
    if (model[j] == -1) {
      for (int k = 0; k < (int)2; k++) {
        vector<int> idx1, idx2;
        model[j] = k;
        cntDetermined++;
        if (satisfied[idx[j + m * (1 - k)][0]] == 0) {
          satisfied[idx[j + m * (1 - k)][0]] = 1;
          idx2.push_back(idx[j + m * (1 - k)][0]);
        }
        idx1.push_back(j);
        while (1) {
          int cnt = cntDetermined;
          bool next = false;
          for (int i = 1; i < (int)m + 1; i++) {
            if (model[i] == -1) {
              next = true;
              findModel(i, idx1, idx2);
            }
          }
          if (cnt == cntDetermined) break;
          if (!next) break;
        }
        st.push(make_pair(idx1, idx2));
        if (fullSearch(j + 1))
          return true;
        else {
          vector<int> idx1 = st.top().first, idx2 = st.top().second;
          st.pop();
          cntDetermined -= idx1.size();
          for (int l = 0; l < (int)idx1.size(); l++) model[idx1[l]] = -1;
          for (int l = 0; l < (int)idx2.size(); l++) satisfied[idx2[l]] = 0;
          continue;
        }
      }
      return false;
    }
  }
}
int main() {
  scanf(\"%d%d\", &n, &m);
  model.resize(m + 1, -1);
  memset(idx, -1, sizeof(idx));
  for (int i = 0; i < (int)n; i++) {
    scanf(\"%d\", &l);
    vector<int> clause(l);
    for (int j = 0; j < (int)l; j++) {
      scanf(\"%d\", &a);
      clause[j] = a;
      if (a < 0) a = abs(a) + m;
      if (idx[a][0] == -1)
        idx[a][0] = i;
      else
        idx[a][1] = i;
    }
    clauses.push_back(clause);
  }
  satisfied.resize(n, 0);
  cntDetermined = 0;
  int lim = 1;
  while (1) {
    int cnt = cntDetermined;
    bool next = false;
    int lim1 = -1;
    for (int i = lim; i < (int)m + 1; i++) {
      if (model[i] == -1) {
        next = true;
        findModel(i);
      }
      if (model[i] == -1 && lim1 == -1) lim1 = i;
    }
    lim = lim1;
    if (cnt == cntDetermined) break;
    if (!next) break;
  }
  if (cntDetermined != m) {
    fullSearch(1);
  }
  if (!isCorrectModel())
    printf(\"NO\n\");
  else {
    printf(\"YES\n\");
    for (int i = 1; i < (int)m + 1; i++) printf(\"%d\", model[i]);
    printf(\"\n\");
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import defaultdict
import itertools
import re

class Ecnf2bootcamp(Basebootcamp):
    def __init__(self, max_variables=10, max_clauses=20):
        self.max_variables = max_variables
        self.max_clauses = max_clauses
    
    def case_generator(self):
        while True:
            m = random.randint(1, self.max_variables)
            n = random.randint(1, self.max_clauses)
            clauses = []
            var_count = defaultdict(int)
            valid = False
            for _ in range(n):
                clause = []
                current_vars = set()
                k = random.randint(1, 3)
                for _ in range(k):
                    available_vars = [var for var in range(1, m+1) if var_count[var] < 2 and var not in current_vars]
                    if not available_vars:
                        break
                    var = random.choice(available_vars)
                    sign = random.choice([1, -1])
                    lit = var * sign
                    clause.append(lit)
                    current_vars.add(var)
                    var_count[var] += 1
                if clause:
                    clauses.append(clause)
                else:
                    continue
            valid = True
            for var in range(1, m+1):
                if var_count[var] > 2:
                    valid = False
                    break
            if not valid:
                continue
            case = {
                "n": len(clauses),
                "m": m,
                "clauses": clauses
            }
            return case
    
    @staticmethod
    def solve_cnf_with_brute_force(n, m, clauses):
        for bits in itertools.product([0, 1], repeat=m):
            assignment = {i+1: bits[i] for i in range(m)}
            all_satisfied = True
            for clause in clauses:
                satisfied = False
                for lit in clause:
                    var = abs(lit)
                    val = assignment[var]
                    if (lit > 0 and val == 1) or (lit < 0 and val == 0):
                        satisfied = True
                        break
                if not satisfied:
                    all_satisfied = False
                    break
            if all_satisfied:
                return (True, bits)
        return (False, None)
    
    @staticmethod
    def prompt_func(question_case):
        m = question_case['m']
        n = question_case['n']
        clauses = question_case['clauses']
        input_lines = [f"{n} {m}"]
        for clause in clauses:
            input_lines.append(f"{len(clause)} " + " ".join(map(str, clause)))
        input_example = '\n'.join(input_lines)
        prompt = f"""In Boolean logic, a formula is in conjunctive normal form (Ecnf2) if it is a conjunction of clauses, where each clause is a disjunction of literals. You are given a Ecnf2 formula where each variable appears in at most two clauses (including its negation).

Your task is to determine if the Ecnf2 is satisfiable. If it is, provide a satisfying assignment for the variables.

Input:
{input_example}

Output:
- If not satisfiable, output "NO".
- If satisfiable, output "YES" followed by a string of {m} digits (0 or 1) representing the values of x1 to xm.

Please place your answer within [answer] and [/answer] tags. For example:
[answer]YES
1101[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        lines = last_match.splitlines()
        if not lines:
            return None
        first_line = lines[0].strip().upper()
        if first_line == "NO":
            return "NO"
        elif first_line == "YES" and len(lines) >= 2:
            assignment = lines[1].strip()
            return ("YES", assignment)
        else:
            parts = last_match.split()
            if len(parts) >= 2 and parts[0].upper() == "YES":
                return ("YES", parts[1])
            elif len(parts) == 1 and parts[0].upper() == "NO":
                return "NO"
        return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        m = identity['m']
        clauses = identity['clauses']
        if solution is None:
            return False
        if solution == "NO":
            is_sat, _ = cls.solve_cnf_with_brute_force(identity['n'], m, clauses)
            return not is_sat
        elif isinstance(solution, tuple) and len(solution) == 2 and solution[0].upper() == "YES":
            assignment_str = solution[1]
            if len(assignment_str) != m:
                return False
            if any(c not in '01' for c in assignment_str):
                return False
            assignment = {i+1: int(assignment_str[i]) for i in range(m)}
            for clause in clauses:
                satisfied = any(
                    (lit > 0 and assignment[abs(lit)] == 1) or 
                    (lit < 0 and assignment[abs(lit)] == 0) 
                    for lit in clause
                )
                if not satisfied:
                    return False
            return True
        else:
            return False
