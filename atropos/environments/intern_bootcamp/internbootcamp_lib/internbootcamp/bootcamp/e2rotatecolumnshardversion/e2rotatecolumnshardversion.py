"""# 

### 谜题描述
This is a harder version of the problem. The difference is only in constraints.

You are given a rectangular n × m matrix a. In one move you can choose any column and cyclically shift elements in this column. You can perform this operation as many times as you want (possibly zero). You can perform this operation to a column multiple times.

After you are done with cyclical shifts, you compute for every row the maximal value in it. Suppose that for i-th row it is equal r_i. What is the maximal possible value of r_1+r_2+…+r_n?

Input

The first line contains an integer t (1 ≤ t ≤ 40), the number of test cases in the input.

The first line of each test case contains integers n and m (1 ≤ n ≤ 12, 1 ≤ m ≤ 2000) — the number of rows and the number of columns in the given matrix a. 

Each of the following n lines contains m integers, the elements of a (1 ≤ a_{i, j} ≤ 10^5).

Output

Print t integers: answers for all test cases in the order they are given in the input.

Example

Input


3
2 3
2 5 7
4 2 4
3 6
4 1 5 2 10 4
8 6 6 4 9 10
5 4 9 5 8 7
3 3
9 9 9
1 1 1
1 1 1


Output


12
29
27

Note

In the first test case you can shift the third column down by one, this way there will be r_1 = 5 and r_2 = 7.

In the second case you can don't rotate anything at all, this way there will be r_1 = r_2 = 10 and r_3 = 9.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
const int M = 15;
const int N = 3005;
const int S = 4505;
using namespace std;
int k;
int T, s;
int n, m;
int dI[N];
int Id[S];
int val[N];
int sum[S];
int f[N][S];
int row[N][M];
const int I = 1000000007;
void out_p(int x) {
  while (x) {
    if (x & 1)
      putchar('1');
    else
      putchar('0');
    x >>= 1;
  }
  putchar(' ');
}
bool comp(int a, int b) { return val[a] > val[b]; }
int main() {
  for (int i = 0; i <= 12; ++i) Id[1 << i] = i + 1;
  cin >> T;
  while (T--) {
    cin >> n >> m;
    memset(f, -63, sizeof(f));
    memset(val, 0, sizeof(val));
    s = (1 << n) - 1, k = min(n, m);
    for (int i = 1; i <= n; ++i)
      for (int j = 1; j <= m; ++j) {
        cin >> row[j][i];
        f[0][0] = 0;
        val[j] = max(val[j], row[j][i]);
      }
    for (int i = 1; i <= m; ++i) dI[i] = i;
    sort(dI + 1, dI + m + 1, comp);
    for (int i = 1; i <= k; ++i) {
      int* now = row[dI[i]];
      for (int p = 1; p <= s; ++p) {
        sum[p] = sum[p - (p & -p)] + now[Id[(p & -p)]];
      }
      for (int p = 1; p <= s; ++p)
        for (int o = 0; o < n; ++o) {
          int q = ((p >> o) | (p << (n - o))) & s;
          sum[p] = max(sum[p], sum[q]);
        }
      f[i][0] = f[i - 1][0];
      for (int p = 1; p <= s; ++p) {
        f[i][p] = f[i - 1][p];
        for (int q = p; q; q = (q - 1) & p)
          f[i][p] = max(f[i][p], f[i - 1][p ^ q] + sum[q]);
      }
    }
    printf(\"%d\n\", f[k][s]);
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def solve_puzzle(test_cases):
    answers = []
    for case in test_cases:
        n = case['n']
        m = case['m']
        matrix = case['matrix']
        columns = []
        for j in range(m):
            col = []
            for i in range(n):
                col.append(matrix[i][j])
            columns.append(col)
        columns.sort(key=lambda col: max(col), reverse=True)
        k = min(n, m)
        selected_columns = columns[:k]
        Id = {}
        for i in range(n):
            mask = 1 << i
            Id[mask] = i + 1
        s = (1 << n) - 1
        prev_f = [-float('inf')] * (s + 1)
        prev_f[0] = 0
        for i in range(1, k + 1):
            current_col = selected_columns[i - 1]
            sum_p = [0] * (s + 1)
            sum_p[0] = 0
            for p in range(1, s + 1):
                lsb = p & -p
                row_val = current_col[Id[lsb] - 1] if lsb in Id else 0
                sum_p[p] = sum_p[p - lsb] + row_val
            for p in range(1, s + 1):
                for o in range(n):
                    shifted = ((p >> o) | (p << (n - o))) & s
                    sum_p[p] = max(sum_p[p], sum_p[shifted])
            current_f = [-float('inf')] * (s + 1)
            current_f[0] = prev_f[0]
            for p in range(s + 1):
                if p == 0:
                    continue
                current_f[p] = prev_f[p]
                q = p
                while q:
                    if (p ^ q) > s or (p ^ q) < 0:
                        q = (q - 1) & p
                        continue
                    if prev_f[p ^ q] + sum_p[q] > current_f[p]:
                        current_f[p] = prev_f[p ^ q] + sum_p[q]
                    q = (q - 1) & p
            prev_f = current_f
        answers.append(prev_f[s] if prev_f[s] != -float('inf') else 0)
    return answers

class E2rotatecolumnshardversionbootcamp(Basebootcamp):
    def __init__(self, t=1, max_n=12, max_m=2000, min_val=1, max_val=10**5):
        self.t = t
        self.max_n = max_n
        self.max_m = max_m
        self.min_val = min_val
        self.max_val = max_val
    
    def case_generator(self):
        test_cases = []
        attempts = 0
        max_attempts = 100 * self.t  # 防止无限循环
        while len(test_cases) < self.t and attempts < max_attempts:
            attempts += 1
            try:
                n = random.randint(1, self.max_n)
                m = random.randint(1, self.max_m)
                matrix = []
                for _ in range(n):
                    row = [random.randint(self.min_val, self.max_val) for _ in range(m)]
                    matrix.append(row)
                case_data = {'n': n, 'm': m, 'matrix': matrix}
                answers = solve_puzzle([case_data])
                test_cases.append({
                    'n': n, 'm': m, 'matrix': matrix, 'answer': answers[0]
                })
            except Exception as e:
                continue
        return {'test_cases': test_cases}
    
    @staticmethod
    def prompt_func(question_case):
        test_cases = question_case['test_cases']
        input_lines = [str(len(test_cases))]
        for case in test_cases:
            input_lines.append(f"{case['n']} {case['m']}")
            for row in case['matrix']:
                input_lines.append(' '.join(map(str, row)))
        input_data = '\n'.join(input_lines)
        prompt = f"""You are given a matrix and can cyclically shift columns to maximize the sum of each row's maximum. Solve the test cases and provide answers within [answer] tags.

Input:
{input_data}

Output your answers as integers, separated by spaces within [answer]. Example:
[answer]1 2 3[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_answer = matches[-1].strip()
        try:
            numbers = list(map(int, re.split(r'[\s,]+', last_answer)))
            return numbers
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not isinstance(solution, list):
            return False
        test_cases = identity['test_cases']
        if len(solution) != len(test_cases):
            return False
        return all(sol == case['answer'] for sol, case in zip(solution, test_cases))
