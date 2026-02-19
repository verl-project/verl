"""# 

### 谜题描述
You are given two tables A and B of size n × m. 

We define a sorting by column as the following: we choose a column and reorder the rows of the table by the value in this column, from the rows with the smallest value to the rows with the largest. In case there are two or more rows with equal value in this column, their relative order does not change (such sorting algorithms are called stable).

You can find this behavior of sorting by column in many office software for managing spreadsheets. Petya works in one, and he has a table A opened right now. He wants to perform zero of more sortings by column to transform this table to table B.

Determine if it is possible to do so, and if yes, find a sequence of columns to sort by. Note that you do not need to minimize the number of sortings.

Input

The first line contains two integers n and m (1 ≤ n, m ≤ 1500) — the sizes of the tables.

Each of the next n lines contains m integers a_{i,j} (1 ≤ a_{i, j} ≤ n), denoting the elements of the table A.

Each of the next n lines contains m integers b_{i, j} (1 ≤ b_{i, j} ≤ n), denoting the elements of the table B.

Output

If it is not possible to transform A into B, print -1.

Otherwise, first print an integer k (0 ≤ k ≤ 5000) — the number of sortings in your solution.

Then print k integers c_1, …, c_k (1 ≤ c_i ≤ m) — the columns, by which Petya needs to perform a sorting.

We can show that if a solution exists, there is one in no more than 5000 sortings.

Examples

Input


2 2
2 2
1 2
1 2
2 2


Output


1
1

Input


3 3
2 3 2
1 3 3
1 1 2
1 1 2
1 3 3
2 3 2


Output


2
1 2

Input


2 2
1 1
2 1
2 1
1 1


Output


-1

Input


4 1
2
2
2
1
1
2
2
2


Output


1
1 

Note

Consider the second example. After the sorting by the first column the table becomes

$$$\begin{matrix} 1&3&3\\\ 1&1&2\\\ 2&3&2. \end{matrix}$$$

After the sorting by the second column the table becomes

$$$\begin{matrix} 1&1&2\\\ 1&3&3\\\ 2&3&2, \end{matrix}$$$

and this is what we need.

In the third test any sorting does not change anything, because the columns are already sorted.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>

using namespace std;
typedef long long ll;

int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);
  int n, m;
  cin >> n >> m;
  vector<vector<int>> a(n, vector<int>(m));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) cin >> a[i][j];
  }
  vector<vector<int>> b(n, vector<int>(m));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) cin >> b[i][j];
  }
  vector<vector<int>> bb;
  for (int i = 0; i < n; ++i) bb.push_back(b[i]);
  sort(bb.begin(), bb.end());
  bb.erase(unique(bb.begin(), bb.end()), bb.end());
  vector<vector<int>> who_a(bb.size());
  vector<vector<int>> who_b(bb.size());
  for (int i = 0; i < n; ++i) {
    int ind = lower_bound(bb.begin(), bb.end(), a[i]) - bb.begin();
    if (ind == (int)bb.size() || bb[ind] != a[i]) {
      cout << -1 << endl;
      return 0;
    }
    who_a[ind].push_back(i);
    ind = lower_bound(bb.begin(), bb.end(), b[i]) - bb.begin();
    who_b[ind].push_back(i);
  }
  vector<int> p(n);
  for (int i = 0; i < (int)bb.size(); ++i) {
    if (who_a[i].size() != who_b[i].size()) {
      cout << -1 << endl;
      return 0;
    }
    for (int j = 0; j < (int)who_a[i].size(); ++j) {
      p[who_a[i][j]] = who_b[i][j];
    }
  }
  vector<int> q(n);
  for (int i = 0; i < n; ++i) q[p[i]] = i;
  vector<bool> good(n, 0);
  vector<int> cnt_bad(m, 0), cnt_kek(m, 0);
  for (int j = 0; j < m; ++j) {
    for (int i = 0; i < n - 1; ++i) {
      if (a[q[i]][j] < a[q[i + 1]][j]) ++cnt_kek[j];
      if (a[q[i]][j] > a[q[i + 1]][j]) ++cnt_bad[j];
    }
  }
  vector<int> ans;
  while (true) {
    bool ok = true;
    for (int i = 0; i < n - 1; ++i) ok &= good[i] || q[i] < q[i + 1];
    if (ok) break;
    int col = -1;
    for (int j = 0; j < m; ++j) {
      if (cnt_bad[j] == 0 && cnt_kek[j] > 0) {
        col = j;
        break;
      }
    }
    if (col == -1) break;
    ans.push_back(col);
    for (int i = 0; i < n - 1; ++i) {
      if (!good[i] && a[q[i]][col] < a[q[i + 1]][col]) {
        good[i] = true;
        for (int j = 0; j < m; ++j) {
            if (a[q[i]][j] < a[q[i + 1]][j]) --cnt_kek[j];
            if (a[q[i]][j] > a[q[i + 1]][j]) --cnt_bad[j];
        }
      }
    }
  }
  bool ok = true;
  for (int i = 0; i < n - 1; ++i) ok &= good[i] || q[i] < q[i + 1];
  if (ok) {
    cout << (int)ans.size() << endl;
    reverse(ans.begin(), ans.end());
    for (int i : ans) cout << i + 1 << \" \";
    cout << endl;
  } else {
    cout << -1 << endl;
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from copy import deepcopy
from typing import List, Union

from bootcamp import Basebootcamp

class Cmatrixsortingbootcamp(Basebootcamp):
    def __init__(self, n: int = 3, m: int = 3, has_solution_prob: float = 0.5):
        self.n = n
        self.m = m
        self.has_solution_prob = has_solution_prob

    def case_generator(self) -> dict:
        if random.random() < self.has_solution_prob:
            A = self._generate_matrix()
            cols = self._generate_sorting_columns()
            B = self._apply_sorting(A, cols)
            return {
                'n': self.n,
                'm': self.m,
                'A': A,
                'B': B,
                'has_solution': True
            }
        else:
            A = self._generate_matrix()
            cols = self._generate_sorting_columns()
            B = self._apply_sorting(A, cols)
            B = self._corrupt_matrix(A, B)
            return {
                'n': self.n,
                'm': self.m,
                'A': A,
                'B': B,
                'has_solution': False
            }

    @staticmethod
    def prompt_func(question_case: dict) -> str:  # 修正参数名称为question_case
        n = question_case['n']
        m = question_case['m']  # 正确获取表格尺寸参数
        prompt = f"""给定两个{n}x{m}表格A和B：
        
表A：
"""
        prompt += '\n'.join(' '.join(map(str, row)) for row in question_case['A'])
        prompt += "\n\n表B：\n"
        prompt += '\n'.join(' '.join(map(str, row)) for row in question_case['B'])
        prompt += "\n\n是否可通过列排序转换？答案格式：\n- 无解：[answer]-1[/answer]\n- 有解：[answer]列序列[/answer] (如[answer]1 2 3[/answer])"
        return prompt

    @staticmethod
    def extract_output(output: str) -> Union[List[int], int, None]:
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        
        if last_match == '-1':
            return -1
        if last_match == '0':
            return []
        
        try:
            return list(map(int, last_match.split()))
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):        
        A = identity['A']
        B = identity['B']
        m = identity['m']
        has_solution = identity['has_solution']

        if not has_solution:
            return solution == -1
        
        if solution == -1:
            return False
        
        if solution == []:
            return A == B
        
        if any(not 1 <= c <= m for c in solution):
            return False

        sorted_table = deepcopy(A)
        for col in solution:
            sorted_table.sort(key=lambda row: row[col-1])
        return sorted_table == B

    def _generate_matrix(self):
        return [[random.randint(1, self.n) for _ in range(self.m)] 
                for _ in range(self.n)]

    def _generate_sorting_columns(self):
        return random.choices(range(1, self.m+1), 
                            k=random.randint(0, self.m))

    def _apply_sorting(self, matrix, columns):
        sorted_mat = deepcopy(matrix)
        for col in columns:
            sorted_mat.sort(key=lambda row: row[col-1])
        return sorted_mat

    def _corrupt_matrix(self, A, B):  # 统一方法名称
        B_prime = deepcopy(B)
        A_rows = {tuple(row) for row in A}
        
        # 确保至少存在一个非法行
        for i in range(self.n):
            if tuple(B_prime[i]) not in A_rows:
                continue
                
            for j in range(self.m):
                for v in range(1, self.n+2):
                    new_row = list(B_prime[i])
                    new_row[j] = v
                    if tuple(new_row) not in A_rows:
                        B_prime[i] = new_row
                        return B_prime
        
        # 最终保护：随机生成全新行
        while True:
            new_row = [random.randint(1, self.n+1) for _ in range(self.m)]
            if tuple(new_row) not in A_rows:
                B_prime[random.randint(0, self.n-1)] = new_row
                return B_prime
