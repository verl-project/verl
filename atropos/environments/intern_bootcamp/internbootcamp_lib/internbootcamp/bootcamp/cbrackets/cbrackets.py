"""# 

### 谜题描述
A two dimensional array is called a bracket array if each grid contains one of the two possible brackets — \"(\" or \")\". A path through the two dimensional array cells is called monotonous if any two consecutive cells in the path are side-adjacent and each cell of the path is located below or to the right from the previous one. 

A two dimensional array whose size equals n × m is called a correct bracket array, if any string formed by writing out the brackets on some monotonous way from cell (1, 1) to cell (n, m) forms a correct bracket sequence. 

Let's define the operation of comparing two correct bracket arrays of equal size (a and b) like that. Let's consider a given two dimensional array of priorities (c) — a two dimensional array of same size, containing different integers from 1 to nm. Let's find such position (i, j) in the two dimensional array, that ai, j ≠ bi, j. If there are several such positions, let's choose the one where number ci, j is minimum. If ai, j = \"(\", then a < b, otherwise a > b. If the position (i, j) is not found, then the arrays are considered equal.

Your task is to find a k-th two dimensional correct bracket array. It is guaranteed that for the given sizes of n and m there will be no less than k two dimensional correct bracket arrays.

Input

The first line contains integers n, m and k — the sizes of the array and the number of the sought correct bracket array (1 ≤ n, m ≤ 100, 1 ≤ k ≤ 1018). Then an array of priorities is given, n lines each containing m numbers, number pi, j shows the priority of character j in line i (1 ≤ pi, j ≤ nm, all pi, j are different).

Please do not use the %lld specificator to read or write 64-bit integers in С++. It is preferred to use the cin, cout streams or the %I64d specificator.

Output

Print the k-th two dimensional correct bracket array.

Examples

Input

1 2 1
1 2


Output

()


Input

2 3 1
1 2 3
4 5 6


Output

(()
())


Input

3 2 2
3 6
1 4
2 5


Output

()
)(
()

Note

In the first sample exists only one correct two-dimensional bracket array.

In the second and in the third samples two arrays exist.

A bracket sequence is called regular if it is possible to obtain correct arithmetic expression by inserting characters «+» and «1» into this sequence. For example, sequences «(())()», «()» and «(()(()))» are regular, while «)(», «(()» and «(()))(» are not.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
int main() {
  int N, M;
  long long int what;
  scanf(\"%d %d %I64i\", &N, &M, &what);
  static int data[205][205];
  static int p[205 * 205];
  int i, j;
  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      scanf(\"%d\", &(data[i][j]));
      (data[i][j])--;
      p[data[i][j]] = i + j;
    }
  }
  static char res[205];
  memset(res, '.', sizeof(res));
  static long long int a[205][205];
  int x, y;
  int K = N + M - 1;
  for (i = 0; i < (N * M); i++) {
    if (res[p[i]] == '.') {
      res[p[i]] = '(';
      memset(a, 0, sizeof(a));
      a[0][0] = 1LL;
      for (x = 1; x <= K; x++) {
        for (y = 0; y <= K; y++) {
          if ((y > 0) && (res[x - 1] != ')')) a[x][y] += a[x - 1][y - 1];
          if (res[x - 1] != '(') a[x][y] += a[x - 1][y + 1];
          if (a[x][y] > 2000000000000000000LL) a[x][y] = 2000000000000000000LL;
        }
      }
      if (a[K][0] < what) {
        res[p[i]] = ')';
        what -= a[K][0];
      }
    }
  }
  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      printf(\"%c\", res[i + j]);
    }
    printf(\"\n\");
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

def generate_correct_array(n, m, k, priority):
    size = n * m
    p = [0] * size
    for i in range(n):
        for j in range(m):
            val = priority[i][j] - 1  # Convert to 0-based index
            p[val] = i + j
    res = ['.'] * (n + m - 1)
    K = n + m - 1
    for i in range(size):
        s = p[i]
        if res[s] == '.':
            res[s] = '('
            a = [[0] * (K + 2) for _ in range(K + 2)]
            a[0][0] = 1
            for x in range(1, K + 1):
                for y in range(K + 1):
                    a[x][y] = 0
                    current_char = res[x-1] if (x-1) < len(res) else '.'
                    if y > 0 and current_char != ')':
                        a[x][y] += a[x-1][y-1]
                    if current_char != '(':
                        a[x][y] += a[x-1][y+1]
                    if a[x][y] > 1e18:
                        a[x][y] = 1e18
            total = a[K][0]
            if total < k:
                res[s] = ')'
                k -= total
    result = []
    for i in range(n):
        row = []
        for j in range(m):
            s = i + j
            row.append(res[s])
        result.append(''.join(row))
    return result

class Cbracketsbootcamp(Basebootcamp):
    def __init__(self, n=2, m=3, k=1, priority=None):
        super().__init__()
        self.n = n
        self.m = m
        self.k = k
        if priority is None:
            # Generate a random priority matrix with 1..n*m
            elements = list(range(1, n * m + 1))
            random.shuffle(elements)
            self.priority = []
            idx = 0
            for i in range(n):
                row = elements[idx:idx + m]
                self.priority.append(row)
                idx += m
        else:
            self.priority = [row.copy() for row in priority]
        self.validate_priority()

    def validate_priority(self):
        elements = []
        for row in self.priority:
            elements.extend(row)
        if len(elements) != self.n * self.m:
            raise ValueError(f"Priority matrix size does not match n={self.n} and m={self.m}.")
        if len(set(elements)) != len(elements):
            raise ValueError("Priority matrix contains duplicate values.")
        expected = set(range(1, self.n * self.m + 1))
        if set(elements) != expected:
            raise ValueError(f"Priority matrix elements must be unique and cover 1 to {self.n*self.m}.")

    def case_generator(self):
        # Generate a new random priority matrix each time to ensure diversity
        elements = list(range(1, self.n * self.m + 1))
        random.shuffle(elements)
        priority = []
        idx = 0
        for i in range(self.n):
            row = elements[idx:idx + self.m]
            priority.append(row)
            idx += self.m
        # Ensure k is valid by setting to 1 (guaranteed by problem constraints)
        return {
            'n': self.n,
            'm': self.m,
            'k': 1,
            'priority': priority
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        k = question_case['k']
        priority = question_case['priority']
        prompt = f"""You are to solve the k-th correct two-dimensional bracket array problem. The task is to find the k-th smallest correct bracket array based on the given priority matrix.

Input:
{n} {m} {k}
Priority matrix:
"""
        for row in priority:
            prompt += ' '.join(map(str, row)) + '\n'
        prompt += f"""
A correct bracket array satisfies that every possible monotonous path from (1,1) to ({n},{m}) forms a valid bracket sequence. The k-th array is determined by comparing arrays using the priority matrix to find the earliest differing cell with the smallest priority.

Output the correct {n}x{m} bracket array. Each row must contain exactly {m} parentheses. Place your answer between [answer] and [/answer], with each row on a separate line.

Example format:
[answer]
()
[/answer]

Now, provide the answer for the given input:"""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        content = matches[-1].strip()
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        return lines

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            if not solution:
                return False
            # Validate solution dimensions
            n = identity['n']
            m = identity['m']
            if len(solution) != n:
                return False
            for row in solution:
                if len(row) != m:
                    return False
            # Generate correct answer
            correct = generate_correct_array(
                identity['n'],
                identity['m'],
                identity['k'],
                identity['priority']
            )
            return solution == correct
        except Exception as e:
            return False
