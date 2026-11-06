"""# 

### 谜题描述
As we know, DZY loves playing games. One day DZY decided to play with a n × m matrix. To be more precise, he decided to modify the matrix with exactly k operations.

Each modification is one of the following:

  1. Pick some row of the matrix and decrease each element of the row by p. This operation brings to DZY the value of pleasure equal to the sum of elements of the row before the decreasing. 
  2. Pick some column of the matrix and decrease each element of the column by p. This operation brings to DZY the value of pleasure equal to the sum of elements of the column before the decreasing. 



DZY wants to know: what is the largest total value of pleasure he could get after performing exactly k modifications? Please, help him to calculate this value.

Input

The first line contains four space-separated integers n, m, k and p (1 ≤ n, m ≤ 103; 1 ≤ k ≤ 106; 1 ≤ p ≤ 100).

Then n lines follow. Each of them contains m integers representing aij (1 ≤ aij ≤ 103) — the elements of the current row of the matrix.

Output

Output a single integer — the maximum possible total pleasure value DZY could get.

Examples

Input

2 2 2 2
1 3
2 4


Output

11


Input

2 2 5 2
1 3
2 4


Output

11

Note

For the first sample test, we can modify: column 2, row 2. After that the matrix becomes:
    
    
      
    1 1  
    0 0  
      
    

For the second sample test, we can modify: column 2, row 2, row 1, column 1, column 2. After that the matrix becomes:
    
    
      
    -3 -3  
    -2 -2  
      
    

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const long long N = 1005;
const long long M = 1e6 + 5;
const long long inf = 1e15;
long long row[N], col[N];
long long ansr[M], ansc[M];
priority_queue<long long> q[2];
int main() {
  long long n, m, k, p, t;
  scanf(\"%I64d%I64d%I64d%I64d\", &n, &m, &k, &p);
  for (long long i = 1; i <= n; i++) {
    for (long long j = 1; j <= m; j++) {
      scanf(\"%I64d\", &t);
      row[i] += t;
      col[j] += t;
    }
  }
  for (long long i = 1; i <= n; i++) {
    q[0].push(row[i]);
  }
  for (long long i = 1; i <= m; i++) {
    q[1].push(col[i]);
  }
  for (long long i = 1; i <= k; i++) {
    long long t1 = q[0].top(), t2 = q[1].top();
    ansr[i] = ansr[i - 1] + t1;
    q[0].pop();
    ansc[i] = ansc[i - 1] + t2;
    q[1].pop();
    q[0].push(t1 - p * m);
    q[1].push(t2 - p * n);
  }
  long long ans = -inf;
  for (long long i = 0; i <= k; i++) {
    ans = max(ans, ansr[i] + ansc[k - i] - 1ll * i * (k - i) * p);
  }
  printf(\"%I64d\n\", ans);
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import heapq
import random
import re

class Ddzylovesmodificationbootcamp(Basebootcamp):
    def __init__(self, n_bounds=(2, 5), m_bounds=(2, 5), k_bounds=(1, 10), p_bounds=(1, 5), a_bounds=(1, 10)):
        self.n_bounds = n_bounds
        self.m_bounds = m_bounds
        self.k_bounds = k_bounds
        self.p_bounds = p_bounds
        self.a_min, self.a_max = a_bounds

    def case_generator(self):
        n = random.randint(*self.n_bounds)
        m = random.randint(*self.m_bounds)
        k = random.randint(*self.k_bounds)
        p = random.randint(*self.p_bounds)
        
        matrix = [[random.randint(self.a_min, self.a_max) for _ in range(m)] for _ in range(n)]
        correct_answer = self.calculate_max_pleasure(n, m, k, p, matrix)
        
        return {
            'n': n, 'm': m, 'k': k, 'p': p,
            'matrix': matrix, 'correct_answer': correct_answer
        }

    @staticmethod
    def calculate_max_pleasure(n, m, k, p, matrix):
        row_sums = [sum(row) for row in matrix]
        col_sums = [sum(col) for col in zip(*matrix)]

        row_heap = [-s for s in row_sums]
        heapq.heapify(row_heap)
        col_heap = [-s for s in col_sums]
        heapq.heapify(col_heap)

        ansr = [0] * (k+1)
        for i in range(1, k+1):
            if not row_heap:
                ansr[i] = ansr[i-1]
                continue
            current = -heapq.heappop(row_heap)
            ansr[i] = ansr[i-1] + current
            heapq.heappush(row_heap, -(current - p*m))

        ansc = [0] * (k+1)
        for i in range(1, k+1):
            if not col_heap:
                ansc[i] = ansc[i-1]
                continue
            current = -heapq.heappop(col_heap)
            ansc[i] = ansc[i-1] + current
            heapq.heappush(col_heap, -(current - p*n))

        max_total = -float('inf')
        for i in range(0, k+1):
            j = k - i
            if j < 0:
                continue
            current = ansr[i] + ansc[j] - i*j*p
            max_total = max(max_total, current)
        return max_total

    @staticmethod
    def prompt_func(question_case) -> str:
        input_lines = [
            f"{question_case['n']} {question_case['m']} {question_case['k']} {question_case['p']}"
        ]
        input_lines += [' '.join(map(str, row)) for row in question_case['matrix']]
        input_str = '\n'.join(input_lines)
        
        return f"""DZY 有一个 {question_case['n']} 行 {question_case['m']} 列的矩阵，需要执行恰好 {question_case['k']} 次操作：
1. 选择一行，将该行所有元素减少 {question_case['p']}，获得该行操作前的元素和作为愉悦值
2. 选择一列，将该列所有元素减少 {question_case['p']}，获得该列操作前的元素和作为愉悦值

请计算可以获得的最大总愉悦值。

输入数据：
{input_str}

请将最终答案数值放在 [answer] 和 [/answer] 标签之间，例如：[answer]11[/answer]。"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except (ValueError, IndexError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
