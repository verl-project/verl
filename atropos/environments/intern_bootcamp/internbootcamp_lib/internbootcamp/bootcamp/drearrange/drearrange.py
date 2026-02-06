"""# 

### 谜题描述
Koa the Koala has a matrix A of n rows and m columns. Elements of this matrix are distinct integers from 1 to n ⋅ m (each number from 1 to n ⋅ m appears exactly once in the matrix).

For any matrix M of n rows and m columns let's define the following:

  * The i-th row of M is defined as R_i(M) = [ M_{i1}, M_{i2}, …, M_{im} ] for all i (1 ≤ i ≤ n). 
  * The j-th column of M is defined as C_j(M) = [ M_{1j}, M_{2j}, …, M_{nj} ] for all j (1 ≤ j ≤ m). 



Koa defines S(A) = (X, Y) as the spectrum of A, where X is the set of the maximum values in rows of A and Y is the set of the maximum values in columns of A.

More formally:

  * X = \{ max(R_1(A)), max(R_2(A)), …, max(R_n(A)) \} 
  * Y = \{ max(C_1(A)), max(C_2(A)), …, max(C_m(A)) \}



Koa asks you to find some matrix A' of n rows and m columns, such that each number from 1 to n ⋅ m appears exactly once in the matrix, and the following conditions hold:

  * S(A') = S(A) 
  * R_i(A') is bitonic for all i (1 ≤ i ≤ n) 
  * C_j(A') is bitonic for all j (1 ≤ j ≤ m) 

An array t (t_1, t_2, …, t_k) is called bitonic if it first increases and then decreases.

More formally: t is bitonic if there exists some position p (1 ≤ p ≤ k) such that: t_1 < t_2 < … < t_p > t_{p+1} > … > t_k.

Help Koa to find such matrix or to determine that it doesn't exist.

Input

The first line of the input contains two integers n and m (1 ≤ n, m ≤ 250) — the number of rows and columns of A.

Each of the ollowing n lines contains m integers. The j-th integer in the i-th line denotes element A_{ij} (1 ≤ A_{ij} ≤ n ⋅ m) of matrix A. It is guaranteed that every number from 1 to n ⋅ m appears exactly once among elements of the matrix.

Output

If such matrix doesn't exist, print -1 on a single line.

Otherwise, the output must consist of n lines, each one consisting of m space separated integers — a description of A'.

The j-th number in the i-th line represents the element A'_{ij}.

Every integer from 1 to n ⋅ m should appear exactly once in A', every row and column in A' must be bitonic and S(A) = S(A') must hold.

If there are many answers print any.

Examples

Input


3 3
3 5 6
1 7 9
4 8 2


Output


9 5 1
7 8 2
3 6 4


Input


2 2
4 1
3 2


Output


4 1
3 2


Input


3 4
12 10 8 6
3 4 5 7
2 11 9 1


Output


12 8 6 1
10 11 9 2
3 4 5 7

Note

Let's analyze the first sample:

For matrix A we have:

    * Rows: 
      * R_1(A) = [3, 5, 6]; max(R_1(A)) = 6 
      * R_2(A) = [1, 7, 9]; max(R_2(A)) = 9 
      * R_3(A) = [4, 8, 2]; max(R_3(A)) = 8 

    * Columns: 
      * C_1(A) = [3, 1, 4]; max(C_1(A)) = 4 
      * C_2(A) = [5, 7, 8]; max(C_2(A)) = 8 
      * C_3(A) = [6, 9, 2]; max(C_3(A)) = 9 

  * X = \{ max(R_1(A)), max(R_2(A)), max(R_3(A)) \} = \{ 6, 9, 8 \} 
  * Y = \{ max(C_1(A)), max(C_2(A)), max(C_3(A)) \} = \{ 4, 8, 9 \} 
  * So S(A) = (X, Y) = (\{ 6, 9, 8 \}, \{ 4, 8, 9 \}) 



For matrix A' we have:

    * Rows: 
      * R_1(A') = [9, 5, 1]; max(R_1(A')) = 9 
      * R_2(A') = [7, 8, 2]; max(R_2(A')) = 8 
      * R_3(A') = [3, 6, 4]; max(R_3(A')) = 6 

    * Columns: 
      * C_1(A') = [9, 7, 3]; max(C_1(A')) = 9 
      * C_2(A') = [5, 8, 6]; max(C_2(A')) = 8 
      * C_3(A') = [1, 2, 4]; max(C_3(A')) = 4 

  * Note that each of this arrays are bitonic. 
  * X = \{ max(R_1(A')), max(R_2(A')), max(R_3(A')) \} = \{ 9, 8, 6 \} 
  * Y = \{ max(C_1(A')), max(C_2(A')), max(C_3(A')) \} = \{ 9, 8, 4 \} 
  * So S(A') = (X, Y) = (\{ 9, 8, 6 \}, \{ 9, 8, 4 \}) 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
inline int read() {
  int x = 0, f = 1;
  char c = getchar();
  while (!isdigit(c)) {
    if (c == '-') f = -1;
    c = getchar();
  }
  while (isdigit(c)) {
    x = (x << 1) + (x << 3) + (c ^ 48);
    c = getchar();
  }
  return x * f;
}
inline void print(int x) {
  if (x < 0) x = -x, putchar('-');
  if (x >= 10) print(x / 10);
  putchar(x % 10 + 48);
}
int n, m;
int a[251][251];
int ans[251][251];
queue<pair<int, int> > q;
bool c[100010], r[100010];
int main() {
  n = read(), m = read();
  for (int i = 1; i <= n; ++i)
    for (int j = 1; j <= m; ++j) a[i][j] = read();
  for (int i = 1; i <= n; ++i) {
    int x = -1;
    for (int j = 1; j <= m; ++j) {
      x = max(x, a[i][j]);
    }
    c[x] = 1;
  }
  for (int i = 1; i <= m; ++i) {
    int x = -1;
    for (int j = 1; j <= n; ++j) x = max(x, a[j][i]);
    r[x] = 1;
  }
  int x = 0, y = 0;
  for (int i = n * m; i; --i) {
    x += c[i], y += r[i];
    if (r[i] || c[i]) {
      ans[x][y] = i;
      if (r[i]) {
        for (int j = x - 1; j; --j) q.push(make_pair(j, y));
      }
      if (c[i]) {
        for (int j = y - 1; j; --j) q.push(make_pair(x, j));
      }
    } else {
      pair<int, int> res = q.front();
      q.pop();
      ans[res.first][res.second] = i;
    }
  }
  for (int i = 1; i <= n; ++i) {
    for (int j = 1; j <= m; ++j) {
      print(ans[i][j]);
      putchar(' ');
    }
    putchar('\n');
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re
from collections import deque

class Drearrangebootcamp(Basebootcamp):
    def __init__(self, n=3, m=3):
        self.n = n
        self.m = m

    def case_generator(self):
        """
        生成保证存在解的原矩阵案例
        """
        while True:
            # 生成随机矩阵并确保存在解
            n, m = self.n, self.m
            total = n * m
            elements = list(range(1, total + 1))
            random.shuffle(elements)
            original = [elements[i*m:(i+1)*m] for i in range(n)]
            
            # 尝试生成解矩阵
            solution = self._generate_solution(original)
            if solution is not None:
                # 提取原矩阵的X和Y
                X = list({max(row) for row in original})
                Y = list({max(col) for col in zip(*original)})
                return {
                    'n': n, 
                    'm': m, 
                    'matrix': original, 
                    'X': X,
                    'Y': Y,
                    '_solution': solution  # 内部保存用于验证
                }

    @staticmethod
    def _generate_solution(original):
        """参考代码算法实现，返回解矩阵或None"""
        n, m = len(original), len(original[0])
        c = dict()  # 行最大值标记
        r = dict()  # 列最大值标记
        
        # 计算原矩阵的行和列最大值
        for i in range(n):
            max_row = max(original[i])
            c[max_row] = True
        for j in range(m):
            max_col = max(original[i][j] for i in range(n))
            r[max_col] = True
        
        ans = [[0]*m for _ in range(n)]
        q = deque()
        x = 0
        y = 0
        
        for num in range(n*m, 0, -1):
            is_row_max = c.get(num, False)
            is_col_max = r.get(num, False)
            x += is_row_max
            y += is_col_max
            
            if is_row_max or is_col_max:
                ans_x = x - 1
                ans_y = y - 1
                ans[ans_x][ans_y] = num
                # 填充队列
                if is_row_max:
                    for j in range(ans_y-1, -1, -1):
                        q.append( (ans_x, j) )
                if is_col_max:
                    for i in range(ans_x-1, -1, -1):
                        q.append( (i, ans_y) )
            else:
                if not q:
                    return None  # 无解
                i, j = q.popleft()
                ans[i][j] = num
        
        # 验证生成的解矩阵
        if Drearrangebootcamp._validate_solution(ans, original):
            return ans
        return None

    @classmethod
    def _validate_solution(cls, solution, original):
        """验证解矩阵是否满足所有条件"""
        # 元素唯一性
        flat = [num for row in solution for num in row]
        if len(set(flat)) != len(flat) or set(flat) != set(range(1, len(flat)+1)):
            return False
        
        # Bitonic验证
        for row in solution:
            if not cls.is_bitonic(row):
                return False
        for col in zip(*solution):
            if not cls.is_bitonic(col):
                return False
        
        # 谱集验证
        X_sol = {max(row) for row in solution}
        Y_sol = {max(col) for col in zip(*solution)}
        X_ori = {max(row) for row in original}
        Y_ori = {max(col) for col in zip(*original)}
        return X_sol == X_ori and Y_sol == Y_ori

    @staticmethod
    def is_bitonic(arr):
        if len(arr) <= 1:
            return True
        peak = arr.index(max(arr))
        # 递增部分
        for i in range(1, peak+1):
            if arr[i] <= arr[i-1]:
                return False
        # 递减部分
        for i in range(peak, len(arr)-1):
            if arr[i] <= arr[i+1]:
                return False
        return True

    @staticmethod
    def prompt_func(question_case) -> str:
        matrix = question_case['matrix']
        n, m = question_case['n'], question_case['m']
        matrix_str = '\n'.join(' '.join(map(str, row)) for row in matrix)
        return f"""Koa the Koala has a {n}x{m} matrix with distinct numbers 1-{n*m}. Find a matrix A' where:

1. S(A') = S(A) (same row/column max sets)
2. All rows/columns are bitonic
3. Output format: n lines with m numbers, or -1

Input:
{n} {m}
{matrix_str}

Output your answer within [answer] and [/answer] tags."""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution == '-1':
            return False  # 原案例确保存在解
        
        try:
            # 转换并验证解矩阵格式
            matrix = []
            for line in solution.split('\n'):
                if not line.strip():
                    continue
                row = list(map(int, line.strip().split()))
                matrix.append(row)
            
            return cls._validate_solution(matrix, identity['matrix'])
        except:
            return False
