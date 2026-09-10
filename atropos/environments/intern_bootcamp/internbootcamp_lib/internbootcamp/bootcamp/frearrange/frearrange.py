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
int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  int n, m;
  cin >> n >> m;
  vector<vector<int>> mat(n, vector<int>(m));
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j) cin >> mat[i][j];
  vector<int> h(n * m + 1);
  vector<int> v(n * m + 1);
  for (int i = 0; i < n; ++i) {
    int a = 0;
    for (int j = 0; j < m; ++j) a = max(a, mat[i][j]);
    h[a] = 1;
  }
  for (int i = 0; i < m; ++i) {
    int a = 0;
    for (int j = 0; j < n; ++j) a = max(a, mat[j][i]);
    v[a] = 1;
  }
  vector<vector<int>> fin(n, vector<int>(m));
  queue<pair<int, int>> q;
  int x = -1, y = -1;
  for (int u = n * m; u >= 1; --u) {
    x += h[u];
    y += v[u];
    if (h[u] || v[u]) {
      fin[x][y] = u;
    } else {
      int qx, qy;
      tie(qx, qy) = q.front();
      q.pop();
      fin[qx][qy] = u;
    }
    if (h[u])
      for (int i = y - 1; i >= 0; --i) q.push({x, i});
    if (v[u])
      for (int i = x - 1; i >= 0; --i) q.push({i, y});
  }
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j) cout << fin[i][j] << \" \n\"[j + 1 == m];
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import deque
from bootcamp import Basebootcamp

def generate_solution(n, m, mat):
    size = n * m
    h = [0] * (size + 2)
    v = [0] * (size + 2)
    
    # 标记行列最大值
    for row in mat:
        h[max(row)] = 1
    for j in range(m):
        max_col = max(mat[i][j] for i in range(n))
        v[max_col] = 1
    
    fin = [[0]*m for _ in range(n)]
    q = deque()
    x, y = -1, -1
    
    for u in range(size, 0, -1):
        dx = 1 if h[u] else 0
        dy = 1 if v[u] else 0
        x += dx
        y += dy
        
        if h[u] or v[u]:
            fin[x][y] = u
            # 填充队列（严格参考C++算法）
            if h[u]:
                for j in range(y-1, -1, -1):
                    q.append((x, j))
            if v[u]:
                for i in range(x-1, -1, -1):
                    q.append((i, y))
        else:
            if not q:
                return None
            i, j = q.popleft()
            fin[i][j] = u
            
    # 检查所有元素填充
    if any(0 in row for row in fin):
        return None
    return fin

def is_bitonic(arr):
    if len(arr) <= 1:
        return True
    peak = 0
    # 寻找递增阶段
    while peak < len(arr)-1 and arr[peak] < arr[peak+1]:
        peak += 1
    # 检查递减阶段
    for i in range(peak, len(arr)-1):
        if arr[i] <= arr[i+1]:
            return False
    return True

class Frearrangebootcamp(Basebootcamp):
    def __init__(self, n=3, m=3):
        self.n = n
        self.m = m
    
    def case_generator(self):
        # 生成包含有效解或无效解的随机案例
        elements = list(range(1, self.n*self.m +1))
        random.shuffle(elements)
        mat = [elements[i*self.m : (i+1)*self.m] for i in range(self.n)]
        solution = generate_solution(self.n, self.m, mat)
        return {
            'n': self.n,
            'm': self.m,
            'matrix': mat,
            'solution': solution
        }
    
    @staticmethod
    def prompt_func(question_case):
        matrix = '\n'.join(' '.join(map(str, row)) for row in question_case['matrix'])
        return f"""给定一个{question_case['n']}x{question_case['m']}矩阵，元素为1-{question_case['n']*question_case['m']}的排列。请构造满足以下条件的新矩阵：
1. 每行、每列的最大值集合与原矩阵相同
2. 每行、每列都是严格双调序列（先严格递增后严格递减）
3. 包含所有数字

若不存在输出-1。将答案放在[answer]和[/answer]之间。

输入矩阵：
{matrix}"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        content = matches[-1].strip()
        
        if content == '-1':
            return -1
        
        matrix = []
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                row = list(map(int, line.split()))
                if not row:
                    continue
                matrix.append(row)
            except:
                continue
        return matrix if matrix else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 无解验证
        if solution == -1:
            return identity['solution'] is None
        
        # 格式验证
        try:
            if len(solution) != identity['n'] or any(len(row)!=identity['m'] for row in solution):
                return False
            nums = set()
            for row in solution:
                for num in row:
                    nums.add(num)
            if nums != set(range(1, identity['n']*identity['m'] +1)):
                return False
        except:
            return False
        
        # S(A')验证
        orig_X = {max(r) for r in identity['matrix']}
        orig_Y = {max(identity['matrix'][i][j] for i in range(identity['n'])) for j in range(identity['m'])}
        sol_X = {max(r) for r in solution}
        sol_Y = {max(solution[i][j] for i in range(identity['n'])) for j in range(identity['m'])}
        if sol_X != orig_X or sol_Y != orig_Y:
            return False
        
        # 双调性验证
        for row in solution:
            if not is_bitonic(row):
                return False
        for col in zip(*solution):
            if not is_bitonic(col):
                return False
        
        return True
