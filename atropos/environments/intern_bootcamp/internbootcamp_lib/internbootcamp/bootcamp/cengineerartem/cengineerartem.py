"""# 

### 谜题描述
Artem is building a new robot. He has a matrix a consisting of n rows and m columns. The cell located on the i-th row from the top and the j-th column from the left has a value a_{i,j} written in it. 

If two adjacent cells contain the same value, the robot will break. A matrix is called good if no two adjacent cells contain the same value, where two cells are called adjacent if they share a side. 

Artem wants to increment the values in some cells by one to make a good.

More formally, find a good matrix b that satisfies the following condition — 

  * For all valid (i,j), either b_{i,j} = a_{i,j} or b_{i,j} = a_{i,j}+1. 



For the constraints of this problem, it can be shown that such a matrix b always exists. If there are several such tables, you can output any of them. Please note that you do not have to minimize the number of increments.

Input

Each test contains multiple test cases. The first line contains the number of test cases t (1 ≤ t ≤ 10). Description of the test cases follows.

The first line of each test case contains two integers n, m (1 ≤ n ≤ 100, 1 ≤ m ≤ 100) — the number of rows and columns, respectively.

The following n lines each contain m integers. The j-th integer in the i-th line is a_{i,j} (1 ≤ a_{i,j} ≤ 10^9).

Output

For each case, output n lines each containing m integers. The j-th integer in the i-th line is b_{i,j}.

Example

Input


3
3 2
1 2
4 5
7 8
2 2
1 1
3 3
2 2
1 3
2 2


Output


1 2
5 6
7 8
2 1
4 3
2 4
3 2

Note

In all the cases, you can verify that no two adjacent cells have the same value and that b is the same as a with some values incremented by one. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin
t = int(stdin.readline())
for _ in xrange(t):
    n,m = map(int,stdin.readline().split())
    for i in xrange(n):
        a = map(int,stdin.readline().split())
        for j in xrange(m):
            x = a[j]
            if (i+j)%2 != x%2:
                x+=1
            print x,
        print \"\"
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Cengineerartembootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=100, min_m=1, max_m=100, seed=None):
        super().__init__()
        self.min_n = min_n
        self.max_n = max_n
        self.min_m = min_m
        self.max_m = max_m
        self.rng = random.Random(seed)
    
    def case_generator(self):
        n = self.rng.randint(self.min_n, self.max_n)
        m = self.rng.randint(self.min_m, self.max_m)
        
        # 确保生成的b矩阵满足奇偶棋盘模式
        b = []
        for i in range(n):
            row = []
            for j in range(m):
                base_value = self.rng.randint(1, 10**3)  # 缩小数值范围便于测试
                parity = (i + j) % 2
                if (base_value % 2) != parity:
                    base_value += 1
                row.append(base_value)
            b.append(row)
        
        # 生成对应的a矩阵（确保值不低于1）
        a = []
        for i in range(n):
            row_a = []
            for j in range(m):
                if self.rng.choice([True, False]) and b[i][j] > 1:
                    row_a.append(b[i][j] - 1)
                else:
                    row_a.append(b[i][j])
            a.append(row_a)
        
        return {
            'n': n,
            'm': m,
            'a': a,
            'expected_parity_pattern': [[(i+j)%2 for j in range(m)] for i in range(n)]  # 添加校验模式
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        matrix_str = '\n'.join(' '.join(map(str, row)) for row in question_case['a'])
        return f"""Artem需要调整一个矩阵使其相邻单元格不重复。每个单元格可以选择保持原值或+1。

当前矩阵（{question_case['n']}行×{question_case['m']}列）：
{matrix_str}

请输出修改后的矩阵，确保：
1. 相邻单元格（上下左右）的值不同
2. 每个单元格只能是原值或原值+1
3. 将最终答案用[answer]标签包裹，例如：
[answer]
1 2
3 4
[/answer]"""

    @staticmethod
    def extract_output(output):
        # 支持多答案块和大小写标签
        pattern = re.compile(r'\[answer\](.*?)\[/?answer\]', re.DOTALL | re.IGNORECASE)
        matches = pattern.findall(output)
        if not matches:
            return None
        
        # 处理最后一个答案块
        last_answer = matches[-1].strip()
        matrix = []
        for line in last_answer.split('\n'):
            line = line.strip()
            if line:
                try:
                    matrix.append( list(map(int, line.split())) )
                except ValueError:
                    continue
        return matrix if matrix else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 维度校验
        if len(solution) != identity['n'] or any(len(row)!=identity['m'] for row in solution):
            return False
        
        a = identity['a']
        # 值域校验
        for i in range(identity['n']):
            for j in range(identity['m']):
                if solution[i][j] not in {a[i][j], a[i][j]+1}:
                    return False
        
        # 相邻校验
        directions = [(-1,0),(1,0),(0,-1),(0,1)]
        for i in range(identity['n']):
            for j in range(identity['m']):
                for dx, dy in directions:
                    x, y = i+dx, j+dy
                    if 0 <= x < identity['n'] and 0 <= y < identity['m']:
                        if solution[i][j] == solution[x][y]:
                            return False
        return True
