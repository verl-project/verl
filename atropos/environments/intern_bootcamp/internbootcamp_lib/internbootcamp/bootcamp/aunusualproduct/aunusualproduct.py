"""# 

### 谜题描述
Little Chris is a huge fan of linear algebra. This time he has been given a homework about the unusual square of a square matrix.

The dot product of two integer number vectors x and y of size n is the sum of the products of the corresponding components of the vectors. The unusual square of an n × n square matrix A is defined as the sum of n dot products. The i-th of them is the dot product of the i-th row vector and the i-th column vector in the matrix A.

Fortunately for Chris, he has to work only in GF(2)! This means that all operations (addition, multiplication) are calculated modulo 2. In fact, the matrix A is binary: each element of A is either 0 or 1. For example, consider the following matrix A:

<image>

The unusual square of A is equal to (1·1 + 1·0 + 1·1) + (0·1 + 1·1 + 1·0) + (1·1 + 0·1 + 0·0) = 0 + 1 + 1 = 0.

However, there is much more to the homework. Chris has to process q queries; each query can be one of the following: 

  1. given a row index i, flip all the values in the i-th row in A; 
  2. given a column index i, flip all the values in the i-th column in A; 
  3. find the unusual square of A. 



To flip a bit value w means to change it to 1 - w, i.e., 1 changes to 0 and 0 changes to 1.

Given the initial matrix A, output the answers for each query of the third type! Can you solve Chris's homework?

Input

The first line of input contains an integer n (1 ≤ n ≤ 1000), the number of rows and the number of columns in the matrix A. The next n lines describe the matrix: the i-th line contains n space-separated bits and describes the i-th row of A. The j-th number of the i-th line aij (0 ≤ aij ≤ 1) is the element on the intersection of the i-th row and the j-th column of A.

The next line of input contains an integer q (1 ≤ q ≤ 106), the number of queries. Each of the next q lines describes a single query, which can be one of the following: 

  * 1 i — flip the values of the i-th row; 
  * 2 i — flip the values of the i-th column; 
  * 3 — output the unusual square of A. 



Note: since the size of the input and output could be very large, don't use slow output techniques in your language. For example, do not use input and output streams (cin, cout) in C++.

Output

Let the number of the 3rd type queries in the input be m. Output a single string s of length m, where the i-th symbol of s is the value of the unusual square of A for the i-th query of the 3rd type as it appears in the input.

Examples

Input

3
1 1 1
0 1 1
1 0 0
12
3
2 3
3
2 2
2 2
1 3
3
3
1 2
2 1
1 1
3


Output

01001

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
from os import read

unusual = False
n = input()
for i in xrange(0,n*2,2):
    unusual ^=  int(raw_input()[i])
q = input()
for _ in xrange(q):
    inp = raw_input()
    if inp=='3':
        sys.stdout.write('1' if unusual else '0')
    else:
        unusual ^= True
print
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp
import re
from copy import deepcopy

class Aunusualproductbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = max(1, params.get('n', 3))
        self.q = max(1, params.get('q', 5))
    
    def case_generator(self):
        n = self.n
        initial_u = random.choice([0, 1])
        diagonal = []
        sum_d = 0
        
        # 生成对角线元素
        if n == 1:
            diagonal = [initial_u]
        else:
            for _ in range(n - 1):
                bit = random.randint(0, 1)
                diagonal.append(bit)
                sum_d += bit
            sum_d %= 2
            last_bit = (initial_u - sum_d) % 2
            diagonal.append(last_bit)
        
        # 生成完整矩阵
        matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(diagonal[i])
                else:
                    row.append(random.randint(0, 1))
            matrix.append(row)
        
        # 生成查询序列
        queries = []
        expected_output = []
        current_u = initial_u
        for _ in range(self.q):
            op_type = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]
            if op_type == 3:
                queries.append('3')
                expected_output.append(str(current_u))
            else:
                i = random.randint(1, n)
                queries.append(f"{op_type} {i}")
                current_u ^= 1  # 严格符合题目中的位运算规律
        
        return {
            'n': n,
            'matrix': matrix,
            'queries': queries,
            'expected_output': ''.join(expected_output)
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        input_lines = [
            str(question_case['n']),
            *[' '.join(map(str, row)) for row in question_case['matrix']],
            str(len(question_case['queries'])),
            *question_case['queries']
        ]
        return (
            "Solve the matrix query problem in GF(2). Each flip operation (row/column) toggles bits. "
            f"Input:\n" + '\n'.join(input_lines) + 
            "\nOutput all '3' query results as a continuous string. Enclose answer in [answer][/answer]."
        )
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*([01]+)\s*\[/answer\]', output, re.IGNORECASE)
        return matches[-1] if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        """双重验证：1.直接比较预期输出 2. 动态重新计算"""
        # 直接比较预期输出
        if solution == identity['expected_output']:
            return True
        
        # 动态重新计算
        n = identity['n']
        matrix = deepcopy(identity['matrix'])
        queries = identity['queries']
        
        # 初始化unusual值
        unusual = sum(matrix[i][i] for i in range(n)) % 2
        expected = []
        
        for q in queries:
            if q == '3':
                expected.append(str(unusual))
            else:
                op, i = q.split()
                i = int(i) - 1  # 转换为0-based索引
                if op == '1':
                    for j in range(n):
                        matrix[i][j] ^= 1
                        if i == j:
                            unusual ^= 1
                elif op == '2':
                    for j in range(n):
                        matrix[j][i] ^= 1
                        if j == i:
                            unusual ^= 1
        return solution == ''.join(expected)
