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
size = int(raw_input())
final_parity  = 0
output = ''
for i in xrange(size):
  final_parity ^= int(raw_input()[2*i])
num_queries = int(raw_input())
while num_queries > 0:
  query_line = raw_input()
  if query_line[0]=='3':
    output+=chr(final_parity+48)
  else:
    final_parity ^= 1
  num_queries -= 1
print output
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cunusualproductbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.pop('n', 3)
        self.q = params.pop('q', 5)
        self.random_seed = params.pop('random_seed', None)
        super().__init__(**params)
    
    def case_generator(self):
        if self.random_seed is not None:
            random.seed(self.random_seed)
        
        n = self.n
        q = self.q
        
        # 生成全随机矩阵
        matrix = []
        initial_S = 0
        for i in range(n):
            row = [random.choice([0, 1]) for _ in range(n)]
            matrix.append(row)
            initial_S ^= row[i]  # 对角线元素异或
        
        queries = []
        type3_count = 0
        for _ in range(q):
            if type3_count == 0 and len(queries) == q - 1:
                query_type = 3
            else:
                query_type = random.choices([1, 2, 3], weights=[2, 2, 1], k=1)[0]
            
            if query_type == 3:
                queries.append({'type': 3})
                type3_count += 1
            else:
                i = random.randint(1, n)
                queries.append({'type': query_type, 'i': i})
        
        return {
            'n': n,
            'matrix': matrix,
            'queries': queries,
            'flip_count': 0  # 初始翻转次数
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        prompt = (
            "Little Chris needs to process matrix operations in GF(2). The unusual square is the XOR sum of diagonal elements.\n\n"
            f"Matrix Size: {question_case['n']}x{question_case['n']}\n"
            "Matrix Values:\n"
        )
        for row in question_case['matrix']:
            prompt += ' '.join(map(str, row)) + '\n'
        prompt += f"\nQueries ({len(question_case['queries'])}):\n"
        for q in question_case['queries']:
            if q['type'] == 3:
                prompt += "3\n"
            else:
                prompt += f"{q['type']} {q['i']}\n"
        prompt += (
            "\nOutput the binary results of all type 3 queries as a continuous string.\n"
            "Enclose your answer with [answer] and [/answer] tags.\n"
            "Example: [answer]0110[/answer]"
        )
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 正确计算初始对角线异或和
        initial_S = sum(row[i] for i, row in enumerate(identity['matrix'])) % 2
        flip_count = 0
        
        expected = []
        for query in identity['queries']:
            if query['type'] == 3:
                expected.append(str((initial_S + flip_count) % 2))
            else:
                flip_count += 1
        return solution == ''.join(expected)
