"""# 

### 谜题描述
A mad scientist Dr.Jubal has made a competitive programming task. Try to solve it!

You are given integers n,k. Construct a grid A with size n × n consisting of integers 0 and 1. The very important condition should be satisfied: the sum of all elements in the grid is exactly k. In other words, the number of 1 in the grid is equal to k.

Let's define:

  * A_{i,j} as the integer in the i-th row and the j-th column. 
  * R_i = A_{i,1}+A_{i,2}+...+A_{i,n} (for all 1 ≤ i ≤ n). 
  * C_j = A_{1,j}+A_{2,j}+...+A_{n,j} (for all 1 ≤ j ≤ n). 
  * In other words, R_i are row sums and C_j are column sums of the grid A. 
  * For the grid A let's define the value f(A) = (max(R)-min(R))^2 + (max(C)-min(C))^2 (here for an integer sequence X we define max(X) as the maximum value in X and min(X) as the minimum value in X). 



Find any grid A, which satisfies the following condition. Among such grids find any, for which the value f(A) is the minimum possible. Among such tables, you can find any.

Input

The input consists of multiple test cases. The first line contains a single integer t (1 ≤ t ≤ 100) — the number of test cases. Next t lines contain descriptions of test cases.

For each test case the only line contains two integers n, k (1 ≤ n ≤ 300, 0 ≤ k ≤ n^2).

It is guaranteed that the sum of n^2 for all test cases does not exceed 10^5.

Output

For each test case, firstly print the minimum possible value of f(A) among all tables, for which the condition is satisfied.

After that, print n lines contain n characters each. The j-th character in the i-th line should be equal to A_{i,j}.

If there are multiple answers you can print any.

Example

Input


4
2 2
3 8
1 0
4 16


Output


0
10
01
2
111
111
101
0
0
0
1111
1111
1111
1111

Note

In the first test case, the sum of all elements in the grid is equal to 2, so the condition is satisfied. R_1 = 1, R_2 = 1 and C_1 = 1, C_2 = 1. Then, f(A) = (1-1)^2 + (1-1)^2 = 0, which is the minimum possible value of f(A).

In the second test case, the sum of all elements in the grid is equal to 8, so the condition is satisfied. R_1 = 3, R_2 = 3, R_3 = 2 and C_1 = 3, C_2 = 2, C_3 = 3. Then, f(A) = (3-2)^2 + (3-2)^2 = 2. It can be proven, that it is the minimum possible value of f(A).

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/usr/bin/env python
from __future__ import division, print_function

import os
import sys
from io import BytesIO, IOBase

if sys.version_info[0] < 3:
    from __builtin__ import xrange as range
    from future_builtins import ascii, filter, hex, map, oct, zip


def main():
    t = int(input())

    for _ in range(t):
        n, k = map(int, input().split())

        A = [[0] * n for _ in range(n)]

        m = k // n

        idx = list(range(m))
        for i in range(n):
            for j in idx:
                A[i][(i + j) % n] = 1

        for i in range(k % n):
            A[i][(n - 1 + i) % n] = 1

        if k % n == 0:
            print(0)
        else:
            print(2)

        for row in A:
            print(*row, sep='')


# region fastio

BUFSIZE = 8192


class FastIO(IOBase):
    newlines = 0

    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = \"x\" in file.mode or \"r\" not in file.mode
        self.write = self.buffer.write if self.writable else None

    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()

    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b\"\n\") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()

    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)


class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode(\"ascii\"))
        self.read = lambda: self.buffer.read().decode(\"ascii\")
        self.readline = lambda: self.buffer.readline().decode(\"ascii\")


def print(*args, **kwargs):
    \"\"\"Prints the values to a stream, or to sys.stdout by default.\"\"\"
    sep, file = kwargs.pop(\"sep\", \" \"), kwargs.pop(\"file\", sys.stdout)
    at_start = True
    for x in args:
        if not at_start:
            file.write(sep)
        file.write(str(x))
        at_start = False
    file.write(kwargs.pop(\"end\", \"\n\"))
    if kwargs.pop(\"flush\", False):
        file.flush()


if sys.version_info[0] < 3:
    sys.stdin, sys.stdout = FastIO(sys.stdin), FastIO(sys.stdout)
else:
    sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)

input = lambda: sys.stdin.readline().rstrip(\"\r\n\")

# endregion

if __name__ == \"__main__\":
    main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re
from math import isclose

class Dgrid00100bootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=300, fixed_n=None, fixed_k=None, **params):
        """
        增强参数有效性校验的初始化方法
        """
        # 参数校验逻辑
        if fixed_n is not None:
            if not isinstance(fixed_n, int) or fixed_n < 1 or fixed_n > 300:
                raise ValueError("fixed_n must be integer between 1 and 300")
            if fixed_k is not None:
                max_k = fixed_n ** 2
                if not (0 <= fixed_k <= max_k):
                    raise ValueError(f"fixed_k must be between 0 and {max_k} when fixed_n={fixed_n}")
        
        self.n_min = max(1, n_min)
        self.n_max = min(300, n_max)
        self.fixed_n = fixed_n
        self.fixed_k = fixed_k
        self.params = params

    def case_generator(self):
        """生成完全遵循题目约束的测试用例"""
        # 处理固定参数模式
        if self.fixed_n is not None:
            n = self.fixed_n
            max_k = n ** 2
            
            if self.fixed_k is not None:
                # 二次校验参数合法性
                if not (0 <= self.fixed_k <= max_k):
                    raise ValueError(f"Invalid fixed_k={self.fixed_k} for n={n}")
                return {
                    'n': n,
                    'k': self.fixed_k,
                    'expected_f_min': self._calc_expected_f(n, self.fixed_k)
                }
        
        # 动态生成模式
        n = self.fixed_n if self.fixed_n else random.randint(self.n_min, self.n_max)
        max_k = n ** 2
        
        # k生成策略
        if self.fixed_k is not None:
            k = self.fixed_k
        else:
            # 确保生成有效k值的四种情况
            case_types = [
                ('zero', 0.1), 
                ('full', 0.1),
                ('divisible', 0.4),
                ('remainder', 0.4)
            ]
            case_type = random.choices(
                [t[0] for t in case_types],
                weights=[t[1] for t in case_types]
            )[0]
            
            if case_type == 'zero':
                k = 0
            elif case_type == 'full':
                k = max_k
            elif case_type == 'divisible':
                max_m = n if max_k >= n else 0
                m = random.randint(0, max_m)
                k = m * n
                # 确保不超出范围
                k = min(max(k, 0), max_k)
            else:  # remainder
                while True:
                    k = random.randint(1, max_k-1)
                    if k % n != 0 or n == 1:  # 处理n=1的情况
                        break
        
        # 最终合法性校验
        k = max(0, min(k, max_k))
        return {
            'n': n,
            'k': k,
            'expected_f_min': self._calc_expected_f(n, k)
        }

    @staticmethod
    def _calc_expected_f(n, k):
        """精确计算理论最小f值"""
        if k in (0, n*n):
            return 0
        if k % n == 0:
            return 0
        return 2

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        k = question_case['k']
        return f"""Dr. Jubal's Grid Optimization Challenge

Problem Statement:
- Grid dimension: {n}×{n}
- Required number of 1s: {k}

Objective:
Construct a binary grid satisfying:
1. Exactly {k} cells contain 1
2. Minimize f(A) = (max_row_diff)² + (max_col_diff)²
   - row_diff = (maximum row sum) - (minimum row sum)
   - col_diff = (maximum column sum) - (minimum column sum)

Output Format Requirements:
[answer]
<minimal_f_value>
<row_1_representation>
<row_2_representation>
...
[/answer]

Example Valid Output:
[answer]
2
110
101
011
[/answer]"""

    @staticmethod
    def extract_output(output):
        """增强版答案提取，支持多种格式变体"""
        # 预处理：移除多余空白字符
        cleaned = re.sub(r'\s+', ' ', output).lower()
        
        # 寻找最后一个答案块
        answer_blocks = re.findall(
            r'\[answer\](.*?)\[/answer\]', 
            cleaned,
            re.DOTALL
        )
        
        if not answer_blocks:
            return None
        
        content = answer_blocks[-1].strip()
        if not content:
            return None
        
        # 解析内容
        elements = [e.strip() for e in content.split() if e.strip()]
        if len(elements) < 1:
            return None
        
        try:
            f_value = int(elements[0])
            grid = elements[1:]
            return {'f_value': f_value, 'grid': grid}
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """增强验证逻辑"""
        # 基础校验
        if not solution or 'f_value' not in solution or 'grid' not in solution:
            return False
        
        try:
            n = identity['n']
            k = identity['k']
            expected_f = identity['expected_f_min']
            grid = solution['grid']
            claimed_f = solution['f_value']
            
            # 校验维度
            if len(grid) != n:
                return False
            
            # 校验每行格式
            valid_chars = {'0', '1'}
            total_ones = 0
            for row in grid:
                if len(row) != n or not set(row) <= valid_chars:
                    return False
                total_ones += sum(int(c) for c in row)
                
            # 校验总数
            if total_ones != k:
                return False
            
            # 计算行列和
            R = [sum(int(c) for c in row) for row in grid]
            C = [sum(int(row[j]) for row in grid) for j in range(n)]
            
            # 计算实际f值
            actual_f = (max(R)-min(R))**2 + (max(C)-min(C))**2
            
            # 双重校验：声明值与实际值都要匹配
            return claimed_f == expected_f and isclose(actual_f, expected_f, abs_tol=1e-9)
        except Exception as e:
            return False
