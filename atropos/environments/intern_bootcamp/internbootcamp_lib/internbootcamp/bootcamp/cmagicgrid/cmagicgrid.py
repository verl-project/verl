"""# 

### 谜题描述
Let us define a magic grid to be a square matrix of integers of size n × n, satisfying the following conditions. 

  * All integers from 0 to (n^2 - 1) inclusive appear in the matrix exactly once. 
  * [Bitwise XOR](https://en.wikipedia.org/wiki/Bitwise_operation#XOR) of all elements in a row or a column must be the same for each row and column. 



You are given an integer n which is a multiple of 4. Construct a magic grid of size n × n.

Input

The only line of input contains an integer n (4 ≤ n ≤ 1000). It is guaranteed that n is a multiple of 4.

Output

Print a magic grid, i.e. n lines, the i-th of which contains n space-separated integers, representing the i-th row of the grid.

If there are multiple answers, print any. We can show that an answer always exists.

Examples

Input


4


Output


8 9 1 13
3 12 7 5
0 2 4 11
6 10 15 14

Input


8


Output


19 55 11 39 32 36 4 52
51 7 35 31 12 48 28 20
43 23 59 15 0 8 16 44
3 47 27 63 24 40 60 56
34 38 6 54 17 53 9 37
14 50 30 22 49 5 33 29
2 10 18 46 41 21 57 13
26 42 62 58 1 45 25 61

Note

In the first example, XOR of each row and each column is 13.

In the second example, XOR of each row and each column is 60.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division, print_function
def main():
    n=int(input())
    res=[[0]*n for _ in range(n)]
    x=0
    for k in range(n//4):
        for i in range(n):
            for j in range(4):
                res[i][j+k*4]=x
                x+=1
    for item in res:
        print(*item,sep=\" \")
    

######## Python 2 and 3 footer by Pajenegod and c1729

# Note because cf runs old PyPy3 version which doesn't have the sped up
# unicode strings, PyPy3 strings will many times be slower than pypy2.
# There is a way to get around this by using binary strings in PyPy3
# but its syntax is different which makes it kind of a mess to use.

# So on cf, use PyPy2 for best string performance.

py2 = round(0.5)
if py2:
    from future_builtins import ascii, filter, hex, map, oct, zip
    range = xrange

import os, sys
from io import IOBase, BytesIO

BUFSIZE = 8192
class FastIO(BytesIO):
    newlines = 0

    def __init__(self, file):
        self._file = file
        self._fd = file.fileno()
        self.writable = \"x\" in file.mode or \"w\" in file.mode
        self.write = super(FastIO, self).write if self.writable else None

    def _fill(self):
        s = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
        self.seek((self.tell(), self.seek(0,2), super(FastIO, self).write(s))[0])
        return s

    def read(self):
        while self._fill(): pass
        return super(FastIO,self).read()

    def readline(self):
        while self.newlines == 0:
            s = self._fill(); self.newlines = s.count(b\"\n\") + (not s)
        self.newlines -= 1
        return super(FastIO, self).readline()

    def flush(self):
        if self.writable:
            os.write(self._fd, self.getvalue())
            self.truncate(0), self.seek(0)

class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        if py2:
            self.write = self.buffer.write
            self.read = self.buffer.read
            self.readline = self.buffer.readline
        else:
            self.write = lambda s:self.buffer.write(s.encode('ascii'))
            self.read = lambda:self.buffer.read().decode('ascii')
            self.readline = lambda:self.buffer.readline().decode('ascii')


sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
input = lambda: sys.stdin.readline().rstrip('\r\n')

# Cout implemented in Python
import sys
class ostream:
    def __lshift__(self,a):
        sys.stdout.write(str(a))
        return self
cout = ostream()
endl = '\n'

# Read all remaining integers in stdin, type is given by optional argument, this is fast
def readnumbers(zero = 0):
    conv = ord if py2 else lambda x:x
    A = []; numb = zero; sign = 1; i = 0; s = sys.stdin.buffer.read()
    try:
        while True:
            if s[i] >= b'0' [0]:
                numb = 10 * numb + conv(s[i]) - 48
            elif s[i] == b'-' [0]: sign = -1
            elif s[i] != b'\r' [0]:
                A.append(sign*numb)
                numb = zero; sign = 1
            i += 1
    except:pass
    if s and s[-1] >= b'0' [0]:
        A.append(sign*numb)
    return A

if __name__== \"__main__\":
  main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Cmagicgridbootcamp(Basebootcamp):
    def __init__(self, **params):
        """
        支持两种模式：
        1. 固定n模式：直接指定n参数
        2. 随机模式：通过min_n/max_n指定范围（默认4-1000）
        """
        if 'n' in params:  # 固定n模式
            self.n = params['n']
            if not (self.n %4 ==0 and 4<=self.n<=1000):
                raise ValueError("n must be multiple of 4 (4 <= n <= 1000)")
            self.mode = 'fixed'
        else:  # 随机模式
            self.min_n = max(4, ((params.get('min_n',4)+3)//4)*4)
            self.max_n = min(1000, (params.get('max_n',1000)//4)*4)
            if self.min_n > self.max_n:
                raise ValueError("Invalid range for n generation")
            self.mode = 'random'
    
    def case_generator(self):
        """动态生成不同n值的案例"""
        if self.mode == 'fixed':
            n = self.n
        else:
            possible_n = list(range(self.min_n, self.max_n+1, 4))
            n = random.choice(possible_n)
        return {'n': n}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        prompt = f"""You need to construct a {n}x{n} magic grid where:
1. All integers 0-{n*n-1} appear exactly once
2. All rows have the same XOR
3. All columns have the same XOR as rows

Generate the grid with {n} rows of {n} space-separated integers. Put your answer between [answer] and [/answer] tags.

Example for n=4:
[answer]
8 9 1 13
3 12 7 5
0 2 4 11
6 10 15 14
[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        # 提取最后一个answer块并转换为二维列表
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        last_answer = answer_blocks[-1].strip()
        try:
            return [list(map(int, line.split())) for line in last_answer.split('\n') if line.strip()]
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        # 结构验证
        if len(solution)!=n or any(len(row)!=n for row in solution):
            return False
        # 数值范围验证
        flat = [num for row in solution for num in row]
        if set(flat) != set(range(n*n)):
            return False
        # 异或一致性验证
        row_xors = [0]*n
        col_xors = [0]*n
        for i in range(n):
            for j in range(n):
                row_xors[i] ^= solution[i][j]
                col_xors[j] ^= solution[i][j]
        return len(set(row_xors+col_xors)) == 1

# 使用示例（随机模式）
# bootcamp = Cmagicgridbootcamp(min_n=8, max_n=16)  # 生成8/12/16的案例
# print(bootcamp.case_generator())  # e.g. {'n': 12}
# print(bootcamp.case_generator())  # e.g. {'n': 8}

# # 使用示例（固定模式）
# bootcamp_fixed = Cmagicgridbootcamp(n=4)
# print(bootcamp_fixed.case_generator())  # 总是返回 {'n': 4}
