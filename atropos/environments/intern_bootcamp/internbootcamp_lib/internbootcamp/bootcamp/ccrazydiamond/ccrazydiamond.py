"""# 

### 谜题描述
You are given a permutation p of integers from 1 to n, where n is an even number. 

Your goal is to sort the permutation. To do so, you can perform zero or more operations of the following type: 

  * take two indices i and j such that 2 ⋅ |i - j| ≥ n and swap p_i and p_j. 



There is no need to minimize the number of operations, however you should use no more than 5 ⋅ n operations. One can show that it is always possible to do that.

Input

The first line contains a single integer n (2 ≤ n ≤ 3 ⋅ 10^5, n is even) — the length of the permutation. 

The second line contains n distinct integers p_1, p_2, …, p_n (1 ≤ p_i ≤ n) — the given permutation.

Output

On the first line print m (0 ≤ m ≤ 5 ⋅ n) — the number of swaps to perform.

Each of the following m lines should contain integers a_i, b_i (1 ≤ a_i, b_i ≤ n, |a_i - b_i| ≥ n/2) — the indices that should be swapped in the corresponding swap.

Note that there is no need to minimize the number of operations. We can show that an answer always exists.

Examples

Input


2
2 1


Output


1
1 2

Input


4
3 4 1 2


Output


4
1 4
1 4
1 3
2 4


Input


6
2 5 3 1 4 6


Output


3
1 5
2 5
1 4

Note

In the first example, when one swap elements on positions 1 and 2, the array becomes sorted.

In the second example, pay attention that there is no need to minimize number of swaps.

In the third example, after swapping elements on positions 1 and 5 the array becomes: [4, 5, 3, 1, 2, 6]. After swapping elements on positions 2 and 5 the array becomes [4, 2, 3, 1, 5, 6] and finally after swapping elements on positions 1 and 4 the array becomes sorted: [1, 2, 3, 4, 5, 6].

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def main():
    inp = readnumbers()
    ii = 0

    n = inp[ii]
    ii += 1

    P = [p-1 for p in inp[ii:ii+n]]
    ii += n

    where = [0]*n
    for i in range(n):
        where[P[i]] = i
    
    Z = []
    def swapper(ind1,ind2):
        if ind1>ind2:
            ind1,ind2 = ind2,ind1
        
        swaps = []
        
        left = 0
        right = n-1

        if ind1!=left and 2*abs(ind2-left)>=n:
            swaps.append((ind2,left))
            ind2 = left

        if ind2!=right and 2*abs(ind1-right)>=n:
            swaps.append((ind1,right))
            ind1 = right

        if ind1!=left and 2*abs(ind2-left)>=n:
            swaps.append((ind2,left))
            ind2 = left

        assert 2*abs(ind1-ind2)>=n
        swaps.append((ind1,ind2))
        swaps += swaps[:-1][::-1]
        
        assert len(swaps)&1

        for l,r in swaps:
            where[P[r]],where[P[l]],P[l],P[r] = l,r,P[r],P[l]
            Z.append((l,r))

    for p in reversed(range(n)):
        indx = where[p]
        indy = p
        if indx==indy:
            continue
        swapper(indx,indy)
    print len(Z)
    print '\n'.join(str(x+1)+' '+str(y+1) for x,y in Z)


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

class Ccrazydiamondbootcamp(Basebootcamp):
    def __init__(self, n=4):
        if n % 2 != 0:
            raise ValueError("n must be an even integer.")
        self.n = n
    
    def case_generator(self):
        n = self.n
        p = list(range(1, n + 1))
        random.shuffle(p)
        return {'n': n, 'p': p.copy()}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        p = question_case['p']
        problem = f"""You are given a permutation of integers from 1 to {n} (n is even). Your goal is to sort it using swaps between indices i and j where 2*|i-j| ≥ {n}. 

Problem Instance:
- n = {n}
- Permutation: {p}

Rules:
1. Each swap must satisfy 2*|i-j| ≥ {n} (i.e., |i-j| ≥ {n//2}).
2. Use at most {5*n} swaps.
3. The sorted permutation must be [1, 2, ..., {n}].

Output Format:
- First line: m (number of swaps)
- Next m lines: pairs of indices (a_i, b_i), each satisfying the swap rule.

Put your answer within [answer] and [/answer] tags. Example:

[answer]
3
1 5
2 5
1 4
[/answer]"""
        return problem
    
    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        last_answer = answer_blocks[-1].strip()
        lines = [line.strip() for line in last_answer.split('\n') if line.strip()]
        
        if not lines:
            return None
        
        # Parse m from first line
        m_line = lines[0]
        if not re.fullmatch(r'\s*\d+\s*', m_line):
            return None
        try:
            m = int(m_line)
        except ValueError:
            return None
        
        if m < 0:
            return None
        
        # Validate swap lines count
        if len(lines) < 1 + m:
            return None
        swap_lines = lines[1:1+m]
        
        # Parse swaps
        swaps = []
        for line in swap_lines:
            if not re.fullmatch(r'\s*\d+\s+\d+\s*', line):
                return None
            a, b = map(int, line.split())
            swaps.append((a, b))
        
        return {'m': m, 'swaps': swaps}
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or 'm' not in solution or 'swaps' not in solution:
            return False
        m = solution['m']
        swaps = solution['swaps']
        n = identity['n']
        original_p = identity['p'].copy()
        
        # Check m constraints
        if m != len(swaps) or m > 5 * n or m < 0:
            return False
        
        # Apply swaps and validate
        current_p = original_p.copy()
        for a, b in swaps:
            if a < 1 or a > n or b < 1 or b > n:
                return False
            if abs(a - b) < n // 2:
                return False
            i, j = a-1, b-1
            current_p[i], current_p[j] = current_p[j], current_p[i]
        
        return current_p == list(range(1, n+1))
