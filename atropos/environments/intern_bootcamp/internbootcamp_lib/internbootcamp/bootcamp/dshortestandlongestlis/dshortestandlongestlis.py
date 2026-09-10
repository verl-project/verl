"""# 

### 谜题描述
Gildong recently learned how to find the [longest increasing subsequence](https://en.wikipedia.org/wiki/Longest_increasing_subsequence) (LIS) in O(nlog{n}) time for a sequence of length n. He wants to test himself if he can implement it correctly, but he couldn't find any online judges that would do it (even though there are actually many of them). So instead he's going to make a quiz for you about making permutations of n distinct integers between 1 and n, inclusive, to test his code with your output.

The quiz is as follows.

Gildong provides a string of length n-1, consisting of characters '<' and '>' only. The i-th (1-indexed) character is the comparison result between the i-th element and the i+1-st element of the sequence. If the i-th character of the string is '<', then the i-th element of the sequence is less than the i+1-st element. If the i-th character of the string is '>', then the i-th element of the sequence is greater than the i+1-st element.

He wants you to find two possible sequences (not necessarily distinct) consisting of n distinct integers between 1 and n, inclusive, each satisfying the comparison results, where the length of the LIS of the first sequence is minimum possible, and the length of the LIS of the second sequence is maximum possible.

Input

Each test contains one or more test cases. The first line contains the number of test cases t (1 ≤ t ≤ 10^4).

Each test case contains exactly one line, consisting of an integer and a string consisting of characters '<' and '>' only. The integer is n (2 ≤ n ≤ 2 ⋅ 10^5), the length of the permutation you need to find. The string is the comparison results explained in the description. The length of the string is n-1.

It is guaranteed that the sum of all n in all test cases doesn't exceed 2 ⋅ 10^5.

Output

For each test case, print two lines with n integers each. The first line is the sequence with the minimum length of the LIS, and the second line is the sequence with the maximum length of the LIS. If there are multiple answers, print any one of them. Each sequence should contain all integers between 1 and n, inclusive, and should satisfy the comparison results.

It can be shown that at least one answer always exists.

Example

Input


3
3 &lt;&lt;
7 &gt;&gt;&lt;&gt;&gt;&lt;
5 &gt;&gt;&gt;&lt;


Output


1 2 3
1 2 3
5 4 3 7 2 1 6
4 3 1 7 5 2 6
4 3 2 1 5
5 4 2 1 3

Note

In the first case, 1 2 3 is the only possible answer.

In the second case, the shortest length of the LIS is 2, and the longest length of the LIS is 3. In the example of the maximum LIS sequence, 4 '3' 1 7 '5' 2 '6' can be one of the possible LIS.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys 
# sys.setrecursionlimit(10**6) 
from sys import stdin, stdout
import bisect            #c++ upperbound
import math
import heapq
def modinv(n,p):
    return pow(n,p-2,p)
def cin():
    return map(int,sin().split())
def ain():                           #takes array as input
    return list(map(int,sin().split()))
def sin():
    return input()
def inin():
    return int(input())
import math  
def Divisors(n) : 
    l = []  
    for i in range(1, int(math.sqrt(n) + 1)) :
        if (n % i == 0) : 
            if (n // i == i) : 
                l.append(i) 
            else : 
                l.append(i)
                l.append(n//i)
    return l

\"\"\"*******************************************************\"\"\"
def main():
    t=inin()
    for _ in range(t):
        s=sin()
        s=s.split()
        n=int(s[0])
        s=s[1]
        # print(n,s)
        b=[]
        a=[]
        for i in range(n,0,-1):
            b.append(i)
        i=0
        while(i<n-1):
            if(s[i]==\"<\"):
                x=0
                for j in range(0,n-i-1):
                    if(s[i+j]==\"<\"):
                        x+=1
                    else:
                        break
                c=[]
                for j in range(i,i+x+1):
                    c.append(b[j])
                c=c[::-1]
                # print(i,i+x)
                for j in range(i,i+x+1):
                    b[j]=c[j-i]
                i=i+x
            i+=1
        for i in range(1,n+1):
            a.append(i)
        i=0
        while(i<n-1):
            if(s[i]==\">\"):
                x=0
                for j in range(0,n-i-1):
                    if(s[i+j]==\">\"):
                        x+=1
                    else:
                        break
                c=[]
                for j in range(i,i+x+1):
                    c.append(a[j])
                c=c[::-1]
                # print(i,i+x)
                for j in range(i,i+x+1):
                    a[j]=c[j-i]
                i=i+x
            i+=1
        # print(*b)
        # print(*a)
        for i in b:
            print i,
        for i in a:
            print i,
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
import random
import bisect
from bootcamp import Basebootcamp

def compute_lis_length(sequence):
    tails = []
    for num in sequence:
        idx = bisect.bisect_left(tails, num)
        if idx == len(tails):
            tails.append(num)
        else:
            tails[idx] = num
    return len(tails)

def generate_min_sequence(n, s):
    b = list(range(n, 0, -1))
    i = 0
    while i < n-1:
        if s[i] == '<':
            x = 0
            j = 0
            while j < (n-1 - i) and s[i + j] == '<':
                x += 1
                j += 1
            sub = b[i:i+x+1][::-1]
            b[i:i+x+1] = sub
            i += x
        i += 1
    return b

def generate_max_sequence(n, s):
    a = list(range(1, n+1))
    i = 0
    while i < n-1:
        if s[i] == '>':
            x = 0
            j = 0
            while j < (n-1 - i) and s[i + j] == '>':
                x += 1
                j += 1
            sub = a[i:i+x+1][::-1]
            a[i:i+x+1] = sub
            i += x
        i += 1
    return a

class Dshortestandlongestlisbootcamp(Basebootcamp):  # 修正类名
    def __init__(self, min_n=2, max_n=10):
        self.min_n = min_n
        self.max_n = max_n

    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        s = ''.join(random.choice(['<', '>']) for _ in range(n-1))
        return {
            'n': n,
            's': s,
            'expected_min': generate_min_sequence(n, s),
            'expected_max': generate_max_sequence(n, s)
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        s = question_case['s']
        return f"""Given n={n} and comparison string '{s}', generate two permutations:
1. Minimal LIS length permutation
2. Maximal LIS length permutation
Format your answer as:
[answer]
a1 a2 ... an
b1 b2 ... bn
[/answer]"""

    @staticmethod
    def extract_output(output):
        start = output.rfind('[answer]')
        end = output.rfind('[/answer]')
        if start == -1 or end == -1: return None
        lines = [l.strip() for l in output[start+8:end].strip().split('\n')]
        if len(lines) < 2: return None
        
        try:
            return (
                list(map(int, lines[0].split())),
                list(map(int, lines[1].split()))
            )
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or len(solution) != 2: return False
        min_seq, max_seq = solution
        n, s = identity['n'], identity['s']
        
        # 验证序列合法性
        def validate(seq):
            if sorted(seq) != list(range(1, n+1)): return False
            for i in range(n-1):
                if (s[i] == '<' and not seq[i] < seq[i+1]) or \
                   (s[i] == '>' and not seq[i] > seq[i+1]):
                    return False
            return True
        
        return (validate(min_seq) and 
                validate(max_seq) and
                compute_lis_length(min_seq) == compute_lis_length(identity['expected_min']) and
                compute_lis_length(max_seq) == compute_lis_length(identity['expected_max']))
