"""# 

### 谜题描述
This problem differs from the previous one only in the absence of the constraint on the equal length of all numbers a_1, a_2, ..., a_n.

A team of SIS students is going to make a trip on a submarine. Their target is an ancient treasure in a sunken ship lying on the bottom of the Great Rybinsk sea. Unfortunately, the students don't know the coordinates of the ship, so they asked Meshanya (who is a hereditary mage) to help them. He agreed to help them, but only if they solve his problem.

Let's denote a function that alternates digits of two numbers f(a_1 a_2 ... a_{p - 1} a_p, b_1 b_2 ... b_{q - 1} b_q), where a_1 ... a_p and b_1 ... b_q are digits of two integers written in the decimal notation without leading zeros.

In other words, the function f(x, y) alternately shuffles the digits of the numbers x and y by writing them from the lowest digits to the older ones, starting with the number y. The result of the function is also built from right to left (that is, from the lower digits to the older ones). If the digits of one of the arguments have ended, then the remaining digits of the other argument are written out. Familiarize with examples and formal definitions of the function below.

For example: $$$f(1111, 2222) = 12121212 f(7777, 888) = 7787878 f(33, 44444) = 4443434 f(555, 6) = 5556 f(111, 2222) = 2121212$$$

Formally,

  * if p ≥ q then f(a_1 ... a_p, b_1 ... b_q) = a_1 a_2 ... a_{p - q + 1} b_1 a_{p - q + 2} b_2 ... a_{p - 1} b_{q - 1} a_p b_q; 
  * if p < q then f(a_1 ... a_p, b_1 ... b_q) = b_1 b_2 ... b_{q - p} a_1 b_{q - p + 1} a_2 ... a_{p - 1} b_{q - 1} a_p b_q. 



Mishanya gives you an array consisting of n integers a_i, your task is to help students to calculate ∑_{i = 1}^{n}∑_{j = 1}^{n} f(a_i, a_j) modulo 998 244 353.

Input

The first line of the input contains a single integer n (1 ≤ n ≤ 100 000) — the number of elements in the array. The second line of the input contains n integers a_1, a_2, ..., a_n (1 ≤ a_i ≤ 10^9) — the elements of the array.

Output

Print the answer modulo 998 244 353.

Examples

Input


3
12 3 45


Output


12330

Input


2
123 456


Output


1115598

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division, print_function
def main():
    def power(x, y, p) : 
        res = 1     # Initialize result 
    
        # Update x if it is more 
        # than or equal to p 
        x = x % p  
      
        while (y > 0) : 
              
            # If y is odd, multiply 
            # x with result 
            if ((y & 1) == 1) : 
                res = (res * x) % p 
     
            # y must be even now 
            y = y >> 1      # y = y/2 
            x = (x * x) % p 
              
        return res
    def count_next_smaller_elements(xs):
        ys = sorted((x,i) for i,x in enumerate(xs))
        zs = [0] * len(ys)
    
        for i in range(1, len(ys)):
            zs[ys[i][1]] = zs[ys[i-1][1]]
            if ys[i][0] != ys[i-1][0]: zs[ys[i][1]] += 1
        ts = [0] * (zs[ys[-1][1]]+1)
        us = [0] * len(xs)
    
        for i in range(len(xs)-1,-1,-1):
            x = zs[i]+1
            while True:
                us[i] += ts[x-1]
                x -= (x & (-x))
                if x <= 0: break
    
                x = zs[i]+1
            while True:
                x += (x & (-x))
                if x > len(ts): break
                ts[x-1] += 1

        return us
    mod=998244353
    n=int(input())
    l1=list(input().split())
    arr=[0]*10
    for item in l1:
        arr[len(item)-1]+=1
    ans=0
    for item in l1:
        for i in range(1,11):
            if arr[i-1]==0:
                continue
            x=len(item)
            res=0
            
            if x<=i:
                for j in range(x-1,-1,-1):
                    res=(res+int(item[j])*pow(10,(x-j-1)*2,mod))%mod
                    res=(res+int(item[j])*pow(10,(x-j)*2-1,mod))%mod
                ans=(ans+arr[i-1]*res)%mod
            else :
                x-=1
                i-=1
                j=0
                while x>i:
                    res=(res+2*int(item[j])*pow(10,(x-i)+(2*(i+1))-1,mod))%mod
                    x-=1
                    j+=1

                i+=1
                x=len(item)
                for j in range(i-1,-1,-1):
                    res=(res+int(item[j+(x-i)])*pow(10,(i-j-1)*2,mod))%mod
                    res=(res+int(item[j+(x-i)])*pow(10,(i-j)*2-1,mod))%mod
                ans=(ans+arr[i-1]*res)%mod
    print(ans)

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
import re
from bootcamp import Basebootcamp

class D2submarineintherybinskseahardeditionbootcamp(Basebootcamp):
    MOD = 998244353

    def __init__(self, max_n=100, max_digits=10, **kwargs):
        super().__init__(**kwargs)
        self.max_n = max_n
        self.max_digits = max_digits

    def case_generator(self):
        n = random.randint(1, self.max_n)
        a = [self._generate_number_str() for _ in range(n)]
        return {'n': n, 'a': a}

    def _generate_number_str(self):
        digits = random.randint(1, self.max_digits)
        first = str(random.randint(1, 9))
        rest = ''.join(str(random.randint(0, 9)) for _ in range(digits-1)) if digits > 1 else ''
        return first + rest

    @staticmethod
    def prompt_func(question_case) -> str:
        input_n = question_case['n']
        input_a = ' '.join(question_case['a'])
        return f"""你需要计算所有i,j对的函数f(a_i,a_j)之和模998244353。函数f交替拼接两个数字的各位：
- 当a_i长度≥a_j时，先取a_i的前几位，再交替剩余位
- 否则先取a_j的前几位，再交替剩余位
例如：f(1111,2222)=12121212，f(7777,888)=7787878

输入：
{input_n}
{input_a}

将最终答案放入[answer]标签。例如：[answer]12345[/answer]。请确保只放置数字答案。"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return int(matches[-1].strip()) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False
        try:
            expected = cls._compute_answer(identity['a'])
            return solution % cls.MOD == expected
        except:
            return False

    @classmethod
    def _compute_answer(cls, a_list):
        mod = cls.MOD
        len_cnt = [0]*10
        for num in a_list:
            len_cnt[len(num)-1] += 1

        total = 0
        for num in a_list:
            x = len(num)
            for i in range(1, 11):
                cnt = len_cnt[i-1]
                if cnt == 0:
                    continue
                res = 0

                if x <= i:
                    # 处理x <= i的情况
                    for j in range(x):
                        digit = int(num[x-1-j])  # 从低位到高位
                        pow1 = pow(10, 2*j, mod)
                        pow2 = pow(10, 2*j+1, mod)
                        res = (res + digit * (pow1 + pow2)) % mod
                    res = res * cnt % mod

                else:
                    # 处理x > i的情况
                    # 参考代码对应分支调整
                    xx = x - 1
                    ii = i - 1
                    j = 0
                    res = 0
                    # 处理前面超出的位数
                    while xx > ii:
                        digit = int(num[j])
                        exponent = (xx - ii) + 2*(ii + 1) - 1
                        res = (res + 2 * digit * pow(10, exponent, mod)) % mod
                        xx -= 1
                        j += 1
                    # 处理交替部分
                    for jj in range(ii, -1, -1):
                        pos = j + (ii - jj)
                        digit = int(num[pos])
                        pow1 = pow(10, 2*jj, mod)
                        pow2 = pow(10, 2*jj +1, mod)
                        res = (res + digit * (pow1 + pow2)) % mod
                    res = res * cnt % mod

                total = (total + res) % mod
        return total
