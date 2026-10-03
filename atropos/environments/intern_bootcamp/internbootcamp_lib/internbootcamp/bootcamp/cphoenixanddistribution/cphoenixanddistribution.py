"""# 

### 谜题描述
Phoenix has a string s consisting of lowercase Latin letters. He wants to distribute all the letters of his string into k non-empty strings a_1, a_2, ..., a_k such that every letter of s goes to exactly one of the strings a_i. The strings a_i do not need to be substrings of s. Phoenix can distribute letters of s and rearrange the letters within each string a_i however he wants.

For example, if s =  baba and k=2, Phoenix may distribute the letters of his string in many ways, such as: 

  * ba and ba
  * a and abb
  * ab and ab
  * aa and bb



But these ways are invalid: 

  * baa and ba
  * b and ba
  * baba and empty string (a_i should be non-empty) 



Phoenix wants to distribute the letters of his string s into k strings a_1, a_2, ..., a_k to minimize the lexicographically maximum string among them, i. e. minimize max(a_1, a_2, ..., a_k). Help him find the optimal distribution and print the minimal possible value of max(a_1, a_2, ..., a_k).

String x is lexicographically less than string y if either x is a prefix of y and x ≠ y, or there exists an index i (1 ≤ i ≤ min(|x|, |y|)) such that x_i < y_i and for every j (1 ≤ j < i) x_j = y_j. Here |x| denotes the length of the string x.

Input

The input consists of multiple test cases. The first line contains an integer t (1 ≤ t ≤ 1000) — the number of test cases. Each test case consists of two lines.

The first line of each test case consists of two integers n and k (1 ≤ k ≤ n ≤ 10^5) — the length of string s and the number of non-empty strings, into which Phoenix wants to distribute letters of s, respectively.

The second line of each test case contains a string s of length n consisting only of lowercase Latin letters.

It is guaranteed that the sum of n over all test cases is ≤ 10^5.

Output

Print t answers — one per test case. The i-th answer should be the minimal possible value of max(a_1, a_2, ..., a_k) in the i-th test case.

Example

Input


6
4 2
baba
5 2
baacb
5 3
baacb
5 3
aaaaa
6 4
aaxxzz
7 1
phoenix


Output


ab
abbc
b
aa
x
ehinopx

Note

In the first test case, one optimal solution is to distribute baba into ab and ab. 

In the second test case, one optimal solution is to distribute baacb into abbc and a.

In the third test case, one optimal solution is to distribute baacb into ac, ab, and b.

In the fourth test case, one optimal solution is to distribute aaaaa into aa, aa, and a.

In the fifth test case, one optimal solution is to distribute aaxxzz into az, az, x, and x.

In the sixth test case, one optimal solution is to distribute phoenix into ehinopx.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division, print_function
 
import os,sys
from io import BytesIO, IOBase
 
if sys.version_info[0] < 3:
    from __builtin__ import xrange as range
    from future_builtins import ascii, filter, hex, map, oct, zip
 
 
def ii():  return int(input())
def si():  return input()
def mi():  return map(int,input().split(\" \"))
def msi(): return map(str,input().split(\" \"))
def li():  return list(mi())
 
def dmain():
    sys.setrecursionlimit(1000000)
    threading.stack_size(1024000)
    thread = threading.Thread(target=main)
    thread.start()
    
import math
def isPowerOfTwo (x): return (x and (not(x & (x - 1))) )
 
def gcd(x, y):
    while y:
        x, y = y, x % y
    return x
    
def checkPrime(n) : # Check Prime Number or not 
    if (n <= 1) : return False
    if (n <= 3) : return True
    if (n % 2 == 0 or n % 3 == 0) : return False
    i = 5
    while(i * i <= n) : 
        if (n % i == 0 or n % (i + 2) == 0) : 
            return False
        i = i + 6
    return True
 
def read():
    sys.stdin = open('input.txt', 'r')  
    sys.stdout = open('output.txt', 'w') 

def main():
    
    for _ in range(ii()):
        n,k=mi()
        s=si()
        s=\"\".join(sorted(s))
        b=list(set(s))
        if s.count(min(s))<k or n==k:
            print(s[k-1])
        else:
            if(s[k] != s[n-1]):
                print(s[0]+s[k:n])
            else:
                t = ''
                t+=s[0]
                for i in range(int((n-1)/k)):
                    t+=s[n-1]
                print(t)  
# region fastio
# template taken from https://github.com/cheran-senthil/PyRival/blob/master/templates/template.py
 
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
    #read()
    main()
    #dmain()
 
# Comment Read()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import string
import re
from collections import Counter
from bootcamp import Basebootcamp

class Cphoenixanddistributionbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__(**params)
        self.params = params
        self.min_length = params.get('min_length', 1)
        self.max_length = params.get('max_length', 20)
        self.force_diverse_cases = params.get('force_diverse_cases', False)

    def case_generator(self):
        if self.force_diverse_cases and random.random() < 0.5:
            return self._generate_edge_case()
        return self._generate_random_case()

    def _generate_random_case(self):
        n = random.randint(self.min_length, self.max_length)
        s = ''.join(random.choices(string.ascii_lowercase, k=n))
        k = random.randint(1, max(1, n//2)) if n > 1 else 1
        return {'n': n, 'k': k, 's': s}

    def _generate_edge_case(self):
        case_type = random.choice([1, 2, 3, 4])
        
        if case_type == 1:  # k=1的特殊情况
            s = ''.join(sorted(random.choices(string.ascii_lowercase, k=random.randint(5, 10))))
            return {'n': len(s), 'k': 1, 's': s}
        
        elif case_type == 2:  # 所有字符相同的情况
            char = random.choice(string.ascii_lowercase)
            n = random.randint(5, 15)
            k = random.randint(1, n)
            return {'n': n, 'k': k, 's': char * n}
        
        elif case_type == 3:  # 需要均匀分配的情况
            base_char = random.choice(string.ascii_lowercase)
            other_char = chr(ord(base_char) + 1)
            s = base_char * 5 + other_char * 10
            k = random.randint(3, 5)
            return {'n': len(s), 'k': k, 's': ''.join(random.sample(s, len(s)))}
        
        else:  # 首字符不满足k需求的情况
            first_char = 'a'
            rest_chars = ''.join(random.choices(string.ascii_lowercase[1:], k=random.randint(8, 15)))
            s = first_char * 3 + rest_chars
            k = 5  # 大于首字符数量(3)
            return {'n': len(s), 'k': k, 's': ''.join(random.sample(s, len(s)))}

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k = question_case['k']
        s = question_case['s']
        problem = (
            f"## 字符串优化分配问题\n\n"
            f"给定字符串 s = '{s}'\n"
            f"需要分成 k = {k} 个非空子串\n\n"
            "### 规则说明\n"
            "1. 必须使用全部字符\n2. 允许重新排列每个子串\n3. 找到使最大子串字典序最小的方案\n\n"
            "请将最终答案放在[answer]和[/answer]标签之间，例如：[answer]abc[/answer]"
        )
        return problem

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*([a-z]+)\s*\[/answer\]', output)
        return matches[-1] if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        
        # 验证字符匹配
        input_chars = Counter(identity['s'])
        output_chars = Counter(solution)
        if input_chars != output_chars:
            return False

        # 验证正确性
        try:
            sorted_s = ''.join(sorted(identity['s']))
            k = identity['k']
            n = identity['n']
            
            if Counter(solution) != Counter(cls.compute_correct_answer(identity['s'], k)):
                return False

            # 验证字典序正确性
            parts = cls.split_into_parts(sorted_s, k)
            expected_max = max(parts)
            return solution == expected_max
        except Exception:
            return False

    @classmethod
    def compute_correct_answer(cls, s, k):
        sorted_s = ''.join(sorted(s))
        n = len(sorted_s)
        first_char_count = sorted_s.count(sorted_s[0])

        if first_char_count < k or n == k:
            return sorted_s[k-1]
        else:
            if sorted_s[k] != sorted_s[-1]:
                return sorted_s[0] + sorted_s[k:]
            else:
                repeat = (n - 1) // k
                return sorted_s[0] + sorted_s[-1] * repeat

    @staticmethod
    def split_into_parts(sorted_s, k):
        # 辅助方法用于验证分割逻辑
        parts = []
        if Counter(sorted_s) == Counter(sorted_s[0]*len(sorted_s)):
            base = sorted_s[0]
            per_part = len(sorted_s) // k
            remainder = len(sorted_s) % k
            for i in range(k):
                parts.append(base * (per_part + (1 if i < remainder else 0)))
        else:
            parts = [sorted_s[0]] * k
            remaining = sorted_s[k:]
            for i in range(len(remaining)):
                parts[i % k] += remaining[i]
            parts = [''.join(sorted(p)) for p in parts]
        return parts
