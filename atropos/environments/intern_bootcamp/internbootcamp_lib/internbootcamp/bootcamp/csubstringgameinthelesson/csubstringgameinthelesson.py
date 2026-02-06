"""# 

### 谜题描述
Mike and Ann are sitting in the classroom. The lesson is boring, so they decided to play an interesting game. Fortunately, all they need to play this game is a string s and a number k (0 ≤ k < |s|).

At the beginning of the game, players are given a substring of s with left border l and right border r, both equal to k (i.e. initially l=r=k). Then players start to make moves one by one, according to the following rules:

  * A player chooses l^{\prime} and r^{\prime} so that l^{\prime} ≤ l, r^{\prime} ≥ r and s[l^{\prime}, r^{\prime}] is lexicographically less than s[l, r]. Then the player changes l and r in this way: l := l^{\prime}, r := r^{\prime}.
  * Ann moves first.
  * The player, that can't make a move loses.



Recall that a substring s[l, r] (l ≤ r) of a string s is a continuous segment of letters from s that starts at position l and ends at position r. For example, \"ehn\" is a substring (s[3, 5]) of \"aaaehnsvz\" and \"ahz\" is not.

Mike and Ann were playing so enthusiastically that they did not notice the teacher approached them. Surprisingly, the teacher didn't scold them, instead of that he said, that he can figure out the winner of the game before it starts, even if he knows only s and k.

Unfortunately, Mike and Ann are not so keen in the game theory, so they ask you to write a program, that takes s and determines the winner for all possible k.

Input

The first line of the input contains a single string s (1 ≤ |s| ≤ 5 ⋅ 10^5) consisting of lowercase English letters.

Output

Print |s| lines.

In the line i write the name of the winner (print Mike or Ann) in the game with string s and k = i, if both play optimally

Examples

Input


abba


Output


Mike
Ann
Ann
Mike


Input


cba


Output


Mike
Mike
Mike

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
# target Expert  

# Author : raj1307 - Raj Singh
# Date   : 18.09.19

from __future__ import division, print_function

import os,sys
from io import BytesIO, IOBase

if sys.version_info[0] < 3:
    from __builtin__ import xrange as range
    from future_builtins import ascii, filter, hex, map, oct, zip


def ii(): return int(input())
def si(): return input()
def mi(): return map(int,input().strip().split(\" \"))
def li(): return list(mi())

def dmain():
    sys.setrecursionlimit(100000000)
    threading.stack_size(40960000)
    thread = threading.Thread(target=main)
    thread.start()
    
#from collections import deque, Counter, OrderedDict,defaultdict
#from heapq import nsmallest, nlargest, heapify,heappop ,heappush, heapreplace
#from math import ceil,floor,log,sqrt,factorial,pi
#from bisect import bisect,bisect_left,bisect_right,insort,insort_left,insort_right
#from decimal import *,threading
#from itertools import permutations

abc='abcdefghijklmnopqrstuvwxyz'
abd={'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25}
mod=1000000007
#mod=998244353
inf = float(\"inf\")
vow=['a','e','i','o','u']
dx,dy=[-1,1,0,0],[0,0,1,-1]
def getKey(item): return item[0] 
def sort2(l):return sorted(l, key=getKey)
def d2(n,m,num):return [[num for x in range(m)] for y in range(n)]
def isPowerOfTwo (x): return (x and (not(x & (x - 1))) )
def decimalToBinary(n): return bin(n).replace(\"0b\",\"\")
def ntl(n):return [int(i) for i in str(n)]

def powerMod(x,y,p):
    res = 1
    x %= p
    while y > 0:
        if y&1:
            res = (res*x)%p
        y = y>>1
        x = (x*x)%p
    return res

def gcd(x, y):
    while y:
        x, y = y, x % y
    return x
    
def isPrime(n) : # Check Prime Number or not 
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
 
    
 
    #for _ in range(ii()):
        

    s=si()


    #print('Mike')
    n=len(s)


    ans=['*']*n

    ans[0]=s[0]

    for i in range(1,n):

        ans[i]=min(ans[i-1],s[i])



    for i in range(n):


        if s[i]>ans[i]:
            print('Ann')
        else:
            print('Mike')

    exit()

    for i in range(1,n):
        f=0

        for j in range(i,-1,-1):


            if s[j]<s[i]:
                print('Ann')
                f=1


        if f==0:
            print('Mike')


    








    

























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
    #read()
    main()
    #dmain()

# Comment Read()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from string import ascii_lowercase
import re
from bootcamp import Basebootcamp

class Csubstringgameinthelessonbootcamp(Basebootcamp):
    def __init__(self, min_length=1, max_length=6):
        # 修正参数校验逻辑
        self.min_length = max(1, min(min_length, max_length))  # 确保最小值合法
        self.max_length = max(self.min_length, max_length)  # 确保max >= min
    
    def case_generator(self):
        """生成全范围合法测试用例"""
        # 确保生成长度在合法范围内
        n = random.randint(self.min_length, self.max_length)
        
        # 生成字符串逻辑优化
        if random.random() < 0.25:  # 25%特殊case
            # 生成全相同字符或严格递减序列
            if random.choice([True, False]):
                c = random.choice(ascii_lowercase)
                s = c * n
            else:
                # 生成严格递减字符串如'cba'
                start = random.randint(0, 25-n+1)
                s = ''.join([ascii_lowercase[start+i] for i in range(n)][::-1])
        else:
            s = ''.join(random.choices(ascii_lowercase, k=n))
        
        # 计算结果逻辑
        correct = []
        if not s:
            return {'s': s, 'correct': []}
        
        min_char = s[0]
        correct.append("Mike")  # k=0
        
        for i in range(1, len(s)):
            current_char = s[i]
            # 严格按题目逻辑判断
            if current_char > min_char:
                correct.append("Ann")
            else:
                correct.append("Mike")
                min_char = current_char  # 更新最小值
        
        return {
            's': s,
            'correct': correct
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        s = question_case['s']
        return f"""请根据以下游戏规则，严格按指定格式输出结果：

# 游戏规则
1. 使用字符串s = "{s}"（长度{len(s)}）
2. 对每个起始位置k（0 ≤ k < {len(s)}）判断胜者
3. Ann先手，双方采取最优策略
4. 每次操作必须扩展子串范围且新子串字典序严格更小

# 输出要求
- 输出{len(s)}行，每行一个结果
- 结果只能是Mike或Ann，首字母大写
- 按k从0到{len(s)-1}的顺序输出

将最终答案放在[answer]标签内，示例：
[answer]
Mike
Ann
Ann
Mike
[/answer]

现在开始处理当前字符串："""

    @staticmethod
    def extract_output(output):
        """增强的答案提取逻辑"""
        # 匹配所有可能的答案块
        blocks = re.findall(r'\[answer\][\s]*((?:Mike|Ann\s*)+)[\s]*\[/answer\]', 
                          output, re.IGNORECASE)
        if not blocks:
            return None
        
        # 处理最后一个答案块
        last_block = blocks[-1].strip().upper()
        candidates = []
        for line in last_block.split('\n'):
            line = line.strip()
            if line == 'MIKE':
                candidates.append('Mike')
            elif line == 'ANN':
                candidates.append('Ann')
        
        return candidates if candidates else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        """严格验证逻辑"""
        expected = identity['correct']
        # 双重验证：长度和内容
        return len(solution) == len(expected) and solution == expected

# 单元测试
if __name__ == "__main__":
    # 测试参数校验
    try:
        bootcamp = Csubstringgameinthelessonbootcamp(min_length=3, max_length=2)
    except Exception as e:
        assert str(e) == "max_length must be >= min_length", "参数校验逻辑错误"
    
    # 测试全相同字符情况
    case = {'s': 'aaa', 'correct': ['Mike', 'Mike', 'Mike']}
    assert Csubstringgameinthelessonbootcamp._verify_correction(['Mike','Mike','Mike'], case)
    
    # 测试严格递减字符串
    case = {'s': 'cba', 'correct': ['Mike', 'Mike', 'Mike']}
    assert Csubstringgameinthelessonbootcamp._verify_correction(['Mike']*3, case)
    
    # 测试混合情况
    case = {'s': 'abba', 'correct': ['Mike','Ann','Ann','Mike']}
    assert Csubstringgameinthelessonbootcamp._verify_correction(['Mike','Ann','Ann','Mike'], case)
