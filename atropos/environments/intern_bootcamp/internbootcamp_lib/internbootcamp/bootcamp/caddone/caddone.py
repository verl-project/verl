"""# 

### 谜题描述
You are given an integer n. You have to apply m operations to it.

In a single operation, you must replace every digit d of the number with the decimal representation of integer d + 1. For example, 1912 becomes 21023 after applying the operation once.

You have to find the length of n after applying m operations. Since the answer can be very large, print it modulo 10^9+7.

Input

The first line contains a single integer t (1 ≤ t ≤ 2 ⋅ 10^5) — the number of test cases.

The only line of each test case contains two integers n (1 ≤ n ≤ 10^9) and m (1 ≤ m ≤ 2 ⋅ 10^5) — the initial number and the number of operations. 

Output

For each test case output the length of the resulting number modulo 10^9+7.

Example

Input


5
1912 1
5 6
999 1
88 2
12 100


Output


5
2
6
4
2115

Note

For the first test, 1912 becomes 21023 after 1 operation which is of length 5.

For the second test, 5 becomes 21 after 6 operations which is of length 2.

For the third test, 999 becomes 101010 after 1 operation which is of length 6.

For the fourth test, 88 becomes 1010 after 2 operations which is of length 4.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
testing = len(sys.argv) == 4 and sys.argv[3] == \"myTest\"
if testing:
    cmd = sys.stdout
    from time import time
    start_time = int(round(time() * 1000)) 
    readAll = open(sys.argv[1], 'r').read
    sys.stdout = open(sys.argv[2], 'w')
else:
    readAll = sys.stdin.read

# ############ ---- I/O Functions ---- ############

flush = sys.stdout.flush
class InputData:
    def __init__(self):
        self.lines = readAll().split('\n')
        self.n = len(self.lines)
        self.ii = -1
    def input(self):
        self.ii += 1
        assert self.ii < self.n
        return self.lines[self.ii]
inputData = InputData()
input = inputData.input

def intin():
    return(int(input()))
def intlin():
    return(list(map(int,input().split())))
def chrin():
    return(list(input()))
def strin():
    return input()
def lout(l, sep=\"\n\", toStr=True):
    print(sep.join(map(str, l) if toStr else l))
def dout(*args, **kargs):
    if not testing: return
    if args: print(args[0] if len(args)==1 else args)
    if kargs: print([(k,v) for k,v in kargs.items()])
    
# ############ ---- I/O Functions ---- ############

# from math import ceil
from collections import defaultdict as ddict, Counter
# from heapq import *
# from Queue import Queue
mmap = [0]*(2*10**5+100)
mod = 10**9+7
cnts = [0]*10
k = 0
cnts[k] = 1
for i in xrange(1,2*10**5+100):
    prv = cnts[0]
    for j in xrange(9):
        nxt = cnts[j+1]
        cnts[j+1] = prv
        prv = nxt
    cnts[0] = prv
    cnts[1] = (prv+cnts[1])%mod
    mmap[i] = (sum(cnts))%mod

def main():
    n,m = intlin()
    ans = 0
    # cnts = Counter(map(int,(list(str(n)))))
    # for i in xrange(10):
    #     ans += mmap[m+i]*cnts[i]
    while n > 0:
        ans += mmap[m+n%10]
        n /= 10
    return(ans%mod)

anss = []
for _ in xrange(intin()):
    anss.append(main())
    # anss.append(\"YES\" if main() else \"NO\")
lout(anss)

if testing:
    sys.stdout = cmd
    print(int(round(time() * 1000))  - start_time)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Caddonebootcamp(Basebootcamp):
    _mmap = None
    MOD = 10**9 + 7

    def __init__(self, **params):
        super().__init__(**params)
        self.params = params

    @classmethod
    def _precompute_mmap(cls):
        if cls._mmap is not None:
            return
        max_m = 2 * 10**5 + 10
        cls._mmap = [0] * (max_m + 10)  # 覆盖最大可能m+9的情况
        
        # 初始化状态：0应用0次操作时的位数
        cnts = [0] * 10
        cnts[0] = 1
        
        # 预处理所有可能的操作次数
        for k in range(0, max_m + 10):
            # 当前状态的总位数即为mmap[k]
            cls._mmap[k] = sum(cnts) % cls.MOD
            
            # 如果未达到最大次数，准备下一层状态
            if k >= max_m:
                continue
                
            # 更新下一层状态
            new_cnts = [0] * 10
            for d in range(10):
                next_num = d + 1
                for digit in str(next_num):
                    new_d = int(digit)
                    new_cnts[new_d] = (new_cnts[new_d] + cnts[d]) % cls.MOD
            cnts = new_cnts

    def case_generator(self):
        # 平衡生成随机和边界用例
        if random.random() < 0.2:  # 20%概率生成预定义边界用例
            cases = [
                (1912, 1), (5, 6), (999, 1),
                (88, 2), (12, 100), (1, 200000),
                (10**9, 1), (9, 200000), (0, 1)
            ]
            n, m = random.choice(cases)
        else:  # 80%概率生成随机有效用例
            n = random.randint(1, 10**9)
            m = random.randint(1, 2*10**5)
        
        # 确保n不包含前导零
        return {'n': n, 'm': m}

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        return f"""Given initial number {n} and {m} operations where each digit d becomes the decimal representation of d+1:
(e.g. 9→10 becomes two digits). Compute the final number length modulo 10^9+7.

Rules:
1. Operations are applied simultaneously to all digits
2. 9→10, 5→6 (single-digit→single-digit)
3. Answer must be an integer within [answer]...[/answer] tags.

Example: For input 1912 with 1 operation:
[answer]5[/answer]

Now solve for n={n}, m={m}."""

    @staticmethod
    def extract_output(output):
        # 支持多空格和换行的健壮正则
        pattern = r'\[answer\][\s\n]*(\d+)[\s\n]*\[/answer\]'
        matches = re.findall(pattern, output, re.IGNORECASE | re.DOTALL)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        cls._precompute_mmap()
        n, m = identity['n'], identity['m']
        
        # 处理n=0的特殊情况（虽然题目限制n≥1）
        if n == 0:
            return solution == 1 if m == 0 else 0
        
        total = 0
        while n > 0:
            d = n % 10
            k = m + d
            if k < len(cls._mmap):
                total = (total + cls._mmap[k]) % cls.MOD
            else:
                # 动态计算超出预处理范围的情况（理论上不应发生）
                pass  
            n //= 10
        return total == solution
