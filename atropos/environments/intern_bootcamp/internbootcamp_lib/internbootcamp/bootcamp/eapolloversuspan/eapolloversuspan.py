"""# 

### 谜题描述
Only a few know that Pan and Apollo weren't only battling for the title of the GOAT musician. A few millenniums later, they also challenged each other in math (or rather in fast calculations). The task they got to solve is the following:

Let x_1, x_2, …, x_n be the sequence of n non-negative integers. Find this value: $$$∑_{i=1}^n ∑_{j=1}^n ∑_{k=1}^n (x_i   \&   x_j) ⋅ (x_j   |   x_k)$$$

Here \& denotes the [bitwise and,](https://en.wikipedia.org/wiki/Bitwise_operation#AND) and | denotes the [bitwise or.](https://en.wikipedia.org/wiki/Bitwise_operation#OR)

Pan and Apollo could solve this in a few seconds. Can you do it too? For convenience, find the answer modulo 10^9 + 7.

Input

The first line of the input contains a single integer t (1 ≤ t ≤ 1 000) denoting the number of test cases, then t test cases follow.

The first line of each test case consists of a single integer n (1 ≤ n ≤ 5 ⋅ 10^5), the length of the sequence. The second one contains n non-negative integers x_1, x_2, …, x_n (0 ≤ x_i < 2^{60}), elements of the sequence.

The sum of n over all test cases will not exceed 5 ⋅ 10^5.

Output

Print t lines. The i-th line should contain the answer to the i-th text case.

Example

Input


8
2
1 7
3
1 2 4
4
5 5 5 5
5
6 2 2 1 0
1
0
1
1
6
1 12 123 1234 12345 123456
5
536870912 536870911 1152921504606846975 1152921504606846974 1152921504606846973


Output


128
91
1600
505
0
1
502811676
264880351

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
testing = len(sys.argv) == 4 and sys.argv[3] == \"myTest\"
if testing:
    cmd = sys.stdout
    from time import time
    start_time = int(round(time() * 1000)) 
    input = open(sys.argv[1], 'r').readline
    sys.stdout = open(sys.argv[2], 'w')
else:
    input = sys.stdin.readline
mod = 10**9+7
# from math import ceil
# from collections import defaultdict as dd
# from heapq import *
############ ---- I/O Functions ---- ############
def intin():
    return(int(input()))
def intlin():
    return(list(map(int,input().split())))
def chrin():
    s = input()
    return(list(s[:len(s) - 1]))
def strin():
    s = input()
    return s[:len(s) - 1]
def intlout(l, sep=\" \"):
    print(sep.join(map(str, l)))

def main():
    n = intin()
    a = intlin()
    b = [format(x,'b')[::-1] for x in a]
    lens = [len(x) for x in b]
    maxK = max(lens)
    bcnt = [0]*maxK
    for i in xrange(n):
        x = b[i]
        for k in xrange(lens[i]):
            if x[k]=='1':
                bcnt[k] += 1
    kpowb = [(((1<<k)%mod)*bcnt[k])%mod for k in xrange(maxK)]
    summ = sum(kpowb)%mod   # sum(a)
    ans = 0
    for j in xrange(n):
        x = b[j]
        tmp = 0
        for k in xrange(lens[j]):
            if x[k] == '1':
                tmp += kpowb[k]
        tmp %= mod
        ans = (ans + (tmp*(((a[j]%mod)*n)%mod + summ - tmp)%mod)%mod)%mod
    # print(ans)
    return(ans)

if __name__ == \"__main__\":
    ans = []
    for _ in xrange(intin()):
        ans.append(main())
        # main()
        # print(\"YES\" if main() else \"NO\")
    intlout(ans,'\n')
    # main()

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

MOD = 10**9 + 7

def compute_answer(n, array):
    if n == 0:
        return 0
    a = array
    b = [format(x, 'b')[::-1] for x in a]
    lens = [len(x) for x in b]
    maxK = max(lens) if lens else 0
    bcnt = [0] * maxK
    for i in range(n):
        x = b[i]
        for k in range(len(x)):
            if x[k] == '1':
                if k >= len(bcnt):
                    bcnt += [0] * (k - len(bcnt) + 1)
                bcnt[k] += 1
    kpowb = [ ((1 << k) % MOD) * bcnt[k] % MOD for k in range(len(bcnt)) ]
    summ = sum(kpowb) % MOD
    ans = 0
    for j in range(n):
        xj = a[j] % MOD
        x_bits = b[j]
        tmp = 0
        for k in range(len(x_bits)):
            if x_bits[k] == '1' and k < len(kpowb):
                tmp = (tmp + kpowb[k]) % MOD
        term_part = ( (xj * n) % MOD + (summ - tmp) % MOD ) % MOD
        term = (tmp * term_part) % MOD
        ans = (ans + term) % MOD
    return ans

class Eapolloversuspanbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=1000, max_x=10**18):
        self.min_n = max(1, min_n)
        self.max_n = min(max_n, 5*10**5)
        self.max_x = min(max_x, (1 << 60)-1)  # 确保不超过题目限制
    
    def case_generator(self):
        # 增加更多边界情况
        case_type = random.choice([0, 1, 2, 3])
        n = random.randint(self.min_n, self.max_n)
        
        if case_type == 0:  # 全零
            array = [0]*n
        elif case_type == 1:  # 全相同高位
            base = random.getrandbits(60)
            array = [base]*n
        elif case_type == 2:  # 混合高低位
            array = [random.getrandbits(random.randint(0, 60)) for _ in range(n)]
        else:  # 极大值
            array = [(1 << 60)-1 for _ in range(n)]
        
        return {"n": n, "array": array}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        array = question_case['array']
        case_input = f"1\n{n}\n{' '.join(map(str, array))}"
        problem_desc = (
            "数学挑战：计算三重位运算和\n\n"
            "给定长度为n的非负整数序列，计算：\n"
            "S = ΣΣΣ (x_i & x_j) * (x_j | x_k) mod 1e9+7 (i,j,k从1到n)\n\n"
            "输入格式：\n"
            "第一行t(测试用例数)\n"
            "每个用例两行：n和数组\n\n"
            "当前测试用例输入：\n"
            f"{case_input}\n\n"
            "将答案放在[answer]标签内，例如：[answer] 123 [/answer]"
        )
        return problem_desc
    
    @staticmethod
    def extract_output(output):
        # 更健壮的正则表达式
        matches = re.findall(
            r'\[answer\][\s\n]*(-?\d+)[\s\n]*\[/answer\]', 
            output, 
            re.IGNORECASE | re.DOTALL
        )
        return int(matches[-1]) if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            expected = compute_answer(identity['n'], identity['array'])
            user_ans = int(solution) % MOD
            return user_ans == expected % MOD
        except:
            return False
