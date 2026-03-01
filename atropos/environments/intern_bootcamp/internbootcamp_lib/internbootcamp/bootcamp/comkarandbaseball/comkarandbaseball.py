"""# 

### 谜题描述
Patrick likes to play baseball, but sometimes he will spend so many hours hitting home runs that his mind starts to get foggy! Patrick is sure that his scores across n sessions follow the identity permutation (ie. in the first game he scores 1 point, in the second game he scores 2 points and so on). However, when he checks back to his record, he sees that all the numbers are mixed up! 

Define a special exchange as the following: choose any subarray of the scores and permute elements such that no element of subarray gets to the same position as it was before the exchange. For example, performing a special exchange on [1,2,3] can yield [3,1,2] but it cannot yield [3,2,1] since the 2 is in the same position. 

Given a permutation of n integers, please help Patrick find the minimum number of special exchanges needed to make the permutation sorted! It can be proved that under given constraints this number doesn't exceed 10^{18}.

An array a is a subarray of an array b if a can be obtained from b by deletion of several (possibly, zero or all) elements from the beginning and several (possibly, zero or all) elements from the end.

Input

Each test contains multiple test cases. The first line contains the number of test cases t (1 ≤ t ≤ 100). Description of the test cases follows.

The first line of each test case contains integer n (1 ≤ n ≤ 2 ⋅ 10^5) — the length of the given permutation.

The second line of each test case contains n integers a_{1},a_{2},...,a_{n} (1 ≤ a_{i} ≤ n) — the initial permutation.

It is guaranteed that the sum of n over all test cases does not exceed 2 ⋅ 10^5.

Output

For each test case, output one integer: the minimum number of special exchanges needed to sort the permutation.

Example

Input


2
5
1 2 3 4 5
7
3 2 4 5 1 6 7


Output


0
2

Note

In the first permutation, it is already sorted so no exchanges are needed.

It can be shown that you need at least 2 exchanges to sort the second permutation.

[3, 2, 4, 5, 1, 6, 7]

Perform special exchange on range (1, 5)

[4, 1, 2, 3, 5, 6, 7]

Perform special exchange on range (1, 4)

[1, 2, 3, 4, 5, 6, 7]

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division, print_function
import os
import sys
from io import BytesIO, IOBase
if sys.version_info[0] < 3:
    from __builtin__ import xrange as range
    from future_builtins import ascii, filter, hex, map, oct, zip
# ------------------------------ TEMPLATE ABOVE THIS --------------------------------



def main():
    t=int(input())
    for _ in range(t):
        n=int(input())
        A=list(map(int,input().split()))
        B=A[:]
        B.sort()
        cnt=0
        i=0
        while i<n:
            while i<n and A[i]==B[i]: i+=1
            flag=0
            while i<n and A[i]!=B[i]: 
                i+=1
                flag=1
            if flag: cnt+=1
        
        if cnt<=1: print(cnt)
        else: print(2)


    return






# ------------------------------ TEMPLATE BELOW THIS --------------------------------
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

def input(): return sys.stdin.readline().rstrip(\"\r\n\")
# endregion

if __name__ == \"__main__\":
    main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random

class Comkarandbaseballbootcamp(Basebootcamp):
    def __init__(self, min_n=5, max_n=10, answer_type=None):
        self.min_n = max(1, min_n)
        self.max_n = max(self.min_n, max_n)
        self.answer_type = answer_type  # 先初始化属性
        if self.answer_type is not None and self.answer_type not in [0, 1, 2]:
            raise ValueError("answer_type must be 0, 1, 2, or None")

    def case_generator(self):
        # 确保answer_type有效性
        answer_type = self.answer_type if self.answer_type is not None else random.choice([0, 1, 2])
        
        # 处理最小尺寸约束
        min_n = self.min_n
        max_n = self.max_n
        if answer_type in {1, 2}:
            min_n = max(2, min_n)
            max_n = max(min_n, max_n)
        
        # 生成合法n值
        n = random.randint(min_n, max_n) if min_n <= max_n else min_n

        # 三种答案类型生成策略
        if answer_type == 0:
            arr = list(range(1, n+1))
            answer = 0
        elif answer_type == 1:
            # 生成单次交换的排列
            while True:
                arr = list(range(1, n+1))
                # 随机选择可交换区间
                start = random.randint(0, n-2)
                end = random.randint(start+1, n-1)
                sub = arr[start:end+1]
                # 生成错位排列（循环右移）
                derangement = sub[1:] + sub[:1]
                arr[start:end+1] = derangement
                if self._compute_answer(arr) == 1:
                    answer = 1
                    break
        else:
            # 生成需要两次交换的排列
            while True:
                arr = list(range(1, n+1))
                # 生成第一个错位区间
                start1 = random.randint(0, n-3)
                end1 = random.randint(start1+1, n-2)
                sub1 = arr[start1:end1+1]
                arr[start1:end1+1] = sub1[1:] + sub1[:1]
                
                # 生成第二个错位区间
                start2 = random.randint(end1+1, n-1)
                end2 = random.randint(start2+1, n-1) if start2 < n-1 else start2
                sub2 = arr[start2:end2+1]
                if len(sub2) >= 2:
                    arr[start2:end2+1] = sub2[1:] + sub2[:1]
                
                if self._compute_answer(arr) == 2:
                    answer = 2
                    break

        return {
            'n': n,
            'arr': arr,
            'answer': answer
        }

    @staticmethod
    def _compute_answer(a):
        """根据官方参考代码实现的答案验证逻辑"""
        B = sorted(a)
        cnt = 0
        i, n = 0, len(a)
        while i < n:
            while i < n and a[i] == B[i]:
                i += 1
            flag = 0
            while i < n and a[i] != B[i]:
                i += 1
                flag = 1
            if flag:
                cnt += 1
        return cnt if cnt <= 1 else 2

    @staticmethod
    def prompt_func(question_case):
        return f"""Patrick needs to sort his baseball score records using special exchanges. Given the permutation:
n = {question_case['n']}
{question_case['arr']}

Calculate the minimum number of special exchanges required. Put your final answer within [answer] and [/answer] tags. For example: [answer]2[/answer]"""

    @staticmethod
    def extract_output(output):
        # 逆向搜索最后一个答案标签对
        end_pos = output.rfind('[/answer]')
        if end_pos == -1:
            return None
        start_pos = output.rfind('[answer]', 0, end_pos)
        if start_pos == -1:
            return None
        return output[start_pos+8:end_pos].strip()

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return int(solution) == identity['answer']
        except (ValueError, KeyError):
            return False
