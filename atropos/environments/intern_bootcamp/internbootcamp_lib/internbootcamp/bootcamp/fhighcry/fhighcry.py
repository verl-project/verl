"""# 

### 谜题描述
Disclaimer: there are lots of untranslateable puns in the Russian version of the statement, so there is one more reason for you to learn Russian :)

Rick and Morty like to go to the ridge High Cry for crying loudly — there is an extraordinary echo. Recently they discovered an interesting acoustic characteristic of this ridge: if Rick and Morty begin crying simultaneously from different mountains, their cry would be heard between these mountains up to the height equal the bitwise OR of mountains they've climbed and all the mountains between them. 

Bitwise OR is a binary operation which is determined the following way. Consider representation of numbers x and y in binary numeric system (probably with leading zeroes) x = xk... x1x0 and y = yk... y1y0. Then z = x | y is defined following way: z = zk... z1z0, where zi = 1, if xi = 1 or yi = 1, and zi = 0 otherwise. In the other words, digit of bitwise OR of two numbers equals zero if and only if digits at corresponding positions is both numbers equals zero. For example bitwise OR of numbers 10 = 10102 and 9 = 10012 equals 11 = 10112. In programming languages C/C++/Java/Python this operation is defined as «|», and in Pascal as «or».

Help Rick and Morty calculate the number of ways they can select two mountains in such a way that if they start crying from these mountains their cry will be heard above these mountains and all mountains between them. More formally you should find number of pairs l and r (1 ≤ l < r ≤ n) such that bitwise OR of heights of all mountains between l and r (inclusive) is larger than the height of any mountain at this interval.

Input

The first line contains integer n (1 ≤ n ≤ 200 000), the number of mountains in the ridge.

Second line contains n integers ai (0 ≤ ai ≤ 109), the heights of mountains in order they are located in the ridge.

Output

Print the only integer, the number of ways to choose two different mountains.

Examples

Input

5
3 2 1 6 5


Output

8


Input

4
3 3 3 3


Output

0

Note

In the first test case all the ways are pairs of mountains with the numbers (numbering from one):

(1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)

In the second test case there are no such pairs because for any pair of mountains the height of cry from them is 3, and this height is equal to the height of any mountain.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
range = xrange
input = raw_input

n = int(input())
A = [int(x) for x in input().split()]

L = [0]*n
R = [0]*n

stack = []
for i in range(n):
    while stack and A[stack[-1]] < A[i]:
        stack.pop()
    L[i] = stack[-1] if stack else -1
    stack.append(i)

stack = []
for i in reversed(range(n)):
    while stack and A[stack[-1]] <= A[i]:
        stack.pop()
    R[i] = stack[-1] if stack else n
    stack.append(i)

L2 = [0]*n
R2 = [0]*n

last = [-1]*30
for i in range(n):
    x = -1

    a = A[i]
    j = 0
    while a:
        if a&1:
            last[j] = i
        elif last[j] > x:
            x = last[j]
        
        j += 1
        a >>= 1
    L2[i] = x


last = [n]*30
for i in reversed(range(n)):
    x = n

    a = A[i]
    j = 0
    while a:
        if a&1:
            last[j] = i
        elif last[j] < x:
            x = last[j]

        j += 1
        a >>= 1
    R2[i] = x

for i in range(n):
    L2[i] = max(L[i], L2[i])
    R2[i] = min(R[i], R2[i])
    #if L2[i] == -1:
    #    L2[i] = L[i]
    #if R2[i] == n:
    #    R2[i] = R[i]
ans = 0
for i in range(n):
    # l in (L[i], L2[i]], r in [i,R[i])
    # l in (L[i], i], r in [R2[i], R[i])

    # l in (L[i], L2[i]], r in [R2[i], R[i])

    ans += (L2[i] - L[i]) * (R[i] - i) + (i - L[i]) * (R[i] - R2[i]) - (L2[i] - L[i]) * (R[i] - R2[i])

print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Fhighcrybootcamp(Basebootcamp):
    def __init__(self, n_min=2, n_max=5, a_min=0, a_max=1e9):
        self.n_min = n_min
        self.n_max = n_max
        self.a_min = a_min
        self.a_max = int(a_max)
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        if random.random() < 0.5:
            val = random.randint(self.a_min, self.a_max)
            heights = [val] * n
        else:
            heights = [random.randint(self.a_min, self.a_max) for _ in range(n)]
        
        return {
            'n': n,
            'heights': heights,
            'expected_answer': self._compute_answer(n, heights)
        }
    
    @staticmethod
    def _compute_answer(n, A):
        if n < 2: return 0
        
        # 原参考代码的实现保持不变
        L = [-1]*n
        stack = []
        for i in range(n):
            while stack and A[stack[-1]] < A[i]:
                stack.pop()
            L[i] = stack[-1] if stack else -1
            stack.append(i)
        
        R = [n]*n
        stack = []
        for i in reversed(range(n)):
            while stack and A[stack[-1]] <= A[i]:
                stack.pop()
            R[i] = stack[-1] if stack else n
            stack.append(i)
        
        L2 = [-1]*n
        last = [-1]*60
        for i in range(n):
            x = -1
            a = A[i]
            for j in range(60):
                if a & (1 << j):
                    last[j] = i
                else:
                    if last[j] > x:
                        x = last[j]
            L2[i] = max(L[i], x)
        
        R2 = [n]*n
        last = [n]*60
        for i in reversed(range(n)):
            x = n
            a = A[i]
            for j in range(60):
                if a & (1 << j):
                    last[j] = i
                else:
                    if last[j] < x:
                        x = last[j]
            R2[i] = min(R[i], x)
        
        ans = 0
        for i in range(n):
            ans += (L2[i]-L[i])*(R[i]-i) + (i-L[i])*(R[i]-R2[i]) - (L2[i]-L[i])*(R[i]-R2[i])
        return ans
    
    @staticmethod
    def prompt_func(question_case) -> str:
        # 修正关键错误：从question_case中提取n和heights
        n = question_case['n']
        heights = question_case['heights']
        heights_str = ' '.join(map(str, heights))
        
        return f"""作为高喊山脊的声学研究员，你需要解决以下问题：

# 问题描述
给定{n}座连续排列的山峰，每座山峰的高度分别为：{heights_str}

请找出满足以下条件的不同山峰对(l, r)(1 ≤ l < r ≤ {n})的数量：
- 从第l座到第r座山峰（包含两端）所有山峰高度的按位或值
- 严格大于该区间内任意一座山峰的高度

# 输入格式
第一行：整数n (2 ≤ n ≤ 2e5)
第二行：n个整数表示山峰高度

# 输出格式
单个整数表示符合条件的对数

# 答案格式
请将最终答案放在[answer]和[/answer]标签之间，例如：[answer]42[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return int(matches[-1].strip()) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected_answer']
