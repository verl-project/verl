"""# 

### 谜题描述
Learn, learn and learn again — Valera has to do this every day. He is studying at mathematical school, where math is the main discipline. The mathematics teacher loves her discipline very much and tries to cultivate this love in children. That's why she always gives her students large and difficult homework. Despite that Valera is one of the best students, he failed to manage with the new homework. That's why he asks for your help. He has the following task. A sequence of n numbers is given. A prefix of a sequence is the part of the sequence (possibly empty), taken from the start of the sequence. A suffix of a sequence is the part of the sequence (possibly empty), taken from the end of the sequence. It is allowed to sequentially make two operations with the sequence. The first operation is to take some prefix of the sequence and multiply all numbers in this prefix by  - 1. The second operation is to take some suffix and multiply all numbers in it by  - 1. The chosen prefix and suffix may intersect. What is the maximum total sum of the sequence that can be obtained by applying the described operations?

Input

The first line contains integer n (1 ≤ n ≤ 105) — amount of elements in the sequence. The second line contains n integers ai ( - 104 ≤ ai ≤ 104) — the sequence itself.

Output

The first and the only line of the output should contain the answer to the problem.

Examples

Input

3
-1 -2 -3


Output

6


Input

5
-4 2 0 5 0


Output

11


Input

5
-1 10 -5 10 -2


Output

18

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from math import *
from array import *
n=int(raw_input())
s=raw_input().split()
a=[0 for x in range(n)]
for i in range(0,n):
    a[i]=int(s[i])
s1=sum(a)
s2=0
mx=0
for i in range(0,n):
    s2+=a[i]
    if s2<0 :
        s2=0
    mx=max(s2,mx)
print 2*mx-s1
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Cwonderfulrandomizedsumbootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=10, min_val=-10000, max_val=10000):
        self.n_min = n_min
        self.n_max = n_max
        self.min_val = min_val
        self.max_val = max_val
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        arr = [random.randint(self.min_val, self.max_val) for _ in range(n)]
        return {
            "n": n,
            "array": arr
        }
    
    @staticmethod
    def prompt_func(question_case):
        arr = question_case["array"]
        arr_str = ' '.join(map(str, arr))
        problem_desc = f"""Valera需要解决一个数学问题，请你帮他计算出最大的总和。题目规则如下：

给定一个整数序列，你需要依次进行两个操作：
1. 选择一个前缀（从第一个元素开始的一段连续元素，可以为空），将该前缀中的每个元素乘以-1。
2. 选择一个后缀（从最后一个元素开始的一段连续元素，可以为空），同样乘以-1。

这两个操作的前缀和后缀可以任意相交。你的目标是使得操作后的序列总和尽可能大。请计算出这个最大的可能总和。

输入序列为：
{arr_str}

请将计算得到的整数答案放置在[answer]和[/answer]标签之间，例如：[answer]42[/answer]。"""
        return problem_desc
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        arr = identity["array"]
        s1 = sum(arr)
        s2 = mx = 0
        for num in arr:
            s2 = max(s2 + num, 0)
            mx = max(mx, s2)
        return solution == 2 * mx - s1
