"""# 

### 谜题描述
There are n water tanks in a row, i-th of them contains a_i liters of water. The tanks are numbered from 1 to n from left to right.

You can perform the following operation: choose some subsegment [l, r] (1≤ l ≤ r ≤ n), and redistribute water in tanks l, l+1, ..., r evenly. In other words, replace each of a_l, a_{l+1}, ..., a_r by \frac{a_l + a_{l+1} + ... + a_r}{r-l+1}. For example, if for volumes [1, 3, 6, 7] you choose l = 2, r = 3, new volumes of water will be [1, 4.5, 4.5, 7]. You can perform this operation any number of times.

What is the lexicographically smallest sequence of volumes of water that you can achieve?

As a reminder:

A sequence a is lexicographically smaller than a sequence b of the same length if and only if the following holds: in the first (leftmost) position where a and b differ, the sequence a has a smaller element than the corresponding element in b.

Input

The first line contains an integer n (1 ≤ n ≤ 10^6) — the number of water tanks.

The second line contains n integers a_1, a_2, ..., a_n (1 ≤ a_i ≤ 10^6) — initial volumes of water in the water tanks, in liters.

Because of large input, reading input as doubles is not recommended.

Output

Print the lexicographically smallest sequence you can get. In the i-th line print the final volume of water in the i-th tank.

Your answer is considered correct if the absolute or relative error of each a_i does not exceed 10^{-9}.

Formally, let your answer be a_1, a_2, ..., a_n, and the jury's answer be b_1, b_2, ..., b_n. Your answer is accepted if and only if \frac{|a_i - b_i|}{max{(1, |b_i|)}} ≤ 10^{-9} for each i.

Examples

Input


4
7 5 5 7


Output


5.666666667
5.666666667
5.666666667
7.000000000


Input


5
7 8 8 10 12


Output


7.000000000
8.000000000
8.000000000
10.000000000
12.000000000


Input


10
3 9 5 5 1 7 5 3 8 7


Output


3.000000000
5.000000000
5.000000000
5.000000000
5.000000000
5.000000000
5.000000000
5.000000000
7.500000000
7.500000000

Note

In the first sample, you can get the sequence by applying the operation for subsegment [1, 3].

In the second sample, you can't get any lexicographically smaller sequence.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division,print_function
#from sortedcontainers import SortedList
import sys
#sys.__stdout__.flush()
le=sys.__stdin__.read().split(\"\n\")
le.pop()
le=le[::-1]
n=int(le.pop())
l=list(map(int,le.pop().split()))
su=[l[0]]
cou=[-1,0]
for k in range(1,n):
    nd=1
    ns=l[k]
    while len(cou)>1 and su[-1]*(cou[-1]-cou[-2]+nd)>(su[-1]+ns)*(cou[-1]-cou[-2]):
        nd+=cou[-1]-cou[-2]
        ns+=su[-1]
        su.pop()
        cou.pop()
    cou.append(k)
    su.append(ns)
#print(cou,su)
af=[]
for k in range(len(su)):
    af+=[su[k]/(cou[k+1]-cou[k])]*(cou[k+1]-cou[k])

print(\"\n\".join(map(str,af)))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def solve_water_tanks(input_list):
    n = len(input_list)
    if n == 0:
        return []
    l = input_list
    su = [l[0]]
    cou = [-1, 0]
    for k in range(1, n):
        nd = 1
        ns = l[k]
        while len(cou) > 1 and su[-1] * (cou[-1] - cou[-2] + nd) > (su[-1] + ns) * (cou[-1] - cou[-2]):
            nd += cou[-1] - cou[-2]
            ns += su[-1]
            su.pop()
            cou.pop()
        cou.append(k)
        su.append(ns)
    af = []
    for k in range(len(su)):
        count = cou[k+1] - cou[k]
        avg = su[k] / count
        af.extend([avg] * count)
    return af

class Cwaterbalancebootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=15, min_val=1, max_val=100):
        self.min_n = min_n
        self.max_n = max_n
        self.min_val = min_val
        self.max_val = max_val

    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        input_list = [random.randint(self.min_val, self.max_val) for _ in range(n)]
        output = solve_water_tanks(input_list)
        return {
            'input': input_list,
            'output': output
        }

    @staticmethod
    def prompt_func(question_case):
        input_list = question_case['input']
        n = len(input_list)
        input_str = ' '.join(map(str, input_list))
        prompt = f"""你是一个编程竞赛选手，正在解决一个关于水箱水量优化的问题。请根据题目描述，找到字典序最小的可能序列。

题目描述：

有n个水箱排成一行，第i个水箱初始有a_i升水。你可以进行任意次数的操作：选择一个子段[l, r]，将该区域内的水重新分配，使得每个水箱中的水等于该子段的总水量除以区间长度。例如，初始为[1,3,6,7]，选择子段[2,3]，得到[1,4.5,4.5,7]。你的任务是找到可以通过这些操作得到的字典序最小的水量序列。

字典序定义的补充说明：两个序列从左到右比较第一个不同的元素，较小的元素所在的序列更小。

输入格式：

第一行为整数n，第二行包含n个整数a_1到a_n。

输出格式：

输出n行，每行精确到九位小数，格式为X.XXXXXXXXX（如5.666666667或7.000000000）。

请根据以下输入数据计算答案，并将最终结果按指定格式放在[answer]和[/answer]之间。

输入数据：
n = {n}
初始水量 = {input_str}

请按照以下格式输出答案：
[answer]
值1.xxxxxxxxx
值2.xxxxxxxxx
...
值n.xxxxxxxxx
[/answer]"""
        return prompt

    @staticmethod
    def extract_output(output):
        pattern = r'\[answer\](.*?)\[/answer\]'
        matches = re.findall(pattern, output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1]
        numbers = re.findall(r'\b\d+\.\d{9}\b', last_match)
        try:
            result = [float(num) for num in numbers]
        except:
            return None
        return result if len(result) > 0 else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity['output']
        if solution is None or len(solution) != len(expected):
            return False
        for s, e in zip(solution, expected):
            if not (abs(s - e) <= 1e-9 * max(1, abs(e))):
                return False
        return True
