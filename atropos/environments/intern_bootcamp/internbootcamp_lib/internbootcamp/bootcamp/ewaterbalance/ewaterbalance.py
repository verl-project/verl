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
from sys import stdout

n = input()
a = map(int, raw_input().strip().split())

pre = [0] * (n + 1)
for i in xrange(1, n + 1):
    pre[i] = pre[i - 1] + a[i - 1]

stk = []
for i in xrange(n):
    stk.append((i, i + 1))
    while (len(stk) > 1):
        i1, j1 = stk[-2]
        s1 = pre[j1] - pre[i1]
        l1 = j1 - i1

        i2, j2 = stk[-1]
        s2 = pre[j2] - pre[i2]
        l2 = j2 - i2

        if s1 * l2 > s2 * l1:
            # merge
            stk.pop()
            stk.pop()
            stk.append((i1, j2))
        else:
            break

for i, j in stk:
    mean = (pre[j] - pre[i]) * 1.0 / (j - i)
    smean = \"{0:.20f}\".format(mean)
    for it in xrange(i, j): a[it] = smean

stdout.write(\"\n\".join(a))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from decimal import Decimal, getcontext
from bootcamp import Basebootcamp

getcontext().prec = 20  # 设置高精度计算环境

class Ewaterbalancebootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10, min_val=1, max_val=20):
        self.min_n = min_n
        self.max_n = max_n
        self.min_val = min_val
        self.max_val = max_val
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        a = [random.randint(self.min_val, self.max_val) for _ in range(n)]
        return {'n': n, 'a': a}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = ' '.join(map(str, question_case['a']))
        prompt = f"""你是解决水箱问题的专家，需要通过重新分配水量的操作得到字典序最小的序列。问题描述如下：

有{n}个水箱排成一行，初始水量分别为：{a}升。你可以多次进行以下操作：选择一个子段[l, r]，将该子段内的水量平均分配。例如，子段中所有水箱的水量会被替换为该子段的总水量的平均值。你的目标是找出通过任意次操作后，可能得到的字典序最小的序列。

字典序更小的定义是：比较两个序列，找到第一个不同的位置，该位置上数值较小的序列字典序更小。请确保你的答案在满足条件的情况下，每个数值的绝对或相对误差不超过1e-9。

输入格式：
第一行是一个整数n（1 ≤ n ≤ 1e6），表示水箱数量。
第二行是n个整数（1 ≤ a_i ≤ 1e6），表示初始水量。

输出格式：
输出n行，每行一个浮点数，表示最终各个水箱的水量，必须包含小数点后恰好9位（例如5.666666667）。

请将你的答案放置在[answer]和[/answer]标签之间，例如：

[answer]
5.666666667
5.666666667
5.666666667
7.000000000
[/answer]

现在，请解决当前问题，并按要求输出答案。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        pattern = r'\[answer\](.*?)\[/answer\]'
        matches = re.findall(pattern, output, re.DOTALL)
        if not matches:
            return None
        
        answer_block = matches[-1].strip()
        solution = []
        for line in answer_block.split('\n'):
            line = line.strip()
            if not line:
                continue
            # 严格格式验证：必须符合xxx.xxxxxxxxx格式
            if not re.fullmatch(r'\d+\.\d{9}', line):
                return None
            solution.append(line)
        
        return solution if len(solution) > 0 else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 使用Decimal进行高精度计算
        n = identity['n']
        a = identity['a']
        
        # 生成参考解
        prefix = [Decimal(0)]
        for num in a:
            prefix.append(prefix[-1] + Decimal(num))
        
        stack = []
        for i in range(n):
            # 当前区间为[i, i+1)
            stack.append((i, i+1))
            while len(stack) >= 2:
                # 比较最后两个区间的平均值
                (i1, j1), (i2, j2) = stack[-2], stack[-1]
                # 计算总水量和长度
                sum1 = prefix[j1] - prefix[i1]
                len1 = j1 - i1
                sum2 = prefix[j2] - prefix[i2]
                len2 = j2 - i2
                # 交叉相乘比较以避免除法误差
                if sum1 * len2 > sum2 * len1:  # 等价于 avg1 > avg2
                    # 合并区间
                    merged = (i1, j2)
                    stack.pop()
                    stack.pop()
                    stack.append(merged)
                else:
                    break
        
        # 生成正确结果
        correct = []
        for i, j in stack:
            avg = (prefix[j] - prefix[i]) / (j - i)
            correct.extend([avg] * (j - i))
        
        # 验证答案
        if len(solution) != len(correct):
            return False
        
        try:
            for user_val, correct_val in zip(solution, correct):
                user_dec = Decimal(user_val)
                # 计算允许误差
                max_denom = max(Decimal(1), abs(correct_val))
                error = abs(user_dec - correct_val)
                if error / max_denom > Decimal('1e-9'):
                    return False
        except:
            return False
        
        return True
