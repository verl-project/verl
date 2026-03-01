"""# 

### 谜题描述
Vasya became interested in bioinformatics. He's going to write an article about similar cyclic DNA sequences, so he invented a new method for determining the similarity of cyclic sequences.

Let's assume that strings s and t have the same length n, then the function h(s, t) is defined as the number of positions in which the respective symbols of s and t are the same. Function h(s, t) can be used to define the function of Vasya distance ρ(s, t): 

<image> where <image> is obtained from string s, by applying left circular shift i times. For example, ρ(\"AGC\", \"CGT\") =  h(\"AGC\", \"CGT\") + h(\"AGC\", \"GTC\") + h(\"AGC\", \"TCG\") +  h(\"GCA\", \"CGT\") + h(\"GCA\", \"GTC\") + h(\"GCA\", \"TCG\") +  h(\"CAG\", \"CGT\") + h(\"CAG\", \"GTC\") + h(\"CAG\", \"TCG\") =  1 + 1 + 0 + 0 + 1 + 1 + 1 + 0 + 1 = 6

Vasya found a string s of length n on the Internet. Now he wants to count how many strings t there are such that the Vasya distance from the string s attains maximum possible value. Formally speaking, t must satisfy the equation: <image>.

Vasya could not try all possible strings to find an answer, so he needs your help. As the answer may be very large, count the number of such strings modulo 109 + 7.

Input

The first line of the input contains a single integer n (1 ≤ n ≤ 105).

The second line of the input contains a single string of length n, consisting of characters \"ACGT\".

Output

Print a single number — the answer modulo 109 + 7.

Examples

Input

1
C


Output

1


Input

2
AG


Output

4


Input

3
TTT


Output

1

Note

Please note that if for two distinct strings t1 and t2 values ρ(s, t1) и ρ(s, t2) are maximum among all possible t, then both strings must be taken into account in the answer even if one of them can be obtained by a circular shift of another one.

In the first sample, there is ρ(\"C\", \"C\") = 1, for the remaining strings t of length 1 the value of ρ(s, t) is 0.

In the second sample, ρ(\"AG\", \"AG\") = ρ(\"AG\", \"GA\") = ρ(\"AG\", \"AA\") = ρ(\"AG\", \"GG\") = 4.

In the third sample, ρ(\"TTT\", \"TTT\") = 27

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/bin/python
from __future__ import print_function, division
import sys

it = iter(sys.stdin.read().splitlines())

n = int(next(it))
s = next(it)
k = 0
repeat = False

countA = s.count(\"A\")
countC = s.count(\"C\")
countG = s.count(\"G\")
countT = s.count(\"T\")

maximo = max(countA, countG, countC, countT)
listaC = [countA, countC, countG, countT]

k = 0
for item in listaC:
    if maximo == item:
        k += 1


if not repeat:
    print(pow(k, n, 1000000007))
else:
    print(1)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cdnaalignmentbootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=100000, **kwargs):
        super().__init__(**kwargs)
        self.n_min = n_min
        self.n_max = n_max
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        bases = ['A', 'C', 'G', 'T']
        s = ''.join(random.choices(bases, k=n))
        return {'n': n, 's': s}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        s = question_case['s']
        prompt = (
            "Vasya在研究生物信息学中的循环DNA序列相似性时，定义了一种新的距离度量方法——Vasya距离。给定两个长度同为n的字符串s和t，它们的Vasya距离ρ(s, t)的计算方式如下：\n\n"
            "对于每个i（0 ≤ i < n）次左循环移位后的s_i，和每个j（0 ≤ j < n）次左循环移位后的t_j，计算这两个字符串的h函数值。h函数h(s_i, t_j)统计两个字符串在相同位置上字符相同的数目。将所有i和j对应的h函数值相加，得到ρ(s, t)。\n"
            "例如，当s是'AGC'，t是'CGT'时，ρ的计算包括所有3次左移后的组合，共有3×3=9种情况。每个h函数值相加的结果为6。\n\n"
            "现在，给定一个长度为n的字符串s，要求找出所有可能的字符串t（长度同样为n），使得ρ(s, t)达到所有可能t中的最大值。由于答案可能非常大，请将结果对10^9+7取模。\n\n"
            "输入参数：\n"
            f"- 字符串长度n = {n}\n"
            f"- 字符串s = '{s}'\n\n"
            "你的任务是计算满足条件的t的数量。请将最终答案放在[answer]和[/answer]标签之间。例如，如果答案是123，则应写成[answer]123[/answer]。\n\n"
            "注意：答案必须是一个整数，且已经对10^9+7取模。"
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        # 支持标签大小写混合匹配并提取最后出现的答案
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.IGNORECASE | re.DOTALL)
        if not matches:
            return None
        
        # 提取最后一个答案并处理非数字字符
        last_match = matches[-1].strip()
        digits = ''.join(filter(str.isdigit, last_match))
        if not digits:
            return None
        
        try:
            return int(digits) % (10**9 + 7)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        s = identity['s']
        counts = {
            'A': s.count('A'),
            'C': s.count('C'),
            'G': s.count('G'),
            'T': s.count('T')
        }
        max_count = max(counts.values())
        k = sum(1 for cnt in counts.values() if cnt == max_count)
        mod = 10**9 + 7
        correct_answer = pow(k, n, mod)
        return solution == correct_answer
