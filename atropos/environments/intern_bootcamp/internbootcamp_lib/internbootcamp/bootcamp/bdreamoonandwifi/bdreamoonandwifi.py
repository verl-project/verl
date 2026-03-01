"""# 

### 谜题描述
Dreamoon is standing at the position 0 on a number line. Drazil is sending a list of commands through Wi-Fi to Dreamoon's smartphone and Dreamoon follows them.

Each command is one of the following two types: 

  1. Go 1 unit towards the positive direction, denoted as '+'
  2. Go 1 unit towards the negative direction, denoted as '-'



But the Wi-Fi condition is so poor that Dreamoon's smartphone reports some of the commands can't be recognized and Dreamoon knows that some of them might even be wrong though successfully recognized. Dreamoon decides to follow every recognized command and toss a fair coin to decide those unrecognized ones (that means, he moves to the 1 unit to the negative or positive direction with the same probability 0.5). 

You are given an original list of commands sent by Drazil and list received by Dreamoon. What is the probability that Dreamoon ends in the position originally supposed to be final by Drazil's commands?

Input

The first line contains a string s1 — the commands Drazil sends to Dreamoon, this string consists of only the characters in the set {'+', '-'}. 

The second line contains a string s2 — the commands Dreamoon's smartphone recognizes, this string consists of only the characters in the set {'+', '-', '?'}. '?' denotes an unrecognized command.

Lengths of two strings are equal and do not exceed 10.

Output

Output a single real number corresponding to the probability. The answer will be considered correct if its relative or absolute error doesn't exceed 10 - 9.

Examples

Input

++-+-
+-+-+


Output

1.000000000000


Input

+-+-
+-??


Output

0.500000000000


Input

+++
??-


Output

0.000000000000

Note

For the first sample, both s1 and s2 will lead Dreamoon to finish at the same position  + 1. 

For the second sample, s1 will lead Dreamoon to finish at position 0, while there are four possibilites for s2: {\"+-++\", \"+-+-\", \"+--+\", \"+---\"} with ending position {+2, 0, 0, -2} respectively. So there are 2 correct cases out of 4, so the probability of finishing at the correct position is 0.5. 

For the third sample, s2 could only lead us to finish at positions {+1, -1, -3}, so the probability to finish at the correct position  + 3 is 0.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import math
f = math.factorial
a = raw_input()
b = raw_input()
pos = 0
for e in a:
    if e == '+':
        pos+=1
    else:
        pos-=1
pos2 = 0
cnt = 0
for e in b:
    if e == '+':
        pos2+=1
    elif e == '-':
        pos2-=1
    else:
        cnt+=1
dist = abs(pos2 - pos)
if dist%2 == 0:
    if cnt%2 == 0:
        if dist > cnt:
            print 0.0
        else:
            c = cnt
            d = dist
            num = f(c) / ( f(d+((c-d)/2)) * f((c-d)/2) )
            den = 2**c
            print float(num)/den
    else:
        print 0.0
else:
    if cnt%2 == 0:
        print 0.0
    else:
        if dist > cnt:
            print 0.0
        else:
            c = cnt
            d = dist
            num = f(c) / ( f(d+((c-d)/2)) * f((c-d)/2) )
            den = 2**c
            print float(num)/den
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import math

class Bdreamoonandwifibootcamp(Basebootcamp):
    def __init__(self, max_length=10, p_question=0.3):
        super().__init__()  # 显式调用父类初始化
        self.max_length = max_length
        self.p_question = p_question
    
    def case_generator(self):
        length = random.randint(1, self.max_length)
        s1 = ''.join(random.choices('+-', k=length))
        s2 = ''.join([
            '?' if random.random() < self.p_question else random.choice('+-') 
            for _ in range(length)
        ])
        return {
            's1': s1,
            's2': s2,
            '_target': self._calculate_probability(s1, s2)
        }
    
    @staticmethod
    def prompt_func(case) -> str:
        return f"""Drazil的原始指令：{case['s1']}
接收到的含噪声指令：{case['s2']}
每个?表示随机选择±1。请计算最终位置一致的概率（12位小数格式在[answer]标签中）：
示例：[answer]0.123456789012[/answer]"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](0?\.\d{12}|0|1\.0{12})\[/answer\]', output)
        try:
            return float(matches[-1]) if matches else None
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, case):
        try:
            return math.isclose(solution, case['_target'], rel_tol=1e-12, abs_tol=1e-12)
        except TypeError:
            return False

    @classmethod
    def _calculate_probability(cls, s1, s2):
        """ 精确的概率计算核心 """
        # 计算原始目标位置
        target_pos = sum(1 if c == '+' else -1 for c in s1)
        
        # 解析接收到的指令
        fixed_pos = 0
        unknown_count = 0
        for c in s2:
            if c == '+':
                fixed_pos += 1
            elif c == '-':
                fixed_pos -= 1
            else:
                unknown_count += 1
        
        # 计算需要补偿的位移
        required_offset = target_pos - fixed_pos
        
        # 检查是否可能满足
        if (required_offset + unknown_count) % 2 != 0:
            return 0.0
        if abs(required_offset) > unknown_count:
            return 0.0
        
        # 计算组合数
        k = (required_offset + unknown_count) // 2
        try:
            combinations = math.comb(unknown_count, k)
        except AttributeError:  # 兼容Python <3.10
            combinations = math.factorial(unknown_count) // (
                math.factorial(k) * math.factorial(unknown_count - k))
        
        return combinations / (2 ** unknown_count)
