"""# 

### 谜题描述
Consider the infinite sequence s of positive integers, created by repeating the following steps:

  1. Find the lexicographically smallest triple of positive integers (a, b, c) such that 
    * a ⊕ b ⊕ c = 0, where ⊕ denotes the [bitwise XOR operation](https://en.wikipedia.org/wiki/Bitwise_operation#XOR). 
    * a, b, c are not in s. 
Here triple of integers (a_1, b_1, c_1) is considered to be lexicographically smaller than triple (a_2, b_2, c_2) if sequence [a_1, b_1, c_1] is lexicographically smaller than sequence [a_2, b_2, c_2]. 
  2. Append a, b, c to s in this order. 
  3. Go back to the first step. 



You have integer n. Find the n-th element of s.

You have to answer t independent test cases.

A sequence a is lexicographically smaller than a sequence b if in the first position where a and b differ, the sequence a has a smaller element than the corresponding element in b.

Input

The first line contains a single integer t (1 ≤ t ≤ 10^5) — the number of test cases.

Each of the next t lines contains a single integer n (1≤ n ≤ 10^{16}) — the position of the element you want to know.

Output

In each of the t lines, output the answer to the corresponding test case.

Example

Input


9
1
2
3
4
5
6
7
8
9


Output


1
2
3
4
8
12
5
10
15

Note

The first elements of s are 1, 2, 3, 4, 8, 12, 5, 10, 15, ... 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from datetime import datetime
from random import randint
import sys

__author__ = 'ratmir'

[t] = [int(x) for x in sys.stdin.readline().split()]


def getFirstInTriple(count):
    st2 = 1
    while st2 < 3 * count:
        st2 <<= 2
    st2 >>= 2
    firstInTriple = st2 + count - ((st2 - 1) / 3) - 1
    return firstInTriple


def getValue(position):
    tripleIndex = 1 + (position - 1) / 3
    firstInTriple = getFirstInTriple(tripleIndex)
    if position % 3 == 1:
        return firstInTriple
    if position % 3 == 2:
        value = 1
        res = 0
        while firstInTriple > 0:
            x = firstInTriple & 3
            if x == 1:
                res += value << 1
            if x == 2:
                res += 3 * value
            if x == 3:
                res += value
            value <<= 2
            firstInTriple >>= 2
        return res
    if position % 3 == 0:
        value = 1
        res = 0
        while firstInTriple > 0:
            x = firstInTriple & 3
            if x == 1:
                res += 3 * value
            if x == 3:
                res += value << 1
            if x == 2:
                res += value
            value <<= 2
            firstInTriple >>= 2
        return res

output = \"\"

# start_time = datetime.now()
#
# for j in range(0,10):
#     for i in range(0, 100000):
#         getValue(9999999999900001 + i)
#
# print(datetime.now() - start_time)

for i in range(0, t):
    position = int(sys.stdin.readline())
    output += str(getValue(position)) + \"\n\"

print output
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Cperfecttriplesbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10**6):  # 默认最大值调整为1e6
        self.min_n = min_n
        self.max_n = max_n
    
    def case_generator(self):
        # 生成策略：25%小值，25%中等值，50%参数范围
        rand = random.random()
        if rand < 0.25:
            n = random.randint(1, 10)        # 基础测试样例区
        elif rand < 0.5:
            n = random.randint(100, 10**4)   # 中等规模测试区
        else:
            n = random.randint(self.min_n, self.max_n)
        return {'n': n}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        return f"""Given n = {n}, compute the n-th element in the XOR triple sequence. Rules:
1. Sequence is built by adding lex smallest (a,b,c) with a^b^c=0
2. Each triple's elements are appended in order
3. Sequence starts with 1,2,3,4,8,12,5,10,15...

Put your final answer within [answer] tags like: [answer]42[/answer]"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output, re.IGNORECASE)
        return int(matches[-1]) if matches else None

    @classmethod
    def _get_st2(cls, count):
        """ 优化st2计算：通过位长度快速定位起始点 """
        if count == 0:
            return 0
        target = 3 * count
        bit_len = target.bit_length()
        exponent = (bit_len + 1) // 2  # 4^exponent初始估算
        st2 = 1 << (2 * exponent)
        
        # 精确调整
        while st2 > target:
            exponent -= 1
            st2 >>= 2
        while st2 * 4 <= target:
            st2 <<= 2
        return st2

    @classmethod
    def _getFirstInTriple(cls, count):
        st2 = cls._get_st2(count)
        return st2 + count - (st2 - 1) // 3 - 1

    @classmethod
    def _getValue(cls, position):
        # 保持原算法结构，优化计算效率
        triple_index = (position + 2) // 3
        first = cls._getFirstInTriple(triple_index)
        
        mod = position % 3
        if mod == 1:
            return first
        
        # 公共计算逻辑提取
        res = 0
        value = 1
        f = first
        while f > 0:
            x = f & 3
            if mod == 2:
                res += (value << 1) if x == 1 else (3*value if x ==2 else value)
            else:
                res += (3*value) if x ==1 else (value<<1 if x==3 else value)
            value <<= 2
            f >>= 2
        return res

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return solution == cls._getValue(identity['n'])
        except:
            return False
