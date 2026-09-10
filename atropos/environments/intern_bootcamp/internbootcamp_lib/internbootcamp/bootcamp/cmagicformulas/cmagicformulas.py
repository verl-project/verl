"""# 

### 谜题描述
People in the Tomskaya region like magic formulas very much. You can see some of them below.

Imagine you are given a sequence of positive integer numbers p1, p2, ..., pn. Lets write down some magic formulas:

<image><image>

Here, \"mod\" means the operation of taking the residue after dividing.

The expression <image> means applying the bitwise xor (excluding \"OR\") operation to integers x and y. The given operation exists in all modern programming languages. For example, in languages C++ and Java it is represented by \"^\", in Pascal — by \"xor\".

People in the Tomskaya region like magic formulas very much, but they don't like to calculate them! Therefore you are given the sequence p, calculate the value of Q.

Input

The first line of the input contains the only integer n (1 ≤ n ≤ 106). The next line contains n integers: p1, p2, ..., pn (0 ≤ pi ≤ 2·109).

Output

The only line of output should contain a single integer — the value of Q.

Examples

Input

3
1 2 3


Output

3

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
'''
Created on Apr 25, 2014

@author: szalivako
'''

def sumX(n):
    if (n % 4 == 0):
        return n
    if (n % 4 == 1):
        return 1
    if (n % 4 == 2):
        return n + 1
    if (n % 4 == 3):
        return 0
    return 0

n = int(raw_input())
p = map(int, raw_input().split())

Q = 0
for i in range(n):
    Q ^= p[i]
if ((n - 1) % 2 == 1):
    Q ^= 1

for i in range(2, n + 1):
    z = (n - i + 1) // i
    if (z % 2 == 1):
        Q ^= sumX(i - 1)
    r = (n - i + 1) % i
    if (r > 1):
        Q ^= sumX(r - 1)
    if ((n - i) % 2 == 1):
        Q ^= i

print Q
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cmagicformulasbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10, max_p=100):
        self.min_n = min_n
        self.max_n = max_n
        self.max_p = max_p
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        p = [random.randint(0, self.max_p) for _ in range(n)]
        return {'n': n, 'p': p}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        p = question_case['p']
        problem = (
            "你被给定一个整数序列，需要计算魔法公式Q的值。规则如下：\n\n"
            "1. 初始Q为所有数的异或结果。\n"
            "2. 若n-1为奇数，Q异或1。\n"
            "3. 对每个i从2到n：\n"
            "   a) 计算z=(n-i+1)//i，若z为奇数，Q异或sumX(i-1)。\n"
            "   b) 计算余数r=(n-i+1)%i，若r>1，Q异或sumX(r-1)。\n"
            "   c) 若(n-i)为奇数，Q异或i。\n\n"
            "sumX(k)的定义：\n"
            "- k%4==0: sumX(k)=k\n"
            "- k%4==1: sumX(k)=1\n"
            "- k%4==2: sumX(k)=k+1\n"
            "- k%4==3: sumX(k)=0\n\n"
            "输入参数：\n"
            f"n = {n}\n"
            f"p = {p}\n\n"
            "请计算Q的值，并将答案放在[answer]和[/answer]之间。"
        )
        return problem
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_answer = matches[-1].strip()
        try:
            return int(last_answer)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        p = identity['p']
        
        Q = 0
        for num in p:
            Q ^= num
        
        if (n - 1) % 2 == 1:
            Q ^= 1
        
        for i in range(2, n + 1):
            z = (n - i + 1) // i
            if z % 2 == 1:
                Q ^= cls.sumX(i - 1)
            
            r = (n - i + 1) % i
            if r > 1:
                Q ^= cls.sumX(r - 1)
            
            if (n - i) % 2 == 1:
                Q ^= i
        
        return Q == solution
    
    @staticmethod
    def sumX(k):
        mod = k % 4
        if mod == 0:
            return k
        elif mod == 1:
            return 1
        elif mod == 2:
            return k + 1
        else:  # mod == 3
            return 0
