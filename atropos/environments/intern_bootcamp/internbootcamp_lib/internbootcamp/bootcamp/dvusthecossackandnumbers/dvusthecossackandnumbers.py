"""# 

### 谜题描述
Vus the Cossack has n real numbers a_i. It is known that the sum of all numbers is equal to 0. He wants to choose a sequence b the size of which is n such that the sum of all numbers is 0 and each b_i is either ⌊ a_i ⌋ or ⌈ a_i ⌉. In other words, b_i equals a_i rounded up or down. It is not necessary to round to the nearest integer.

For example, if a = [4.58413, 1.22491, -2.10517, -3.70387], then b can be equal, for example, to [4, 2, -2, -4]. 

Note that if a_i is an integer, then there is no difference between ⌊ a_i ⌋ and ⌈ a_i ⌉, b_i will always be equal to a_i.

Help Vus the Cossack find such sequence!

Input

The first line contains one integer n (1 ≤ n ≤ 10^5) — the number of numbers.

Each of the next n lines contains one real number a_i (|a_i| < 10^5). It is guaranteed that each a_i has exactly 5 digits after the decimal point. It is guaranteed that the sum of all the numbers is equal to 0.

Output

In each of the next n lines, print one integer b_i. For each i, |a_i-b_i|<1 must be met.

If there are multiple answers, print any.

Examples

Input


4
4.58413
1.22491
-2.10517
-3.70387


Output


4
2
-2
-4


Input


5
-6.32509
3.30066
-0.93878
2.00000
1.96321


Output


-6
3
-1
2
2

Note

The first example is explained in the legend.

In the second example, we can round the first and fifth numbers up, and the second and third numbers down. We can round the fourth number neither up, nor down.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#import resource
import sys
#resource.setrlimit(resource.RLIMIT_STACK, [0x100000000, resource.RLIM_INFINITY])
#import threading
#threading.Thread(target=main).start()
#threading.stack_size(2**26)
#sys.setrecursionlimit(10**6)
mod=(10**9)+7
#fact=[1]
#for i in range(1,100001):
#    fact.append((fact[-1]*i)%mod)
#ifact=[0]*100001
#ifact[100000]=pow(fact[100000],mod-2,mod)
#for i in range(100000,0,-1):
#    ifact[i-1]=(i*ifact[i])%mod
from sys import stdin, stdout
import bisect
from bisect import bisect_left as bl
from bisect import bisect_right as br
import itertools
import collections
import math
import heapq
from random import randint as rn
from Queue import Queue as Q
def modinv(n,p):
    return pow(n,p-2,p)
def ncr(n,r,p):
    t=((fact[n])*((ifact[r]*ifact[n-r])%p))%p
    return t
def ain():
    return map(int,sin().split())
def sin():
    return stdin.readline().strip()
def GCD(x,y):
    while(y):
        x, y = y, x % y
    return x
def isprime(x):
    if(x==1):
        return False
    elif(x<4):
        return True
    for i in range(2,int(math.sqrt(x))+1):
        if(x%i==0):
            return False
    return True
\"\"\"**************************************************************************\"\"\"
ans=[]
n=input()
b=[]
x=0
x1=0
y1=0
for i in range(n):
    b.append(float(sin()))
    if(b[-1]>0):
        x+=b[-1]
        x1+=int(b[-1])
    else:
        y1+=int(b[-1])
x=int(math.ceil(x))
x1=x-x1
y1=y1+x
for i in range(n):
    if(b[i]>0):
        if(x1>0 and b[i]!=math.ceil(b[i])):
            x1-=1
            t=math.ceil(b[i])
        else:
            t=math.floor(b[i])
        ans.append(str(int(t)))
    else:
        if(y1>0 and b[i]!=math.floor(b[i])):
            y1-=1
            t=math.floor(b[i])
        else:
            t=math.ceil(b[i])
        ans.append(str(int(t)))
stdout.write(\"\n\".join(ans))
#stdout.write(\"\n\".join(ans))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import random
import re
from bootcamp import Basebootcamp

class Dvusthecossackandnumbersbootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=10, max_int=5, **params):
        super().__init__(**params)
        self.n_min = max(n_min, 1)
        self.n_max = n_max
        self.max_int = max_int  # 控制生成的整数范围

    def case_generator(self):
        """生成满足以下条件的案例：
        1. Σa_i = 0（精确到5位小数）
        2. 所有a_i的小数部分随机分布
        3. 包含至少一个整数a_i（当n>=2时）
        """
        while True:
            n = random.randint(self.n_min, self.n_max)
            if n == 1:
                return {'n': 1, 'a_list': [0.00000]}  # 特例处理
            
            # 生成基础整数集合，确保总和为0
            b = []
            sum_b = 0
            for _ in range(n-1):
                num = random.randint(-self.max_int, self.max_int)
                if num == 0:  # 避免生成过多0
                    num = random.choice([-1, 1])
                b.append(num)
                sum_b += num
            # 添加最后一个数使总和为0
            last_num = -sum_b
            if last_num == 0 and n > 1:  # 避免两个0相邻
                last_num = random.choice([-1, 1])
            b.append(last_num)
            
            # 加入至少一个整数（当n>=2时）
            int_pos = random.randint(0, n-1)
            b[int_pos] = b[int_pos]

            # 生成小数部分
            decimal_parts = []
            positive_deltas = []
            negative_deltas = []
            for num in b:
                if num == 0:
                    decimal = 0.0  # 强制整数
                elif random.random() < 0.2:  # 20%概率生成整数
                    decimal = 0.0
                else:
                    decimal = round(random.uniform(0.00001, 0.99999), 5)
                
                if num > 0:
                    positive_deltas.append(decimal)
                elif num < 0:
                    negative_deltas.append(-decimal)
                decimal_parts.append(decimal)

            # 调整小数部分总和为整数
            total_diff = sum(positive_deltas) + sum(negative_deltas)
            adjust = round(total_diff, 0) - total_diff
            if adjust != 0:
                if positive_deltas:
                    adj_index = random.choice(range(len(positive_deltas)))
                    positive_deltas[adj_index] = round(positive_deltas[adj_index] + adjust, 5)
                elif negative_deltas:
                    adj_index = random.choice(range(len(negative_deltas)))
                    negative_deltas[adj_index] = round(negative_deltas[adj_index] + adjust, 5)

            # 构建最终的a_i列表
            a_list = []
            pos_idx = 0
            neg_idx = 0
            for num in b:
                if num > 0:
                    delta = positive_deltas[pos_idx]
                    pos_idx += 1
                    a = num + delta
                elif num < 0:
                    delta = negative_deltas[neg_idx]
                    neg_idx += 1
                    a = num + delta
                else:
                    a = 0.0
                a_rounded = round(a, 5)
                a_list.append(a_rounded)

            # 最终验证
            if abs(sum(a_list)) < 1e-8:
                return {'n': n, 'a_list': a_list}

    @staticmethod
    def prompt_func(question_case):
        case = question_case
        prompt = (
            "Vus the Cossack需要将以下实数四舍五入为整数，使得总和保持为0。\n"
            "规则：\n"
            "1. 每个数b_i必须是a_i的地板值(floor)或天花板值(ceil)\n"
            "2. 最终Σb_i必须等于0\n"
            "3. |a_i - b_i| < 1 必须成立\n"
            f"输入：\n{case['n']}\n" +
            "\n".join(f"{a:.5f}" for a in case['a_list']) + 
            "\n输出：每行一个整数，包含在[answer]标签内\n"
            "示例:\n[answer]\n3\n-1\n-2\n0\n[/answer]"
        )
        return prompt

    @staticmethod
    def extract_output(output):
        # 匹配最后一个[answer]块内的所有整数
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            # 允许换行符或空格分隔
            solution = list(map(int, re.split(r'[\n\s]+', last_match)))
            return solution
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 基础检查
        if not solution:
            return False
        if len(solution) != identity['n']:
            return False
        
        total = 0
        for b, a in zip(solution, identity['a_list']):
            floor = math.floor(a)
            ceil = math.ceil(a)
            
            # 检查是否有效舍入
            if b not in {floor, ceil}:
                return False
            
            # 检查数学约束
            if not (abs(a - b) < 1 - 1e-8):  # 处理浮点精度
                return False
            
            total += b
        
        return abs(total) < 1e-8  # 允许浮点误差
