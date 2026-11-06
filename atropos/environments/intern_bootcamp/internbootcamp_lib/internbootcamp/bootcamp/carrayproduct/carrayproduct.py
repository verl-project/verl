"""# 

### 谜题描述
You are given an array a consisting of n integers. You can perform the following operations with it: 

  1. Choose some positions i and j (1 ≤ i, j ≤ n, i ≠ j), write the value of a_i ⋅ a_j into the j-th cell and remove the number from the i-th cell; 
  2. Choose some position i and remove the number from the i-th cell (this operation can be performed no more than once and at any point of time, not necessarily in the beginning). 



The number of elements decreases by one after each operation. However, the indexing of positions stays the same. Deleted numbers can't be used in the later operations.

Your task is to perform exactly n - 1 operations with the array in such a way that the only number that remains in the array is maximum possible. This number can be rather large, so instead of printing it you need to print any sequence of operations which leads to this maximum number. Read the output format to understand what exactly you need to print.

Input

The first line contains a single integer n (2 ≤ n ≤ 2 ⋅ 10^5) — the number of elements in the array.

The second line contains n integers a_1, a_2, ..., a_n (-10^9 ≤ a_i ≤ 10^9) — the elements of the array.

Output

Print n - 1 lines. The k-th line should contain one of the two possible operations.

The operation of the first type should look like this: 1~ i_k~ j_k, where 1 is the type of operation, i_k and j_k are the positions of the chosen elements.

The operation of the second type should look like this: 2~ i_k, where 2 is the type of operation, i_k is the position of the chosen element. Note that there should be no more than one such operation.

If there are multiple possible sequences of operations leading to the maximum number — print any of them.

Examples

Input

5
5 -2 0 1 -3


Output

2 3
1 1 2
1 2 4
1 4 5


Input

5
5 2 0 4 0


Output

1 3 5
2 5
1 1 2
1 2 4


Input

2
2 -1


Output

2 2


Input

4
0 -10 0 0


Output

1 1 2
1 2 3
1 3 4


Input

4
0 0 0 0


Output

1 1 2
1 2 3
1 3 4

Note

Let X be the removed number in the array. Let's take a look at all the examples:

The first example has, for example, the following sequence of transformations of the array: [5, -2, 0, 1, -3] → [5, -2, X, 1, -3] → [X, -10, X, 1, -3] → [X, X, X, -10, -3] → [X, X, X, X, 30]. Thus, the maximum answer is 30. Note, that other sequences that lead to the answer 30 are also correct.

The second example has, for example, the following sequence of transformations of the array: [5, 2, 0, 4, 0] → [5, 2, X, 4, 0] → [5, 2, X, 4, X] → [X, 10, X, 4, X] → [X, X, X, 40, X]. The following answer is also allowed: 
    
    
      
    1 5 3  
    1 4 2  
    1 2 1  
    2 3  
    

Then the sequence of transformations of the array will look like this: [5, 2, 0, 4, 0] → [5, 2, 0, 4, X] → [5, 8, 0, X, X] → [40, X, 0, X, X] → [40, X, X, X, X].

The third example can have the following sequence of transformations of the array: [2, -1] → [2, X].

The fourth example can have the following sequence of transformations of the array: [0, -10, 0, 0] → [X, 0, 0, 0] → [X, X, 0, 0] → [X, X, X, 0].

The fifth example can have the following sequence of transformations of the array: [0, 0, 0, 0] → [X, 0, 0, 0] → [X, X, 0, 0] → [X, X, X, 0].

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
#from random import randint as rn
#from Queue import Queue as Q
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
n=input()
a=ain()
for i in range(n):
    a[i]=[a[i],i]
a.sort()
k1=0
t=0
r=0
h=[]
h1=set()
for i in range(n):
    h1.add(i)
for i in range(n):
    if(a[i][0]<0):
        k1=a[i][1]
        t+=1
    elif(a[i][0]==0):
        r+=1
        h1.remove(a[i][1])
        h.append(a[i][1])
if(t%2==1):
    h1.remove(k1)
    h.append(k1)
h1=list(h1)
ans=[]
if(len(h)>0):
    for i in range(len(h)-1):
        ans.append(\"1 \"+str(h[i]+1)+\" \"+str(h[i+1]+1))
    if(len(h)!=n):
        ans.append(\"2 \"+str(h[-1]+1))
if(len(h1)>0):
    for i in range(len(h1)-1):
        ans.append(\"1 \"+str(h1[i]+1)+\" \"+str(h1[i+1]+1))
stdout.write(\"\n\".join(ans))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Carrayproductbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', 5)
        self.min_val = params.get('min_val', -10)
        self.max_val = params.get('max_val', 10)
        self.zero_prob = params.get('zero_prob', 0.2)
    
    def case_generator(self):
        n = self.n
        # 生成包含不同特征的数组（全正、全负、含零等）
        a = []
        for _ in range(n):
            val = 0
            if random.random() > self.zero_prob:
                val = random.randint(self.min_val, self.max_val)
                # 确保不生成0
                while val == 0:
                    val = random.randint(self.min_val, self.max_val)
            a.append(val)
        return {
            'n': n,
            'array': a.copy(),
            'max_value': self.compute_max_value(a)
        }
    
    @staticmethod
    def prompt_func(question_case):
        array_str = ' '.join(map(str, question_case['array']))
        n = question_case['n']
        return f"""Given array [{array_str}], perform {n-1} operations. 

Rules:
1. Use '1 i j' to multiply a[i] and a[j], store at j, remove i
2. Use '2 i' to remove i (max once)

Output format: {n-1} lines of operations (1-based indices). Enclose answer in [answer][/answer].

Example:
[answer]
1 1 2
2 3
[/answer]"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_answer = matches[-1].strip()
        return [line.strip() for line in last_answer.split('\n') if line.strip()]
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            if len(solution) != identity['n'] - 1:
                return False
            simulated = cls.simulate_operations(identity['array'], solution)
            return simulated == identity['max_value']
        except:
            return False
    
    @staticmethod
    def compute_max_value(arr):
        """精确计算理论最大值"""
        if len(arr) == 0:
            return 0
        
        negatives = [x for x in arr if x < 0]
        zeros = [x for x in arr if x == 0]
        positives = [x for x in arr if x > 0]
        
        # 关键修正：处理仅两个元素的情况
        if len(arr) == 2:
            if (arr[0] * arr[1] >= max(arr)):
                return arr[0] * arr[1]
            else:
                return max(arr)
        
        # 正常情况处理
        if len(negatives) % 2 == 1:
            negatives.remove(max(negatives))  # 移除绝对值最小的负数
        
        product = 1
        for x in positives + negatives:
            product *= x
        return product if (len(positives) + len(negatives)) > 0 else 0
    
    @staticmethod
    def simulate_operations(initial_arr, operations):
        """强化模拟校验"""
        exists = set(range(1, len(initial_arr)+1))
        values = {i: v for i, v in enumerate(initial_arr, 1)}
        op2_used = False
        last_valid_value = None
        
        for op in operations:
            parts = op.split()
            if not parts:
                return None
            
            if parts[0] == '1':
                if len(parts) != 3:
                    return None
                try:
                    i, j = map(int, parts[1:])
                except:
                    return None
                if i not in exists or j not in exists or i == j:
                    return None
                values[j] = values[i] * values[j]
                exists.remove(i)
                del values[i]
                
            elif parts[0] == '2':
                if op2_used or len(parts) != 2:
                    return None
                try:
                    i = int(parts[1])
                except:
                    return None
                if i not in exists:
                    return None
                exists.remove(i)
                del values[i]
                op2_used = True
                
            else:
                return None
        
        return values[next(iter(exists))] if len(exists) == 1 else None
