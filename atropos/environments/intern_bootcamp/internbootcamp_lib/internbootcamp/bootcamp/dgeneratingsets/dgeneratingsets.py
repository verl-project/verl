"""# 

### 谜题描述
You are given a set Y of n distinct positive integers y1, y2, ..., yn.

Set X of n distinct positive integers x1, x2, ..., xn is said to generate set Y if one can transform X to Y by applying some number of the following two operation to integers in X:

  1. Take any integer xi and multiply it by two, i.e. replace xi with 2·xi. 
  2. Take any integer xi, multiply it by two and add one, i.e. replace xi with 2·xi + 1. 



Note that integers in X are not required to be distinct after each operation.

Two sets of distinct integers X and Y are equal if they are equal as sets. In other words, if we write elements of the sets in the array in the increasing order, these arrays would be equal.

Note, that any set of integers (or its permutation) generates itself.

You are given a set Y and have to find a set X that generates Y and the maximum element of X is mininum possible.

Input

The first line of the input contains a single integer n (1 ≤ n ≤ 50 000) — the number of elements in Y.

The second line contains n integers y1, ..., yn (1 ≤ yi ≤ 109), that are guaranteed to be distinct.

Output

Print n integers — set of distinct integers that generate Y and the maximum element of which is minimum possible. If there are several such sets, print any of them.

Examples

Input

5
1 2 3 4 5


Output

4 5 2 3 1 


Input

6
15 14 3 13 1 12


Output

12 13 14 7 3 1 


Input

6
9 7 13 17 5 11


Output

4 5 2 6 3 1 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys, math
from sys import stdin, stdout
from collections import deque
from copy import deepcopy
from bisect import bisect_right
from heapq import heapify,heappop,heappush,heapreplace
rem = 10 ** 9 + 7
sys.setrecursionlimit(10 ** 6)
take = lambda: map(int, stdin.readline().split())

n=input()
pin=take()
arr=[]
for i in pin:
    arr.append(-i)

heapify(arr)
done={}
for i in arr:
    done[-i]=1
while(len(arr)):
    x=-heappop(arr)
    y=x/2
    yes=False
    while(y>0):
        if done.get(y,-1)==-1:
            done[y]=1
            yes=True
            break
        y/=2
    #print y,x
    if yes==True:
        del done[x]
        heappush(arr,-y)



for i in done:
    print i,
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Dgeneratingsetsbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__()
        self.n = params.get('n', 5)
        self.max_number = params.get('max_number', 100)
    
    def case_generator(self):
        n = self.n
        max_num = self.max_number
        while True:
            X = random.sample(range(1, max_num + 1), n)
            Y = []
            for x in X:
                current = x
                k = random.randint(0, 10)
                for _ in range(k):
                    op = random.choice([0, 1])
                    if op == 0:
                        current = current * 2
                    else:
                        current = current * 2 + 1
                Y.append(current)
            Y = list(set(Y))
            if len(Y) >= n:
                Y = Y[:n]
                break
        Y.sort()
        return {'y': Y}
    
    @staticmethod
    def prompt_func(question_case):
        Y = question_case['y']
        y_str = ' '.join(map(str, Y))
        prompt = f"你是一个数学专家，现在需要解决一个数学谜题。给定一个集合Y={{ {y_str} }}，请找出一个集合X，使得X可以通过一系列操作生成Y。操作包括：将X中的元素乘以2，或者乘以2加1。X中的元素必须互不相同，并且X的最大元素尽可能小。请将X的元素以空格分隔的形式放在[answer]标签中，例如：[answer]4 5 2 3 1[/answer]"
        return prompt
    
    @staticmethod
    def extract_output(output):
        start = output.rfind('[answer]')
        if start == -1:
            return None
        end = output.find('[/answer]', start)
        if end == -1:
            return None
        content = output[start + len('[answer]'):end].strip()
        if not content:
            return None
        try:
            X = list(map(int, content.split()))
            return X
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        Y = identity['y']
        X = solution
        if len(X) != len(Y):
            return False
        if len(set(X)) != len(X):
            return False
        for y in Y:
            current = y
            found = False
            while current > 0:
                if current in X:
                    found = True
                    break
                if current % 2 == 0:
                    current = current // 2
                else:
                    current = (current - 1) // 2
            if not found:
                return False
        return True
