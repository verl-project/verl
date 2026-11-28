"""# 

### 谜题描述
<image>

William is a huge fan of planning ahead. That is why he starts his morning routine by creating a nested list of upcoming errands.

A valid nested list is any list which can be created from a list with one item \"1\" by applying some operations. Each operation inserts a new item into the list, on a new line, just after one of existing items a_1  .  a_2  .  a_3  .   ⋅⋅⋅   .\,a_k and can be one of two types: 

  1. Add an item a_1  .  a_2  .  a_3  .  ⋅⋅⋅  .  a_k  .  1 (starting a list of a deeper level), or 
  2. Add an item a_1  .  a_2  .  a_3  .  ⋅⋅⋅  .  (a_k + 1) (continuing the current level). 

Operation can only be applied if the list does not contain two identical items afterwards. And also, if we consider every item as a sequence of numbers, then the sequence of items should always remain increasing in lexicographical order. Examples of valid and invalid lists that are shown in the picture can found in the \"Notes\" section.

When William decided to save a Word document with the list of his errands he accidentally hit a completely different keyboard shortcut from the \"Ctrl-S\" he wanted to hit. It's not known exactly what shortcut he pressed but after triggering it all items in the list were replaced by a single number: the last number originally written in the item number.

William wants you to help him restore a fitting original nested list.

Input

Each test contains multiple test cases. The first line contains the number of test cases t (1 ≤ t ≤ 10). Description of the test cases follows.

The first line of each test case contains a single integer n (1 ≤ n ≤ 10^3), which is the number of lines in the list.

Each of the next n lines contains a single integer a_i (1 ≤ a_i ≤ n), which is what remains of William's nested list.

It is guaranteed that in each test case at least one fitting list exists.

It is guaranteed that the sum of values n across all test cases does not exceed 10^3.

Output

For each test case output n lines which represent a valid nested list, which could become the data provided to you by William.

If there are multiple answers, print any.

Example

Input


2
4
1
1
2
3
9
1
1
1
2
2
1
2
1
2


Output


1
1.1
1.2
1.3
1
1.1
1.1.1
1.1.2
1.2
1.2.1
2
2.1
2.2

Note

In the second example test case one example of a fitting list is:

1

1.1

1.1.1

1.1.2

1.2

1.2.1

2

2.1

2.2

This list can be produced by using the sequence of operations shown below: <image>

  1. Original list with a single item 1. 
  2. Insert item 2 by using the insertion operation of the second type after item 1. 
  3. Insert item 1.1 by using the insertion operation of the first type after item 1. 
  4. Insert item 1.2 by using the insertion operation of the second type after item 1.1. 
  5. Insert item 1.1.1 by using the insertion operation of the first type after item 1.1. 
  6. Insert item 1.1.2 by using the insertion operation of the second type after item 1.1.1. 
  7. Insert item 1.2.1 by using the insertion operation of the first type after item 1.2. 
  8. Insert item 2.1 by using the insertion operation of the first type after item 2. 
  9. Insert item 2.2 by using the insertion operation of the second type after item 2.1. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#python2
#from __future__ import print_function
mod=int(1e9+7)
#import resource
#resource.setrlimit(resource.RLIMIT_STACK, [0x100000000, resource.RLIM_INFINITY])
#import threading
#threading.stack_size(2**25)
import sys
#sys.setrecursionlimit(10**5)
#fact=[1]
#for i in range(1,100001):
#    fact.append((fact[-1]*i)%mod)
#ifact=[0]*100001
#ifact[100000]=pow(fact[100000],mod-2,mod)
#for i in range(100000,0,-1):
#    ifact[i-1]=(i*ifact[i])%mod
from random import randint as rn
import bisect
from bisect import bisect_left as bl              #c++ lowerbound bl(array,element)
from bisect import bisect_right as br             #c++ upperbound
import itertools
import collections
import math
import heapq
from types import GeneratorType
def bootstrap(f, stack=[]):
    def wrappedfunc(*args, **kwargs):
        if stack:
            return f(*args, **kwargs)
        else:
            to = f(*args, **kwargs)
            while True:
                if type(to) is GeneratorType:
                    stack.append(to)
                    to = next(to)
                else:
                    stack.pop()
                    if not stack:
                        break
                    to = stack[-1].send(to)
            return to
    return wrappedfunc
def modinv(n,p):
    return pow(n,p-2,p)
def ncr(n,r,p):                        #for using this uncomment the lines calculating fact and ifact
    t=((fact[n])*((ifact[r]*ifact[n-r])%p))%p
    return t
def GCD(x,y):
    while(y):
        x, y = y, x % y
    return x
def ain():                           #takes array as input
    return list(map(int,sin().split()))
def sin():
    return input().strip()
intmax=10**18
intmin=-intmax
#from itertools import permutations
\"\"\"*******************************************************\"\"\"
def main():
    for _ in range(int(sin())):
        n=int(sin())
        l=[]
        for i in range(n):
            s=int(sin())
            if(s==1):
                l.append(str(\"1\"))
            else:
                w=str(s-1)
                while(l[-1]!=w):
                    l.pop()
                l.pop()
                l.append(str(s))
            print \".\".join(l)

######## Python 2 and 3 footer by Pajenegod and c1729
py2 = round(0.5)
if py2:
    from future_builtins import ascii, filter, hex, map, oct, zip
    range = xrange

import os, sys
from io import IOBase, BytesIO

BUFSIZE = 8192
class FastIO(BytesIO):
    newlines = 0
    def __init__(self, file):
        self._file = file
        self._fd = file.fileno()
        self.writable = \"x\" in file.mode or \"w\" in file.mode
        self.write = super(FastIO, self).write if self.writable else None

    def _fill(self):
        s = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
        self.seek((self.tell(), self.seek(0,2), super(FastIO, self).write(s))[0])
        return s
    def read(self):
        while self._fill(): pass
        return super(FastIO,self).read()

    def readline(self):
        while self.newlines == 0:
            s = self._fill(); self.newlines = s.count(b\"\n\") + (not s)
        self.newlines -= 1
        return super(FastIO, self).readline()

    def flush(self):
        if self.writable:
            os.write(self._fd, self.getvalue())
            self.truncate(0), self.seek(0)

class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        if py2:
            self.write = self.buffer.write
            self.read = self.buffer.read
            self.readline = self.buffer.readline
        else:
            self.write = lambda s:self.buffer.write(s.encode('ascii'))
            self.read = lambda:self.buffer.read().decode('ascii')
            self.readline = lambda:self.buffer.readline().decode('ascii')

sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
input = lambda: sys.stdin.readline().rstrip('\r\n')
if __name__ == '__main__':
   main()
#threading.Thread(target=main).start()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Ccompressionandexpansionbootcamp(Basebootcamp):
    def __init__(self, max_n=1000):
        self.max_n = max_n  # Maximum value of n for generated cases

    def case_generator(self):
        desired_n = random.randint(1, self.max_n)
        items = ["1"]
        for _ in range(desired_n - 1):
            parent = items[-1]
            parts = list(map(int, parent.split('.')))
            op_type = random.choice([1, 2])
            if op_type == 1:
                new_part = parts + [1]
            else:
                if len(parts) == 1:
                    new_part = [parts[0] + 1]
                else:
                    new_part = parts[:-1] + [parts[-1] + 1]
            new_item = '.'.join(map(str, new_part))
            items.append(new_item)
        a = [int(item.split('.')[-1]) for item in items]
        return {
            'n': len(items),
            'a': a
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        case_desc = [
            f"Number of test cases: 1\n\n",
            f"Test case 1:\n",
            f"n = {question_case['n']}\n",
            "Sequence:\n" + '\n'.join(map(str, question_case['a'])) + "\n"
        ]
        prompt = (
            "You are a helpful assistant tasked with solving a nested list restoration puzzle. "
            "William's original nested list was transformed into a sequence of integers, which represent "
            "the last number of each item. Your task is to restore a valid nested list that corresponds to this sequence.\n\n"
            "Rules for constructing the nested list:\n"
            "1. Start with a list containing one item '1'.\n"
            "2. Each subsequent item is inserted after an existing item and can be one of two types:\n"
            "   a. Type 1: Append '.1' to create a deeper nested level.\n"
            "   b. Type 2: Increment the last number by 1 to continue the current level.\n"
            "3. All items must be unique and maintain strictly increasing lexicographical order.\n\n"
            "Given the following input sequence, reconstruct a valid nested list:\n\n"
            f"{''.join(case_desc)}\n"
            "Output your answer with each item on a new line enclosed within [answer] and [/answer] tags.\n"
            "Example format:\n"
            "[answer]\n"
            "1\n"
            "1.1\n"
            "1.2\n"
            "[/answer]"
        )
        return prompt

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_answer = matches[-1].strip()
        lines = [line.strip() for line in last_answer.split('\n') if line.strip()]
        return lines if lines else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or len(solution) != identity['n']:
            return False
        try:
            for i in range(len(solution)):
                parts = solution[i].split('.')
                if int(parts[-1]) != identity['a'][i]:
                    return False
            for i in range(1, len(solution)):
                if solution[i] <= solution[i-1]:
                    return False
            if len(set(solution)) != len(solution):
                return False
            return True
        except (ValueError, IndexError):
            return False
