"""# 

### 谜题描述
This is an easier version of the next problem. In this version, q = 0.

A sequence of integers is called nice if its elements are arranged in blocks like in [3, 3, 3, 4, 1, 1]. Formally, if two elements are equal, everything in between must also be equal.

Let's define difficulty of a sequence as a minimum possible number of elements to change to get a nice sequence. However, if you change at least one element of value x to value y, you must also change all other elements of value x into y as well. For example, for [3, 3, 1, 3, 2, 1, 2] it isn't allowed to change first 1 to 3 and second 1 to 2. You need to leave 1's untouched or change them to the same value.

You are given a sequence of integers a_1, a_2, …, a_n and q updates.

Each update is of form \"i x\" — change a_i to x. Updates are not independent (the change stays for the future).

Print the difficulty of the initial sequence and of the sequence after every update.

Input

The first line contains integers n and q (1 ≤ n ≤ 200 000, q = 0), the length of the sequence and the number of the updates.

The second line contains n integers a_1, a_2, …, a_n (1 ≤ a_i ≤ 200 000), the initial sequence.

Each of the following q lines contains integers i_t and x_t (1 ≤ i_t ≤ n, 1 ≤ x_t ≤ 200 000), the position and the new value for this position.

Output

Print q+1 integers, the answer for the initial sequence and the answer after every update.

Examples

Input


5 0
3 7 3 7 3


Output


2


Input


10 0
1 2 1 2 3 1 1 1 50 1


Output


4


Input


6 0
6 6 3 3 4 4


Output


0


Input


7 0
3 3 1 3 2 1 2


Output


4

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import os
import sys
from atexit import register
from io import BytesIO

sys.stdin = BytesIO(os.read(0, os.fstat(0).st_size))
sys.stdout = BytesIO()
register(lambda: os.write(1, sys.stdout.getvalue()))

input = lambda: sys.stdin.readline().rstrip('\r\n')

n,q = map(int,input().split(\" \"))
arr = map(int,input().split(\" \"))
dic = {}
def add(dic,k,v):
    if not dic.has_key(k):
        dic[k] = v
    else:
        dic[k]+= v
for i,v in enumerate(arr):
    add(dic,v,[i])

nodes = []
for k,v in dic.items():
    nodes.append((v[0],v[-1],k))
nodes.sort()
low,high = -1,-1
sects= []
for fe,le,k in nodes:
    if  fe>low and fe<high:
        high = max(le,high)
    else:
        if low != -1:
            sects.append((low,high))
        low,high = fe,le
sects.append((low,high))
ans = 0
for l,h in sects:
    maxv = 0
    for i in range(l,h+1):
        maxv = max(maxv,len(dic[arr[i]]))
    ans += h+1-l-maxv

print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import defaultdict
from bootcamp import Basebootcamp
import re

class G1intoblockseasyversionbootcamp(Basebootcamp):
    def __init__(self, min_n=5, max_n=10, max_val=50):
        self.min_n = min_n
        self.max_n = max_n
        self.max_val = max_val
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        arr = [random.randint(1, self.max_val) for _ in range(n)]
        correct_answer = self.compute_difficulty(arr)
        return {
            "sequence": arr,
            "correct_answer": correct_answer
        }
    
    @staticmethod
    def compute_difficulty(arr):
        position_map = defaultdict(list)
        for idx, num in enumerate(arr):
            position_map[num].append(idx)
        
        if not position_map:
            return 0

        intervals = []
        for num in position_map:
            indices = position_map[num]
            intervals.append((indices[0], indices[-1], num))
        
        intervals.sort()
        merged = []
        for interval in intervals:
            if not merged:
                merged.append(interval)
            else:
                last_start, last_end, _ = merged[-1]
                curr_start, curr_end, _ = interval
                if curr_start <= last_end:
                    merged[-1] = (last_start, max(last_end, curr_end), None)
                else:
                    merged.append(interval)

        total = 0
        for (l, r, _) in merged:
            freq = defaultdict(int)
            for i in range(l, r+1):
                freq[arr[i]] += 1
            max_freq = max(freq.values())
            total += (r - l + 1) - max_freq
        
        return total
    
    @staticmethod
    def prompt_func(question_case):
        seq = ' '.join(map(str, question_case['sequence']))
        return f"""给定一个整数序列，计算使其成为nice序列的最小修改次数。规则：
1. nice序列要求相同数值必须连续出现
2. 修改时必须将所有相同数值改为同一新值

当前序列：{seq}

请将最终答案放入[answer]标签内，如：[answer]5[/answer]。"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\][\s\n]*(\d+)[\s\n]*\[/answer\]', output, re.IGNORECASE)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
