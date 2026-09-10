"""# 

### 谜题描述
Polycarpus is an amateur programmer. Now he is analyzing a friend's program. He has already found there the function rangeIncrement(l, r), that adds 1 to each element of some array a for all indexes in the segment [l, r]. In other words, this function does the following: 
    
    
      
    function rangeIncrement(l, r)  
        for i := l .. r do  
            a[i] = a[i] + 1  
    

Polycarpus knows the state of the array a after a series of function calls. He wants to determine the minimum number of function calls that lead to such state. In addition, he wants to find what function calls are needed in this case. It is guaranteed that the required number of calls does not exceed 105.

Before calls of function rangeIncrement(l, r) all array elements equal zero.

Input

The first input line contains a single integer n (1 ≤ n ≤ 105) — the length of the array a[1... n]. 

The second line contains its integer space-separated elements, a[1], a[2], ..., a[n] (0 ≤ a[i] ≤ 105) after some series of function calls rangeIncrement(l, r). 

It is guaranteed that at least one element of the array is positive. It is guaranteed that the answer contains no more than 105 calls of function rangeIncrement(l, r).

Output

Print on the first line t — the minimum number of calls of function rangeIncrement(l, r), that lead to the array from the input data. It is guaranteed that this number will turn out not more than 105.

Then print t lines — the descriptions of function calls, one per line. Each line should contain two integers li, ri (1 ≤ li ≤ ri ≤ n) — the arguments of the i-th call rangeIncrement(l, r). Calls can be applied in any order.

If there are multiple solutions, you are allowed to print any of them.

Examples

Input

6
1 2 1 1 4 1


Output

5
2 2
5 5
5 5
5 5
1 6


Input

5
1 0 1 0 1


Output

3
1 1
3 3
5 5

Note

The first sample requires a call for the entire array, and four additional calls:

  * one for the segment [2,2] (i.e. the second element of the array), 
  * three for the segment [5,5] (i.e. the fifth element of the array). 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = int(raw_input())
l = raw_input().split()
ans = []
for i in range(n):
    l[i] = int(l[i])
s = []; opened = []
for i in range(n):
    if not len(s) or l[i] > s[-1]:
        s.append(l[i])
        opened.append(i + 1)
    elif l[i] < s[-1]:
        while s and l[i] < s[-1]:
            pp = True
            base = l[i]
            if len(s) > 1:
                base = max(base, s[-2])
            if base == l[i]:
                pp = False
            val = s[-1] - base
            while val:
                ans.append( str(opened[-1]) + \" \" + str(i))
                val -= 1
            if pp:
                s.pop(); opened.pop()
            else:
                break
        if s:
            s[-1] = l[i]
while s:
    base = 0
    if len(s) > 1:
        base = s[-2]
    val = s[-1] - base
    while val:
        ans.append( str(opened[-1]) + \" \" + str(n))
        val -= 1
    s.pop(); opened.pop()
print len(ans)
for s in ans:
    print s
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

def solve(a_list):
    n = len(a_list)
    l = a_list.copy()
    ans = []
    s = []
    opened = []
    for i in range(n):
        current = l[i]
        if not s or current > s[-1]:
            s.append(current)
            opened.append(i + 1)
        elif current < s[-1]:
            while s and current < s[-1]:
                pp = True
                base = current
                if len(s) > 1:
                    base = max(base, s[-2])
                if base == current:
                    pp = False
                val = s[-1] - base
                while val > 0:
                    ans.append(f"{opened[-1]} {i}")
                    val -= 1
                if pp:
                    s.pop()
                    opened.pop()
                else:
                    break
            if s:
                s[-1] = current
    while s:
        base = 0
        if len(s) > 1:
            base = s[-2]
        val = s[-1] - base
        while val > 0:
            ans.append(f"{opened[-1]} {n}")
            val -= 1
        s.pop()
        opened.pop()
    operations = []
    for op in ans:
        li, ri = map(int, op.split())
        operations.append((li, ri))
    return operations

class Crangeincrementsbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10, max_val=5):
        self.min_n = min_n
        self.max_n = max_n
        self.max_val = max_val
    
    def case_generator(self):
        while True:
            n = random.randint(self.min_n, self.max_n)
            a = [random.randint(0, self.max_val) for _ in range(n)]
            if sum(a) == 0:  # 确保至少一个正元素
                continue
            try:
                correct_ops = solve(a)
            except:
                continue  # 防止极端情况异常
            if len(correct_ops) > 1e5:  # 题目保证答案次数不超过1e5
                continue
            return {
                'n': n,
                'a': a,
                'correct_t': len(correct_ops),
                'correct_ops': correct_ops
            }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = ' '.join(map(str, question_case['a']))
        prompt = f"""Polycarpus需要确定最少的函数调用次数，使得初始全为0的数组变成给定的状态。函数rangeIncrement(l, r)每次将下标l到r的元素加1。你需要解决这个问题。

输入格式：
第一行是整数n，表示数组长度。
第二行是n个整数，表示数组的最终状态。

当前的问题实例：
输入的第一行：{n}
输入的第二行：{a}

请你输出最少的调用次数t，并给出每次调用的l和r参数。可能有多个正确答案，只要满足次数最少即可。

请将答案按照以下格式输出，答案必须包含在[answer]和[/answer]之间：

[answer]
t
l1 r1
l2 r2
...
lt rt
[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        import re
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        last_block = answer_blocks[-1].strip()
        lines = [line.strip() for line in last_block.split('\n') if line.strip()]
        if len(lines) < 1:
            return None
        try:
            t = int(lines[0])
            if len(lines) != t + 1:
                return None
            operations = []
            for line in lines[1:t+1]:  # 防止多余行干扰
                parts = line.split()
                if len(parts) != 2:
                    return None
                l = int(parts[0])
                r = int(parts[1])
                operations.append( (l, r) )
            return operations
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False
        # 检查操作次数是否匹配最优解
        if len(solution) != identity['correct_t']:
            return False
        n = identity['n']
        a = identity['a']
        # 模拟操作过程
        simulated = [0] * n
        for l, r in solution:
            if l < 1 or r > n or l > r:  # 参数合法性检查
                return False
            start = l - 1
            end = r - 1
            for i in range(start, end + 1):
                simulated[i] += 1
        return simulated == a
