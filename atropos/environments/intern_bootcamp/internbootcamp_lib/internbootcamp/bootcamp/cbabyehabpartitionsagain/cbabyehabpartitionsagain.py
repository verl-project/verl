"""# 

### 谜题描述
Baby Ehab was toying around with arrays. He has an array a of length n. He defines an array to be good if there's no way to partition it into 2 subsequences such that the sum of the elements in the first is equal to the sum of the elements in the second. Now he wants to remove the minimum number of elements in a so that it becomes a good array. Can you help him?

A sequence b is a subsequence of an array a if b can be obtained from a by deleting some (possibly zero or all) elements. A partitioning of an array is a way to divide it into 2 subsequences such that every element belongs to exactly one subsequence, so you must use all the elements, and you can't share any elements.

Input

The first line contains an integer n (2 ≤ n ≤ 100) — the length of the array a.

The second line contains n integers a_1, a_2, …, a_{n} (1 ≤ a_i ≤ 2000) — the elements of the array a.

Output

The first line should contain the minimum number of elements you need to remove.

The second line should contain the indices of the elements you're removing, separated by spaces.

We can show that an answer always exists. If there are multiple solutions, you can print any.

Examples

Input


4
6 3 9 12


Output


1
2

Input


2
1 2


Output


0

Note

In the first example, you can partition the array into [6,9] and [3,12], so you must remove at least 1 element. Removing 3 is sufficient.

In the second example, the array is already good, so you don't need to remove any elements.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division, print_function

import os
import sys
from io import BytesIO, IOBase
sys.setrecursionlimit(10**5)
if sys.version_info[0] < 3:
    from __builtin__ import xrange as range
    from future_builtins import ascii, filter, hex, map, oct, zip
 
 
def main():
    pass
 
 
# region fastio
 
BUFSIZE = 8192
 
 
class FastIO(IOBase):
    newlines = 0
 
    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = \"x\" in file.mode or \"r\" not in file.mode
        self.write = self.buffer.write if self.writable else None
 
    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()
 
    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b\"\n\") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()
 
    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)
 
 
class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode(\"ascii\"))
        self.read = lambda: self.buffer.read().decode(\"ascii\")
        self.readline = lambda: self.buffer.readline().decode(\"ascii\")
 
 
def print(*args, **kwargs):
    \"\"\"Prints the values to a stream, or to sys.stdout by default.\"\"\"
    sep, file = kwargs.pop(\"sep\", \" \"), kwargs.pop(\"file\", sys.stdout)
    at_start = True
    for x in args:
        if not at_start:
            file.write(sep)
        file.write(str(x))
        at_start = False
    file.write(kwargs.pop(\"end\", \"\n\"))
    if kwargs.pop(\"flush\", False):
        file.flush()
 
 
if sys.version_info[0] < 3:
    sys.stdin, sys.stdout = FastIO(sys.stdin), FastIO(sys.stdout)
else:
    sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)


input = lambda: sys.stdin.readline().rstrip(\"\r\n\")


# import sys
# sys.stdin = open(\"input.txt\", \"r\")
# sys.stdout = open(\"output.txt\", \"w\")
def gcd(x, y):
    \"\"\"greatest common divisor of x and y\"\"\"
    while y:
        x, y = y, x % y
    return x
gcdm = lambda *args: reduce(gcd, args, 0)
def solution():
	n=int(input())
	arr=list(map(int,input().split()))
	sm=sum(arr)
	
	if sm%2!=0:
		print(0)
		return
	else:
		target=sm//2
		dp=[[False for i in range(target+1)] for j in range(n+1)]
 
		for i in range(n+1):
			dp[i][0]=True
			
		for i in range(1,n+1):
			for j in range(1, target+1):
				if arr[i-1]<=j:
					dp[i][j]=dp[i-1][j-arr[i-1]] or dp[i-1][j]
				else:
					dp[i][j]=dp[i-1][j] 
		
		if not dp[-1][-1]:
			print(0)
			return
		for i in range(n):			
			if arr[i]%2!=0:
				print(1)
				print(i+1)
				return
 
		g=gcdm(*arr)
		arr=[ai//g for ai in arr]
		for i in range(n):			
			if arr[i]%2!=0:
				print(1)
				print(i+1)
				return

 
 
	return
 
 
# t=int(input())
# while t:
# 	t-=1
solution()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from math import gcd
from functools import reduce
from bootcamp import Basebootcamp

def solve(n, arr):
    total = sum(arr)
    if total % 2 != 0:
        return (0, [])
    
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    for num in arr:
        for j in range(target, num - 1, -1):
            if dp[j - num]:
                dp[j] = True
    
    if not dp[target]:
        return (0, [])
    
    # Find an odd element
    for idx, num in enumerate(arr):
        if num % 2 != 0:
            return (1, [idx + 1])
    
    # Reduce by GCD
    current_gcd = reduce(gcd, arr)
    normalized = [num // current_gcd for num in arr]
    for idx, num in enumerate(normalized):
        if num % 2 != 0:
            return (1, [idx + 1])
    
    return (0, [])

def is_good_array(arr):
    total = sum(arr)
    if total % 2 != 0:
        return True
    
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    for num in arr:
        for j in range(target, num - 1, -1):
            if dp[j - num]:
                dp[j] = True
    return not dp[target]

class Cbabyehabpartitionsagainbootcamp(Basebootcamp):
    def __init__(self, max_n=100, max_val=2000):
        self.max_n = max_n
        self.max_val = max_val
    
    def case_generator(self):
        while True:
            n = random.randint(2, self.max_n)
            arr = [random.randint(1, self.max_val) for _ in range(n)]
            min_remove, indices = solve(n, arr)
            case = {
                'array': arr.copy(),
                'min_remove': min_remove
            }
            # 验证案例自洽性
            test_arr = [x for i,x in enumerate(arr) if (i+1) not in indices]
            if is_good_array(test_arr) and len(test_arr) == n - min_remove:
                return case
    
    @staticmethod
    def prompt_func(question_case) -> str:
        arr = question_case['array']
        return (
            "As a programming competition participant, determine the minimal elements to remove from "
            f"this array (1-based indices) to make it good:\n\n"
            f"Array length: {len(arr)}\n"
            f"Elements: {' '.join(map(str, arr))}\n\n"
            "Format your answer as:\n[answer]\nk\ni1 i2...ik\n[/answer]\n"
            "Example1: [answer]\\n1\\n3\\n[/answer]\n"
            "Example2 (no removal): [answer]\\n0\\n[/answer]"
        )
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(.*?)\s*\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        
        try:
            lines = [l.strip() for l in matches[-1].split('\n') if l.strip()]
            k = int(lines[0])
            if k < 0:
                return None
            
            if k == 0:
                return (0, []) if len(lines) == 1 else None
            
            if len(lines) >= 2:
                indices = list(map(int, lines[1].split()))
                if len(indices) == k and all(1 <= i <= len(lines[1].split())+k for i in indices):
                    return (k, sorted(indices))
        except:
            pass
        return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        
        k, indices = solution
        if k != identity['min_remove']:
            return False
        
        # Validate indices
        n = len(identity['array'])
        seen = set()
        for idx in indices:
            if not (1 <= idx <= n) or idx in seen:
                return False
            seen.add(idx)
        
        # Verify the remaining array
        remaining = [x for i,x in enumerate(identity['array']) if i+1 not in seen]
        return is_good_array(remaining)

# 关键改进点说明：
# 1. 案例生成增加自洽性检查：确保生成的案例确实满足解的要求
# 2. 完善提取逻辑：严格验证索引范围(1 <= i <= n)
# 3. 增强格式验证：处理带有多余换行符的边缘情况
# 4. 优化提示模板：通过具体示例说明格式要求
