"""# 

### 谜题描述
Ksenia has an array a consisting of n positive integers a_1, a_2, …, a_n. 

In one operation she can do the following: 

  * choose three distinct indices i, j, k, and then 
  * change all of a_i, a_j, a_k to a_i ⊕ a_j ⊕ a_k simultaneously, where ⊕ denotes the [bitwise XOR operation](https://en.wikipedia.org/wiki/Bitwise_operation#XOR). 



She wants to make all a_i equal in at most n operations, or to determine that it is impossible to do so. She wouldn't ask for your help, but please, help her!

Input

The first line contains one integer n (3 ≤ n ≤ 10^5) — the length of a.

The second line contains n integers, a_1, a_2, …, a_n (1 ≤ a_i ≤ 10^9) — elements of a.

Output

Print YES or NO in the first line depending on whether it is possible to make all elements equal in at most n operations.

If it is possible, print an integer m (0 ≤ m ≤ n), which denotes the number of operations you do.

In each of the next m lines, print three distinct integers i, j, k, representing one operation. 

If there are many such operation sequences possible, print any. Note that you do not have to minimize the number of operations.

Examples

Input


5
4 2 1 7 2


Output


YES
1
1 3 4

Input


4
10 4 49 22


Output


NO

Note

In the first example, the array becomes [4 ⊕ 1 ⊕ 7, 2, 4 ⊕ 1 ⊕ 7, 4 ⊕ 1 ⊕ 7, 2] = [2, 2, 2, 2, 2].

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import print_function # for PyPy2
from collections import Counter, OrderedDict
from itertools import permutations as perm
from collections import deque
from sys import stdin
from bisect import *
from heapq import *
import math
 
g   = lambda : stdin.readline().strip()
gl  = lambda : g().split()
gil = lambda : [int(var) for var in gl()]
gfl = lambda : [float(var) for var in gl()]
gcl = lambda : list(g())
gbs = lambda : [int(var) for var in g()]
mod = int(1e9)+7
inf = float(\"inf\")

n, = gil()
a = gil()
xor = 0 
ans = []
if ~n&1 :
	n-=1
	for v in a:
		xor ^= v
if xor :
	print(\"NO\")
else:
	print(\"YES\")
	print(n-2)
	i = 3
	while i<= n:
		# print(i-2, i-1, i)
		ans.append(str(i-2)+\" \"+str(i-1)+\" \"+str(i)+\"\n\")
		i += 2

	i=n-2
	while i > 1:
		# print(i-2, i-1, i)
		ans.append(str(i-2)+\" \"+str(i-1)+\" \"+str(i)+\"\n\")
		i -= 2
	print(\"\".join(ans))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Dpowerfulkseniabootcamp(Basebootcamp):
    def __init__(self, min_n=3, max_n=10, possible=None):
        self.min_n = max(3, min_n)
        self.max_n = max(self.min_n, max_n)
        self.possible = possible

    def case_generator(self):
        if self.possible is not None:
            should_generate_possible = self.possible
        else:
            should_generate_possible = random.choice([True, False])
        
        if should_generate_possible:
            # 确定可行的生成类型
            generate_odd = False
            possible_odd = [x for x in range(self.min_n, self.max_n+1) if x%2 == 1]
            possible_even = [x for x in range(max(4, self.min_n), self.max_n+1) if x%2 == 0]
            
            # 动态选择生成类型
            if possible_odd and possible_even:
                generate_odd = random.choice([True, False])
            elif possible_odd:
                generate_odd = True
            elif possible_even:
                generate_odd = False
            else:
                raise ValueError("No valid n in given range")
            
            if generate_odd:
                n = random.choice(possible_odd)
                elements = [random.randint(1, 10) for _ in range(n)]
                return {'n': n, 'a': elements}
            else:
                n = random.choice(possible_even)
                elements = [random.randint(1, 10) for _ in range(n-1)]
                total_xor = 0
                for num in elements:
                    total_xor ^= num
                elements.append(total_xor)
                return {'n': n, 'a': elements}
        else:
            # 生成不可能的偶数案例
            possible_even = [x for x in range(max(4, self.min_n), self.max_n+1) if x%2 == 0]
            if not possible_even:
                raise ValueError("No even n in given range for impossible case")
            n = random.choice(possible_even)
            elements = [random.randint(1, 10) for _ in range(n-1)]
            total_xor = 0
            for num in elements:
                total_xor ^= num
            elements.append(total_xor ^ random.randint(1, 10))
            return {'n': n, 'a': elements}

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = question_case['a']
        return (
            f"Problem: Given array of {n} integers: {', '.join(map(str, a))}\n"
            "Operation: Choose 3 distinct indices and set all to their XOR\n"
            "Task: Determine if achievable in ≤n operations\n"
            "Answer format:\n"
            "[answer]\n"
            "YES/NO\n"
            "m\n"
            "i j k\n"
            "...\n"
            "[/answer]\n"
            "Note: Indices must be 1-based"
        )

    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        return answer_blocks[-1].strip()

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            lines = [l.strip() for l in solution.split('\n') if l.strip()]
            if not lines:
                return False
            
            # 验证首行有效性
            first_line = lines[0].upper()
            if first_line not in {'YES', 'NO'}:
                return False
            
            # 获取原始数据
            n = identity['n']
            a = identity['a'].copy()
            total_xor = 0
            for num in a:
                total_xor ^= num
            
            # 校验理论正确性
            expected = 'YES' if (n%2 == 1) or (total_xor == 0) else 'NO'
            if first_line != expected:
                return False
            
            # NO情况直接返回正确
            if expected == 'NO':
                return True
            
            # 验证操作步骤
            if len(lines) < 2:
                return False  # 缺失操作数行
            
            try:
                m = int(lines[1])
                if m < 0 or m > n:
                    return False
            except ValueError:
                return False
            
            # 验证每个操作
            operations = []
            arr = a.copy()
            for line in lines[2:2+m]:
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 3:
                    return False
                try:
                    i, j, k = map(int, parts)
                    if len({i, j, k}) != 3 or any(x < 1 or x > n for x in (i, j, k)):
                        return False
                    # 转换为0-based索引
                    i -= 1
                    j -= 1
                    k -= 1
                    # 执行操作
                    xor_val = arr[i] ^ arr[j] ^ arr[k]
                    arr[i] = arr[j] = arr[k] = xor_val
                except (ValueError, IndexError):
                    return False
            
            # 检查最终统一性
            return all(x == arr[0] for x in arr)
        
        except Exception:
            return False
