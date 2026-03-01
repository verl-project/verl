"""# 

### 谜题描述
Andrey thinks he is truly a successful developer, but in reality he didn't know about the binary search algorithm until recently. After reading some literature Andrey understood that this algorithm allows to quickly find a certain number x in an array. For an array a indexed from zero, and an integer x the pseudocode of the algorithm is as follows:

<image>

Note that the elements of the array are indexed from zero, and the division is done in integers (rounding down).

Andrey read that the algorithm only works if the array is sorted. However, he found this statement untrue, because there certainly exist unsorted arrays for which the algorithm find x!

Andrey wants to write a letter to the book authors, but before doing that he must consider the permutations of size n such that the algorithm finds x in them. A permutation of size n is an array consisting of n distinct integers between 1 and n in arbitrary order.

Help Andrey and find the number of permutations of size n which contain x at position pos and for which the given implementation of the binary search algorithm finds x (returns true). As the result may be extremely large, print the remainder of its division by 10^9+7.

Input

The only line of input contains integers n, x and pos (1 ≤ x ≤ n ≤ 1000, 0 ≤ pos ≤ n - 1) — the required length of the permutation, the number to search, and the required position of that number, respectively.

Output

Print a single number — the remainder of the division of the number of valid permutations by 10^9+7.

Examples

Input


4 1 2


Output


6


Input


123 42 24


Output


824071958

Note

All possible permutations in the first test case: (2, 3, 1, 4), (2, 4, 1, 3), (3, 2, 1, 4), (3, 4, 1, 2), (4, 2, 1, 3), (4, 3, 1, 2).

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
'''input
123 42 24


'''

from bisect import bisect_right as bl
from random import randint as R
RI = lambda : [int(_x) for _x in raw_input().split()]
mod = 10**9+7
F = [1]*(1010)
for i in range(1009):
	F[i+1]=(F[i]*(i+1))%mod

n,x,p = RI()

l = 0
r = n


big = n-x
small = x-1
ans = 1

while l < r:
	mid = (l+r)/2
	if mid == p:
		l = mid + 1

		continue
	elif mid < p:
		ans = (ans*small)%mod
		small -= 1
		l = mid + 1
	else:
		ans = (ans*big)%mod
		big -= 1
		r = mid


ans = (ans*F[(big+small)])%mod

print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

MOD = 10**9 + 7

class Cbinarysearchbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__()
        self.mod = MOD
        # 预先计算阶乘数组，最多到2000
        self.max_fact = 2000
        self.fact = [1] * (self.max_fact + 1)
        for i in range(1, self.max_fact + 1):
            self.fact[i] = (self.fact[i-1] * i) % self.mod

    def compute_answer(self, n, x, pos):
        l = 0
        r = n
        big = n - x
        small = x - 1
        ans = 1
        while l < r:
            mid = (l + r) // 2
            if mid == pos:
                l = mid + 1
                continue
            elif mid < pos:
                ans = (ans * small) % self.mod
                small -= 1
                l = mid + 1
            else:
                ans = (ans * big) % self.mod
                big -= 1
                r = mid
        remaining = big + small
        if remaining >= 0:
            ans = (ans * self.fact[remaining]) % self.mod
        else:
            ans = 0
        return ans

    def case_generator(self):
        # Ensure x is between 1 and n, and pos is between 0 and n-1
        n = random.randint(1, 1000)
        x = random.randint(1, n)
        pos = random.randint(0, n - 1)
        correct_answer = self.compute_answer(n, x, pos)
        return {
            'n': n,
            'x': x,
            'pos': pos,
            'correct_answer': correct_answer
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        x = question_case['x']
        pos = question_case['pos']
        prompt = f"给定n={n}, x={x}, pos={pos}。请计算满足以下条件的排列数目，结果模10^9+7。\n"
        prompt += "条件如下：\n"
        prompt += "1. 排列包含1到n的所有整数，每个整数恰好出现一次。\n"
        prompt += "2. 数字x位于位置pos（数组索引从0开始）。\n"
        prompt += "3. 使用以下二分查找实现能够找到x并返回True。\n"
        prompt += "二分查找的伪代码如下：\n"
        prompt += "函数binary_search(a, x):\n"
        prompt += "    l = 0\n"
        prompt += "    r = n\n"
        prompt += "    while l < r:\n"
        prompt += "        mid = (l + r) // 2\n"
        prompt += "        if a[mid] <= x:\n"
        prompt += "            l = mid + 1\n"
        prompt += "        else:\n"
        prompt += "            r = mid\n"
        prompt += "    return a[l-1] == x\n"
        prompt += "请将答案放在[answer]和[/answer]标签之间，例如：\n"
        prompt += "答案：[answer]数值[/answer]"
        return prompt

    @staticmethod
    def extract_output(output):
        pattern = r'\[answer\](\d+)\[/answer\]'
        matches = re.findall(pattern, output)
        if matches:
            return matches[-1]  # 返回最后一个匹配的结果
        else:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False
        expected = identity['correct_answer']
        try:
            solution_int = int(solution)
            return solution_int == expected
        except (ValueError, TypeError):
            return False
