"""# 

### 谜题描述
Petya loves lucky numbers. Everybody knows that lucky numbers are positive integers whose decimal representation contains only the lucky digits 4 and 7. For example, numbers 47, 744, 4 are lucky and 5, 17, 467 are not.

One day Petya dreamt of a lexicographically k-th permutation of integers from 1 to n. Determine how many lucky numbers in the permutation are located on the positions whose indexes are also lucky numbers.

Input

The first line contains two integers n and k (1 ≤ n, k ≤ 109) — the number of elements in the permutation and the lexicographical number of the permutation.

Output

If the k-th permutation of numbers from 1 to n does not exist, print the single number \"-1\" (without the quotes). Otherwise, print the answer to the problem: the number of such indexes i, that i and ai are both lucky numbers.

Examples

Input

7 4


Output

1


Input

4 7


Output

1

Note

A permutation is an ordered set of n elements, where each integer from 1 to n occurs exactly once. The element of permutation in position with index i is denoted as ai (1 ≤ i ≤ n). Permutation a is lexicographically smaller that permutation b if there is such a i (1 ≤ i ≤ n), that ai < bi, and for any j (1 ≤ j < i) aj = bj. Let's make a list of all possible permutations of n elements and sort it in the order of lexicographical increasing. Then the lexicographically k-th permutation is the k-th element of this list of permutations.

In the first sample the permutation looks like that:

1 2 3 4 6 7 5

The only suitable position is 4.

In the second sample the permutation looks like that:

2 1 3 4

The only suitable position is 4.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from math import factorial
import sys

def lucky(x):
	return str(x).count('4')+str(x).count('7')==len(str(x))

def fun(x):
	xs, result = str(x), 0
	for i in range(len(xs)):
		rest = 1<<(len(xs)-i-1)
		if int(xs[i]) == 4:
			continue
		elif int(xs[i]) == 7:
			result += rest
			continue
		elif int(xs[i]) < 4:
			return result
		elif 4 < int(xs[i]) < 7:
			return result + rest
		else:
			return result + rest * 2
	return result + 1

def solve(x):
	now, result = 9, 0
	while now < x:
		result += fun(now)
		now = now*10+9
	return result + fun(x)

n, k = [int(t) for t in raw_input().split()]
m = 1
while factorial(m) < k:
	m += 1
if m > n:
	print -1
	sys.exit(0)
lst = range(n-m+1,n+1)
for i in range(m):
	s = sorted(lst[i:])
	j = 0
	while factorial(m-i-1) < k:
		j += 1
		k -= factorial(m-i-1)
	s[0], s[j] = s[j], s[0]
	lst = lst[:i] + s

#print range(1,n-m+1) + lst
print solve(n-m) + len([1 for a, b in zip(lst, range(n-m+1,n+1)) if lucky(a) and lucky(b)])
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from math import factorial
import random
import re

class Cluckypermutationbootcamp(Basebootcamp):
    def __init__(self, max_m=20):
        self.max_m = max_m  # 控制最大排列长度防止阶乘溢出

    def case_generator(self):
        # 生成有效案例和无效案例的混合
        if random.random() < 0.5:
            # 有效案例：n >= m 且 k <= m!
            m = random.randint(1, min(12, self.max_m))  # 限制m保证阶乘计算不超限
            k = random.randint(1, factorial(m))
            n = random.randint(m, m + 100)
        else:
            # 无效案例：n < m 且 k > (m-1)! 
            m = random.randint(2, min(12, self.max_m))
            k = random.randint(factorial(m-1) + 1, factorial(m)*2)
            n = m - 1
        
        expected = self.calculate_answer(n, k)
        return {'n': n, 'k': k, 'expected': expected}

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k = question_case['k']
        return f"""给定整数n={n}和k={k}，请确定1~n的字典序第k个排列，并统计同时满足以下两个条件的位置数量：
1. 位置索引i是幸运数（由4/7组成的1-based索引）
2. 该位置的元素值a_i也是幸运数

如果第k个排列不存在，输出-1。最终答案放在[answer]标签内，例如：
[answer]0[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(-?\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected']

    @classmethod
    def calculate_answer(cls, n, k):
        # 验证排列是否存在
        m = 1
        while True:
            try:
                if factorial(m) >= k:
                    break
                m += 1
                if m > min(20, n+1):  # 防止无限循环
                    break
            except OverflowError:
                break
        if m > n:
            return -1

        # 生成排列后缀部分
        suffix = list(range(n-m+1, n+1))
        remaining_k = k
        for i in range(m):
            available = sorted(suffix[i:])
            slot_size = factorial(m - i - 1)
            
            # 计算当前块的位置
            pos = 0
            while remaining_k > slot_size:
                remaining_k -= slot_size
                pos += 1
                if pos >= len(available):
                    return -1  # 防止越界

            # 交换元素位置
            available[0], available[pos] = available[pos], available[0]
            # 保持后续元素有序
            suffix = suffix[:i] + available

        # 计算幸运数数量
        count = cls.count_lucky_numbers(n - m)
        
        # 检查后缀部分
        for idx, num in enumerate(suffix, start=n-m+1):
            if cls.is_lucky(idx) and cls.is_lucky(num):
                count += 1
                
        return count

    @staticmethod
    def is_lucky(x):
        return x > 0 and all(c in {'4', '7'} for c in str(x))

    @classmethod
    def count_lucky_numbers(cls, max_num):
        """使用BFS生成所有幸运数"""
        count = 0
        queue = ['4', '7']
        while queue:
            num = queue.pop(0)
            value = int(num)
            if value > max_num:
                continue
            count += 1
            queue.append(num + '4')
            queue.append(num + '7')
        return count
