"""# 

### 谜题描述
Vasily the bear has got a sequence of positive integers a1, a2, ..., an. Vasily the Bear wants to write out several numbers on a piece of paper so that the beauty of the numbers he wrote out was maximum. 

The beauty of the written out numbers b1, b2, ..., bk is such maximum non-negative integer v, that number b1 and b2 and ... and bk is divisible by number 2v without a remainder. If such number v doesn't exist (that is, for any non-negative integer v, number b1 and b2 and ... and bk is divisible by 2v without a remainder), the beauty of the written out numbers equals -1. 

Tell the bear which numbers he should write out so that the beauty of the written out numbers is maximum. If there are multiple ways to write out the numbers, you need to choose the one where the bear writes out as many numbers as possible.

Here expression x and y means applying the bitwise AND operation to numbers x and y. In programming languages C++ and Java this operation is represented by \"&\", in Pascal — by \"and\".

Input

The first line contains integer n (1 ≤ n ≤ 105). The second line contains n space-separated integers a1, a2, ..., an (1 ≤ a1 < a2 < ... < an ≤ 109).

Output

In the first line print a single integer k (k > 0), showing how many numbers to write out. In the second line print k integers b1, b2, ..., bk — the numbers to write out. You are allowed to print numbers b1, b2, ..., bk in any order, but all of them must be distinct. If there are multiple ways to write out the numbers, choose the one with the maximum number of numbers to write out. If there still are multiple ways, you are allowed to print any of them.

Examples

Input

5
1 2 3 4 5


Output

2
4 5


Input

3
1 2 4


Output

1
4

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def main():
	n = input()
	a = map(int, raw_input().split())
	for i in xrange(30, -1, -1):
		f = []
		for j in a:
			if j & (1 << i) > 0: f.append(j)
		if f:
			ans = f[0]
			for j in xrange(1, len(f)): ans &= f[j]
			if ans != 0 and 0 == ans % (1 << i):
				print len(f)
				print \" \".join(map(str, f))
				break

if __name__ == \"__main__\": main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cvasilythebearandsequencebootcamp(Basebootcamp):
    def __init__(self, min_v=1, max_v=30, max_k=5):
        self.min_v = min_v
        self.max_v = max_v
        self.max_k = max_k
    
    def case_generator(self):
        v_max = random.randint(self.min_v, self.max_v)
        base = 1 << v_max  # 2^v_max

        # Generate optimal subset (S_numbers)
        s_numbers = []
        current = base
        s_numbers.append(current)
        remaining = self.max_k - 1  # Already added the first number
        
        # Generate up to max_k numbers in [base, 2*base-1] range
        while remaining > 0 and current < (base << 1) - 1:
            next_num = random.randint(current + 1, (base << 1) - 1)
            s_numbers.append(next_num)
            current = next_num
            remaining -= 1

        # Generate lower numbers (if base allows)
        lower_count = random.randint(0, self.max_k)
        lower_numbers = []
        if base > 1 and lower_count > 0:
            available = list(range(1, base))
            if available:
                lower_numbers = sorted(random.sample(available, k=min(lower_count, len(available))))
        
        # Combine and sort the array
        a = sorted(lower_numbers + s_numbers)
        if not a:  # Fallback if empty (impossible due to s_numbers)
            a = [base]
            s_numbers = [base]

        return {
            'n': len(a),
            'a': a,
            'v_max': v_max,
            'base': base,
            's_count': len(s_numbers)  # Optimal subset size
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        a = question_case['a']
        problem_text = f"""Vasily the bear has a sequence of strictly increasing positive integers. Your task is to select a subset of these numbers such that the beauty of the subset is maximized. The beauty is defined as the maximum non-negative integer v for which the bitwise AND of all selected numbers is divisible by 2^v. If no such v exists (i.e., the bitwise AND is zero), the beauty is -1. Among all possible subsets with maximum beauty, you must choose the one with the largest possible size k. If there are multiple such subsets, any is acceptable.

Input:
- The first line contains an integer n ({n} in this case).
- The second line contains {n} strictly increasing integers: {' '.join(map(str, a))}.

Your output should be two lines:
- The first line is the integer k.
- The second line contains the selected numbers in any order.

Please provide your answer within [answer] and [/answer] tags. For example:

[answer]
2
4 5
[/answer]"""
        return problem_text
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        answer_str = matches[-1].strip()
        lines = [line.strip() for line in answer_str.split('\n') if line.strip()]
        if len(lines) < 2:
            return None
        try:
            k = int(lines[0])
            numbers = list(map(int, lines[1].split()))
            if len(numbers) != k or len(set(numbers)) != k:
                return None
            return {'k': k, 'numbers': sorted(numbers)}
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False
        
        a = identity['a']
        v_max = identity['v_max']
        base = identity['base']
        s_count = identity['s_count']
        sol_k = solution.get('k')
        sol_numbers = solution.get('numbers', [])
        
        # Check k matches optimal subset size
        if sol_k != s_count:
            return False
        
        # Validate numbers exist in array
        if len(sol_numbers) != sol_k or len(set(sol_numbers)) != sol_k:
            return False
        for num in sol_numbers:
            if num not in a:
                return False
        
        # Calculate AND of solution
        current_and = sol_numbers[0]
        for num in sol_numbers[1:]:
            current_and &= num
        
        # Verify AND equals base and v_max
        if current_and != base:
            return False
        
        calculated_v = 0
        temp = current_and
        while temp % 2 == 0 and temp > 0:
            calculated_v += 1
            temp >>= 1
        
        return calculated_v == v_max
