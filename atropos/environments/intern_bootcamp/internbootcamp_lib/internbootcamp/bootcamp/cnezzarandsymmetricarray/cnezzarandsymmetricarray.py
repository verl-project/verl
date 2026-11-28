"""# 

### 谜题描述
Long time ago there was a symmetric array a_1,a_2,…,a_{2n} consisting of 2n distinct integers. Array a_1,a_2,…,a_{2n} is called symmetric if for each integer 1 ≤ i ≤ 2n, there exists an integer 1 ≤ j ≤ 2n such that a_i = -a_j.

For each integer 1 ≤ i ≤ 2n, Nezzar wrote down an integer d_i equal to the sum of absolute differences from a_i to all integers in a, i. e. d_i = ∑_{j = 1}^{2n} {|a_i - a_j|}.

Now a million years has passed and Nezzar can barely remember the array d and totally forget a. Nezzar wonders if there exists any symmetric array a consisting of 2n distinct integers that generates the array d.

Input

The first line contains a single integer t (1 ≤ t ≤ 10^5) — the number of test cases. 

The first line of each test case contains a single integer n (1 ≤ n ≤ 10^5).

The second line of each test case contains 2n integers d_1, d_2, …, d_{2n} (0 ≤ d_i ≤ 10^{12}).

It is guaranteed that the sum of n over all test cases does not exceed 10^5.

Output

For each test case, print \"YES\" in a single line if there exists a possible array a. Otherwise, print \"NO\".

You can print letters in any case (upper or lower).

Example

Input


6
2
8 12 8 12
2
7 7 9 11
2
7 11 7 11
1
1 1
4
40 56 48 40 80 56 80 48
6
240 154 210 162 174 154 186 240 174 186 162 210


Output


YES
NO
NO
NO
NO
YES

Note

In the first test case, a=[1,-3,-1,3] is one possible symmetric array that generates the array d=[8,12,8,12].

In the second test case, it can be shown that there is no symmetric array consisting of distinct integers that can generate array d.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin

rint = lambda: int(stdin.readline())
rints = lambda: [int(x) for x in stdin.readline().split()]
out = []
for _ in range(int(input())):
    n, a, su, ans = rint(), sorted(rints()), 0, 'yes'

    for i in range(n * 2 - 1, -1, -2):
        if a[i] != a[i - 1] or (i > 1 and a[i] == a[i - 2]) or (a[i] - 2 * su) % (2 * n):
            ans = 'no'
            break
        cur = (a[i] - 2 * su) // (n * 2)
        if cur <= 0:
            ans = 'no'
            break
        su += cur
        n -= 1

    out.append(ans)
print('\n'.join(map(str, out)))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Cnezzarandsymmetricarraybootcamp(Basebootcamp):
    def __init__(self, case_type='mixed', min_n=1, max_n=10, value_range=(1, 10000)):
        self.case_type = case_type
        self.min_n = min_n
        self.max_n = max_n
        self.value_range = value_range

    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        if self.case_type == 'valid':
            return self._generate_valid_case(n)
        elif self.case_type == 'invalid':
            return self._generate_robust_invalid_case(n)
        else:
            return random.choice([self._generate_valid_case(n), self._generate_robust_invalid_case(n)])

    def _generate_valid_case(self, n):
        min_val, max_val = self.value_range
        positives = []
        while len(positives) < n:
            num = random.randint(min_val, max_val)
            if num not in positives:
                positives.append(num)
        
        symmetric_array = []
        for num in positives:
            symmetric_array.extend([num, -num])
        random.shuffle(symmetric_array)
        
        d_array = [sum(abs(num - other) for other in symmetric_array) for num in symmetric_array]
        return {'n': n, 'd': d_array}

    def _generate_robust_invalid_case(self, n, max_attempts=10):
        # 策略1：破坏有效案例的约束条件
        for _ in range(max_attempts):
            valid_case = self._generate_valid_case(n)
            d = valid_case['d'].copy()
            sorted_d = sorted(d)
            
            # 破坏方法1：打破配对约束
            last_pair_index = 2*n - 2
            if sorted_d[last_pair_index] == sorted_d[last_pair_index + 1]:
                sorted_d[-1] += 1
                shuffled = sorted_d.copy()
                random.shuffle(shuffled)
                if self.check_case(n, shuffled) == 'NO':
                    return {'n': n, 'd': shuffled}
            
            # 破坏方法2：修改数值导致余数错误
            target_index = random.choice(range(0, 2*n, 2))
            sorted_d[target_index] += 2*n
            shuffled = sorted_d.copy()
            random.shuffle(shuffled)
            if self.check_case(n, shuffled) == 'NO':
                return {'n': n, 'd': shuffled}

        # 策略2：完全随机生成直至找到无效案例
        for _ in range(max_attempts):
            random_d = [random.randint(0, 10**6) for _ in range(2*n)]
            if self.check_case(n, random_d) == 'NO':
                return {'n': n, 'd': random_d}
        
        # 保底策略：构造必定失败的案例
        return {'n': n, 'd': [0]*(2*n)}

    @staticmethod
    def check_case(n, d_list):
        sorted_d = sorted(d_list)
        su = 0
        current_n = n
        valid = True
        
        if len(sorted_d) != 2*current_n:
            return 'NO'
        
        while current_n > 0 and valid:
            i = 2*current_n - 1
            if i < 1 or sorted_d[i] != sorted_d[i-1]:
                valid = False
                break
            
            if i > 1 and sorted_d[i] == sorted_d[i-2]:
                valid = False
                break
            
            total = sorted_d[i] - 2*su
            if total % (2*current_n) != 0:
                valid = False
                break
            
            cur = total // (2*current_n)
            if cur <= 0:
                valid = False
                break
            
            su += cur
            current_n -= 1
        
        return 'YES' if valid and current_n == 0 else 'NO'

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        d = question_case['d']
        return f"""Determine if a symmetric array exists given the parameters:
- n = {n}
- d = {d}

A symmetric array requires:
1. Exactly 2n distinct integers
2. For each element a, there exists -a
3. Each d_i = sum of absolute differences from a_i to all elements

Output YES or NO within [answer] tags:

[answer]
{{ANSWER}}
[/answer]"""

    @staticmethod
    def extract_output(output):
        markers = ['[answer]', '[/answer]']
        try:
            start = output.rindex(markers[0]) + len(markers[0])
            end = output.index(markers[1], start)
            answer = output[start:end].strip().upper()
            return answer if answer in ('YES', 'NO') else None
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        expected = cls.check_case(identity['n'], identity['d'])
        return solution.upper() == expected
