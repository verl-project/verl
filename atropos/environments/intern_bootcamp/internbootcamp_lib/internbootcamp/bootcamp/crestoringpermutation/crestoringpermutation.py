"""# 

### 谜题描述
You are given a sequence b_1, b_2, …, b_n. Find the lexicographically minimal permutation a_1, a_2, …, a_{2n} such that b_i = min(a_{2i-1}, a_{2i}), or determine that it is impossible.

Input

Each test contains one or more test cases. The first line contains the number of test cases t (1 ≤ t ≤ 100).

The first line of each test case consists of one integer n — the number of elements in the sequence b (1 ≤ n ≤ 100).

The second line of each test case consists of n different integers b_1, …, b_n — elements of the sequence b (1 ≤ b_i ≤ 2n).

It is guaranteed that the sum of n by all test cases doesn't exceed 100.

Output

For each test case, if there is no appropriate permutation, print one number -1.

Otherwise, print 2n integers a_1, …, a_{2n} — required lexicographically minimal permutation of numbers from 1 to 2n.

Example

Input


5
1
1
2
4 1
3
4 1 3
4
2 3 4 5
5
1 5 7 2 8


Output


1 2 
-1
4 5 1 2 3 6 
-1
1 3 5 6 7 9 2 4 8 10 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from bisect import bisect
for _ in range(input()):
    n = input()
    a = map(int,raw_input().split())
    if 1 not in a or 2*n in a:
        print -1
        continue
    l = []
    for i in range(1,2*n+1):
        if i not in a:
            l.append(i)
    d = {}
    f = 0
    for i in range(n):
        p = bisect(l,a[i])
        if p==len(l):
            f = 1
            break
        d[a[i]] = l[p]
        l.remove(l[p])
    if f==1:
        print -1
        continue
    for i in range(n):
        print a[i],d[a[i]],
    print
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bisect import bisect_right as bisect
import re
from bootcamp import Basebootcamp

class Crestoringpermutationbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=5, unsolvable_prob=0.2):
        self.min_n = min_n
        self.max_n = max_n
        self.unsolvable_prob = unsolvable_prob
    
    def case_generator(self):
        if random.random() < self.unsolvable_prob:
            n = random.randint(self.min_n, self.max_n)
            case_type = random.choice([1, 2])
            b = []
            if case_type == 1:
                b = [2 * n]
                remaining = list(range(1, 2 * n))
                if n > 1:
                    others = random.sample(remaining, n-1)
                    b.extend(others)
            else:
                possible = list(range(2, 2 * n + 1))
                b = random.sample(possible, k=n)
            random.shuffle(b)
            return {
                'n': n,
                'b': b,
                'expected': -1
            }
        else:
            while True:
                n = random.randint(self.min_n, self.max_n)
                possible_values = list(range(1, 2 * n + 1))
                if 2 * n in possible_values:
                    possible_values.remove(2 * n)
                if 1 not in possible_values:
                    continue
                b = [1]
                if n > 1:
                    remaining = possible_values.copy()
                    remaining.remove(1)
                    others = random.sample(remaining, n-1)
                    b.extend(others)
                if len(set(b)) != n or 2 * n in b or 1 not in b:
                    continue
                sorted_b = sorted(b)
                l = sorted([num for num in range(1, 2 * n + 1) if num not in sorted_b])
                d = {}
                f = 0
                for bi in sorted_b:
                    pos = bisect(l, bi)
                    if pos >= len(l):
                        f = 1
                        break
                    selected = l[pos]
                    d[bi] = selected
                    del l[pos]
                if f:
                    continue
                a = []
                for num in sorted_b:
                    a.append(num)
                    a.append(d[num])
                return {
                    'n': n,
                    'b': b,
                    'expected': a
                }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        b = question_case['b']
        prompt = f"""You are given a sequence b of length {n}. Find the lexicographically smallest permutation a of 1 to {2*n} such that for each i (1 ≤ i ≤ {n}), b[i] is the minimum of a[2i-1] and a[2i]. If impossible, output -1.

Input:
n = {n}
b = {b}

Format your answer as space-separated numbers within [answer] tags. Example:
[answer]1 2 3 4[/answer] or [answer]-1[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        pattern = r'\[answer\](.*?)\[/answer\]'
        matches = re.findall(pattern, output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        if last_match == '-1':
            return -1
        try:
            return list(map(int, last_match.split()))
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        b = identity['b']
        expected = identity.get('expected')
        if solution == -1:
            return expected == -1
        if not isinstance(solution, list) or len(solution) != 2 * n:
            return False
        if set(solution) != set(range(1, 2 * n + 1)):
            return False
        for i in range(n):
            if min(solution[2*i], solution[2*i+1]) != b[i]:
                return False
        if expected != -1:
            return solution == expected
        return False
