"""# 

### 谜题描述
You found a useless array a of 2n positive integers. You have realized that you actually don't need this array, so you decided to throw out all elements of a.

It could have been an easy task, but it turned out that you should follow some rules: 

  1. In the beginning, you select any positive integer x.
  2. Then you do the following operation n times: 
    * select two elements of array with sum equals x; 
    * remove them from a and replace x with maximum of that two numbers. 



For example, if initially a = [3, 5, 1, 2], you can select x = 6. Then you can select the second and the third elements of a with sum 5 + 1 = 6 and throw them out. After this operation, x equals 5 and there are two elements in array: 3 and 2. You can throw them out on the next operation.

Note, that you choose x before the start and can't change it as you want between the operations.

Determine how should you behave to throw out all elements of a.

Input

The first line contains a single integer t (1 ≤ t ≤ 1000) — the number of test cases.

The first line of each test case contains the single integer n (1 ≤ n ≤ 1000).

The second line of each test case contains 2n integers a_1, a_2, ..., a_{2n} (1 ≤ a_i ≤ 10^6) — the initial array a.

It is guaranteed that the total sum of n over all test cases doesn't exceed 1000.

Output

For each test case in the first line print YES if it is possible to throw out all elements of the array and NO otherwise.

If it is possible to throw out all elements, print the initial value of x you've chosen. Print description of n operations next. For each operation, print the pair of integers you remove.

Example

Input


4
2
3 5 1 2
3
1 1 8 8 64 64
2
1 1 2 4
5
1 2 3 4 5 6 7 14 3 11


Output


YES
6
1 5
2 3
NO
NO
YES
21
14 7
3 11
5 6
2 4
3 1

Note

The first test case was described in the statement.

In the second and third test cases, we can show that it is impossible to throw out all elements of array a.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from heapq import heappop, heappush, heapify
def solve():
    n = int(raw_input())
    a = map(int, raw_input().split())
    a.sort(reverse=True)
    d = {a[0]: 0}
    m = 2 * n
    for i in xrange(1, m):
        if a[i-1] > a[i]:
            d[a[i]] = i
    for i in xrange(1, m):
        if i > 1 and a[i-1] == a[i]:
            continue
        p = d.copy()
        s = a[0] + a[i]
        b = [0] * m
        ans = []
        k = 0
        for j in xrange(n):
            while k < m and b[k]:
                k += 1
            if k >= m:
                break
            x = a[k]
            p[x] += 1
            y = s - a[k]
            if x < y or y not in p:
                break
            l = p[y]
            if l >= m or a[l] != y:
                break
            b[l] = 1
            p[y] += 1
            ans.append((x, y))
            s = x
            k += 1
        else:
            print \"YES\"
            print a[0] + a[i]
            for x, y in ans:
                print x, y
            return
    print \"NO\"
            
T = int(raw_input())
for t in xrange(T):
    solve()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import defaultdict

class Carraydestructionbootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=3, elem_min=1, elem_max=10, solvable_prob=0.5):
        self.n_min = n_min
        self.n_max = n_max
        self.elem_min = elem_min
        self.elem_max = elem_max
        self.solvable_prob = solvable_prob

    def case_generator(self):
        generate_solvable = random.random() < self.solvable_prob
        if generate_solvable:
            # 构造可解案例
            while True:
                n = random.randint(self.n_min, self.n_max)
                size = 2 * n
                a = sorted([random.randint(self.elem_min, self.elem_max) for _ in range(size)], reverse=True)
                # 确保可解
                if Carraydestructionbootcamp.check_solvable(n, a):
                    random.shuffle(a)  # 打乱数组顺序
                    return {'n': n, 'a': a}
        else:
            # 生成不可解案例
            while True:
                n = random.randint(self.n_min, self.n_max)
                a = [random.randint(self.elem_min, self.elem_max) for _ in range(2 * n)]
                if not Carraydestructionbootcamp.check_solvable(n, a.copy()):
                    return {'n': n, 'a': a}

    @staticmethod
    def check_solvable(n, a_original):
        a = sorted(a_original.copy(), reverse=True)
        m = 2 * n
        d = {}
        prev = None
        for i in range(m):
            if i == 0 or a[i] != prev:
                d[a[i]] = i
                prev = a[i]
        
        for i in range(1, m):
            if i > 1 and a[i] == a[i-1]:
                continue
            p = d.copy()
            s = a[0] + a[i]
            b = [0] * m
            k = 0
            valid = True
            for _ in range(n):
                while k < m and b[k]:
                    k += 1
                if k >= m:
                    valid = False
                    break
                x = a[k]
                if x not in p:
                    valid = False
                    break
                p[x] += 1
                y = s - x
                if y not in p or x < y:
                    valid = False
                    break
                l = p[y]
                if l >= m or a[l] != y:
                    valid = False
                    break
                b[k] = 1
                b[l] = 1
                s = x
                k += 1
            if valid:
                return True
        return False

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = question_case['a']
        case_str = ' '.join(map(str, a))
        prompt = (
            "You are given an array elimination puzzle. Your task is to determine if it's possible to remove all elements of the array by following specific rules.\n\n"
            "Rules:\n"
            "1. Choose a positive integer x initially.\n"
            "2. Perform n operations. In each operation:\n"
            "   - Select two elements whose sum equals the current x.\n"
            "   - Remove them and update x to the maximum of the two selected elements.\n\n"
            f"Input:\nThe array has {2*n} elements: {case_str}\n\n"
            "Output:\n"
            "If possible, output YES followed by the initial x and the pairs removed in each operation.\n"
            "If not possible, output NO.\n\n"
            "Format your answer as:\n"
            "[answer]\n"
            "YES\n<initial_x>\n<element1> <element2>\n...\n[/answer]\n"
            "or\n"
            "[answer]\nNO\n[/answer]\n"
            "Include ONLY the [answer] block in your response."
        )
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        return last_match

    @classmethod
    def _verify_correction(cls, solution, identity):
        lines = solution.strip().split('\n')
        if not lines:
            return False
        first_line = lines[0].strip().upper()
        n = identity['n']
        a = identity['a']
        if first_line == 'NO':
            return not cls.check_solvable(n, a)
        elif first_line == 'YES':
            if len(lines) < 2:
                return False
            try:
                initial_x = int(lines[1].strip())
            except ValueError:
                return False
            steps = []
            for line in lines[2:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    return False
                try:
                    a1, a2 = map(int, parts)
                except ValueError:
                    return False
                steps.append((a1, a2))
            if len(steps) != n:
                return False
            current_x = initial_x
            remaining = a.copy()
            for step in steps:
                a1, a2 = step
                if a1 + a2 != current_x:
                    return False
                if a1 not in remaining or a2 not in remaining:
                    return False
                try:
                    remaining.remove(a1)
                    remaining.remove(a2)
                except ValueError:
                    return False
                current_x = max(a1, a2)
            return len(remaining) == 0
        else:
            return False
