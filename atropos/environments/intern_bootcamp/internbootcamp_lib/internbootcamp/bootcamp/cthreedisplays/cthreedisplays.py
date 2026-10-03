"""# 

### 谜题描述
It is the middle of 2018 and Maria Stepanovna, who lives outside Krasnokamensk (a town in Zabaikalsky region), wants to rent three displays to highlight an important problem.

There are n displays placed along a road, and the i-th of them can display a text with font size s_i only. Maria Stepanovna wants to rent such three displays with indices i < j < k that the font size increases if you move along the road in a particular direction. Namely, the condition s_i < s_j < s_k should be held.

The rent cost is for the i-th display is c_i. Please determine the smallest cost Maria Stepanovna should pay.

Input

The first line contains a single integer n (3 ≤ n ≤ 3 000) — the number of displays.

The second line contains n integers s_1, s_2, …, s_n (1 ≤ s_i ≤ 10^9) — the font sizes on the displays in the order they stand along the road.

The third line contains n integers c_1, c_2, …, c_n (1 ≤ c_i ≤ 10^8) — the rent costs for each display.

Output

If there are no three displays that satisfy the criteria, print -1. Otherwise print a single integer — the minimum total rent cost of three displays with indices i < j < k such that s_i < s_j < s_k.

Examples

Input

5
2 4 5 4 10
40 30 20 10 40


Output

90


Input

3
100 101 100
2 4 5


Output

-1


Input

10
1 2 3 4 5 6 7 8 9 10
10 13 11 14 15 12 13 13 18 13


Output

33

Note

In the first example you can, for example, choose displays 1, 4 and 5, because s_1 < s_4 < s_5 (2 < 4 < 10), and the rent cost is 40 + 10 + 40 = 90.

In the second example you can't select a valid triple of indices, so the answer is -1.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = int(raw_input())
size = map(int, raw_input().split())
cost = map(int, raw_input().split())
ret = float('inf')
for second in xrange(1, n - 1):
    first = third = float('inf')
    for i in xrange(n):
        if i < second and size[i] < size[second] and cost[i] < first:
            first = cost[i]
        elif i > second and size[second] < size[i] and cost[i] < third:
            third = cost[i]
    ret = min(ret, first + cost[second] + third)
print ret if ret != float('inf') else -1
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cthreedisplaysbootcamp(Basebootcamp):
    def __init__(self, n_min=3, n_max=3000, s_min=1, s_max=10**9, c_min=1, c_max=10**8):
        self.n_min = n_min
        self.n_max = n_max
        self.s_min = s_min
        self.s_max = s_max
        self.c_min = c_min
        self.c_max = c_max
    
    def case_generator(self):
        # 50%概率生成有解案例
        has_solution = random.choice([True, False])
        n = random.randint(self.n_min, self.n_max)
        s = []
        if has_solution:
            # 强制生成至少一个有效三元组
            i = random.randint(0, n-3)
            j = random.randint(i+1, n-2)
            k = random.randint(j+1, n-1)
            
            # 确保s_i < s_j < s_k
            s_i = random.randint(self.s_min, self.s_max - 2)
            s_j = random.randint(s_i + 1, self.s_max - 1)
            s_k = random.randint(s_j + 1, self.s_max)
            
            s = [random.randint(self.s_min, self.s_max) for _ in range(n)]
            s[i] = s_i
            s[j] = s_j
            s[k] = s_k
        else:
            # 生成严格递减数组确保无解
            s = sorted([random.randint(self.s_min, self.s_max) for _ in range(n)], reverse=True)
        
        c = [random.randint(self.c_min, self.c_max) for _ in range(n)]
        expected = self._solve(s, c)
        
        # 二次验证无解生成合法性
        if has_solution and expected == -1:
            return self.case_generator()  # 重新生成
        
        return {'n': n, 's': s, 'c': c, 'expected': expected}
    
    @staticmethod
    def _solve(s, c):
        n = len(s)
        min_total = float('inf')
        for j in range(1, n-1):
            s_j = s[j]
            min_left = min_right = float('inf')
            
            # 查找左半最小成本
            for i in range(j):
                if s[i] < s_j and c[i] < min_left:
                    min_left = c[i]
            
            # 查找右半最小成本
            for k in range(j+1, n):
                if s[k] > s_j and c[k] < min_right:
                    min_right = c[k]
            
            if min_left != float('inf') and min_right != float('inf'):
                current_total = min_left + c[j] + min_right
                if current_total < min_total:
                    min_total = current_total
        
        return min_total if min_total != float('inf') else -1
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        s = ' '.join(map(str, question_case['s']))
        c = ' '.join(map(str, question_case['c']))
        return (
            f"Maria needs to rent three displays with strictly increasing font sizes along a road. "
            f"Find the minimal total cost for indices i < j < k where s_i < s_j < s_k. If impossible, output -1.\n\n"
            f"Input:\n{n}\n{s}\n{c}\n\n"
            f"Put your final answer within [answer]...[/answer]."
        )
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected']
