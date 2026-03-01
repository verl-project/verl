"""# 

### 谜题描述
There are n knights sitting at the Round Table at an equal distance from each other. Each of them is either in a good or in a bad mood.

Merlin, the wizard predicted to King Arthur that the next month will turn out to be particularly fortunate if the regular polygon can be found. On all vertices of the polygon knights in a good mood should be located. Otherwise, the next month will bring misfortunes.

A convex polygon is regular if all its sides have same length and all his angles are equal. In this problem we consider only regular polygons with at least 3 vertices, i. e. only nondegenerated.

On a picture below some examples of such polygons are present. Green points mean knights in a good mood. Red points mean ones in a bad mood.

<image>

King Arthur knows the knights' moods. Help him find out if the next month will be fortunate or not.

Input

The first line contains number n, which is the number of knights at the round table (3 ≤ n ≤ 105). The second line contains space-separated moods of all the n knights in the order of passing them around the table. \"1\" means that the knight is in a good mood an \"0\" means that he is in a bad mood.

Output

Print \"YES\" without the quotes if the following month will turn out to be lucky. Otherwise, print \"NO\".

Examples

Input

3
1 1 1


Output

YES

Input

6
1 0 1 1 1 0


Output

YES

Input

6
1 0 0 1 0 1


Output

NO

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
# -*- coding: utf-8 -*-
n = input()
status = map(int, raw_input().split())
frt = False
ors = lambda x, y: x or y
ands = lambda x, y: x and y
b = False
for l in range(1, n/3+1):
    if n%l!=0: continue
    for i in range(l):
        b = True
        for j in range(i, n, l):
            if status[j] == 0:
                b = False; break
        if b: break
    if b: break
print 'YES' if b else 'NO'
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Croundtableknightsbootcamp(Basebootcamp):
    def __init__(self, min_n=3, max_n=100, has_solution=None):
        self.min_n = min_n
        self.max_n = max_n
        self.has_solution = has_solution  # None: random, True: yes, False: no
    
    def case_generator(self):
        generate_solution = self.has_solution if self.has_solution is not None else random.choice([True, False])
        n = random.randint(self.min_n, self.max_n)
        
        if generate_solution:
            possible_ls = [l for l in range(1, n//3 +1) if n % l == 0 and (n//l) >=3]
            if not possible_ls:
                l = 1  # 当n本身是素数时确保至少有一个有效分割
            else:
                l = random.choice(possible_ls)
            i = random.randint(0, l-1)
            status = [0] * n
            for j in range(i, n, l):
                status[j] = 1
            # 其他位置随机设置，但不影响存在解
            for j in range(n):
                if j % l != i % l:
                    status[j] = random.randint(0, 1)
        else:
            while True:
                status = [random.choice([0, 1]) for _ in range(n)]
                # 允许全0情况，因为全0必然无解
                if not self.__class__._is_lucky(n, status):
                    break
        
        return {'n': n, 'status': status}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        status = ' '.join(map(str, question_case['status']))
        return f"""Determine if the knights' moods allow forming a regular polygon. Knights are arranged around a round table. 
A regular polygon requires at least 3 vertices (knights in good mood). 

Input:
{n}
{status}

Output your answer (YES/NO) within [answer]...[/answer]."""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](YES|NO)\[/answer\]', output, re.IGNORECASE)
        return matches[-1].strip().upper() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        correct = 'YES' if cls._is_lucky(identity['n'], identity['status']) else 'NO'
        return (solution or '').upper() == correct

    @staticmethod
    def _is_lucky(n, status):
        for l in range(1, n//3 +1):
            if n % l != 0:
                continue
            k = n // l
            if k < 3:
                continue
            for i in range(l):
                if all(status[j] == 1 for j in range(i, n, l)):
                    return True
        return False
