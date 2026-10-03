"""# 

### 谜题描述
Amr loves Chemistry, and specially doing experiments. He is preparing for a new interesting experiment.

Amr has n different types of chemicals. Each chemical i has an initial volume of ai liters. For this experiment, Amr has to mix all the chemicals together, but all the chemicals volumes must be equal first. So his task is to make all the chemicals volumes equal.

To do this, Amr can do two different kind of operations. 

  * Choose some chemical i and double its current volume so the new volume will be 2ai
  * Choose some chemical i and divide its volume by two (integer division) so the new volume will be <image>



Suppose that each chemical is contained in a vessel of infinite volume. Now Amr wonders what is the minimum number of operations required to make all the chemicals volumes equal?

Input

The first line contains one number n (1 ≤ n ≤ 105), the number of chemicals.

The second line contains n space separated integers ai (1 ≤ ai ≤ 105), representing the initial volume of the i-th chemical in liters.

Output

Output one integer the minimum number of operations required to make all the chemicals volumes equal.

Examples

Input

3
4 8 2


Output

2

Input

3
3 5 6


Output

5

Note

In the first sample test, the optimal solution is to divide the second chemical volume by two, and multiply the third chemical volume by two to make all the volumes equal 4.

In the second sample test, the optimal solution is to divide the first chemical volume by two, and divide the second and the third chemical volumes by two twice to make all the volumes equal 1.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
N=input()
A=map(int, raw_input().split())
c=[0]*100001
d=[0]*100001
u=[0]*100001
v=[0]*100001
for i in A:
    x=i
    t=0
    if x*2<=100000:
        u[x*2]+=1
        v[x*2]+=1
    while x>0:
        if x%2==1:
            u[x-1]+=t+2
            v[x-1]+=1
        c[x]+=t
        d[x]+=1
        x/=2
        t+=1
    c[x]+=t
    d[x]+=1
for i in xrange(1, 100001):
    if v[i]!=0:
        x=i
        while x<=100000:
            c[x]+=u[i]
            d[x]+=v[i]
            u[i]+=v[i]
            x*=2
print min(j for i, j in zip(d, c) if i==N)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import deque
from bootcamp import Basebootcamp

class Camrandchemistrybootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__()
        self.n = params.get('n', 10)
        self.max_volume = params.get('max_volume', 100000)
    
    def case_generator(self):
        n = self.n
        max_volume = self.max_volume
        
        # 选择一个目标数，确保在生成实例时总是有解
        target = random.randint(1, max_volume)
        numbers = []
        correct_operations = 0
        
        for _ in range(n):
            current = target
            steps = random.randint(0, 10)  # 调整随机步骤的范围
            
            # 生成随机的操作序列
            for _ in range(steps):
                choice = random.choice(['mult', 'div'])
                if choice == 'mult':
                    current *= 2
                else:
                    if current % 2 == 0:
                        current = current // 2
                    else:
                        current *= 2  # 如果是奇数，只能乘以2
            
            numbers.append(current)
            
            # 计算从当前数到目标数的最小操作次数
            visited = set()
            queue = deque()
            queue.append((current, 0))
            visited.add(current)
            found = False
            
            while queue:
                num, ops = queue.popleft()
                if num == target:
                    correct_operations += ops
                    found = True
                    break
                next_num = num * 2
                if next_num <= 10**6 and next_num not in visited:
                    visited.add(next_num)
                    queue.append((next_num, ops + 1))
                if num % 2 == 0:
                    next_num = num // 2
                    if next_num not in visited:
                        visited.add(next_num)
                        queue.append((next_num, ops + 1))
            
            if not found:
                # 如果无法到达目标数，重新生成当前数
                while True:
                    current = target
                    steps = random.randint(0, 10)
                    for _ in range(steps):
                        choice = random.choice(['mult', 'div'])
                        if choice == 'mult':
                            current *= 2
                        else:
                            if current % 2 == 0:
                                current = current // 2
                            else:
                                current *= 2
                    # 计算最小操作次数
                    visited = set()
                    queue = deque()
                    queue.append((current, 0))
                    visited.add(current)
                    found_inner = False
                    while queue:
                        num_inner, ops_inner = queue.popleft()
                        if num_inner == target:
                            correct_operations += ops_inner
                            found_inner = True
                            break
                        next_num_inner = num_inner * 2
                        if next_num_inner <= 10**6 and next_num_inner not in visited:
                            visited.add(next_num_inner)
                            queue.append((next_num_inner, ops_inner + 1))
                        if num_inner % 2 == 0:
                            next_num_inner = num_inner // 2
                            if next_num_inner not in visited:
                                visited.add(next_num_inner)
                                queue.append((next_num_inner, ops_inner + 1))
                    if found_inner:
                        numbers.append(current)
                        break
        
        identity = {'numbers': numbers, 'correct_operations': correct_operations}
        return identity
    
    @staticmethod
    def prompt_func(question_case):
        numbers = question_case['numbers']
        n = len(numbers)
        prompt = f"Camrandchemistry有{n}种不同的化学品，初始体积分别为：{numbers}。他需要通过乘以2或除以2（整数除法）操作，使所有化学品体积相等。请计算最小的操作次数，并将答案放在[answer]标签中，例如：[answer]5[/answer]"
        return prompt
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](\d+)\[/answer\]', output)
        return int(matches[-1]) if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        correct = identity['correct_operations']
        return solution == correct
