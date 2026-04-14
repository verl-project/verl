"""# 

### 谜题描述
This time the Berland Team Olympiad in Informatics is held in a remote city that can only be reached by one small bus. Bus has n passenger seats, seat i can be occupied only by a participant from the city ai.

Today the bus has completed m trips, each time bringing n participants. The participants were then aligned in one line in the order they arrived, with people from the same bus standing in the order of their seats (i. e. if we write down the cities where the participants came from, we get the sequence a1, a2, ..., an repeated m times).

After that some teams were formed, each consisting of k participants form the same city standing next to each other in the line. Once formed, teams left the line. The teams were formed until there were no k neighboring participants from the same city.

Help the organizers determine how many participants have left in the line after that process ended. We can prove that answer doesn't depend on the order in which teams were selected.

Input

The first line contains three integers n, k and m (1 ≤ n ≤ 105, 2 ≤ k ≤ 109, 1 ≤ m ≤ 109).

The second line contains n integers a1, a2, ..., an (1 ≤ ai ≤ 105), where ai is the number of city, person from which must take seat i in the bus. 

Output

Output the number of remaining participants in the line.

Examples

Input

4 2 5
1 2 3 1


Output

12


Input

1 9 10
1


Output

1


Input

3 2 10
1 2 1


Output

0

Note

In the second example, the line consists of ten participants from the same city. Nine of them will form a team. At the end, only one participant will stay in the line.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, k, m = tuple([int(x) for x in raw_input().rstrip().split()])

a = [int(x) for x in raw_input().rstrip().split()]

stack = []
for aa in a:
    if len(stack) == 0 or aa != stack[-1][0]:
        stack.append([aa, 1])
    else:
        stack[-1][1] += 1
        if stack[-1][1] == k:
            stack.pop()

rem = 0
start, end = 0, len(stack) - 1
if m > 1:
    while (end - start + 1) > 1 and stack[start][0] == stack[end][0]:
        join = stack[start][1] + stack[end][1]
        if join < k:
            break
        elif join % k == 0:
            start += 1
            end -= 1
            rem += join
        else:
            stack[start][1] = join % k
            stack[end][1] = 0
            rem += (join / k) * k

ls = 0
len_stack = end - start + 1
for ss in stack[start: end + 1]:
    ls += ss[1]

if len_stack == 0:
    print 0
elif len_stack == 1:
    r = (m * stack[start][1]) % k
    if r == 0:
        print 0
    else:
        print r + rem
else:
    print ls * m + rem
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import deque
from bootcamp import Basebootcamp

class Dteamsformationbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.max_n = params.get('max_n', 100)
        self.max_k = params.get('max_k', 100)
        self.max_m = params.get('max_m', 100)
        self.city_max = params.get('city_max', 50)
    
    def case_generator(self):
        n = random.randint(1, self.max_n)
        k = random.randint(2, self.max_k)
        m = random.randint(1, self.max_m)
        a = [random.randint(1, self.city_max) for _ in range(n)]
        return {'n': n, 'k': k, 'm': m, 'a': a}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        case = question_case
        input_lines = f"{case['n']} {case['k']} {case['m']}\n{' '.join(map(str, case['a']))}"
        return f"""根据以下规则解决问题：
[规则]
1. 巴士往返m次形成总队列
2. 移除所有连续的k个同城参与者
3. 返回最终剩余人数

[输入]
{input_lines}

答案放入[answer]标签内，如：[answer]42[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return matches[-1] if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            ans = int(solution)
        except:
            return False
        
        n, k, m = identity['n'], identity['k'], identity['m']
        a = identity['a']
        
        # 压缩原始序列
        stack = []
        for city in a:
            if stack and stack[-1][0] == city:
                stack[-1] = (city, stack[-1][1] + 1)
                if stack[-1][1] == k:
                    stack.pop()
            else:
                stack.append((city, 1))
        if not stack:
            return ans == 0
        
        # 处理多趟次合并
        q = deque(stack)
        cycle_len = len(q)
        total_cycles = m
        removed = 0
        
        # 首尾合并处理
        while len(q) >= 2 and q[0][0] == q[-1][0]:
            front_city, front_cnt = q[0]
            back_city, back_cnt = q[-1]
            
            total = front_cnt + back_cnt
            if total < k:
                break
                
            if total % k == 0:
                removed += total * (total_cycles - 1)
                q.popleft()
                q.pop()
                total_cycles = 1  # 剩余部分只能处理一次
            else:
                new_cnt = total % k
                removed += (total - new_cnt) * (total_cycles - 1)
                q[0] = (front_city, new_cnt)
                q.pop()
                total_cycles = 1
                break
        
        # 计算最终结果
        if len(q) == 0:
            final = 0
        elif len(q) == 1:
            total = q[0][1] * total_cycles
            remainder = total % k
            final = remainder + removed
            final = final if remainder != 0 else removed
        else:
            base_sum = sum(cnt for city, cnt in q)
            final = base_sum * total_cycles + removed
        
        return ans == final
