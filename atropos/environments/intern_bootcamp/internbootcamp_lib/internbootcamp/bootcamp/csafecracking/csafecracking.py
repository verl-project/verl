"""# 

### 谜题描述
Right now you are to solve a very, very simple problem — to crack the safe. Four positive integers stand one by one on a circle protecting the safe. You know that to unlock this striking safe you have to make all four numbers equal to one. Operations are as follows: you may choose two adjacent numbers and increase both by one; you may choose two adjacent even numbers and divide both by two. Nothing else. Crack the safe!

Input

The single line of the input contains four space-separated integer positive numbers not greater than 109 each — four numbers on the circle in consecutive order.

Output

The output should contain \"-1\" (quotes for clarity) if the safe is secure, that is it's impossible to crack it. Otherwise, output should contain the sequence of operations (one operations per line) leading to unlocking the safe. You don't have to minimize the number of operations, but it should not exceed 1000. To make things clear, assume numbers stand on positions 1 through 4. Each operation is encoded by two symbols. If the following operation is dividing then first symbol is '/'; otherwise it's '+' (addition). The second symbol is the position of the first number in pair in consecutive order. (see samples for clarification).

If there are several solutions, output any of them.

Examples

Input

1 1 1 1


Output



Input

1 2 4 2


Output

/2
/3


Input

3 3 1 1


Output

+1
/1
/1


Input

2 1 2 4


Output

/3
/4

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys


no = [int(n) for n in sys.stdin.readline().split()]

def prev(i):
    return (i + 3) % 4

def add(i):
    no[i] += 1
    no[(i+1) % 4] += 1
    print \"+%d\" % (i+1)

def div(i):
    no[i] /= 2
    no[(i+1) % 4] /= 2
    print \"/%d\" % (i+1)

while max(no) > 2:
    big = max(no)
    i = no.index(big)
    n = (i + 1) % 4

    if no[i] % 2 == 1 and no[n] % 2 == 1:
        add(i)
    elif no[i] % 2 == 1 and no[n] % 2 == 0:
        add(prev(i))
    elif no[i] % 2 == 0 and no[n] % 2 == 1:
        add(n)
    div(i)


for i in xrange(4):
    if no[i] == 2 and no[(i+1) % 4] == 2:
        div(i)

for i in xrange(4):
    if no[prev(i)] == 1 and no[i] == 2 and no[(i+1) % 4] == 1:
        add(prev(i))
        add(i)
        div(prev(i))
        div(i)

# print no
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Csafecrackingbootcamp(Basebootcamp):
    def __init__(self, min_reverse_steps=3, max_reverse_steps=15):
        super().__init__()
        self.min_reverse_steps = min_reverse_steps
        self.max_reverse_steps = max_reverse_steps
    
    def case_generator(self):
        steps = random.randint(self.min_reverse_steps, self.max_reverse_steps)
        current = [1, 1, 1, 1]
        for _ in range(steps):
            i = random.randint(0, 3)
            # 动态调整操作概率，优先产生可逆的乘法操作
            if random.random() < 0.6 and all(x % 2 == 0 for x in (current[i], current[(i+1)%4])):
                # 尝试除法逆向操作（即正向的乘法）
                temp_i = current[i] * 2
                temp_j = current[(i + 1) % 4] * 2
                if temp_i <= 1e9 and temp_j <= 1e9:
                    current[i] = temp_i
                    current[(i + 1) % 4] = temp_j
                    continue
            # 默认进行加法逆向操作
            current[i] += 1
            current[(i + 1) % 4] += 1
        return {'numbers': current}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        nums = question_case['numbers']
        prompt = (
            "You are a safe cracker. Four numbers in a circle: "
            f"{nums[0]}, {nums[1]}, {nums[2]}, {nums[3]}.\n\n"
            "**Rules**:\n"
            "1. '+X': Add 1 to numbers at positions X and X%4+1 (1-based).\n"
            "2. '/X': Divide (if both even) at positions X and X%4+1.\n\n"
            "**Goal**: Make all numbers 1 with ≤1000 steps. Output '-1' if impossible.\n"
            "**Format**: Put your answer between [answer] and [/answer], one operation per line.\n"
            "Example:\n[answer]\n+1\n/2\n[/answer]"
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_answer = matches[-1].strip().split('\n')
        valid_ops = []
        for line in last_answer:
            line = line.strip()
            if re.fullmatch(r'^[+/][1-4]$', line):
                valid_ops.append(line)
        return valid_ops if valid_ops else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False  # 生成的案例均有解，不能接受-1
        
        nums = list(identity['numbers'])
        if len(solution) > 1000:
            return False
        
        for op in solution:
            if len(op) != 2 or op[0] not in ('+', '/'):
                return False
            pos = int(op[1]) - 1  # 转0-based
            if not (0 <= pos < 4):
                return False
            next_pos = (pos + 1) % 4
            
            if op[0] == '+':
                nums[pos] += 1
                nums[next_pos] += 1
            else:
                if nums[pos] % 2 != 0 or nums[next_pos] % 2 != 0:
                    return False
                nums[pos] //= 2
                nums[next_pos] //= 2
        
        return all(n == 1 for n in nums)
