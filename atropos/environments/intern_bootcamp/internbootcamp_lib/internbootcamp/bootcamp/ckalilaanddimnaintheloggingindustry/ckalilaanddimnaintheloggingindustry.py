"""# 

### 谜题描述
Kalila and Dimna are two jackals living in a huge jungle. One day they decided to join a logging factory in order to make money. 

The manager of logging factory wants them to go to the jungle and cut n trees with heights a1, a2, ..., an. They bought a chain saw from a shop. Each time they use the chain saw on the tree number i, they can decrease the height of this tree by one unit. Each time that Kalila and Dimna use the chain saw, they need to recharge it. Cost of charging depends on the id of the trees which have been cut completely (a tree is cut completely if its height equal to 0). If the maximum id of a tree which has been cut completely is i (the tree that have height ai in the beginning), then the cost of charging the chain saw would be bi. If no tree is cut completely, Kalila and Dimna cannot charge the chain saw. The chainsaw is charged in the beginning. We know that for each i < j, ai < aj and bi > bj and also bn = 0 and a1 = 1. Kalila and Dimna want to cut all the trees completely, with minimum cost. 

They want you to help them! Will you?

Input

The first line of input contains an integer n (1 ≤ n ≤ 105). The second line of input contains n integers a1, a2, ..., an (1 ≤ ai ≤ 109). The third line of input contains n integers b1, b2, ..., bn (0 ≤ bi ≤ 109).

It's guaranteed that a1 = 1, bn = 0, a1 < a2 < ... < an and b1 > b2 > ... > bn.

Output

The only line of output must contain the minimum cost of cutting all the trees completely.

Please, do not write the %lld specifier to read or write 64-bit integers in С++. It is preferred to use the cin, cout streams or the %I64d specifier.

Examples

Input

5
1 2 3 4 5
5 4 3 2 0


Output

25


Input

6
1 2 3 10 20 30
6 5 4 3 2 0


Output

138

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division
from collections import deque
n = int(raw_input())
A = list(map(int,raw_input().split()))
B = list(map(int,raw_input().split()))
dp = [9999999999999999999 for x in range(n)]
dp[0] = 0
lines = deque()
lines.appendleft([B[0],0])

def solve(a,b):
  return (a[1]-b[1])/(b[0]-a[0])

for i in range(1,n):
  while len(lines)>=2 and solve(lines[-1],lines[-2]) <= A[i]:
    lines.pop()
  dp[i] = lines[-1][1]+lines[-1][0]*A[i]
  while len(lines)>=2 and solve(lines[1],lines[0]) >= solve(lines[0],[B[i],dp[i]]):
    lines.popleft()
  lines.appendleft([B[i],dp[i]])
print dp[n-1]
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from collections import deque
import random
from bootcamp import Basebootcamp

class Ckalilaanddimnaintheloggingindustrybootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n_min = params.get('n_min', 1)
        self.n_max = params.get('n_max', 10)
        self.a_step_min = params.get('a_step_min', 1)
        self.a_step_max = params.get('a_step_max', 100)
        self.b_step_min = params.get('b_step_min', 1)
        self.b_step_max = params.get('b_step_max', 100)
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        
        # Generate strictly increasing a
        a = [1]
        current = 1
        for _ in range(1, n):
            step = random.randint(self.a_step_min, self.a_step_max)
            current += step
            a.append(current)
        
        # Generate strictly decreasing b with bn=0
        b = [0]
        prev_b = 0
        for _ in range(n-1):
            step = random.randint(self.b_step_min, self.b_step_max)
            prev_b += step
            b.insert(0, prev_b)

        return {
            'n': n,
            'a': a,
            'b': b,
            'correct_answer': self._calculate_min_cost(n, a, b)
        }

    @staticmethod
    def _calculate_min_cost(n, A, B):
        if n == 0:
            return 0
        dp = [0]*n
        lines = deque()
        lines.append((B[0], 0))  # (k, b)

        def get_x(line1, line2):
            k1, b1 = line1
            k2, b2 = line2
            if k1 == k2:
                return float('-inf')
            return (b2 - b1) / (k1 - k2)

        for i in range(1, n):
            # Remove outdated lines from end
            while len(lines) >= 2 and get_x(lines[-1], lines[-2]) <= A[i]:
                lines.pop()
            
            # Calculate current dp value
            best_k, best_b = lines[-1]
            dp[i] = best_b + best_k * A[i]

            # Maintain convex hull from front
            new_line = (B[i], dp[i])
            while len(lines) >= 2:
                x1 = get_x(lines[0], new_line)
                x2 = get_x(lines[0], lines[1])
                if x1 >= x2:
                    lines.popleft()
                else:
                    break
            lines.appendleft(new_line)

        return dp[-1]

    @staticmethod
    def prompt_func(question_case) -> str:
        return f"""Kalila and Dimna need to cut down all trees with minimum cost. 

Input:
n = {question_case['n']}
a = {question_case['a']}
b = {question_case['b']}

Calculate the minimal total cost. Put your final answer within [answer]...[/answer]."""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except (ValueError, TypeError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
