"""# 

### 谜题描述
Little Vasya had n boxes with balls in the room. The boxes stood in a row and were numbered with numbers from 1 to n from left to right.

Once Vasya chose one of the boxes, let's assume that its number is i, took all balls out from it (it is guaranteed that this box originally had at least one ball), and began putting balls (one at a time) to the boxes with numbers i + 1, i + 2, i + 3 and so on. If Vasya puts a ball into the box number n, then the next ball goes to box 1, the next one goes to box 2 and so on. He did it until he had no balls left in his hands. It is possible that Vasya puts multiple balls to the same box, and it is also possible that one or more balls will go to the box number i. If i = n, Vasya puts the first ball into the box number 1, then the next ball goes to box 2 and so on. 

For example, let's suppose that initially Vasya had four boxes, and the first box had 3 balls, the second one had 2, the third one had 5 and the fourth one had 4 balls. Then, if i = 3, then Vasya will take all five balls out of the third box and put them in the boxes with numbers: 4, 1, 2, 3, 4. After all Vasya's actions the balls will lie in the boxes as follows: in the first box there are 4 balls, 3 in the second one, 1 in the third one and 6 in the fourth one.

At this point Vasya has completely forgotten the original arrangement of the balls in the boxes, but he knows how they are arranged now, and the number x — the number of the box, where he put the last of the taken out balls.

He asks you to help to find the initial arrangement of the balls in the boxes.

Input

The first line of the input contains two integers n and x (2 ≤ n ≤ 105, 1 ≤ x ≤ n), that represent the number of the boxes and the index of the box that got the last ball from Vasya, correspondingly. The second line contains n space-separated integers a1, a2, ..., an, where integer ai (0 ≤ ai ≤ 109, ax ≠ 0) represents the number of balls in the box with index i after Vasya completes all the actions. 

Please, do not use the %lld specifier to read or write 64-bit integers in С++. It is preferred to use the cin, cout streams or the %I64d specifier.

Output

Print n integers, where the i-th one represents the number of balls in the box number i before Vasya starts acting. Separate the numbers in the output by spaces. If there are multiple correct solutions, you are allowed to print any of them.

Examples

Input

4 4
4 3 1 6


Output

3 2 5 4 

Input

5 2
3 2 0 2 7


Output

2 1 4 1 6 

Input

3 3
2 3 1


Output

1 2 3 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n,x = map(int,raw_input().split())
a = map(int,raw_input().split())
m = x-1
resorted = []
min = 100000000000
indMin = -1
for i in range(0,n):
    if a[m] < min:
        min = a[m]
        indMin = m
    if m > 0:
        m = m - 1
    else:
        m = n - 1    

begin = indMin
subst = a[begin]
temp = list(q-subst for q in a)
if begin == x-1:
    temp[indMin] = subst*n
    print ' '.join(str(el) for el in temp)  
else:
    count = 0
    if begin < x-1:
        for k in range(begin+1,x):
            temp[k] = temp[k] -1
            count = count + 1
    else:
        if begin != n-1:
            i = begin + 1
        else:
            i = 0    
        while i != x:
            count = count + 1
            temp[i] = temp[i] - 1
            if i != n-1:
                i = i+1
            else:
                i = 0
    temp[indMin] = subst*n + count
    print ' '.join(str(el) for el in temp)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cballbootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=5, ball_max=10):
        self.min_n = min_n
        self.max_n = max_n
        self.ball_max = ball_max

    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        original = []
        max_attempts = 10
        for _ in range(max_attempts):
            original = [random.randint(0, self.ball_max) for _ in range(n)]
            if sum(original) > 0:
                break
        else:
            original = [0] * n
            idx = random.randint(0, n-1)
            original[idx] = 1

        possible_i = [i+1 for i in range(n) if original[i] > 0]
        i = random.choice(possible_i)
        k = original[i-1]

        simulated = original.copy()
        simulated[i-1] = 0
        current_pos = i + 1
        last_x = None

        for _ in range(k):
            current_pos = (current_pos - 1) % n
            current_pos += 1
            target_pos = current_pos
            simulated[target_pos - 1] += 1
            last_x = target_pos
            current_pos += 1

        return {
            'n': n,
            'x': last_x,
            'a': simulated,
            'expected': original
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        x = question_case['x']
        a = question_case['a']
        input_lines = f"{n} {x}\n{' '.join(map(str, a))}"
        prompt = f"""You are a programming problem solver. Please solve the following puzzle based on the given input.

Problem Description:
Vasya took all balls from one box and distributed them cyclically starting from the next box. Given the final ball counts and the last box that received a ball, determine the initial ball counts.

Input Format:
- First line: n (number of boxes) and x (last box that received a ball)
- Second line: a_1 a_2 ... a_n (final ball counts in each box)

Output Format:
Print the initial ball counts as space-separated integers. Place your answer within [answer] and [/answer] tags.

Example Input:
4 4
4 3 1 6

Example Output:
[answer]3 2 5 4[/answer]

Current Problem Input:
{input_lines}

Please provide your solution inside the [answer] tags:"""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            numbers = list(map(int, last_match.split()))
            return numbers
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected']
