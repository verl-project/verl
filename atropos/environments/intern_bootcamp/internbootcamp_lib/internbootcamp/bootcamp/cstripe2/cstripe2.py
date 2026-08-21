"""# 

### 谜题描述
Once Bob took a paper stripe of n squares (the height of the stripe is 1 square). In each square he wrote an integer number, possibly negative. He became interested in how many ways exist to cut this stripe into three pieces so that the sum of numbers from each piece is equal to the sum of numbers from any other piece, and each piece contains positive integer amount of squares. Would you help Bob solve this problem?

Input

The first input line contains integer n (1 ≤ n ≤ 105) — amount of squares in the stripe. The second line contains n space-separated numbers — they are the numbers written in the squares of the stripe. These numbers are integer and do not exceed 10000 in absolute value.

Output

Output the amount of ways to cut the stripe into three non-empty pieces so that the sum of numbers from each piece is equal to the sum of numbers from any other piece. Don't forget that it's allowed to cut the stripe along the squares' borders only.

Examples

Input

4
1 2 3 3


Output

1


Input

5
1 2 3 4 5


Output

0

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n=input()
c={}
a=map(int,raw_input().split())
w,s,t=0,sum(a),0
for x in a[:-1]:
  v=s-t-x
  if v+v==t+x:w+=c.get(v,0)
  t+=x
  c[t]=c.get(t,0)+1
print (0,w)[n>2]
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cstripe2bootcamp(Basebootcamp):
    def __init__(self, n_min=3, n_max=100, allow_unsolvable=True, force_solvable=False):
        self.n_min = max(3, n_min)
        self.n_max = n_max
        self.allow_unsolvable = allow_unsolvable
        self.force_solvable = force_solvable
    
    def case_generator(self):
        generate_solvable = self.force_solvable or not self.allow_unsolvable
        
        if generate_solvable:
            n = random.randint(self.n_min, self.n_max)
            if n < 3:
                n = 3

            total_sum = random.randint(-3000, 3000) * 3  # 确保总和是3的倍数
            target = total_sum // 3

            def generate_segment(length, segment_target):
                if length == 0:
                    return []
                while True:
                    # 最后一个元素在有效范围内
                    last = random.randint(-10000, 10000)
                    needed_sum = segment_target - last
                    # 生成前length-1个元素
                    if length-1 == 0:
                        if needed_sum == 0:
                            return [last]
                        else:
                            continue
                    elements = [random.randint(-10000, 10000) for _ in range(length-1)]
                    adj = needed_sum - sum(elements)
                    if -10000 <= adj <= 10000:
                        elements.append(adj)
                        # 再次校验所有元素范围
                        if all(-10000 <= x <= 10000 for x in elements):
                            return elements + [last]

            # 生成三段（确保每段长度至少1）
            len1 = random.randint(1, n-2)
            len2 = random.randint(1, n - len1 -1)
            len3 = n - len1 - len2

            part1 = generate_segment(len1, target)
            part2 = generate_segment(len2, target)
            part3 = generate_segment(len3, target)

            array = part1 + part2 + part3
            return {'n': n, 'array': array}
        else:
            n = random.randint(self.n_min, self.n_max)
            array = [random.randint(-10000, 10000) for _ in range(n)]
            return {'n': n, 'array': array}

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        array = question_case['array']
        array_str = ' '.join(map(str, array))
        return f"""Bob's paper stripe has {n} squares with numbers: {array_str}. Find how many ways to cut it into three non-empty parts with equal sums. Put your answer in [answer]...[/answer]. Example: [answer]1[/answer]."""

    @staticmethod
    def extract_output(output):
        answers = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(answers[-1]) if answers else None

    @classmethod
    def compute_solution(cls, n, array):
        if n < 3:
            return 0
        total = sum(array)
        if total % 3 != 0:
            return 0
        target = total // 3
        first_cut = 0
        count = 0
        current = 0
        for i in range(n-1):
            current += array[i]
            if current == 2 * target:
                count += first_cut
            if current == target:
                first_cut += 1
        return count

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            user_ans = int(solution)
        except:
            return False
        n = identity['n']
        array = identity['array']
        return user_ans == cls.compute_solution(n, array)
