"""# 

### 谜题描述
A team of students from the city S is sent to the All-Berland Olympiad in Informatics. Traditionally, they go on the train. All students have bought tickets in one carriage, consisting of n compartments (each compartment has exactly four people). We know that if one compartment contain one or two students, then they get bored, and if one compartment contain three or four students, then the compartment has fun throughout the entire trip.

The students want to swap with other people, so that no compartment with students had bored students. To swap places with another person, you need to convince him that it is really necessary. The students can not independently find the necessary arguments, so they asked a sympathetic conductor for help. The conductor can use her life experience to persuade any passenger to switch places with some student.

However, the conductor does not want to waste time persuading the wrong people, so she wants to know what is the minimum number of people necessary to persuade her to change places with the students. Your task is to find the number. 

After all the swaps each compartment should either have no student left, or have a company of three or four students. 

Input

The first line contains integer n (1 ≤ n ≤ 106) — the number of compartments in the carriage. The second line contains n integers a1, a2, ..., an showing how many students ride in each compartment (0 ≤ ai ≤ 4). It is guaranteed that at least one student is riding in the train.

Output

If no sequence of swapping seats with other people leads to the desired result, print number \"-1\" (without the quotes). In another case, print the smallest number of people you need to persuade to swap places.

Examples

Input

5
1 2 2 4 3


Output

2


Input

3
4 1 1


Output

2


Input

4
0 3 0 4


Output

0

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = int(raw_input())
A = map(int, raw_input().split())

C = [0]*5
for a in A:
  C[a] += 1

cost = 0
t = min(C[1], C[2])
cost += t
C[1] -= t
C[2] -= t
C[3] += t

if C[2] != 0: # C[1] == 0
  t = C[2]/3
  cost += 2*t
  C[2] -= 3*t
  C[3] += 2*t
  if C[2] == 1:
    if C[4] > 0:
      cost += 1
    elif C[3] > 1:
      cost += 2
    else:
      cost = -1
  elif C[2] == 2:
    cost += 2
else: # C[2] == 0
  t = C[1]/3
  cost += 2*t
  C[1] -= 3*t
  C[3] += t
  if C[1] == 1:
    if C[3] > 0:
      cost += 1
    elif C[4] > 1:
      cost += 2
    else:
      cost = -1
  elif C[1] == 2:
    if C[4] > 0:
      cost += 2
    elif C[3] > 1:
      cost += 2
    else:
      cost = -1

print cost
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re

from bootcamp import Basebootcamp

def compute_min_people(n, a):
    C = [0] * 5
    for num in a:
        C[num] += 1
    cost = 0

    # 处理1和2的情况
    t = min(C[1], C[2])
    cost += t
    C[1] -= t
    C[2] -= t
    C[3] += t

    if C[2] != 0:
        # 处理剩下的2
        t = C[2] // 3
        cost += 2 * t
        C[2] -= 3 * t
        C[3] += 2 * t

        if C[2] == 1:
            if C[4] > 0:
                cost += 1
                C[4] -= 1
                C[3] += 1
            elif C[3] > 1:
                cost += 2
                C[3] -= 2
                C[3] += 2
            else:
                return -1
        elif C[2] == 2:
            if C[4] >= 1:
                cost += 2
                C[4] -= 1
                C[3] += 1
            else:
                return -1
    else:
        # 处理剩下的1
        t = C[1] // 3
        cost += 2 * t
        C[1] -= 3 * t
        C[3] += t

        if C[1] == 1:
            if C[3] > 0:
                cost += 1
                C[3] -= 1
                C[3] += 1
            elif C[4] >= 2:
                cost += 2
                C[4] -= 2
                C[3] += 2
            else:
                return -1
        elif C[1] == 2:
            if C[4] >= 1:
                cost += 2
                C[4] -= 1
                C[3] += 1
            elif C[3] >= 2:
                cost += 2
                C[3] -= 2
                C[3] += 2
            else:
                return -1

    # 检查所有隔间是否满足条件
    for i in range(5):
        if i not in (0, 3, 4) and C[i] != 0:
            return -1

    return cost

class Ecompartmentsbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__()
        self.params = params

    def case_generator(self):
        n = random.randint(1, 100)
        a = [random.randint(0, 4) for _ in range(n)]
        while sum(a) == 0:
            a = [random.randint(0,4) for _ in range(n)]
        return {
            'n': n,
            'a': a
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = ' '.join(map(str, question_case['a']))
        prompt = (
            f"在一个火车车厢里有{n}个隔间，每个隔间的学生人数分别是：{a}。"
            "你需要通过换人，使得每个隔间要么没有学生，要么有3或4个学生。"
            "每个换人操作需要说服一个人。"
            "请问，最少需要说服多少人换座位？"
            "\n"
            "请将答案放在[answer]标签内，例如：\n"
            "[answer]2[/answer]"
        )
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if matches:
            extracted = matches[-1].strip()
            try:
                return int(extracted)
            except ValueError:
                return None
        else:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False
        n = identity['n']
        a = identity['a']
        correct = compute_min_people(n, a)
        if correct == -1:
            return solution == -1
        try:
            solution_int = int(solution)
            return solution_int == correct
        except:
            return False
