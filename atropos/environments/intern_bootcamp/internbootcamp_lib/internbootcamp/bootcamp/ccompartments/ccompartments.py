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
import sys
inp = sys.stdin

#n, m, g = map(int, sys.stdin.readline().strip().split())
#inp = open(\"input.txt\")
#n, m = map(int, inp.readline().strip().split())
n = int(inp.readline().strip())
a = map(int, inp.readline().strip().split())

counts = [0, 0, 0, 0, 0]

for i in a:
    counts[i] += 1

res = min(counts[1], counts[2])
counts[1] -= res
counts[2] -= res
counts[3] += res

if counts[1] > 0:
    tres = counts[1] / 3
    res += tres * 2
    counts[1] -= tres * 3
    counts[3] += tres
elif counts[2] > 0:
    tres = counts[2] / 3
    res += tres * 2
    counts[3] += tres * 2
    counts[2] -= tres * 3

if counts[1] == 0:
    if counts[2] == 1:
        if counts[4] > 0:
            res += 1
        elif counts[3] > 1:
            res += 2
        else:
            res = -1
    elif counts[2] == 2:
        res += 2
elif counts[1] == 1:
    if counts[2] == 1:
        res += 1
    elif counts[2] == 2:
        if counts[4] > 0:
            res += 2
        elif counts[3] > 0:
            res += 3
        else:
            res = -1
    elif counts[2] == 0:
        if counts[3] > 0:
            res += 1
        elif counts[4] > 1:
            res += 2
        else:
            res = -1
elif counts[1] == 2:
    if counts[2] == 0:
        if counts[4] > 0:
            res += 2
        elif counts[3] > 1:
            res += 2
        else:
            res = -1
    elif counts[2] == 1:
        res += 2
    elif counts[2] == 2:
        res += 2

print res
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Ccompartmentsbootcamp(Basebootcamp):
    def __init__(self, max_compartments=10, **params):
        super().__init__(**params)
        self.max_compartments = max_compartments

    def case_generator(self):
        while True:
            n = random.randint(1, self.max_compartments)
            a = [random.randint(0, 4) for _ in range(n)]
            if sum(a) >= 1:
                return {"n": n, "a": a}

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case["n"]
        a = question_case["a"]
        a_str = " ".join(map(str, a))
        problem = (
            "A team of students is traveling in a train with several compartments. Each compartment currently has some students.\n"
            "They need to rearrange seats such that each compartment ends with 0, 3, or 4 students. Each swap requires convincing a non-student.\n"
            "Your task is to find the minimal number of non-students to convince, or output -1 if impossible.\n\n"
            f"Input Format:\n- First line: {n} (number of compartments)\n- Second line: {a_str} (students per compartment)\n\n"
            "Output the minimal number of swaps or -1. Place your answer within [answer]...[/answer] tags."
        )
        return problem

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_answer = matches[-1].strip()
        try:
            return int(last_answer)
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        a = identity["a"]
        correct = cls.compute_min_persuasion(a)
        return solution == correct

    @staticmethod
    def compute_min_persuasion(a):
        counts = [0] * 5
        for num in a:
            counts[num] += 1
        res = 0

        # Pair 1s and 2s optimally
        t = min(counts[1], counts[2])
        res += t
        counts[1] -= t
        counts[2] -= t
        counts[3] += t

        # Process remaining 1s or 2s
        if counts[1] > 0:
            # Handle groups of 3 remaining 1s
            t = counts[1] // 3
            res += t * 2
            counts[3] += t
            counts[1] %= 3
        elif counts[2] > 0:
            # Handle groups of 3 remaining 2s
            t = counts[2] // 3
            res += t * 2
            counts[3] += t * 2
            counts[2] %= 3

        remaining_1 = counts[1]
        remaining_2 = counts[2]

        # Handle remaining cases
        if remaining_1 == 0:
            if remaining_2 == 1:
                if counts[4] >= 1:
                    res += 1
                elif counts[3] >= 2:
                    res += 2
                else:
                    return -1
            elif remaining_2 == 2:
                res += 2
            elif remaining_2 > 0:
                return -1
        elif remaining_1 == 1:
            if remaining_2 == 1:
                res += 1
            elif remaining_2 == 2:
                if counts[4] >= 1:
                    res += 2
                elif counts[3] >= 1:
                    res += 3
                else:
                    return -1
            elif remaining_2 == 0:
                if counts[3] >= 1:
                    res += 1
                elif counts[4] >= 2:
                    res += 2
                else:
                    return -1
            else:
                return -1
        elif remaining_1 == 2:
            if remaining_2 == 0:
                if counts[4] >= 1:
                    res += 2
                elif counts[3] >= 2:
                    res += 2
                else:
                    return -1
            elif remaining_2 in (1, 2):
                res += 2
            else:
                return -1
        else:
            return -1

        # Check if any remaining students after processing
        if counts[1] > 0 or counts[2] > 0:
            return -1
        return res
