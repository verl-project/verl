"""# 

### 谜题描述
Jon Snow is on the lookout for some orbs required to defeat the white walkers. There are k different types of orbs and he needs at least one of each. One orb spawns daily at the base of a Weirwood tree north of the wall. The probability of this orb being of any kind is equal. As the north of wall is full of dangers, he wants to know the minimum number of days he should wait before sending a ranger to collect the orbs such that the probability of him getting at least one of each kind of orb is at least <image>, where ε < 10 - 7.

To better prepare himself, he wants to know the answer for q different values of pi. Since he is busy designing the battle strategy with Sam, he asks you for your help.

Input

First line consists of two space separated integers k, q (1 ≤ k, q ≤ 1000) — number of different kinds of orbs and number of queries respectively.

Each of the next q lines contain a single integer pi (1 ≤ pi ≤ 1000) — i-th query.

Output

Output q lines. On i-th of them output single integer — answer for i-th query.

Examples

Input

1 1
1


Output

1


Input

2 2
1
2


Output

2
2

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import random
import time

seed = int(time.time())
random.seed(seed)

def solve(k, p):
    pass

def main():
    xs = [int(x) for x in raw_input().strip().split()]
    k = xs[0]
    dp = [[0.0 for _ in range(k+1)] for _ in range(10000)]
    for d in range(1, 10000):
        dp[d][1] = 1.0
        for ki in range(2, min(d, k)+1):
            dp[d][ki] = (float(k-ki+1)/k)*dp[d-1][ki-1] + (float(ki-1)/k)*dp[d-1][ki]
    for _ in range(xs[1]):
        p = int(raw_input().strip())
        for d in range(k, 10000):
            if dp[d][k] > float(p)/2000:
                print d
                break

if '__main__'==__name__:
    main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Djonandorbsbootcamp(Basebootcamp):
    def __init__(self, max_k=10, max_q=5, max_p=1000, d_max=10000):
        self.max_k = max_k
        self.max_q = max_q
        self.max_p = max_p
        self.d_max = d_max

    def case_generator(self):
        k = random.randint(1, self.max_k)
        q = random.randint(1, self.max_q)
        queries = [random.randint(1, self.max_p) for _ in range(q)]

        # Precompute dynamic programming table with correct recurrence
        dp = [[0.0] * (k + 1) for _ in range(self.d_max + 1)]
        dp[0][0] = 1.0

        for d in range(1, self.d_max + 1):
            for ki in range(0, min(k, d) + 1):
                if ki == 0:
                    dp[d][ki] = 0.0
                else:
                    term1 = 0.0
                    if ki - 1 <= (d - 1) and ki - 1 >= 0:
                        term1 = dp[d - 1][ki - 1] * ((k - (ki - 1)) / k)
                    term2 = 0.0
                    if ki <= (d - 1):
                        term2 = dp[d - 1][ki] * (ki / k)
                    dp[d][ki] = term1 + term2

        answers = []
        for p in queries:
            threshold = p / 2000.0
            answer = None
            for candidate_d in range(k, self.d_max + 1):
                if dp[candidate_d][k] >= threshold:
                    answer = candidate_d
                    break
            if answer is None:
                raise ValueError(f"No solution found for p={p}, k={k}")
            answers.append(answer)

        return {
            'k': k,
            'queries': queries,
            'answers': answers
        }

    @staticmethod
    def prompt_func(question_case):
        k = question_case['k']
        queries = question_case['queries']
        q = len(queries)
        input_lines = [f"{k} {q}"] + [str(p) for p in queries]
        input_example = '\n'.join(input_lines)

        prompt = f"""Jon Snow needs to collect {k} different types of magical orbs. Each day, one orb appears at the base of the Weirwood tree, with each type equally likely. He wants to determine the minimum number of days he must wait to ensure the probability of collecting at least one of each orb type is at least the specified threshold for each query.

The threshold for the i-th query is p_i/2000. For each query, calculate the smallest number of days required.

Input Format:
The first line contains two integers, k (number of orb types) and q (number of queries).
The next q lines each contain an integer p_i (the threshold parameter for the query).

Output Format:
Output q lines, each containing the minimum number of days for the corresponding query.

Example Input:
{input_example}

Please provide your answers for all queries, each on a new line, enclosed within [answer] and [/answer] tags. For example:

[answer]
{question_case['answers'][0]}
...
[/answer]

Ensure each value is correctly formatted and in order."""
        return prompt

    @staticmethod
    def extract_output(output):
        import re
        answer_blocks = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        last_block = answer_blocks[-1].strip()
        answers = []
        for line in last_block.split('\n'):
            line = line.strip()
            if line:
                try:
                    answers.append(int(line))
                except ValueError:
                    pass
        return answers if answers else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity.get('answers', [])
        if not solution or len(solution) != len(expected):
            return False
        return all(s == e for s, e in zip(solution, expected))
