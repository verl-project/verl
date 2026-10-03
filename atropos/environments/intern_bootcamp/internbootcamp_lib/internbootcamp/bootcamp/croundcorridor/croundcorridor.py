"""# 

### 谜题描述
Amugae is in a very large round corridor. The corridor consists of two areas. The inner area is equally divided by n sectors, and the outer area is equally divided by m sectors. A wall exists between each pair of sectors of same area (inner or outer), but there is no wall between the inner area and the outer area. A wall always exists at the 12 o'clock position.

<image>

The inner area's sectors are denoted as (1,1), (1,2), ..., (1,n) in clockwise direction. The outer area's sectors are denoted as (2,1), (2,2), ..., (2,m) in the same manner. For a clear understanding, see the example image above.

Amugae wants to know if he can move from one sector to another sector. He has q questions.

For each question, check if he can move between two given sectors.

Input

The first line contains three integers n, m and q (1 ≤ n, m ≤ 10^{18}, 1 ≤ q ≤ 10^4) — the number of sectors in the inner area, the number of sectors in the outer area and the number of questions.

Each of the next q lines contains four integers s_x, s_y, e_x, e_y (1 ≤ s_x, e_x ≤ 2; if s_x = 1, then 1 ≤ s_y ≤ n, otherwise 1 ≤ s_y ≤ m; constraints on e_y are similar). Amague wants to know if it is possible to move from sector (s_x, s_y) to sector (e_x, e_y).

Output

For each question, print \"YES\" if Amugae can move from (s_x, s_y) to (e_x, e_y), and \"NO\" otherwise.

You can print each letter in any case (upper or lower).

Example

Input


4 6 3
1 1 2 3
2 6 1 2
2 6 2 4


Output


YES
NO
YES

Note

Example is shown on the picture in the statement.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def gcd(a, b):
	if a == 0:
		return b
	return gcd(b % a, a)

def process(t, x, tn, tm):  # tn = total / n
	if t == 1:
		return tn * x
	else:
		return tm * x


def main():
	n, m, q = map(int, raw_input().split())
	gc = gcd(n, m)
	total = n * m / gc
	tn = total / n
	tm = total / m
	gc = tn * tm / gcd(tn, tm)
	# print '!', total, tn, tm, gc
	for i in range(q):
		t1, x, t2, y = map(int, raw_input().split())
		x = process(t1, x - 1, tn, tm)
		y = process(t2, y - 1, tn, tm)
		# print '!', x, y
		if x / gc == y / gc:
			print 'YES'
		else:
			print 'NO'


if __name__ == '__main__':
	main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import re
import random
from math import gcd
from bootcamp import Basebootcamp

class Croundcorridorbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', 4)
        self.m = params.get('m', 6)
        self.q = params.get('q', 3)

    def case_generator(self):
        n = self.n
        m = self.m
        q = self.q

        g = gcd(n, m)
        total = (n * m) // g
        tn = total // n  # m // g
        tm = total // m  # n // g
        current_gcd = gcd(tn, tm)
        gc = (tn * tm) // current_gcd

        queries = []

        for _ in range(q):
            s_x = random.choice([1, 2])
            e_x = random.choice([1, 2])

            s_y = random.randint(1, n) if s_x == 1 else random.randint(1, m)
            e_y = random.randint(1, n) if e_x == 1 else random.randint(1, m)

            # Calculate processed coordinates
            x = (s_y - 1) * tn if s_x == 1 else (s_y - 1) * tm
            y = (e_y - 1) * tn if e_x == 1 else (e_y - 1) * tm

            block_x = x // gc
            block_y = y // gc

            answer = 'YES' if block_x == block_y else 'NO'
            queries.append({
                'input': [s_x, s_y, e_x, e_y],
                'answer': answer
            })

        case = {
            'n': n,
            'm': m,
            'q': q,
            'queries': queries
        }
        return case

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        m = question_case['m']
        q = question_case['q']
        queries = question_case['queries']

        prompt = (
            f"Croundcorridor is in a large circular corridor divided into an inner area and an outer area. The inner area is divided into {n} sectors, numbered (1,1) to (1,{n}) clockwise. The outer area is divided into {m} sectors, numbered (2,1) to (2,{m}) clockwise. There are walls between adjacent sectors of the same area, but no walls between inner and outer sectors. A wall is always present at the 12 o'clock position.\n\n"
            f"Croundcorridor wants to determine if he can move from one sector to another. You will be given {q} queries. For each query, you must output YES if movement is possible and NO otherwise.\n\n"
            "Queries:\n"
        )

        for idx, query in enumerate(queries, 1):
            s_x, s_y, e_x, e_y = query['input']
            start_area = "inner" if s_x == 1 else "outer"
            end_area = "inner" if e_x == 1 else "outer"
            prompt += (
                f"Query {idx}: Start at sector ({s_x},{s_y}) in the {start_area} area. End at sector ({e_x},{e_y}) in the {end_area} area.\n"
            )

        prompt += (
            "\nOutput your answers as a space-separated list within [answer] tags. Example: [answer]YES NO YES[/answer]"
        )
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.IGNORECASE | re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        answers = []
        for part in last_match.split():
            normalized = part.upper()
            if normalized in ('YES', 'NO'):
                answers.append(normalized)
        return ' '.join(answers) if answers else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        solution_answers = solution.split()
        expected_answers = [q['answer'] for q in identity['queries']]
        if len(solution_answers) != len(expected_answers):
            return False
        for sol, exp in zip(solution_answers, expected_answers):
            if sol != exp.upper():
                return False
        return True
