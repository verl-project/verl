"""# 

### 谜题描述
On the Literature lesson Sergei noticed an awful injustice, it seems that some students are asked more often than others.

Seating in the class looks like a rectangle, where n rows with m pupils in each. 

The teacher asks pupils in the following order: at first, she asks all pupils from the first row in the order of their seating, then she continues to ask pupils from the next row. If the teacher asked the last row, then the direction of the poll changes, it means that she asks the previous row. The order of asking the rows looks as follows: the 1-st row, the 2-nd row, ..., the n - 1-st row, the n-th row, the n - 1-st row, ..., the 2-nd row, the 1-st row, the 2-nd row, ...

The order of asking of pupils on the same row is always the same: the 1-st pupil, the 2-nd pupil, ..., the m-th pupil.

During the lesson the teacher managed to ask exactly k questions from pupils in order described above. Sergei seats on the x-th row, on the y-th place in the row. Sergei decided to prove to the teacher that pupils are asked irregularly, help him count three values:

  1. the maximum number of questions a particular pupil is asked, 
  2. the minimum number of questions a particular pupil is asked, 
  3. how many times the teacher asked Sergei. 



If there is only one row in the class, then the teacher always asks children from this row.

Input

The first and the only line contains five integers n, m, k, x and y (1 ≤ n, m ≤ 100, 1 ≤ k ≤ 1018, 1 ≤ x ≤ n, 1 ≤ y ≤ m).

Output

Print three integers:

  1. the maximum number of questions a particular pupil is asked, 
  2. the minimum number of questions a particular pupil is asked, 
  3. how many times the teacher asked Sergei. 

Examples

Input

1 3 8 1 1


Output

3 2 3

Input

4 2 9 4 2


Output

2 1 1

Input

5 5 25 4 3


Output

1 1 1

Input

100 100 1000000000000000000 100 100


Output

101010101010101 50505050505051 50505050505051

Note

The order of asking pupils in the first test: 

  1. the pupil from the first row who seats at the first table, it means it is Sergei; 
  2. the pupil from the first row who seats at the second table; 
  3. the pupil from the first row who seats at the third table; 
  4. the pupil from the first row who seats at the first table, it means it is Sergei; 
  5. the pupil from the first row who seats at the second table; 
  6. the pupil from the first row who seats at the third table; 
  7. the pupil from the first row who seats at the first table, it means it is Sergei; 
  8. the pupil from the first row who seats at the second table; 



The order of asking pupils in the second test: 

  1. the pupil from the first row who seats at the first table; 
  2. the pupil from the first row who seats at the second table; 
  3. the pupil from the second row who seats at the first table; 
  4. the pupil from the second row who seats at the second table; 
  5. the pupil from the third row who seats at the first table; 
  6. the pupil from the third row who seats at the second table; 
  7. the pupil from the fourth row who seats at the first table; 
  8. the pupil from the fourth row who seats at the second table, it means it is Sergei; 
  9. the pupil from the third row who seats at the first table; 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def times(x,y):
    if n == 1:
        rounds = k / m
        left = k - rounds * m
        if left >= y:
            rounds += 1
        return rounds
    rounds = k / ((n-1)*m)
    left = k - (rounds * (n-1)* m)
    if x == 1:
        if rounds % 2 == 1:
            t = (rounds / 2) + 1 
        else:
            t = rounds / 2 
    elif x == n:
        t = rounds / 2
    else:
        t = rounds
    if rounds % 2 == 0:
        # from top
        if (left >= ((x - 1) * m + y)):
            t += 1  
    else:
        # from bottom
        if (left >= ((n-x) * m + y)):
            t += 1
    return t


n,m,k,x,y = map(long, raw_input().split())
maximum = 0
minimum = k
for i in xrange(1,n+1):
    for j in xrange(1,m+1):
        ans = times(i,j)
        if ans > maximum:
            maximum = ans
        if ans < minimum:
            minimum = ans

print maximum, minimum, times(x,y)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Cunfairpollbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n_min = params.get('n_min', 1)
        self.n_max = params.get('n_max', 100)
        self.m_min = params.get('m_min', 1)
        self.m_max = params.get('m_max', 100)

    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        m = random.randint(self.m_min, self.m_max)
        x = random.randint(1, n)
        y = random.randint(1, m)
        k = random.randint(1, 10**18)
        return {
            'n': n,
            'm': m,
            'k': k,
            'x': x,
            'y': y,
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        k = question_case['k']
        x = question_case['x']
        y = question_case['y']
        prompt = (
            f"During literature class, pupils are seated in a grid of {n} rows and {m} columns. "
            f"The teacher asks questions in a specific order: starts from the first row to the {n}th row, "
            f"then reverses direction and asks from the {n-1}th row back to the second row, repeating this pattern. "
            f"Each row is asked from the first to the {m}th pupil. Exactly {k} questions were asked.\n\n"
            f"Sergei sits in row {x}, column {y}. Compute:\n"
            f"1. Maximum questions asked to any pupil\n"
            f"2. Minimum questions asked\n"
            f"3. Questions asked to Sergei\n\n"
            "Format your answer as three integers within [answer] and [/answer], e.g., [answer] 3 2 3 [/answer]."
        )
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            parts = list(map(int, last_match.split()))
            if len(parts) == 3:
                return tuple(parts)
        except ValueError:
            pass
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        m = identity['m']
        k = identity['k']
        x = identity['x']
        y = identity['y']

        def calculate(i, j):
            if n == 1:
                rounds = k // m
                left = k % m
                return rounds + 1 if left >= j else rounds
            else:
                cycle = (n - 1) * m
                rounds = k // cycle
                left = k % cycle
                if i == 1:
                    t = (rounds + 1) // 2 if rounds % 2 else rounds // 2
                elif i == n:
                    t = rounds // 2
                else:
                    t = rounds
                if rounds % 2 == 0:
                    required = (i - 1) * m + j
                else:
                    required = (n - i) * m + j
                if left >= required:
                    t += 1
                return t

        max_count = 0
        min_count = float('inf')
        ser_count = calculate(x, y)

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cnt = calculate(i, j)
                max_count = max(max_count, cnt)
                min_count = min(min_count, cnt)

        expected = (max_count, min_count, ser_count)
        return solution == expected
