"""# 

### 谜题描述
Ryouko is an extremely forgetful girl, she could even forget something that has just happened. So in order to remember, she takes a notebook with her, called Ryouko's Memory Note. She writes what she sees and what she hears on the notebook, and the notebook became her memory.

Though Ryouko is forgetful, she is also born with superb analyzing abilities. However, analyzing depends greatly on gathered information, in other words, memory. So she has to shuffle through her notebook whenever she needs to analyze, which is tough work.

Ryouko's notebook consists of n pages, numbered from 1 to n. To make life (and this problem) easier, we consider that to turn from page x to page y, |x - y| pages should be turned. During analyzing, Ryouko needs m pieces of information, the i-th piece of information is on page ai. Information must be read from the notebook in order, so the total number of pages that Ryouko needs to turn is <image>.

Ryouko wants to decrease the number of pages that need to be turned. In order to achieve this, she can merge two pages of her notebook. If Ryouko merges page x to page y, she would copy all the information on page x to y (1 ≤ x, y ≤ n), and consequently, all elements in sequence a that was x would become y. Note that x can be equal to y, in which case no changes take place.

Please tell Ryouko the minimum number of pages that she needs to turn. Note she can apply the described operation at most once before the reading. Note that the answer can exceed 32-bit integers.

Input

The first line of input contains two integers n and m (1 ≤ n, m ≤ 105).

The next line contains m integers separated by spaces: a1, a2, ..., am (1 ≤ ai ≤ n).

Output

Print a single integer — the minimum number of pages Ryouko needs to turn.

Examples

Input

4 6
1 2 3 4 3 2


Output

3


Input

10 5
9 4 3 8 8


Output

6

Note

In the first sample, the optimal solution is to merge page 4 to 3, after merging sequence a becomes {1, 2, 3, 3, 3, 2}, so the number of pages Ryouko needs to turn is |1 - 2| + |2 - 3| + |3 - 3| + |3 - 3| + |3 - 2| = 3.

In the second sample, optimal solution is achieved by merging page 9 to 4.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
R = lambda: map(int, raw_input().split())
(n, m), a = R(), R()
v = [[] for i in range(n + 1)]
ans = 0
for i in range(1, m):
	if a[i] != a[i - 1]:
		v[a[i - 1]].append(a[i])
		v[a[i]].append(a[i - 1])
		ans += abs(a[i] - a[i - 1])

max_change = 0
for i in range(1, n + 1):
	l = len(v[i])
	v[i].sort()
	suma = sumb = 0
	for x in v[i]:
		suma += abs(x - i)
	left, right = 0, l - 1
	while left < right:
		sumb += v[i][right] - v[i][left]
		left += 1
		right -= 1
	max_change = max(max_change, suma - sumb)
print ans - max_change
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Aryoukosmemorynotebootcamp(Basebootcamp):
    def __init__(self, max_n=100, max_m=100, **params):
        self.max_n = max_n
        self.max_m = max_m
    
    def case_generator(self):
        while True:
            n = random.randint(2, self.max_n)
            m = random.randint(2, self.max_m)
            a = [random.randint(1, n) for _ in range(m)]
            if any(a[i] != a[i-1] for i in range(1, m)):
                correct_answer = self.calculate_min_pages(n, m, a)
                return {
                    'n': n,
                    'm': m,
                    'a': a,
                    'correct_answer': correct_answer
                }

    @staticmethod
    def calculate_min_pages(n, m, a):
        if m < 2:
            return 0
        v = [[] for _ in range(n + 1)]
        ans = 0
        for i in range(1, m):
            prev = a[i-1]
            curr = a[i]
            if prev != curr:
                v[prev].append(curr)
                v[curr].append(prev)
                ans += abs(curr - prev)
        max_change = 0
        for i in range(1, n + 1):
            neighbors = v[i]
            if not neighbors:
                continue
            sorted_nb = sorted(neighbors)
            suma = sum(abs(x - i) for x in sorted_nb)
            sumb = 0
            left = 0
            right = len(sorted_nb) - 1
            while left < right:
                sumb += sorted_nb[right] - sorted_nb[left]
                left += 1
                right -= 1
            current_change = suma - sumb
            if current_change > max_change:
                max_change = current_change
        return ans - max_change

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        a = question_case['a']
        problem = (
            "Ryouko's Memory Note Problem\n\n"
            "Background:\n"
            "Ryouko has a notebook with {n} pages, numbered 1 to {n}. She needs to read {m} pieces of information in order. The sequence of pages she visits is {a}.\n"
            "She can merge one page (x) into another (y), which replaces all occurrences of x in the sequence with y. This operation can be done at most once.\n\n"
            "Task:\n"
            "Calculate the minimal total pages she needs to turn after this operation. The total is the sum of absolute differences between consecutive pages in the modified sequence.\n\n"
            "Output Requirements:\n"
            "Your answer must be a single integer inside [answer] and [/answer] tags. Example: [answer]42[/answer].\n"
            "Provide the minimal possible total pages turned.\n"
        ).format(n=n, m=m, a=a)
        return problem

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity.get('correct_answer', None)
