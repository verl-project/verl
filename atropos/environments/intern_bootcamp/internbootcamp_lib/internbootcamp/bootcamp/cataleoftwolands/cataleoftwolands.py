"""# 

### 谜题描述
The legend of the foundation of Vectorland talks of two integers x and y. Centuries ago, the array king placed two markers at points |x| and |y| on the number line and conquered all the land in between (including the endpoints), which he declared to be Arrayland. Many years later, the vector king placed markers at points |x - y| and |x + y| and conquered all the land in between (including the endpoints), which he declared to be Vectorland. He did so in such a way that the land of Arrayland was completely inside (including the endpoints) the land of Vectorland.

Here |z| denotes the absolute value of z.

Now, Jose is stuck on a question of his history exam: \"What are the values of x and y?\" Jose doesn't know the answer, but he believes he has narrowed the possible answers down to n integers a_1, a_2, ..., a_n. Now, he wants to know the number of unordered pairs formed by two different elements from these n integers such that the legend could be true if x and y were equal to these two values. Note that it is possible that Jose is wrong, and that no pairs could possibly make the legend true.

Input

The first line contains a single integer n (2 ≤ n ≤ 2 ⋅ 10^5) — the number of choices.

The second line contains n pairwise distinct integers a_1, a_2, ..., a_n (-10^9 ≤ a_i ≤ 10^9) — the choices Jose is considering.

Output

Print a single integer number — the number of unordered pairs \\{x, y\} formed by different numbers from Jose's choices that could make the legend true.

Examples

Input


3
2 5 -3


Output


2


Input


2
3 6


Output


1

Note

Consider the first sample. For the pair \{2, 5\}, the situation looks as follows, with the Arrayland markers at |2| = 2 and |5| = 5, while the Vectorland markers are located at |2 - 5| = 3 and |2 + 5| = 7:

<image>

The legend is not true in this case, because the interval [2, 3] is not conquered by Vectorland. For the pair \{5, -3\} the situation looks as follows, with Arrayland consisting of the interval [3, 5] and Vectorland consisting of the interval [2, 8]:

<image>

As Vectorland completely contains Arrayland, the legend is true. It can also be shown that the legend is true for the pair \{2, -3\}, for a total of two pairs.

In the second sample, the only pair is \{3, 6\}, and the situation looks as follows:

<image>

Note that even though Arrayland and Vectorland share 3 as endpoint, we still consider Arrayland to be completely inside of Vectorland.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = int(raw_input())
a = list(map(lambda x: abs(int(x)), raw_input().split()))
a.sort()

ans = 0
from bisect import bisect_right
for i in xrange(n):
    x = bisect_right(a, 2*a[i])
    ans += x - i - 1
print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import bisect
import random
import re
from bootcamp import Basebootcamp

class Cataleoftwolandsbootcamp(Basebootcamp):
    def __init__(self, max_n=5, min_n=2, min_val=0, max_val=20, allow_negative=True):
        self.max_n = max_n
        self.min_n = min_n
        self.min_val = min_val
        self.max_val = max_val
        self.allow_negative = allow_negative

    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        existing = set()
        a = []
        while len(a) < n:
            num = random.randint(self.min_val, self.max_val)
            if self.allow_negative and random.choice([True, False]):
                num *= -1
            if num not in existing:
                a.append(num)
                existing.add(num)
        return {'n': n, 'a': a}

    @staticmethod
    def prompt_func(question_case):
        a_list = question_case['a']
        a_str = ' '.join(map(str, a_list))
        return f"""You are tasked with solving a historical mathematics problem about Vectorland and Arrayland. 

**Problem Statement:**
Two integers x and y form a valid pair if the Arrayland interval [min(|x|, |y|), max(|x|, |y|)] is fully contained within the Vectorland interval [min(|x−y|, |x+y|), max(|x−y|, |x+y|)]. 

**Input Format:**
- First line: Integer n (2 ≤ n ≤ 2×10^5)
- Second line: n distinct integers (space-separated)

**Sample Input 1:**
3
2 5 -3

**Sample Output 1:**
2

**Your Input:**
{question_case['n']}
{a_str}

Calculate the answer and put ONLY THE FINAL INTEGER within [answer] tags like: [answer]5[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        a = list(map(abs, identity['a']))
        a_sorted = sorted(a)
        n = identity['n']
        ans = 0
        for i in range(n):
            threshold = 2 * a_sorted[i]
            pos = bisect.bisect_right(a_sorted, threshold)
            ans += (pos - i - 1)
        return solution == ans
