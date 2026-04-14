"""# 

### 谜题描述
One day in the IT lesson Anna and Maria learned about the lexicographic order.

String x is lexicographically less than string y, if either x is a prefix of y (and x ≠ y), or there exists such i (1 ≤ i ≤ min(|x|, |y|)), that xi < yi, and for any j (1 ≤ j < i) xj = yj. Here |a| denotes the length of the string a. The lexicographic comparison of strings is implemented by operator < in modern programming languages​​.

The teacher gave Anna and Maria homework. She gave them a string of length n. They should write out all substrings of the given string, including the whole initial string, and the equal substrings (for example, one should write out the following substrings from the string \"aab\": \"a\", \"a\", \"aa\", \"ab\", \"aab\", \"b\"). The resulting strings should be sorted in the lexicographical order. The cunning teacher doesn't want to check all these strings. That's why she said to find only the k-th string from the list. Help Anna and Maria do the homework.

Input

The first line contains a non-empty string that only consists of small Latin letters (\"a\"-\"z\"), whose length does not exceed 105. The second line contains the only integer k (1 ≤ k ≤ 105).

Output

Print the string Anna and Maria need — the k-th (in the lexicographical order) substring of the given string. If the total number of substrings is less than k, print a string saying \"No such line.\" (without the quotes).

Examples

Input

aa
2


Output

a


Input

abc
5


Output

bc


Input

abab
7


Output

b

Note

In the second sample before string \"bc\" follow strings \"a\", \"ab\", \"abc\", \"b\".

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from heapq import heappush
from heapq import heappop
s = raw_input()
n = input()
#n, = map(int, raw_input().strip())
h = []
ls = len(s)
if ls * (ls + 1) / 2 < n:
    print \"No such line.\"
    exit(0)
for i in xrange(ls):
    heappush(h, (s[i], i))
while n > 0:
    now = heappop(h)
    n = n -1
    if n == 0:
        print now[0]
        exit(0)
    if now[1] < ls - 1:    
        heappush(h, (now[0] + s[now[1] + 1], now[1] + 1))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import string
import heapq
from bootcamp import Basebootcamp

class Dstringbootcamp(Basebootcamp):
    def __init__(self, s_len_min=3, s_len_max=10, k_min=1, k_max=100):
        self.s_len_min = s_len_min
        self.s_len_max = s_len_max
        self.k_min = k_min
        self.k_max = k_max

    def case_generator(self):
        n = random.randint(self.s_len_min, self.s_len_max)
        s = ''.join(random.choices(string.ascii_lowercase, k=n))
        total = n * (n + 1) // 2
        # Ensure k does not exceed the total number of substrings
        k = random.randint(self.k_min, min(self.k_max, total))
        return {
            's': s,
            'k': k
        }

    @staticmethod
    def prompt_func(question_case):
        s = question_case['s']
        k = question_case['k']
        prompt = f"Anna and Maria are given the string: {s}\n"
        prompt += f"They need to find the {k}-th lexicographically smallest substring. "
        prompt += "All possible substrings, including duplicates, should be considered and sorted in lex order. "
        prompt += "If the total number of substrings is less than {k}, output 'No such line.' "
        prompt += "Please provide your answer within [answer] and [/answer] tags."
        return prompt

    @staticmethod
    def extract_output(output):
        start = output.rfind('[answer]')
        if start == -1:
            return None
        end = output.find('[/answer]', start)
        if end == -1:
            return None
        return output[start+8:end].strip()

    @classmethod
    def _verify_correction(cls, solution, identity):
        s = identity['s']
        k = identity['k']
        n = len(s)
        total = n * (n + 1) // 2
        if k > total:
            return solution == "No such line."
        heap = []
        for i in range(len(s)):
            heapq.heappush(heap, (s[i], i))
        current = None
        for _ in range(k):
            current = heapq.heappop(heap)
            if current[1] < len(s) - 1:
                next_char = s[current[1] + 1]
                heapq.heappush(heap, (current[0] + next_char, current[1] + 1))
        expected = current[0] if k <= total else "No such line."
        return solution == expected
