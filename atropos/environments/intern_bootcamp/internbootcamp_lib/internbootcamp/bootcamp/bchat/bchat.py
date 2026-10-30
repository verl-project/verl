"""# 

### 谜题描述
There are times you recall a good old friend and everything you've come through together. Luckily there are social networks — they store all your message history making it easy to know what you argued over 10 years ago.

More formal, your message history is a sequence of messages ordered by time sent numbered from 1 to n where n is the total number of messages in the chat.

Each message might contain a link to an earlier message which it is a reply to. When opening a message x or getting a link to it, the dialogue is shown in such a way that k previous messages, message x and k next messages are visible (with respect to message x). In case there are less than k messages somewhere, they are yet all shown.

Digging deep into your message history, you always read all visible messages and then go by the link in the current message x (if there is one) and continue reading in the same manner.

Determine the number of messages you'll read if your start from message number t for all t from 1 to n. Calculate these numbers independently. If you start with message x, the initial configuration is x itself, k previous and k next messages. Messages read multiple times are considered as one.

Input

The first line contains two integers n and k (1 ≤ n ≤ 105, 0 ≤ k ≤ n) — the total amount of messages and the number of previous and next messages visible.

The second line features a sequence of integers a1, a2, ..., an (0 ≤ ai < i), where ai denotes the i-th message link destination or zero, if there's no link from i. All messages are listed in chronological order. It's guaranteed that the link from message x goes to message with number strictly less than x.

Output

Print n integers with i-th denoting the number of distinct messages you can read starting from message i and traversing the links while possible.

Examples

Input

6 0
0 1 1 2 3 2


Output

1 2 2 3 3 3 


Input

10 1
0 1 0 3 4 5 2 3 7 0


Output

2 3 3 4 5 6 6 6 8 2 


Input

2 2
0 1


Output

2 2 

Note

Consider i = 6 in sample case one. You will read message 6, then 2, then 1 and then there will be no link to go.

In the second sample case i = 6 gives you messages 5, 6, 7 since k = 1, then 4, 5, 6, then 2, 3, 4 and then the link sequence breaks. The number of distinct messages here is equal to 6.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys


def main():
    n, k = map(int, sys.stdin.readline().split())
    x = list(map(int, sys.stdin.readline().split()))
    for i in range(n):
        x[i] -= 1
    a = [0] * n
    for i in range(n):
        c = 1 + (i - max(0, i - k)) + (min(n - 1, i + k) - i)
        if x[i] != -1:
            if i - x[i] > 2 * k:
                c += a[x[i]]
            else:
                c = a[x[i]] + (min(n - 1, i + k)  - min(n - 1, x[i] + k))
        a[i] = c

    print(\" \".join(map(str, a)))


main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

def compute_answer(n, k, links):
    x = [a - 1 for a in links]
    a = [0] * n
    for i in range(n):
        left = max(0, i - k)
        right = min(n - 1, i + k)
        visible = right - left + 1
        c = visible
        if x[i] != -1:
            if (i - x[i]) > 2 * k:
                c += a[x[i]]
            else:
                overlap_right = min(n - 1, x[i] + k)
                current_right = min(n - 1, i + k)
                additional = current_right - overlap_right
                c = a[x[i]] + additional
        a[i] = c
    return a

class Bchatbootcamp(Basebootcamp):
    def __init__(self, max_n=10, **kwargs):
        super().__init__(**kwargs)
        self.max_n = max_n
    
    def case_generator(self):
        n = random.randint(1, self.max_n)
        k = random.randint(0, n)
        links = []
        for i in range(1, n + 1):
            if i == 1:
                ai = 0
            else:
                choices = [0] + list(range(1, i))
                ai = random.choice(choices)
            links.append(ai)
        correct_output = compute_answer(n, k, links)
        return {
            'n': n,
            'k': k,
            'links': links,
            'correct_output': correct_output
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k = question_case['k']
        links = question_case['links']
        prompt = f"""You are analyzing your social network message history. Each message is numbered from 1 to {n} in chronological order. Each message may contain a link to an earlier message (strictly less than its own number) or no link. 

When you open a message x, you will see up to {k} previous messages, message x itself, and up to {k} next messages. If there are fewer than {k} messages in either direction, all available messages are shown. You repeatedly follow any links in the current message until there are no more links to follow. Each message is counted only once, regardless of how many times it is viewed.

Your task is to determine, for each starting message from 1 to {n}, the number of distinct messages read.

Input parameters:
- n = {n}
- k = {k}
- Links: {links} (each element a_i corresponds to the message linked by message i, where i ranges from 1 to {n}. A value of 0 indicates no link.)

Output a sequence of {n} integers separated by spaces, where the i-th integer corresponds to the result for starting at message i. Place your answer within [answer] and [/answer] tags. For example: [answer]1 2 3 4 5 6[/answer]"""

        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            solution = list(map(int, last_match.split()))
            return solution
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_output']
