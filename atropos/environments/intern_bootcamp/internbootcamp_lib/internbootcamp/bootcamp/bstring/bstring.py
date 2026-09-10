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
from heapq import heapify
S=raw_input()
n=input()
q=[]
N=len(S)
if N*(N+1)/2<n:
    print \"No such line.\"
    exit()
i=0
while i < N:
    q.append((S[i],i))
    i+=1
heapify(q)
while n:
    n-=1
    t=heappop(q)
    if not n :
        print t[0]
    if t[1]+1 < N:
        heappush(q,(t[0]+S[t[1]+1],t[1]+1))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from heapq import heappush, heappop, heapify
from bootcamp import Basebootcamp

class Bstringbootcamp(Basebootcamp):
    def __init__(self, max_str_length=10, max_k=100000):
        self.max_str_length = max_str_length
        self.max_k = max_k
    
    def case_generator(self):
        # Generate random string (length >=1)
        length = random.randint(1, self.max_str_length)
        chars = [random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(length)]
        s = ''.join(chars)
        
        # Calculate total possible substrings
        total_substrings = length * (length + 1) // 2
        
        # Generate k with controlled distribution
        if random.random() < 0.7:  # 70% valid k
            k = random.randint(1, min(total_substrings, self.max_k))
        else:  # 30% invalid k
            k = random.randint(
                min(total_substrings + 1, self.max_k),
                self.max_k  # Ensure k never exceeds problem constraints
            )
        return {'s': s, 'k': k}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        s = question_case['s']
        k = question_case['k']
        prompt = (
            f"In an IT lesson, Anna and Maria learned about lexicographic order. Their homework is to find the k-th lexicographically smallest substring of a given string.\n\n"
            f"**Rules:**\n"
            f"1. All possible contiguous substrings are considered (including duplicates from different starting positions). For example, 'aa' has substrings: 'a'(pos0), 'a'(pos1), 'aa'.\n"
            f"2. Substrings are ordered lexicographically as defined by the '<' operator.\n"
            f"3. If there are fewer than k substrings, output \"No such line.\"\n\n"
            f"**Input:**\nString: {s}\nk: {k}\n\n"
            f"**Task:**\nOutput the k-th substring. Enclose your answer in [answer][/answer] tags.\n\n"
            f"**Example:**\nInput: aa\nk=2\nAnswer: [answer]a[/answer]"
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        s = identity['s']
        k = identity['k']
        
        # Edge case: k exceeds maximum possible
        total = len(s) * (len(s)+1) // 2
        if k > total or k > identity.get('max_k', 10**5):
            return solution == "No such line."
        
        # Heap-based k-th smallest calculation
        heap = [(s[i], i) for i in range(len(s))]
        heapify(heap)
        
        result = None
        for _ in range(k):
            curr, pos = heappop(heap)
            result = curr
            if pos + 1 < len(s):
                heappush(heap, (curr + s[pos+1], pos+1))
        
        return solution == result
