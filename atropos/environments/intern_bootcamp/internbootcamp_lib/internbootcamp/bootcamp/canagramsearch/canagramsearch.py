"""# 

### 谜题描述
A string t is called an anagram of the string s, if it is possible to rearrange letters in t so that it is identical to the string s. For example, the string \"aab\" is an anagram of the string \"aba\" and the string \"aaa\" is not.

The string t is called a substring of the string s if it can be read starting from some position in the string s. For example, the string \"aba\" has six substrings: \"a\", \"b\", \"a\", \"ab\", \"ba\", \"aba\".

You are given a string s, consisting of lowercase Latin letters and characters \"?\". You are also given a string p, consisting of lowercase Latin letters only. Let's assume that a string is good if you can obtain an anagram of the string p from it, replacing the \"?\" characters by Latin letters. Each \"?\" can be replaced by exactly one character of the Latin alphabet. For example, if the string p = «aba», then the string \"a??\" is good, and the string «?bc» is not. 

Your task is to find the number of good substrings of the string s (identical substrings must be counted in the answer several times).

Input

The first line is non-empty string s, consisting of no more than 105 lowercase Latin letters and characters \"?\". The second line is non-empty string p, consisting of no more than 105 lowercase Latin letters. Please note that the length of the string p can exceed the length of the string s.

Output

Print the single number representing the number of good substrings of string s.

Two substrings are considered different in their positions of occurrence are different. Thus, if some string occurs several times, then it should be counted the same number of times.

Examples

Input

bb??x???
aab


Output

2


Input

ab?c
acb


Output

2

Note

Consider the first sample test. Here the string s has two good substrings: \"b??\" (after we replace the question marks we get \"baa\"), \"???\" (after we replace the question marks we get \"baa\").

Let's consider the second sample test. Here the string s has two good substrings: \"ab?\" (\"?\" can be replaced by \"c\"), \"b?c\" (\"?\" can be replaced by \"a\").

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
s = raw_input ()
p = raw_input ()
n = len (s)
m = len (p)

countInP = [0] * 26
for i in p:
	countInP[ord (i) - ord ('a')] += 1

ans = 0

for i in xrange (n):
	if s[i] != '?': countInP[ord (s[i]) - ord ('a')] -= 1
	if i < m - 1 : continue
	if i - m >= 0 and s[i - m] != '?': countInP[ord (s[i - m]) - ord ('a')] += 1
	if all (k >= 0 for k in countInP): ans += 1
print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import string
from collections import defaultdict
from bootcamp import Basebootcamp

def calculate_answer(s, p):
    m = len(p)
    n = len(s)
    if m == 0 or n < m:
        return 0
    
    # Initialize frequency counter for p
    count_p = defaultdict(int)
    for c in p:
        count_p[c] += 1
    
    # Initialize sliding window parameters
    current_counts = defaultdict(int)
    required = len(count_p)
    formed = 0
    ans = 0
    q_count = 0  # number of '?' in current window
    
    left = 0
    for right in range(n):
        # Add right character
        char = s[right]
        if char == '?':
            q_count += 1
        else:
            current_counts[char] += 1
            if current_counts[char] == count_p.get(char, 0):
                formed += 1
        
        # Maintain window size m
        if right - left + 1 > m:
            # Remove left character
            left_char = s[left]
            if left_char == '?':
                q_count -= 1
            else:
                if current_counts[left_char] == count_p.get(left_char, 0):
                    formed -= 1
                current_counts[left_char] -= 1
            left += 1
        
        # Check window validity when window size is exactly m
        if right - left + 1 == m:
            # Calculate needed characters
            needed = sum(max(0, count_p[c] - current_counts[c]) for c in count_p)
            if needed <= q_count and formed == required:
                ans += 1
    
    return ans

class Canagramsearchbootcamp(Basebootcamp):
    def __init__(self, max_s_length=10, max_p_length=5, **kwargs):
        super().__init__(**kwargs)
        self.max_s_length = max_s_length
        self.max_p_length = max_p_length
    
    def case_generator(self):
        # Generate valid length combination
        m = random.randint(1, self.max_p_length)
        max_s_len = max(m + random.randint(-2,3), 1)  # allow s shorter than p
        s_len = random.randint(1, self.max_s_length)
        
        # Generate p with random letters
        p = ''.join(random.choices(string.ascii_lowercase, k=m))
        
        # Generate s with letters and ?'s
        s = []
        num_q = random.randint(0, min(s_len, 3))
        for _ in range(s_len):
            if random.random() < 0.3 and num_q > 0:
                s.append('?')
                num_q -= 1
            else:
                s.append(random.choice(string.ascii_lowercase))
        
        return {
            's': ''.join(s),
            'p': p,
            '_answer': calculate_answer(''.join(s), p)  # precompute answer
        }
    
    @staticmethod
    def prompt_func(question_case):
        return f"""Given strings:
s = "{question_case['s']}" (with '?' wildcards)
p = "{question_case['p']}"

Count how many substrings of s (with length {len(question_case['p'])} if possible) can become anagrams of p after replacing '?'s. 

Output format: 
[answer]NUMBER[/answer]

Example: If answer is 5:
[answer]5[/answer]"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](\d+)', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['_answer']
