"""# 

### 谜题描述
Mahmoud wrote a message s of length n. He wants to send it as a birthday present to his friend Moaz who likes strings. He wrote it on a magical paper but he was surprised because some characters disappeared while writing the string. That's because this magical paper doesn't allow character number i in the English alphabet to be written on it in a string of length more than ai. For example, if a1 = 2 he can't write character 'a' on this paper in a string of length 3 or more. String \"aa\" is allowed while string \"aaa\" is not.

Mahmoud decided to split the message into some non-empty substrings so that he can write every substring on an independent magical paper and fulfill the condition. The sum of their lengths should be n and they shouldn't overlap. For example, if a1 = 2 and he wants to send string \"aaa\", he can split it into \"a\" and \"aa\" and use 2 magical papers, or into \"a\", \"a\" and \"a\" and use 3 magical papers. He can't split it into \"aa\" and \"aa\" because the sum of their lengths is greater than n. He can split the message into single string if it fulfills the conditions.

A substring of string s is a string that consists of some consecutive characters from string s, strings \"ab\", \"abc\" and \"b\" are substrings of string \"abc\", while strings \"acb\" and \"ac\" are not. Any string is a substring of itself.

While Mahmoud was thinking of how to split the message, Ehab told him that there are many ways to split it. After that Mahmoud asked you three questions: 

  * How many ways are there to split the string into substrings such that every substring fulfills the condition of the magical paper, the sum of their lengths is n and they don't overlap? Compute the answer modulo 109 + 7. 
  * What is the maximum length of a substring that can appear in some valid splitting? 
  * What is the minimum number of substrings the message can be spit in? 



Two ways are considered different, if the sets of split positions differ. For example, splitting \"aa|a\" and \"a|aa\" are considered different splittings of message \"aaa\".

Input

The first line contains an integer n (1 ≤ n ≤ 103) denoting the length of the message.

The second line contains the message s of length n that consists of lowercase English letters.

The third line contains 26 integers a1, a2, ..., a26 (1 ≤ ax ≤ 103) — the maximum lengths of substring each letter can appear in.

Output

Print three lines.

In the first line print the number of ways to split the message into substrings and fulfill the conditions mentioned in the problem modulo 109 + 7.

In the second line print the length of the longest substring over all the ways.

In the third line print the minimum number of substrings over all the ways.

Examples

Input

3
aab
2 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1


Output

3
2
2


Input

10
abcdeabcde
5 5 5 5 4 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1


Output

401
4
3

Note

In the first example the three ways to split the message are: 

  * a|a|b
  * aa|b
  * a|ab



The longest substrings are \"aa\" and \"ab\" of length 2.

The minimum number of substrings is 2 in \"a|ab\" or \"aa|b\".

Notice that \"aab\" is not a possible splitting because the letter 'a' appears in a substring of length 3, while a1 = 2.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from itertools import imap
#!/usr/bin/env python3
def ri():
    return imap(int, raw_input().split())


n = int(raw_input())
s = raw_input()
s = [ord(x) - ord(u'a') for x in s]
a = list(ri())

T = [0 for i in xrange(n)]

maxs = 0
mins = 0
lasts = 0
for i in xrange(len(s)):
    if i == 0:
        maxs = 1
        T[i] = 1
        mins +=1
        continue
    mm = i
    for j in xrange(i, max(i-a[s[i]], -1), -1):
        length = i-j+1
        flag = 0
        for k in xrange(j, i+1):
            if a[s[k]] < length:
                flag =1
                break
        if flag == 1:
            break
        else:
            mm = j
            if j == 0:
                adding = 1
            else:
                adding = T[j-1]
            T[i] += adding
            T[i] = T[i]%(10**9 + 7)
    maxs = max(maxs, i-mm+1)
    if mm > lasts:
        mins+=1
        lasts = i

print T[n-1]
print maxs
print mins
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from collections import defaultdict
from bootcamp import Basebootcamp

def solve(n, s_str, a_list):
    s = [ord(c) - ord('a') for c in s_str]
    a = a_list
    MOD = 10**9 +7
    if n ==0:
        return (0,0,0)
    T = [0]*n
    maxs =0
    mins=0
    lasts=0
    for i in range(n):
        if i ==0:
            maxs =1
            T[i] =1
            mins =1
            lasts=0
        else:
            mm =i
            start_j = i
            end_j = max(i - a[s[i]], -1)
            for j in range(start_j, end_j, -1):
                if j <0:
                    break
                length = i -j +1
                flag =0
                for k in range(j, i+1):
                    if a[s[k]] < length:
                        flag=1
                        break
                if flag:
                    break
                else:
                    mm =j
                    if j ==0:
                        adding =1
                    else:
                        adding = T[j-1]
                    T[i] = (T[i] + adding) % MOD
            maxs = max(maxs, i - mm +1)
            if mm > lasts:
                mins +=1
                lasts =i
    ways = T[-1] % MOD if n else 0
    return (ways, maxs, mins)

class Cmahmoudandamessagebootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=10, char_set='abcdefghijklmnopqrstuvwxyz'):
        self.n_min = n_min
        self.n_max = n_max
        self.char_set = char_set
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        s = ''.join(random.choices(self.char_set, k=n))
        lengths = []
        remaining = n
        while remaining > 0:
            l = random.randint(1, remaining)
            lengths.append(l)
            remaining -= l
        
        max_length = defaultdict(int)
        start = 0
        for l in lengths:
            substring = s[start:start+l]
            for c in substring:
                if max_length[c] < l:
                    max_length[c] = l
            start += l
        
        a = [0]*26
        for i in range(26):
            c = chr(ord('a') + i)
            a[i] = max_length.get(c, 1)
        
        ways, max_sub, min_split = solve(n, s, a)
        return {
            'n': n,
            's': s,
            'a': a,
            'expected_ways': ways,
            'expected_max': max_sub,
            'expected_min': min_split,
        }
    
    @staticmethod
    def prompt_func(question_case):
        prompt = f"""Mahmoud wrote a message s of length n and wants to split it into non-overlapping substrings according to the magical paper's constraints. Each substring must satisfy that for every character in it, the character's corresponding a_i value is at least the length of the substring. Your task is to determine three things:

1. The number of valid ways to split the string modulo 10^9 +7.
2. The maximum possible length of a substring in any valid splitting.
3. The minimum number of substrings required in a valid splitting.

The input is as follows:

The first line contains an integer n ({question_case['n']} in this case) — the length of the message.
The second line contains the string "{question_case['s']}".
The third line contains 26 integers: {' '.join(map(str, question_case['a']))}.

Write your answers as three lines, each containing the respective result. Please enclose your answer within [answer] and [/answer] tags. For example:

[answer]
42
5
3
[/answer]

Please provide your solution:"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        lines = [line.strip() for line in last_match.split('\n') if line.strip()]
        if len(lines) !=3:
            return None
        try:
            ways = int(lines[0])
            max_len = int(lines[1])
            min_split = int(lines[2])
        except:
            return None
        return (ways, max_len, min_split)
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or len(solution)!=3:
            return False
        expected_ways = identity['expected_ways']
        expected_max = identity['expected_max']
        expected_min = identity['expected_min']
        return (solution[0] == expected_ways and
                solution[1] == expected_max and
                solution[2] == expected_min)
