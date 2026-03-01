"""# 

### 谜题描述
One day little Vasya found mom's pocket book. The book had n names of her friends and unusually enough, each name was exactly m letters long. Let's number the names from 1 to n in the order in which they are written.

As mom wasn't home, Vasya decided to play with names: he chose three integers i, j, k (1 ≤ i < j ≤ n, 1 ≤ k ≤ m), then he took names number i and j and swapped their prefixes of length k. For example, if we take names \"CBDAD\" and \"AABRD\" and swap their prefixes with the length of 3, the result will be names \"AABAD\" and \"CBDRD\".

You wonder how many different names Vasya can write instead of name number 1, if Vasya is allowed to perform any number of the described actions. As Vasya performs each action, he chooses numbers i, j, k independently from the previous moves and his choice is based entirely on his will. The sought number can be very large, so you should only find it modulo 1000000007 (109 + 7).

Input

The first input line contains two integers n and m (1 ≤ n, m ≤ 100) — the number of names and the length of each name, correspondingly. Then n lines contain names, each name consists of exactly m uppercase Latin letters.

Output

Print the single number — the number of different names that could end up in position number 1 in the pocket book after the applying the procedures described above. Print the number modulo 1000000007 (109 + 7).

Examples

Input

2 3
AAB
BAA


Output

4


Input

4 5
ABABA
BCGDG
AAAAA
YABSA


Output

216

Note

In the first sample Vasya can get the following names in the position number 1: \"AAB\", \"AAA\", \"BAA\" and \"BAB\".

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, m = map(int, raw_input().split())
score = [raw_input() for i in range(n)]
sets = [set(i) for i in zip(*score)]
ans = 1
for i in sets:
	ans = ans * len(i) % 1000000007
print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cpocketbookbootcamp(Basebootcamp):
    def __init__(self, n=2, m=3):
        if n < 1 or m < 1:
            raise ValueError("n and m must be at least 1")
        self.n = n
        self.m = m
    
    def case_generator(self):
        names = [[] for _ in range(self.n)]
        s_k_list = []
        
        for k in range(self.m):
            max_possible = min(self.n, 26)
            s_k = random.randint(1, max_possible)
            s_k_list.append(s_k)
            chars = random.sample('ABCDEFGHIJKLMNOPQRSTUVWXYZ', s_k)
            
            for i in range(self.n):
                if i < s_k:
                    c = chars[i]
                else:
                    c = random.choice(chars)
                names[i].append(c)
        
        names = [''.join(lst) for lst in names]
        return {
            'n': self.n,
            'm': self.m,
            'names': names,
            's_k_list': s_k_list
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        input_lines = [f"{question_case['n']} {question_case['m']}"] + question_case['names']
        input_str = '\n'.join(input_lines)
        
        prompt = f"""You are participating in a programming competition. Solve the following puzzle based on the described rules.

Problem Description:
Vasya found his mother's pocket book with n names, each exactly m letters long. The names are numbered 1 to n. Vasya can perform the following operation any number of times: choose two names (i and j, where i < j) and a length k, then swap the first k letters between them. The goal is to determine how many distinct names can end up in position 1 after any number of such operations. The answer should be given modulo 1000000007 (1e9+7).

Rules and Notes:
1. Each swap operation can be performed any number of times, with any valid i, j, and k.
2. The result depends on the different possible characters that can appear in each position of the first name.
3. For each position, the number of possible characters is equal to the number of distinct characters in that position across all names. The total answer is the product of these counts for all positions, modulo 1e9+7.

Input Format:
- The first line contains two integers n and m.
- The next n lines each contain a string of m uppercase letters.

Your Task:
Compute the correct answer for the provided input and write it inside [answer] and [/answer] tags. For example, if the answer is 4, write [answer]4[/answer].

Input Provided:
{input_str}

Please provide your answer within the tags as described."""
        return prompt
    
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
        product = 1
        mod = 10**9 + 7
        
        # 动态计算每列实际的不同字符数
        names = identity['names']
        m = identity['m']
        for k in range(m):
            column_chars = set(name[k] for name in names)
            product = (product * len(column_chars)) % mod
        
        return solution == product
