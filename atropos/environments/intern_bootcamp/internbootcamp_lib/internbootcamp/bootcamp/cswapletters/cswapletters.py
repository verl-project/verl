"""# 

### 谜题描述
Monocarp has got two strings s and t having equal length. Both strings consist of lowercase Latin letters \"a\" and \"b\". 

Monocarp wants to make these two strings s and t equal to each other. He can do the following operation any number of times: choose an index pos_1 in the string s, choose an index pos_2 in the string t, and swap s_{pos_1} with t_{pos_2}.

You have to determine the minimum number of operations Monocarp has to perform to make s and t equal, and print any optimal sequence of operations — or say that it is impossible to make these strings equal.

Input

The first line contains one integer n (1 ≤ n ≤ 2 ⋅ 10^{5}) — the length of s and t.

The second line contains one string s consisting of n characters \"a\" and \"b\". 

The third line contains one string t consisting of n characters \"a\" and \"b\". 

Output

If it is impossible to make these strings equal, print -1.

Otherwise, in the first line print k — the minimum number of operations required to make the strings equal. In each of the next k lines print two integers — the index in the string s and the index in the string t that should be used in the corresponding swap operation. 

Examples

Input


4
abab
aabb


Output


2
3 3
3 2


Input


1
a
b


Output


-1


Input


8
babbaabb
abababaa


Output


3
2 6
1 3
7 8

Note

In the first example two operations are enough. For example, you can swap the third letter in s with the third letter in t. Then s =  \"abbb\", t =  \"aaab\". Then swap the third letter in s and the second letter in t. Then both s and t are equal to \"abab\".

In the second example it's impossible to make two strings equal.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import os
import sys
from atexit import register
from io import BytesIO
sys.stdin = BytesIO(os.read(0, os.fstat(0).st_size))
sys.stdout = BytesIO()
register(lambda: os.write(1, sys.stdout.getvalue()))
input = lambda: sys.stdin.readline().rstrip('\r\n')

def get(arr):
	ret = []
	for i in range(len(arr)/2):
		ret.append([str(arr[2*i]),str(arr[2*i+1])])
	return ret

n = int(input())
s1 = input()
s2 = input()
ab = []
ba = []
for i in range(n):
	if s1[i]!=s2[i]:
		if s1[i] == \"a\":
			ab.append(i+1)
		else:
			ba.append(i+1)

ans = len(ab)/2+len(ba)/2
res = get(ab)+get(ba)

if len(ab)%2 == 1 and len(ba)%2 == 1:
	ans += 2
	res += [[str(ab[-1]),str(ab[-1])],[str(ab[-1]),str(ba[-1])]]
elif len(ab)%2 != len(ba)%2:
	ans = -1
print ans
if ans!=-1:
	for i in res:
		print \" \".join(i)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Cswaplettersbootcamp(Basebootcamp):
    def __init__(self, n=4):
        self.n = n
    
    def case_generator(self):
        # Decide whether to generate an impossible case
        make_impossible = random.choice([True, False])
        
        s = list(''.join(random.choice(['a', 'b']) for _ in range(self.n)))
        t = list(''.join(random.choice(['a', 'b']) for _ in range(self.n)))
        
        current_total_a = ''.join(s).count('a') + ''.join(t).count('a')
        
        if make_impossible:
            # Ensure total_a is odd
            if current_total_a % 2 == 0:
                # Flip a character
                if random.choice([True, False]):
                    idx = random.randint(0, self.n-1)
                    s[idx] = 'a' if s[idx] == 'b' else 'b'
                else:
                    idx = random.randint(0, self.n-1)
                    t[idx] = 'a' if t[idx] == 'b' else 'b'
        else:
            # Ensure total_a is even
            if current_total_a % 2 != 0:
                # Flip a character to make even
                if random.choice([True, False]):
                    idx = random.randint(0, self.n-1)
                    s[idx] = 'a' if s[idx] == 'b' else 'b'
                else:
                    idx = random.randint(0, self.n-1)
                    t[idx] = 'a' if t[idx] == 'b' else 'b'
                # Check again and flip if still odd
                current_total_a = ''.join(s).count('a') + ''.join(t).count('a')
                if current_total_a % 2 != 0:
                    if random.choice([True, False]):
                        idx = random.randint(0, self.n-1)
                        s[idx] = 'a' if s[idx] == 'b' else 'b'
                    else:
                        idx = random.randint(0, self.n-1)
                        t[idx] = 'a' if t[idx] == 'b' else 'b'
        
        return {
            'n': self.n,
            's': ''.join(s),
            't': ''.join(t)
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        s = question_case['s']
        t = question_case['t']
        prompt = f"""You are Monocarp trying to make two strings s and t equal. Both strings have a length of {n} and consist only of lowercase letters 'a' and 'b'. You can perform any number of the following operation: choose an index pos_1 in string s and an index pos_2 in string t, then swap the character at s[pos_1] with the character at t[pos_2]. Your goal is to determine the minimum number of operations required to make the two strings equal, and provide one such optimal sequence of operations. If it is impossible, output -1.

Input:

The first line contains the integer n: {n}.
The second line is string s: {s}.
The third line is string t: {t}.

Output:

If impossible, output -1. Otherwise, output the minimum number of operations k on the first line, followed by k lines each containing two integers pos_1 and pos_2 (1-based indices).

Please format your answer as follows:

[answer]
k
pos_1 pos_2
...
[/answer]

Ensure your answer is enclosed within [answer] and [/answer] tags."""
        return prompt
    
    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        content = answer_blocks[-1].strip()
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return None
        first_line = lines[0]
        if first_line == '-1':
            return -1
        try:
            k = int(first_line)
        except ValueError:
            return None
        if k < 0:
            return None
        remaining_lines = lines[1:]
        if len(remaining_lines) != k:
            return None
        operations = []
        for line in remaining_lines:
            parts = line.split()
            if len(parts) != 2:
                return None
            try:
                pos1 = int(parts[0])
                pos2 = int(parts[1])
                operations.append((pos1, pos2))
            except ValueError:
                return None
        return (k, operations)
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        s = identity['s']
        t = identity['t']
        n = identity['n']

        def calculate_min_operations(s, t, n):
            count_a_s = s.count('a')
            count_a_t = t.count('a')
            total_a = count_a_s + count_a_t
            if total_a % 2 != 0:
                return -1, None
            ab = []
            ba = []
            for i in range(n):
                sc = s[i]
                tc = t[i]
                if sc != tc:
                    if sc == 'a':
                        ab.append(i + 1)
                    else:
                        ba.append(i + 1)
            if (len(ab) % 2) != (len(ba) % 2):
                return -1, None
            operations = []
            ans = (len(ab) // 2) + (len(ba) // 2)
            for i in range(0, len(ab) - 1, 2):
                operations.append((ab[i], ab[i + 1]))
            for i in range(0, len(ba) - 1, 2):
                operations.append((ba[i], ba[i + 1]))
            if len(ab) % 2 == 1 and len(ba) % 2 == 1:
                ans += 2
                a_last = ab[-1]
                b_last = ba[-1]
                operations.append((a_last, a_last))
                operations.append((a_last, b_last))
            return ans, operations

        correct_ans, correct_ops = calculate_min_operations(s, t, n)
        if correct_ans == -1:
            return solution == -1
        if solution == -1:
            return False
        user_k, user_ops = solution
        if user_k != correct_ans:
            return False
        if len(user_ops) != user_k:
            return False
        for op in user_ops:
            pos1, pos2 = op
            if not (1 <= pos1 <= n and 1 <= pos2 <= n):
                return False
        s_list = list(s)
        t_list = list(t)
        for pos1_s, pos2_t in user_ops:
            pos1 = pos1_s - 1
            pos2 = pos2_t - 1
            s_list[pos1], t_list[pos2] = t_list[pos2], s_list[pos1]
        return s_list == t_list
