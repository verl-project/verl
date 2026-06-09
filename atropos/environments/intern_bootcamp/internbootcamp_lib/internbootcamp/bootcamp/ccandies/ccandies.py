"""# 

### 谜题描述
Consider a sequence of digits of length 2^k [a_1, a_2, …, a_{2^k}]. We perform the following operation with it: replace pairs (a_{2i+1}, a_{2i+2}) with (a_{2i+1} + a_{2i+2})mod 10 for 0≤ i<2^{k-1}. For every i where a_{2i+1} + a_{2i+2}≥ 10 we get a candy! As a result, we will get a sequence of length 2^{k-1}.

Less formally, we partition sequence of length 2^k into 2^{k-1} pairs, each consisting of 2 numbers: the first pair consists of the first and second numbers, the second of the third and fourth …, the last pair consists of the (2^k-1)-th and (2^k)-th numbers. For every pair such that sum of numbers in it is at least 10, we get a candy. After that, we replace every pair of numbers with a remainder of the division of their sum by 10 (and don't change the order of the numbers).

Perform this operation with a resulting array until it becomes of length 1. Let f([a_1, a_2, …, a_{2^k}]) denote the number of candies we get in this process. 

For example: if the starting sequence is [8, 7, 3, 1, 7, 0, 9, 4] then:

After the first operation the sequence becomes [(8 + 7)mod 10, (3 + 1)mod 10, (7 + 0)mod 10, (9 + 4)mod 10] = [5, 4, 7, 3], and we get 2 candies as 8 + 7 ≥ 10 and 9 + 4 ≥ 10.

After the second operation the sequence becomes [(5 + 4)mod 10, (7 + 3)mod 10] = [9, 0], and we get one more candy as 7 + 3 ≥ 10. 

After the final operation sequence becomes [(9 + 0) mod 10] = [9]. 

Therefore, f([8, 7, 3, 1, 7, 0, 9, 4]) = 3 as we got 3 candies in total.

You are given a sequence of digits of length n s_1, s_2, … s_n. You have to answer q queries of the form (l_i, r_i), where for i-th query you have to output f([s_{l_i}, s_{l_i+1}, …, s_{r_i}]). It is guaranteed that r_i-l_i+1 is of form 2^k for some nonnegative integer k.

Input

The first line contains a single integer n (1 ≤ n ≤ 10^5) — the length of the sequence.

The second line contains n digits s_1, s_2, …, s_n (0 ≤ s_i ≤ 9).

The third line contains a single integer q (1 ≤ q ≤ 10^5) — the number of queries.

Each of the next q lines contains two integers l_i, r_i (1 ≤ l_i ≤ r_i ≤ n) — i-th query. It is guaranteed that r_i-l_i+1 is a nonnegative integer power of 2.

Output

Output q lines, in i-th line output single integer — f([s_{l_i}, s_{l_i + 1}, …, s_{r_i}]), answer to the i-th query.

Examples

Input


8
8 7 3 1 7 0 9 4
3
1 8
2 5
7 7


Output


3
1
0


Input


6
0 1 2 3 3 5
3
1 2
1 4
3 6


Output


0
0
1

Note

The first example illustrates an example from the statement.

f([7, 3, 1, 7]) = 1: sequence of operations is [7, 3, 1, 7] → [(7 + 3)mod 10, (1 + 7)mod 10] = [0, 8] and one candy as 7 + 3 ≥ 10 → [(0 + 8) mod 10] = [8], so we get only 1 candy.

f([9]) = 0 as we don't perform operations with it.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = input()
a = map(int,raw_input().split())
f = [ [ 0 for j in range(21) ] for i in range(n) ]
d = [ [ a[i] for j in range(21) ] for i in range(n) ]

for j in range(1, 21):
	for i in range(n):
		if i+(1<<(j-1)) >= n:
			continue
		d[i][j] = d[i][j-1] + d[i+(1<<(j-1))][j-1]
		f[i][j] = f[i][j-1] + f[i+(1<<(j-1))][j-1]
		if d[i][j] >= 10:
			f[i][j] += 1
			d[i][j] -= 10

for _ in range(input()):
	l, r = map(int,raw_input().split())
	print f[l-1][len(bin(r-l+1))-3]
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Ccandiesbootcamp(Basebootcamp):
    def __init__(self, max_n=16, max_q=3):
        self.max_n = max(1, max_n)  # Ensure max_n ≥ 1
        self.max_q = max(1, max_q)  # Ensure max_q ≥ 1
    
    def case_generator(self):
        n = random.randint(1, self.max_n)
        s = [random.randint(0, 9) for _ in range(n)]
        
        # Precompute all valid query lengths (powers of two)
        possible_lengths = []
        current_length = 1
        while current_length <= n:
            possible_lengths.append(current_length)
            current_length <<= 1
        
        # Generate queries with valid ranges
        queries = []
        for _ in range(min(self.max_q, 10)):  # Limit query count
            if not possible_lengths:
                break
            length = random.choice(possible_lengths)
            max_l = n - length + 1
            if max_l < 1:
                continue  # Skip if no valid range for this length
            l = random.randint(1, max_l)
            r = l + length - 1
            queries.append({'l': l, 'r': r})
        
        return {
            'n': n,
            's': s,
            'queries': queries
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        s = ' '.join(map(str, question_case['s']))
        q = len(question_case['queries'])
        queries = '\n'.join([f"{query['l']} {query['r']}" for query in question_case['queries']])
        
        return f"""You are given a sequence of digits and must answer queries about the number of candies obtained during a specific merging process. 

Input:
{n}
{s}
{q}
{queries}

Output each answer on a separate line within [answer] tags. Example:
[answer]
3
0
1
[/answer]"""

    @staticmethod
    def extract_output(output):
        last_answer = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not last_answer:
            return None
        answers = []
        for line in last_answer[-1].strip().splitlines():
            cleaned = line.strip()
            if cleaned.isdigit():
                answers.append(int(cleaned))
        return answers if answers else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or len(solution) != len(identity['queries']):
            return False
        s = identity['s']
        for i, query in enumerate(identity['queries']):
            l, r = query['l'], query['r']
            expected = cls.compute_f(s[l-1:r])
            if solution[i] != expected:
                return False
        return True
    
    @staticmethod
    def compute_f(subarray):
        total_candies = 0
        current = subarray.copy()
        while len(current) > 1:
            new_level = []
            level_candies = 0
            for i in range(0, len(current), 2):
                pair_sum = current[i] + current[i+1]
                if pair_sum >= 10:
                    level_candies += 1
                new_level.append(pair_sum % 10)
            total_candies += level_candies
            current = new_level
        return total_candies
