"""# 

### 谜题描述
You are given n pairs of integers (a_1, b_1), (a_2, b_2), …, (a_n, b_n). All of the integers in the pairs are distinct and are in the range from 1 to 2 ⋅ n inclusive.

Let's call a sequence of integers x_1, x_2, …, x_{2k} good if either 

  * x_1 < x_2 > x_3 < … < x_{2k-2} > x_{2k-1} < x_{2k}, or 
  * x_1 > x_2 < x_3 > … > x_{2k-2} < x_{2k-1} > x_{2k}. 



You need to choose a subset of distinct indices i_1, i_2, …, i_t and their order in a way that if you write down all numbers from the pairs in a single sequence (the sequence would be a_{i_1}, b_{i_1}, a_{i_2}, b_{i_2}, …, a_{i_t}, b_{i_t}), this sequence is good.

What is the largest subset of indices you can choose? You also need to construct the corresponding index sequence i_1, i_2, …, i_t.

Input

The first line contains single integer n (2 ≤ n ≤ 3 ⋅ 10^5) — the number of pairs.

Each of the next n lines contain two numbers — a_i and b_i (1 ≤ a_i, b_i ≤ 2 ⋅ n) — the elements of the pairs.

It is guaranteed that all integers in the pairs are distinct, that is, every integer from 1 to 2 ⋅ n is mentioned exactly once.

Output

In the first line print a single integer t — the number of pairs in the answer.

Then print t distinct integers i_1, i_2, …, i_t — the indexes of pairs in the corresponding order.

Examples

Input


5
1 7
6 4
2 10
9 8
3 5


Output


3
1 5 3


Input


3
5 4
3 2
6 1


Output


3
3 2 1

Note

The final sequence in the first example is 1 < 7 > 3 < 5 > 2 < 10.

The final sequence in the second example is 6 > 1 < 3 > 2 < 5 > 4.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys,os,math
from collections import Counter, defaultdict
import bisect
from sys import stdin, stdout


# n, k = map(int, raw_input().split())
# da = map(int, raw_input().split())
# db = map(int, raw_input().split())


n = map(int, raw_input().split())[0]
da = [tuple(map(int, raw_input().split())) for j in range(n)]

sm, bi = [], []
for i,(a,b) in enumerate(da):
    if a<b:
        sm.append((a,b,i+1))
    else:
        bi.append((a,b,i+1))
if len(sm) > len(bi):
    print len(sm)
    sm.sort(key=lambda k:k[1], reverse=True)
    stdout.write(' '.join(map(lambda k:str(k[-1]), sm)))
else:
    print len(bi)
    bi.sort(key=lambda k:k[1])
    stdout.write(' '.join(map(lambda k:str(k[-1]), bi)))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Ddirtydeedsdonedirtcheapbootcamp(Basebootcamp):
    def __init__(self, n=5):
        self.n = n
    
    def case_generator(self):
        n = self.n
        numbers = list(range(1, 2 * n + 1))
        random.shuffle(numbers)
        pairs = []
        for i in range(n):
            a = numbers[2 * i]
            b = numbers[2 * i + 1]
            if random.choice([True, False]):
                a, b = b, a
            pairs.append((a, b))
        return {
            'n': n,
            'pairs': pairs
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        pairs = question_case['pairs']
        n = question_case['n']
        pairs_list = "\n".join([f"{a} {b}" for a, b in pairs])
        prompt = f"""You are given {n} pairs of integers. All integers in the pairs are distinct and each integer is between 1 and {2*n} inclusive. Your task is to select the largest subset of pairs and arrange them in order such that the resulting sequence alternates between increasing and decreasing (or vice versa).

For example, a valid sequence might look like 1 < 7 > 3 < 5 > 2 < 10 or 6 > 1 < 3 > 2 < 5 > 4. The goal is to include as many pairs as possible.

Input:
{n}
{pairs_list}

Your answer should be two lines. The first line contains the number of pairs selected, t. The second line contains t distinct indices (1-based) in the order they should be arranged. Please place your answer within [answer] and [/answer] tags.

Example:
[answer]
3
1 5 3
[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        last_answer = answer_blocks[-1].strip()
        lines = [line.strip() for line in last_answer.split('\n') if line.strip()]
        if len(lines) < 2:
            return None
        try:
            t = int(lines[0])
            indices = list(map(int, lines[1].split()))
            if len(indices) != t or len(set(indices)) != t:
                return None
            return indices
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            pairs = identity['pairs']
            n = identity['n']
            sm_count = 0
            bi_count = 0
            for a, b in pairs:
                if a < b:
                    sm_count += 1
                else:
                    bi_count += 1
            t_max = max(sm_count, bi_count)
            if len(solution) != t_max:
                return False
            
            target_group_is_sm = sm_count > bi_count  # 严格大于才选SM组
            
            for idx in solution:
                if idx < 1 or idx > len(pairs):
                    return False
                a, b = pairs[idx - 1]
                if target_group_is_sm:
                    if a >= b:
                        return False
                else:
                    if a <= b:
                        return False
            
            sequence = []
            for idx in solution:
                sequence.extend(pairs[idx - 1])
            
            return cls.is_valid_sequence(sequence)
        except:
            return False

    @staticmethod
    def is_valid_sequence(seq):
        if len(seq) < 2 or len(seq) % 2 != 0:
            return False
        current_relation = seq[0] < seq[1]
        for i in range(1, len(seq)-1):
            next_relation = seq[i] < seq[i+1]
            if next_relation == current_relation:
                return False
            current_relation = next_relation
        return True
