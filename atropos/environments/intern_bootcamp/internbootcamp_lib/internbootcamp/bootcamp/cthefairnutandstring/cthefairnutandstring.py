"""# 

### 谜题描述
The Fair Nut found a string s. The string consists of lowercase Latin letters. The Nut is a curious guy, so he wants to find the number of strictly increasing sequences p_1, p_2, …, p_k, such that: 

  1. For each i (1 ≤ i ≤ k), s_{p_i} = 'a'. 
  2. For each i (1 ≤ i < k), there is such j that p_i < j < p_{i + 1} and s_j = 'b'. 



The Nut is upset because he doesn't know how to find the number. Help him.

This number should be calculated modulo 10^9 + 7.

Input

The first line contains the string s (1 ≤ |s| ≤ 10^5) consisting of lowercase Latin letters.

Output

In a single line print the answer to the problem — the number of such sequences p_1, p_2, …, p_k modulo 10^9 + 7.

Examples

Input

abbaa


Output

5

Input

baaaa


Output

4

Input

agaa


Output

3

Note

In the first example, there are 5 possible sequences. [1], [4], [5], [1, 4], [1, 5].

In the second example, there are 4 possible sequences. [2], [3], [4], [5].

In the third example, there are 3 possible sequences. [1], [3], [4].

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import math
import collections
import bisect
import heapq
import time
import random
import itertools
import sys

mod=1000000007
s=str(raw_input())
pos=1
n=len(s)

lagataara=1

p=[]

for i in xrange(0,n):
	if s[i]=='a':
		lagataara=lagataara+1
	if s[i]=='b':
		pos=(pos*lagataara)%mod
		lagataara=1
pos=(pos*lagataara+mod)%mod
print(pos-1)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Cthefairnutandstringbootcamp(Basebootcamp):
    def __init__(self, max_blocks=5, max_a_per_block=5, max_other_chars=3):
        self.max_blocks = max_blocks
        self.max_a_per_block = max_a_per_block
        self.max_other_chars = max_other_chars
    
    def case_generator(self):
        k = random.randint(1, self.max_blocks)
        m_list = []
        
        # 确保至少存在一个块有a（避免全零情况）
        while True:
            m_list = [random.randint(0, self.max_a_per_block) for _ in range(k)]
            if sum(m_list) > 0:  # 保证至少有一个a存在
                break
        
        s_parts = []
        for i in range(k):
            a_part = 'a' * m_list[i]
            other_num = random.randint(0, self.max_other_chars)
            other_chars = ''.join(random.choices(
                [c for c in 'cdefghijklmnopqrstuvwxyz' if c not in {'a', 'b'}],
                k=other_num
            ))
            combined = list(a_part + other_chars)
            random.shuffle(combined)  # 混合a和其他字符
            s_parts.append(''.join(combined))
            
            if i < k - 1:
                s_parts.append('b' * random.randint(1, 2))  # 插入1~2个b作为分隔
        
        s = ''.join(s_parts)
        
        # 严格按题目规则计算答案
        mod = 10**9 + 7
        product = 1
        for m in m_list:
            product = (product * (m + 1)) % mod
        answer = (product - 1) % mod
        
        return {
            's': s,
            'correct_answer': answer
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        s = question_case['s']
        return f"""You are given a string composed of lowercase letters. Find the number of strictly increasing index sequences where:
1. Each index points to an 'a'
2. Between consecutive indices there's at least one 'b'

Calculate modulo 1e9+7.

Input: {s}

Format your final answer as [answer]number[/answer]. Example: [answer]5[/answer]"""

    @staticmethod
    def extract_output(output):
        import re
        answers = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(answers[-1]) if answers else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
