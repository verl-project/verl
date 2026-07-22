"""# 

### 谜题描述
Vasya became interested in bioinformatics. He's going to write an article about similar cyclic DNA sequences, so he invented a new method for determining the similarity of cyclic sequences.

Let's assume that strings s and t have the same length n, then the function h(s, t) is defined as the number of positions in which the respective symbols of s and t are the same. Function h(s, t) can be used to define the function of Vasya distance ρ(s, t): 

<image> where <image> is obtained from string s, by applying left circular shift i times. For example, ρ(\"AGC\", \"CGT\") =  h(\"AGC\", \"CGT\") + h(\"AGC\", \"GTC\") + h(\"AGC\", \"TCG\") +  h(\"GCA\", \"CGT\") + h(\"GCA\", \"GTC\") + h(\"GCA\", \"TCG\") +  h(\"CAG\", \"CGT\") + h(\"CAG\", \"GTC\") + h(\"CAG\", \"TCG\") =  1 + 1 + 0 + 0 + 1 + 1 + 1 + 0 + 1 = 6

Vasya found a string s of length n on the Internet. Now he wants to count how many strings t there are such that the Vasya distance from the string s attains maximum possible value. Formally speaking, t must satisfy the equation: <image>.

Vasya could not try all possible strings to find an answer, so he needs your help. As the answer may be very large, count the number of such strings modulo 109 + 7.

Input

The first line of the input contains a single integer n (1 ≤ n ≤ 105).

The second line of the input contains a single string of length n, consisting of characters \"ACGT\".

Output

Print a single number — the answer modulo 109 + 7.

Examples

Input

1
C


Output

1


Input

2
AG


Output

4


Input

3
TTT


Output

1

Note

Please note that if for two distinct strings t1 and t2 values ρ(s, t1) и ρ(s, t2) are maximum among all possible t, then both strings must be taken into account in the answer even if one of them can be obtained by a circular shift of another one.

In the first sample, there is ρ(\"C\", \"C\") = 1, for the remaining strings t of length 1 the value of ρ(s, t) is 0.

In the second sample, ρ(\"AG\", \"AG\") = ρ(\"AG\", \"GA\") = ρ(\"AG\", \"AA\") = ρ(\"AG\", \"GG\") = 4.

In the third sample, ρ(\"TTT\", \"TTT\") = 27

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
MOD=1000000007
n = int( raw_input() )
s = raw_input()
d = dict()
d['A'] = d['C'] = d['G'] = d['T'] = 0
for i in s:
	d[i] += 1
maxv = max ( d.values() )
num = d.values().count(maxv)
ans = 1
for i in s:
	ans *= num
	ans %= MOD
print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

MOD = 10**9 + 7

class Adnaalignmentbootcamp(Basebootcamp):
    def __init__(self, n=None, min_n=1, max_n=100000, seed=None):
        super().__init__()
        if n is not None and not (1 <= n <= 10**5):
            raise ValueError("n must be between 1 and 1e5")
        self.n = n
        self.min_n = max(1, min(min_n, 10**5))
        self.max_n = min(max_n, 10**5)
        self.rng = random.Random(seed)
    
    def case_generator(self):
        chars = ['A', 'C', 'G', 'T']
        n = self.n if self.n is not None else self.rng.randint(self.min_n, self.max_n)
        s = ''.join(self.rng.choices(chars, k=n))
        return {'n': n, 's': s}
    
    @staticmethod
    def prompt_func(question_case):
        case = question_case
        return f"""给定DNA字符串长度n={case['n']}，原始字符串s={case['s']}，计算使得Vasya距离ρ(s,t)最大的不同字符串t的数量（模{MOD}）。

Vasya距离定义：遍历s和t的所有循环移位组合，统计相同位置字符匹配的总次数。

答案请用[answer]答案[/answer]标记。例如：[answer]42[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            user_ans = int(solution.strip())
        except:
            return False
        
        s = identity['s']
        counts = {'A':0, 'C':0, 'G':0, 'T':0}
        for c in s:
            counts[c] += 1
        
        max_freq = max(counts.values())
        num_options = sum(1 for v in counts.values() if v == max_freq)
        correct_ans = pow(num_options, len(s), MOD)
        
        return user_ans == correct_ans
