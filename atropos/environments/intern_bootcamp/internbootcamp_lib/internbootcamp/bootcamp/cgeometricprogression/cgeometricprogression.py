"""# 

### 谜题描述
Polycarp loves geometric progressions very much. Since he was only three years old, he loves only the progressions of length three. He also has a favorite integer k and a sequence a, consisting of n integers.

He wants to know how many subsequences of length three can be selected from a, so that they form a geometric progression with common ratio k.

A subsequence of length three is a combination of three such indexes i1, i2, i3, that 1 ≤ i1 < i2 < i3 ≤ n. That is, a subsequence of length three are such groups of three elements that are not necessarily consecutive in the sequence, but their indexes are strictly increasing.

A geometric progression with common ratio k is a sequence of numbers of the form b·k0, b·k1, ..., b·kr - 1.

Polycarp is only three years old, so he can not calculate this number himself. Help him to do it.

Input

The first line of the input contains two integers, n and k (1 ≤ n, k ≤ 2·105), showing how many numbers Polycarp's sequence has and his favorite number.

The second line contains n integers a1, a2, ..., an ( - 109 ≤ ai ≤ 109) — elements of the sequence.

Output

Output a single number — the number of ways to choose a subsequence of length three, such that it forms a geometric progression with a common ratio k.

Examples

Input

5 2
1 1 2 2 4


Output

4

Input

3 1
1 1 1


Output

1

Input

10 3
1 2 6 2 3 6 9 18 3 9


Output

6

Note

In the first sample test the answer is four, as any of the two 1s can be chosen as the first element, the second element can be any of the 2s, and the third element of the subsequence must be equal to 4.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from collections import defaultdict


def geometric(k, lst):
    d = defaultdict(int)
    total = 0

    for i, num in enumerate(lst):
        
        if num % k == 0:
            total += d[num // k, 2]
            d[num, 2] += d[num // k, 1]
        d[num, 1] += 1

    return total


_, k = map(int, raw_input().split())
lst = map(int, raw_input().split())

print geometric(k, lst)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from collections import defaultdict
from bootcamp import Basebootcamp

def geometric(k, lst):
    d = defaultdict(int)
    total = 0
    for num in lst:
        if k != 0 and num % k == 0:
            total += d.get((num // k, 2), 0)
            d[(num, 2)] += d.get((num // k, 1), 0)
        d[(num, 1)] += 1
    return total

class Cgeometricprogressionbootcamp(Basebootcamp):
    def __init__(self, min_n=3, max_n=12, min_k=1, max_k=4, a_min=-8, a_max=8):
        self.min_n = min_n
        self.max_n = max_n
        self.min_k = min_k
        self.max_k = max_k
        self.a_min = a_min
        self.a_max = a_max
    
    def case_generator(self):
        k = random.randint(self.min_k, self.max_k)
        x = random.choice([num for num in range(self.a_min, self.a_max + 1) if num != 0])
        m1 = random.randint(1, 3)
        m2 = random.randint(1, 3)
        m3 = random.randint(1, 3)
        a = [x] * m1 + [x * k] * m2 + [x * k * k] * m3
        
        # Add non-interfering noise elements
        noise_count = random.randint(0, 2)
        for _ in range(noise_count):
            while True:
                noise = random.randint(self.a_min * 3, self.a_max * 3)
                if noise not in {x, x * k, x * k * k}:
                    a.append(noise)
                    break
        
        random.shuffle(a)  # 打乱顺序不影响正确性，验证时会动态计算
        
        return {
            'n': len(a),
            'k': k,
            'a': a
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k = question_case['k']
        a = question_case['a']
        problem_desc = (
            "Polycarp loves geometric progressions of length three. Help him find how many such subsequences exist in his sequence with a common ratio of k.\n\n"
            "Rules:\n"
            "1. Indices must be strictly increasing (i < j < k).\n"
            "2. Elements must form a geometric progression: a[j] = a[i] * k and a[k] = a[j] * k.\n\n"
            "Input Parameters:\n"
            f"- n (array length) = {n}\n"
            f"- k (common ratio) = {k}\n"
            f"- Sequence: {a}\n\n"
            "Output the exact number of valid subsequences. Enclose your answer within [answer] and [/answer] tags."
        )
        return problem_desc
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return int(matches[-1].strip()) if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        k = identity['k']
        a = identity['a']
        return solution == geometric(k, a)
