"""# 

### 谜题描述
Permutation p is an ordered set of integers p1, p2, ..., pn, consisting of n distinct positive integers, each of them doesn't exceed n. We'll denote the i-th element of permutation p as pi. We'll call number n the size or the length of permutation p1, p2, ..., pn.

You have a sequence of integers a1, a2, ..., an. In one move, you are allowed to decrease or increase any number by one. Count the minimum number of moves, needed to build a permutation from this sequence.

Input

The first line contains integer n (1 ≤ n ≤ 3·105) — the size of the sought permutation. The second line contains n integers a1, a2, ..., an ( - 109 ≤ ai ≤ 109).

Output

Print a single number — the minimum number of moves.

Please, do not use the %lld specifier to read or write 64-bit integers in C++. It is preferred to use the cin, cout streams or the %I64d specifier.

Examples

Input

2
3 0


Output

2


Input

3
-1 -1 2


Output

6

Note

In the first sample you should decrease the first number by one and then increase the second number by one. The resulting permutation is (2, 1).

In the second sample you need 6 moves to build permutation (1, 3, 2).

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def abs(x):
 if x<0:
  return 0-x
 return x

n = input()
a = map(int,raw_input().split())
a.sort()
sum=0
for i in range(1,n+1):
 sum+=abs(a[i-1]-i)
print sum
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Cbuildingpermutationbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.min_n = params.get('min_n', 2)
        self.max_n = params.get('max_n', 10)
        self.value_range = params.get('value_range', (-10, 10))
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        permutation = list(range(1, n+1))
        a = [x + random.randint(*self.value_range) for x in permutation]
        random.shuffle(a)
        return {
            'n': n,
            'a': a
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = question_case['a']
        prompt = (
            f"你有一个整数序列：{a}。你的任务是计算将其转换为一个排列所需的最小移动次数。排列是指包含1到{n}每个数恰好一次的序列。"
            f"移动是指将一个数增加或减少1的次数。例如，将3变成2需要1次移动。"
            f"请计算将该序列转换为排列所需的最小总移动次数，并将你的答案放在[answer]标签中。"
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        start_tag = "[answer]"
        end_tag = "[/answer]"
        start = output.rfind(start_tag)
        if start == -1:
            return None
        end = output.find(end_tag, start + len(start_tag))
        if end == -1:
            return None
        answer_str = output[start + len(start_tag):end].strip()
        return answer_str
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        a = identity['a']
        a_sorted = sorted(a)
        target = list(range(1, n+1))
        expected = sum(abs(a_sorted[i] - target[i]) for i in range(n))
        try:
            return int(solution) == expected
        except ValueError:
            return False
