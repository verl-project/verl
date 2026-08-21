"""# 

### 谜题描述
Vasya bought the collected works of a well-known Berland poet Petya in n volumes. The volumes are numbered from 1 to n. He thinks that it does not do to arrange the book simply according to their order. Vasya wants to minimize the number of the disposition’s divisors — the positive integers i such that for at least one j (1 ≤ j ≤ n) is true both: j mod i = 0 and at the same time p(j) mod i = 0, where p(j) is the number of the tome that stands on the j-th place and mod is the operation of taking the division remainder. Naturally, one volume can occupy exactly one place and in one place can stand exactly one volume.

Help Vasya — find the volume disposition with the minimum number of divisors.

Input

The first line contains number n (1 ≤ n ≤ 100000) which represents the number of volumes and free places.

Output

Print n numbers — the sought disposition with the minimum divisor number. The j-th number (1 ≤ j ≤ n) should be equal to p(j) — the number of tome that stands on the j-th place. If there are several solutions, print any of them.

Examples

Input

2


Output

2 1 


Input

3


Output

1 3 2 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/usr/bin/python

n = int(raw_input())
r = range(2, n + 2)
r[-1] = 1
print ' '.join(map(str, r))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import math
from bootcamp import Basebootcamp

class Cdispositionbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', random.randint(1, 100000))

    def case_generator(self):
        n = random.randint(1, 100000)
        if n == 1:
            p = [1]
        else:
            p = [i + 1 for i in range(n)]
            p[-1] = 1
        return {'n': n, 'p': p}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        return (
            f"你有一个包含{n}本的书籍集，编号从1到{n}。你需要将这些书籍排列成一个序列p，使得除数的数目最少。"
            f"除数i的定义是：存在至少一个位置j（1 ≤ j ≤ {n}），使得j和p(j)都能被i整除。"
            f"你的任务是找到这样的排列p，并将其输出为一个由空格分隔的整数序列。"
            f"请将答案放在[answer]标签内，例如：[answer]2 3 1[/answer]"
        )
    
    @staticmethod
    def extract_output(output):
        start = output.rfind('[answer]')
        if start == -1:
            return None
        end = output.find('[/answer]', start)
        if end == -1:
            return None
        answer_str = output[start + len('[answer]'):end].strip()
        try:
            solution = list(map(int, answer_str.split()))
        except ValueError:
            return None
        return solution
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        if len(solution) != n:
            return False
        if sorted(solution) != list(range(1, n+1)):
            return False
        divisors = set()
        for j in range(1, n+1):
            p_j = solution[j-1]
            g = math.gcd(j, p_j)
            factors = set()
            for i in range(1, int(math.sqrt(g)) + 1):
                if g % i == 0:
                    factors.add(i)
                    factors.add(g // i)
            divisors.update(factors)
        return len(divisors) == 1
