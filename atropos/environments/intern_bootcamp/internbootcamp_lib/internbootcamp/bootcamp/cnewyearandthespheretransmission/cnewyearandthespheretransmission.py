"""# 

### 谜题描述
There are n people sitting in a circle, numbered from 1 to n in the order in which they are seated. That is, for all i from 1 to n-1, the people with id i and i+1 are adjacent. People with id n and 1 are adjacent as well.

The person with id 1 initially has a ball. He picks a positive integer k at most n, and passes the ball to his k-th neighbour in the direction of increasing ids, that person passes the ball to his k-th neighbour in the same direction, and so on until the person with the id 1 gets the ball back. When he gets it back, people do not pass the ball any more.

For instance, if n = 6 and k = 4, the ball is passed in order [1, 5, 3, 1]. 

Consider the set of all people that touched the ball. The fun value of the game is the sum of the ids of people that touched it. In the above example, the fun value would be 1 + 5 + 3 = 9.

Find and report the set of possible fun values for all choices of positive integer k. It can be shown that under the constraints of the problem, the ball always gets back to the 1-st player after finitely many steps, and there are no more than 10^5 possible fun values for given n.

Input

The only line consists of a single integer n (2 ≤ n ≤ 10^9) — the number of people playing with the ball.

Output

Suppose the set of all fun values is f_1, f_2, ..., f_m.

Output a single line containing m space separated integers f_1 through f_m in increasing order.

Examples

Input


6


Output


1 5 9 21


Input


16


Output


1 10 28 64 136

Note

In the first sample, we've already shown that picking k = 4 yields fun value 9, as does k = 2. Picking k = 6 results in fun value of 1. For k = 3 we get fun value 5 and with k = 1 or k = 5 we get 21. 

<image>

In the second sample, the values 1, 10, 28, 64 and 136 are achieved for instance for k = 16, 8, 4, 10 and 11, respectively.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = input()
d = []

for i in xrange(1, int(n ** 0.5) + 1):
    if n % i == 0 and i * i != n:
        d.append(i)
        d.append(n // i)
    if i * i == n:
        d.append(i)

ans = []
for g in d:
    x = (n * (n - g + 2)) // (2 * g)
    ans.append(x)

ans.sort()

print ' '.join(map(str, ans))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from math import isqrt
from bootcamp import Basebootcamp

class Cnewyearandthespheretransmissionbootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=10**6, case_types=None):
        """
        改进后的初始化参数，支持生成多种数论特征案例
        
        参数:
            min_n (int): 最小n值（≥2）
            max_n (int): 最大n值（需≥min_n）
            case_types (list[str]): 指定生成的案例类型（prime/square/random），默认全类型
        """
        self.min_n = max(2, min_n)
        self.max_n = max(self.min_n, max_n)
        self.case_types = case_types or ['prime', 'square', 'random']

    @staticmethod
    def is_prime(n):
        """Miller-Rabin素数检测快速实现"""
        if n < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            if n % p == 0:
                return n == p
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in [2, 325, 9375, 28178, 450775, 9780504, 1795265022]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n -1:
                continue
            for _ in range(s-1):
                x = pow(x, 2, n)
                if x == n-1:
                    break
            else:
                return False
        return True

    def case_generator(self):
        """改进的案例生成器，支持质数/平方数/随机数三类特征"""
        case_type = random.choice(self.case_types)
        n = self.min_n
        
        if case_type == 'prime':
            # 生成区间内最大质数
            candidates = [i for i in range(self.max_n, self.min_n-1, -1) if self.is_prime(i)]
            n = random.choice(candidates) if candidates else random.randint(self.min_n, self.max_n)
        
        elif case_type == 'square':
            # 生成完全平方数
            min_sq = isqrt(self.min_n)
            max_sq = isqrt(self.max_n)
            if min_sq**2 < self.min_n:
                min_sq += 1
            if max_sq**2 > self.max_n:
                max_sq -= 1
            if min_sq > max_sq:
                n = random.randint(self.min_n, self.max_n)
            else:
                sq = random.randint(min_sq, max_sq)
                n = sq ** 2
        
        else:  # random
            n = random.randint(self.min_n, self.max_n)
        
        return {'n': n}

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        return f"""## 题目描述

{n}人围坐成一圈（编号1~{n}），1号选手选择正整数k（1≤k≤{n}）开始传球：
1. 每次将球传给**顺时针方向**的第k个邻居（如当前在m号，则传给(m-1 +k) mod {n} +1）
2. 当球**首次回到1号**时停止
3. 统计**所有触球选手的编号之和**作为趣味值

## 任务要求
找出所有可能的趣味值，按升序输出

## 输出格式
将答案按升序排列，置于[answer]标签内，如：
[answer]1 5 9 21[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]([\s\d]+)\[/answer\]', output)
        if not matches:
            return None
        try:
            last = matches[-1].replace('\n', ' ').split()
            return list(map(int, last))
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        divisors = set()
        for i in range(1, isqrt(n)+1):
            if n % i == 0:
                divisors.update([i, n//i])
        
        ans = []
        for g in divisors:
            ans.append(n * (n - g + 2) // (2 * g))
        ans = sorted(list(set(ans)))  # 去重后排序
        
        return solution == ans
