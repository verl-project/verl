"""# 

### 谜题描述
Vasya's got a birthday coming up and his mom decided to give him an array of positive integers a of length n.

Vasya thinks that an array's beauty is the greatest common divisor of all its elements. His mom, of course, wants to give him as beautiful an array as possible (with largest possible beauty). Unfortunately, the shop has only one array a left. On the plus side, the seller said that he could decrease some numbers in the array (no more than by k for each number).

The seller can obtain array b from array a if the following conditions hold: bi > 0; 0 ≤ ai - bi ≤ k for all 1 ≤ i ≤ n.

Help mom find the maximum possible beauty of the array she will give to Vasya (that seller can obtain).

Input

The first line contains two integers n and k (1 ≤ n ≤ 3·105; 1 ≤ k ≤ 106). The second line contains n integers ai (1 ≤ ai ≤ 106) — array a.

Output

In the single line print a single number — the maximum possible beauty of the resulting array.

Examples

Input

6 1
3 6 10 12 13 16


Output

3


Input

5 3
8 21 52 15 77


Output

7

Note

In the first sample we can obtain the array:

3 6 9 12 12 15

In the second sample we can obtain the next array:

7 21 49 14 77

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, k = map(int, raw_input().split())
v = list(map(int, raw_input().split()))

sol, stop = min(v), False

while not stop:
    stop = True
    for x in v:
        while x % sol > k:
            sol -= 1
            stop = False

print sol
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
import math
from functools import reduce
from bootcamp import Basebootcamp

class Evasyaandbeautifularraysbootcamp(Basebootcamp):
    def __init__(self, min_d=1, max_d=50, min_n=1, max_n=10, min_k=1, max_k=10, a_max=1000):
        self.min_d = min_d
        self.max_d = max_d
        self.min_n = min_n
        self.max_n = max_n
        self.min_k = min_k
        self.max_k = max_k
        self.a_max = a_max

    def case_generator(self):
        while True:
            n = random.randint(self.min_n, self.max_n)
            k = random.randint(self.min_k, self.max_k)
            d = random.randint(self.min_d, self.max_d)
            
            a = []
            # 确保至少一个元素调整到d（保证最终GCD为d的锚点）
            anchor_value = d + random.randint(0, min(k, self.a_max - d))
            a.append(anchor_value)
            
            # 生成其他元素（必须能被调整为d的倍数且满足k限制）
            for _ in range(n - 1):
                max_m = (self.a_max - k) // d
                if max_m < 1:
                    break  # 重新生成整个案例
                m = random.randint(1, max_m)
                t = random.randint(0, min(k, self.a_max - d * m))
                a.append(d * m + t)
            else:  # 仅当所有元素成功生成时才退出循环
                # 双重验证生成数据的正确性
                final_gcd = d
                for num in a:
                    adjusted = num - random.randint(0, min(k, num - 1))
                    if adjusted % d != 0:
                        break
                else:
                    random.shuffle(a)
                    return {
                        'n': n,
                        'k': k,
                        'a': a,
                        'd': d
                    }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k = question_case['k']
        a = ' '.join(map(str, question_case['a']))
        problem_text = f"""你正在解决一个数学优化问题。给定一个数组，通过减少每个元素最多k点（减少后必须仍为正整数），使得数组的最大公约数达到最大值。找出这个最大可能的公约数。

输入：
第一行包含两个整数n和k，分别表示数组长度和最大减少量。
第二行包含n个正整数，即原数组的元素。

输出：
最大可能的美观值（最大公约数）。

示例：
输入：
6 1
3 6 10 12 13 16
输出：3

请解决以下问题：

输入：
{n} {k}
{a}

将答案放在[answer]和[/answer]之间。"""
        return problem_text

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_answer = matches[-1].strip()
        try:
            return int(last_answer)
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 验证算法正确性的三重保障
        d = identity['d']
        a = identity['a']
        k = identity['k']
        
        # 1. 检查solution能否整除所有元素调整后的值
        for num in a:
            max_reduce = num - solution * (num // solution)
            if max_reduce > k or num - solution * (num // solution) < 0:
                return False
        return True
