"""# 

### 谜题描述
Ashishgup and FastestFinger play a game. 

They start with a number n and play in turns. In each turn, a player can make any one of the following moves:

  * Divide n by any of its odd divisors greater than 1. 
  * Subtract 1 from n if n is greater than 1. 



Divisors of a number include the number itself.

The player who is unable to make a move loses the game.

Ashishgup moves first. Determine the winner of the game if both of them play optimally.

Input

The first line contains a single integer t (1 ≤ t ≤ 100) — the number of test cases. The description of the test cases follows.

The only line of each test case contains a single integer — n (1 ≤ n ≤ 10^9).

Output

For each test case, print \"Ashishgup\" if he wins, and \"FastestFinger\" otherwise (without quotes).

Example

Input


7
1
2
3
4
5
6
12


Output


FastestFinger
Ashishgup
Ashishgup
FastestFinger
Ashishgup
FastestFinger
Ashishgup

Note

In the first test case, n = 1, Ashishgup cannot make a move. He loses.

In the second test case, n = 2, Ashishgup subtracts 1 on the first move. Now n = 1, FastestFinger cannot make a move, so he loses.

In the third test case, n = 3, Ashishgup divides by 3 on the first move. Now n = 1, FastestFinger cannot make a move, so he loses.

In the last test case, n = 12, Ashishgup divides it by 3. Now n = 4, FastestFinger is forced to subtract 1, and Ashishgup gets 3, so he wins by dividing it by 3.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import math
test_cases=int(input())
res=[]
for i in range(test_cases):
    n=int(input())
    t=0
    while n%2==0:
        n=n//2
        t+=1
    if t==0:
        if n==1:
            res.append(\"FastestFinger\")
        else:
            res.append(\"Ashishgup\")
    elif t==1:
       
        if n==1:
            res.append(\"Ashishgup\")
        else:
            cp=0
            for j in range(3,int(math.floor(math.sqrt(n)))+1,2):
                #print(j)
                if n%j ==0:
                    cp=1
                    break
            if cp==1:
                res.append(\"Ashishgup\")
            else:
                res.append(\"FastestFinger\")
    else:
        if n==1:
            res.append(\"FastestFinger\")
        else:
            res.append(\"Ashishgup\")
            '''cp=0
            for j in range(3,n//2,2):
                if n%j ==0:
                    cp=1
                    break
            if cp==1:
                res.append(\"FastestFinger\")
            else:
                res.append(\"Ashishgup\")'''
for i in range(test_cases):
    print(res[i])
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import math
import random
from bootcamp import Basebootcamp

class Cnumbergamebootcamp(Basebootcamp):
    def __init__(self, **params):
        self.min_n = max(1, params.get('min_n', 1))
        self.max_n = params.get('max_n', 10**9)
        self.params = params

    def case_generator(self):
        def generate_valid_n():
            for _ in range(1000):  # 增加尝试次数
                case_type = random.choice([
                    'edge_1', 'edge_2', 'prime', 'power_of_2',
                    'two_times_prime', 'two_times_composite', 'complex_case'
                ])

                if case_type == 'edge_1':
                    n = 1
                elif case_type == 'edge_2':
                    n = 2
                elif case_type == 'prime':
                    n = self._generate_odd_prime()
                elif case_type == 'power_of_2':
                    max_exp = math.floor(math.log2(self.max_n))
                    if max_exp < 2:
                        continue  # 无法生成合理2^n
                    exp = random.randint(2, max_exp)
                    n = 2 ** exp
                elif case_type == 'two_times_prime':
                    max_p = self.max_n // 2
                    min_p = max(3, self.min_n // 2)
                    if min_p > max_p:
                        continue
                    p = self._generate_odd_prime(min_p, max_p)
                    n = 2 * p
                elif case_type == 'two_times_composite':
                    max_c = self.max_n // 2
                    if max_c < 9:  # 最小奇合数9
                        continue
                    c = self._generate_odd_composite(3, max_c)
                    n = 2 * c
                elif case_type == 'complex_case':
                    max_t = math.floor(math.log2(self.max_n))
                    if max_t < 2:
                        continue
                    t = random.randint(2, min(5, max_t))
                    base = 2 ** t
                    remaining = self.max_n // base
                    if remaining < 3:
                        continue
                    factors = self._generate_odd_composite(3, remaining)
                    n = base * factors

                if self.min_n <= n <= self.max_n:
                    return n
            return random.randint(self.min_n, self.max_n)

        n = generate_valid_n()
        return {
            'n': n,
            'correct_answer': self.get_correct_answer(n)
        }

    def _generate_odd_prime(self, min_p=3, max_p=None):
        max_p = max_p or self.max_n // 2
        if min_p % 2 == 0:
            min_p += 1
        attempts = 0
        while attempts < 1000:
            p = random.randint(min_p, max_p)
            if p % 2 == 0:
                continue
            if self.is_prime(p):
                return p
            attempts += 1
        return 3  # fallback

    def _generate_odd_composite(self, min_val=9, max_val=None):
        max_val = max_val or self.max_n // 2
        while True:
            num = random.randint(min_val, max_val)
            if num % 2 == 0:
                continue
            if not self.is_prime(num):
                return num

    @staticmethod
    def is_prime(num):
        if num < 2:
            return False
        if num % 2 == 0:
            return num == 2
        for i in range(3, int(math.isqrt(num)) + 1, 2):
            if num % i == 0:
                return False
        return True

    @staticmethod
    def get_correct_answer(n):
        original_n = n
        t = 0
        while n % 2 == 0:
            n = n // 2
            t += 1
        k = n

        if t == 0:
            return "FastestFinger" if k == 1 else "Ashishgup"
        elif t == 1:
            if k == 1:
                return "Ashishgup"
            is_prime = Cnumbergamebootcamp.is_prime(k)
            return "FastestFinger" if is_prime else "Ashishgup"
        else:
            return "FastestFinger" if k == 1 else "Ashishgup"

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        return f"""Ashishgup和FastestFinger在玩一个数字游戏。规则如下：

- 初始数字为{n}。玩家轮流进行操作，Ashishgup先手。
- 每个回合可以选择以下操作之一：
  a) 将当前数除以一个大于1的奇数因子
  b) 当当前数>1时，减去1

无法操作的玩家失败。两人都采用最优策略。

请分析游戏过程并输出获胜者姓名，将答案置于[answer]标签内。例如：
[answer]Ashishgup[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(.*?)\s*\[/answer\]', output, re.IGNORECASE)
        if not matches:
            return None
        last_answer = matches[-1].strip().capitalize()
        return last_answer if last_answer in ['Ashishgup', 'FastestFinger'] else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity.get('correct_answer')
