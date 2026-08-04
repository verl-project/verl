"""# 

### 谜题描述
Vasya plays The Elder Trolls III: Morrowindows. He has a huge list of items in the inventory, however, there is no limits on the size of things. Vasya does not know the total amount of items but he is sure that are not more than x and not less than 2 items in his inventory. A new patch for the game appeared to view inventory in n different modes. Displaying in mode i is a partition of all inventory items on pages, each of which (except for maybe the last one) shows exactly ai items. In addition, each mode shows how many pages bi is in a complete list. Great! Perhaps this information will be enough for Vasya to find the required number. Moreover, it is very interesting, what is the fewest number of modes in which Vasya can see inventory to determine the number of items in it?

Vasya cannot use the information that was received while looking on inventory in some mode for selection of next actions. I. e. Vasya chooses some set of modes first, and then sees all the results and determines the size.

Knowing the number of ai, x and assuming that Vasya is very smart, check whether he can uniquely determine the number of items in his inventory, and how many modes he will need to do that if he knows numbers ai, x and he is able to know number bi after viewing items in mode i.

Input

The first line contains two integers n and x (0 ≤ n ≤ 105, 2 ≤ x ≤ 109). The second line contains integers ai (1 ≤ ai ≤ 109). Some numbers among all ai may be equal.

Output

Output the fewest amount of modes required to uniquely determine amount of items in the inventory. If there is no solution output  - 1.

Examples

Input

2 4
2 3


Output

2


Input

1 4
2


Output

-1

Note

In the second example Vasya is not able to determine items count uniquely because 3 items, as well as 4 items, can be displayed on two pages.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n,x = map(int,raw_input().split())
a = set(map(int,raw_input().split()))
if 1 in a and x>2: print 1
elif x>1300000: print -1
else:
    pr = range(x)
    for i in xrange(2,x):
        if not pr[i]: continue
        ii=i*i
        if ii>x: break
        pr[ii::i]=[0]*len(pr[ii::i])     
    pr = set(filter(None,pr)[1:])
    print -1 if len(pr-a) else len(pr)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Emorrowindowsbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__()
        self.n = params.get('n', 0)
        self.x = params.get('x', 2)
        self.a = params.get('a', [])

    def case_generator(self):
        n = random.randint(0, 105)
        x = random.randint(2, 10**9)
        a = [random.randint(1, 10**9) for _ in range(n)]
        return {'n': n, 'x': x, 'a': a}

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        x = question_case['x']
        a = question_case['a']
        a_str = ', '.join(map(str, a))
        prompt = (
            f"Vasya在游戏中的库存物品数量介于2和{x}之间。他有{n}种模式，每种模式的ai值分别是：{a_str}。每种模式会显示页面总数bi。"
            f"请确定Vasya需要查看多少种模式才能唯一确定物品数量。如果无法确定，请输出-1。"
            f"你的答案应放在[answer]标签中，例如：[answer]2[/answer]。"
        )
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        correct = cls.compute_correct_answer(identity['n'], identity['x'], identity['a'])
        return solution == correct

    @staticmethod
    def compute_correct_answer(n, x, a):
        a_set = set(a)
        if 1 in a_set and x > 2:
            return 1
        if x <= 2:
            return 1
        # 使用筛法计算质数，但限制在x不超过1e6，以避免内存问题
        max_sieve_x = min(x, 10**6)
        sieve = [True] * (max_sieve_x + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(max_sieve_x**0.5) + 1):
            if sieve[i]:
                sieve[i*i : max_sieve_x+1 : i] = [False] * len(sieve[i*i : max_sieve_x+1 : i])
        primes = [i for i, is_prime in enumerate(sieve) if is_prime]
        # 检查所有质数是否都在a_set中
        for p in primes:
            if p > x:
                break
            if p not in a_set:
                return -1
        return len(primes)
