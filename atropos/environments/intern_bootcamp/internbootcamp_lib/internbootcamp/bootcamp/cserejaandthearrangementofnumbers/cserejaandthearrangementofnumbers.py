"""# 

### 谜题描述
Let's call an array consisting of n integer numbers a1, a2, ..., an, beautiful if it has the following property:

  * consider all pairs of numbers x, y (x ≠ y), such that number x occurs in the array a and number y occurs in the array a; 
  * for each pair x, y must exist some position j (1 ≤ j < n), such that at least one of the two conditions are met, either aj = x, aj + 1 = y, or aj = y, aj + 1 = x. 



Sereja wants to build a beautiful array a, consisting of n integers. But not everything is so easy, Sereja's friend Dima has m coupons, each contains two integers qi, wi. Coupon i costs wi and allows you to use as many numbers qi as you want when constructing the array a. Values qi are distinct. Sereja has no coupons, so Dima and Sereja have made the following deal. Dima builds some beautiful array a of n elements. After that he takes wi rubles from Sereja for each qi, which occurs in the array a. Sereja believed his friend and agreed to the contract, and now he is wondering, what is the maximum amount of money he can pay.

Help Sereja, find the maximum amount of money he can pay to Dima.

Input

The first line contains two integers n and m (1 ≤ n ≤ 2·106, 1 ≤ m ≤ 105). Next m lines contain pairs of integers. The i-th line contains numbers qi, wi (1 ≤ qi, wi ≤ 105).

It is guaranteed that all qi are distinct.

Output

In a single line print maximum amount of money (in rubles) Sereja can pay.

Please, do not use the %lld specifier to read or write 64-bit integers in С++. It is preferred to use the cin, cout streams or the %I64d specifier.

Examples

Input

5 2
1 2
2 3


Output

5


Input

100 3
1 2
2 1
3 1


Output

4


Input

1 2
1 1
2 100


Output

100

Note

In the first sample Sereja can pay 5 rubles, for example, if Dima constructs the following array: [1, 2, 1, 2, 2]. There are another optimal arrays for this test.

In the third sample Sereja can pay 100 rubles, if Dima constructs the following array: [2].

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import itertools

n, m = map(int, raw_input().split())
Q = [0]*m
W = [0]*m
for i in range(m):
  Q[i], W[i] = map(int, raw_input().split())

k = 0
while True:
  k += 1
  l = 0
  if k%2 == 1:
    l = k*(k-1)/2+1
  else:
    l = k*k/2
  if l > n:
    break
k -= 1
W.sort()
W.reverse()
if k > m:
  print sum(W)
else:
  print sum(W[:k])
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cserejaandthearrangementofnumbersbootcamp(Basebootcamp):
    def __init__(self, max_n=2000000, max_m=100000, default_n=100, default_m=10):
        self.max_n = max_n
        self.max_m = max_m
        self.default_n = default_n
        self.default_m = default_m
    
    def case_generator(self):
        n = random.randint(1, self.default_n)
        m = random.randint(1, self.default_m)
        q_list = []
        while len(q_list) < m:
            q = random.randint(1, 10**5)
            if q not in q_list:
                q_list.append(q)
        w_list = [random.randint(1, 10**5) for _ in range(m)]
        
        k = 0
        while True:
            k_candidate = k + 1
            if k_candidate % 2 == 1:
                l = (k_candidate * (k_candidate - 1)) // 2 + 1
            else:
                l = (k_candidate ** 2) // 2
            if l > n:
                break
            k = k_candidate
        w_sorted = sorted(w_list, reverse=True)
        if k <= m:
            correct_sum = sum(w_sorted[:k])
        else:
            correct_sum = sum(w_sorted)
        coupons = list(zip(q_list, w_list))
        return {
            'n': n,
            'm': m,
            'coupons': coupons,
            'correct_sum': correct_sum
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        coupons = question_case['coupons']
        lines = [f"{n} {m}"]
        for q, w in coupons:
            lines.append(f"{q} {w}")
        problem_instance = "\n".join(lines)
        prompt = f"""问题描述：

定义一个由n个整数组成的数组为“美丽数组”，当且仅当满足以下条件：

考虑数组中的所有不同的数对x和y（x≠y），其中x和y都出现在数组中。对于每一对x和y，必须存在至少一个位置j（1 ≤j <n），使得aj=x且aj+1=y，或者aj=y且aj+1=x。

Dima会构造这样的美丽数组a，包含n个元素。Sereja需要支付的金额等于数组中所有不同qi对应的wi的总和。你的任务是计算Sereja可能支付的最大金额。

输入格式：
第一行是两个整数n和m，分别表示数组的长度和优惠券的数量。
接下来m行，每行两个整数qi和wi，表示每个优惠券允许使用的数字和对应的费用。

输出格式：
输出一个整数，表示Sereja可能支付的最大金额。

请根据以下输入数据求解问题：

{problem_instance}

请将最终答案放入[answer]标签中，例如：[answer]12345[/answer]。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity.get('correct_sum', None)
