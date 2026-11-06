"""# 

### 谜题描述
Lena is the most economical girl in Moscow. So, when her dad asks her to buy some food for a trip to the country, she goes to the best store — \"PriceFixed\". Here are some rules of that store:

  * The store has an infinite number of items of every product. 
  * All products have the same price: 2 rubles per item. 
  * For every product i there is a discount for experienced buyers: if you buy b_i items of products (of any type, not necessarily type i), then for all future purchases of the i-th product there is a 50\% discount (so you can buy an item of the i-th product for 1 ruble!). 



Lena needs to buy n products: she must purchase at least a_i items of the i-th product. Help Lena to calculate the minimum amount of money she needs to spend if she optimally chooses the order of purchasing. Note that if she wants, she can buy more items of some product than needed.

Input

The first line contains a single integer n (1 ≤ n ≤ 100 000) — the number of products.

Each of next n lines contains a product description. Each description consists of two integers a_i and b_i (1 ≤ a_i ≤ 10^{14}, 1 ≤ b_i ≤ 10^{14}) — the required number of the i-th product and how many products you need to buy to get the discount on the i-th product. 

The sum of all a_i does not exceed 10^{14}.

Output

Output the minimum sum that Lena needs to make all purchases. 

Examples

Input


3
3 4
1 3
1 5


Output


8


Input


5
2 7
2 8
1 2
2 4
1 8


Output


12

Note

In the first example, Lena can purchase the products in the following way:

  1. one item of product 3 for 2 rubles, 
  2. one item of product 1 for 2 rubles, 
  3. one item of product 1 for 2 rubles, 
  4. one item of product 2 for 1 ruble (she can use the discount because 3 items are already purchased), 
  5. one item of product 1 for 1 ruble (she can use the discount because 4 items are already purchased). 



In total, she spends 8 rubles. It can be proved that it is impossible to spend less.

In the second example Lena can purchase the products in the following way:

  1. one item of product 1 for 2 rubles, 
  2. two items of product 2 for 2 rubles for each, 
  3. one item of product 5 for 2 rubles, 
  4. one item of product 3 for 1 ruble, 
  5. two items of product 4 for 1 ruble for each, 
  6. one item of product 1 for 1 ruble. 



In total, she spends 12 rubles.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division,print_function
from heapq import*
import sys
A,B=[],[]
le = sys.__stdin__.read().split(\"\n\")[::-1]
for zorg in range(int(le.pop())):
    a,b= list(map(int,le.pop().split()))
    A.append(a)
    B.append(b)
n=len(A)
s=sum(A)
pr=sorted((b,a) for a,b in zip(A,B))
r=0#to return
nbb=0#number bougth
for b,a in pr:
    x1=max(min(b,s)-nbb,0)
    nbb+=x1
    r+=2*x1
    x2=min(a,s-nbb)
    r+=x2
    nbb+=x2

print(r)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from heapq import *
from bootcamp import Basebootcamp

class Dpricefixedbootcamp(Basebootcamp):
    def __init__(self, max_n=1000, max_a=10**5, max_b=10**5, max_total_a=10**14):
        self.max_n = max_n
        self.max_a = max_a
        self.max_b = max_b
        self.max_total_a = max_total_a
    
    def case_generator(self):
        n = random.randint(1, self.max_n)
        products = []
        sum_prev = 0
        for i in range(n):
            remaining_products = n - i - 1
            max_a_possible = min(self.max_a, self.max_total_a - sum_prev - remaining_products)
            if max_a_possible < 1:
                a_i = 1
            else:
                a_i = random.randint(1, max_a_possible)
            b_i = random.randint(1, self.max_b)
            products.append((a_i, b_i))
            sum_prev += a_i
        correct_total = self._calculate_min_cost(n, products)
        return {
            'n': n,
            'products': products,
            'correct_total': correct_total
        }
    
    @staticmethod
    def _calculate_min_cost(n, products):
        A = [a for a, b in products]
        B = [b for a, b in products]
        total = sum(A)
        sorted_pr = sorted(zip(B, A), key=lambda x: x[0])
        cost = 0
        bought = 0
        for b, a in sorted_pr:
            full_price_steps = max(min(b, total) - bought, 0)
            bought += full_price_steps
            cost += 2 * full_price_steps
            discount_steps = min(a, total - bought)
            cost += discount_steps
            bought += discount_steps
        return cost
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        products = question_case['products']
        problem_lines = "\n".join(f"{a} {b}" for a, b in products)
        prompt = f"""Lena needs to buy {n} different products for her trip, each with a required quantity and a discount threshold. The store's pricing rules are as follows:

- Each item initially costs 2 rubles.
- For product i, if Lena has bought at least b_i items in total (from any products), subsequent purchases of product i cost 1 ruble each.

Your task is to compute the minimum total cost. The input format is:

{n}
{problem_lines}

Output the minimal total rubles. Place your answer within [answer] and [/answer] tags. For example: [answer]12[/answer]."""
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
        return solution == identity['correct_total']
