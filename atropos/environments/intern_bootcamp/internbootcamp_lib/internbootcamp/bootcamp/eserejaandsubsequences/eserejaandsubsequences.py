"""# 

### 谜题描述
Sereja has a sequence that consists of n positive integers, a1, a2, ..., an. 

First Sereja took a piece of squared paper and wrote all distinct non-empty non-decreasing subsequences of sequence a. Then for each sequence written on the squared paper, Sereja wrote on a piece of lines paper all sequences that do not exceed it.

A sequence of positive integers x = x1, x2, ..., xr doesn't exceed a sequence of positive integers y = y1, y2, ..., yr, if the following inequation holds: x1 ≤ y1, x2 ≤ y2, ..., xr ≤ yr.

Now Sereja wonders, how many sequences are written on the lines piece of paper. Help Sereja, find the required quantity modulo 1000000007 (109 + 7). 

Input

The first line contains integer n (1 ≤ n ≤ 105). The second line contains n integers a1, a2, ..., an (1 ≤ ai ≤ 106).

Output

In the single line print the answer to the problem modulo 1000000007 (109 + 7).

Examples

Input

1
42


Output

42


Input

3
1 2 2


Output

13


Input

5
1 2 3 4 5


Output

719

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = input()
a = map(int, raw_input().split())
mod = int(1e+9) + 7
size = max(a)
tree = [0] * (size + 1)

def query(index):
  res = 0
  while index:
    res = (res + tree[index]) % mod
    index -= index & -index
  return res

def update(index, delta):
  while index <= size:
    tree[index] = (tree[index] + delta) % mod
    index += index & -index

def query_one(index):
  res = tree[index]
  bot = index - (index & -index)
  index -= 1
  while index > bot:
    res -= tree[index]
    if res < 0: res += mod
    index -= index & -index
  return res


for x in a:
  value = query(x) * x + x
  update(x, value - query_one(x))
print query(size)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Eserejaandsubsequencesbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=5, min_val=1, max_val=10, seed=None):
        self.min_n = min_n
        self.max_n = max_n
        self.min_val = min_val
        self.max_val = max_val
        self.seed = seed
        if seed is not None:
            random.seed(seed)
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        a = [random.randint(self.min_val, self.max_val) for _ in range(n)]
        answer = self.compute_answer(a)
        return {
            'n': n,
            'a': a,
            'answer': answer
        }
    
    @staticmethod
    def compute_answer(a):
        mod = 10**9 + 7
        if not a:
            return 0
        max_val = max(a)
        tree = [0] * (max_val + 2)  # Extra space to avoid index issues
        last = {}
        total = 0
        
        for x in a:
            # Query sum of all elements <= x
            sum_prev = 0
            idx = x
            while idx > 0:
                sum_prev = (sum_prev + tree[idx]) % mod
                idx -= idx & -idx
            
            # Calculate new contribution
            new_contrib = (sum_prev * x + x) % mod
            delta = (new_contrib - last.get(x, 0)) % mod
            
            # Update Fenwick tree
            idx = x
            while idx <= max_val:
                tree[idx] = (tree[idx] + delta) % mod
                idx += idx & -idx
            
            # Update last and total
            last[x] = new_contrib
            total = (total + delta) % mod
        
        return total
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = question_case['a']
        a_str = ' '.join(map(str, a))
        problem_text = f"""Sereja有一个由n个正整数组成的序列a。你需要解决以下问题：

问题描述：
找出所有不同的非空非递减子序列y。然后，对于每个y，计算所有可能的序列x的数量，其中x的长度与y相同，并且每个对应的元素x_i ≤ y_i。所有x的数量的总和模1000000007即为答案。

子序列定义：
- 子序列的元素保持原序列中的相对顺序，但可以删除某些元素。
- 非递减：子序列中的每个元素不小于前一个元素。
- 不同的子序列由它们的元素序列决定，即相同的元素序列被视为同一个子序列，即使它们来自原序列的不同位置。

输入格式：
- 第一行包含整数n（1 ≤ n ≤ 1e5）。
- 第二行包含n个正整数a_1, a_2, ..., a_n（1 ≤ a_i ≤ 1e6）。

你的任务：
给定n和序列a，计算最终的答案并以模1e9+7输出。

输入样例：
{n}
{a_str}

请按照上述输入样例的格式，计算出正确的答案，并将最终答案用[answer]和[/answer]标签包裹。例如：[answer]42[/answer]。"""
        return problem_text
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        digits = re.sub(r'\D', '', last_match)
        try:
            return int(digits) % (10**9 + 7)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        correct_answer = identity.get('answer')
        return solution == correct_answer
