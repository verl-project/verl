"""# 

### 谜题描述
You are given three bags. Each bag contains a non-empty multiset of numbers. You can perform a number of operations on these bags. In one operation, you can choose any two non-empty bags, and choose one number from each of the bags. Let's say that you choose number a from the first bag and number b from the second bag. Then, you remove b from the second bag and replace a with a-b in the first bag. Note that if there are multiple occurrences of these numbers, then you shall only remove/replace exactly one occurrence.

You have to perform these operations in such a way that you have exactly one number remaining in exactly one of the bags (the other two bags being empty). It can be shown that you can always apply these operations to receive such a configuration in the end. Among all these configurations, find the one which has the maximum number left in the end.

Input

The first line of the input contains three space-separated integers n_1, n_2 and n_3 (1 ≤ n_1, n_2, n_3 ≤ 3⋅10^5, 1 ≤ n_1+n_2+n_3 ≤ 3⋅10^5) — the number of numbers in the three bags.

The i-th of the next three lines contain n_i space-separated integers a_{{i,1}}, a_{{i,2}}, ..., a_{{i,{{n_i}}}} (1 ≤ a_{{i,j}} ≤ 10^9) — the numbers in the i-th bag.

Output

Print a single integer — the maximum number which you can achieve in the end.

Examples

Input


2 4 1
1 2
6 3 4 5
5


Output


20

Input


3 2 2
7 5 4
2 9
7 1


Output


29

Note

In the first example input, let us perform the following operations:

[1, 2], [6, 3, 4, 5], [5]

[-5, 2], [3, 4, 5], [5] (Applying an operation to (1, 6))

[-10, 2], [3, 4], [5] (Applying an operation to (-5, 5))

[2], [3, 4], [15] (Applying an operation to (5, -10))

[-1], [4], [15] (Applying an operation to (2, 3))

[-5], [], [15] (Applying an operation to (-1, 4))

[], [], [20] (Applying an operation to (15, -5))

You can verify that you cannot achieve a bigger number. Hence, the answer is 20.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division,print_function
from heapq import*
import sys

le = sys.__stdin__.read().split(\"\n\")[::-1]
af = []
n=list(map(int,le.pop().split()))
l=[]
for k in range(3):
    l.append(list(map(int,le.pop().split())))
m=sorted(list(map(min,l)))
s=sorted(list(map(sum,l)))
print(max(s[2]+s[1]-s[0],sum(s)-2*(m[0]+m[1])))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Cthreebagsbootcamp(Basebootcamp):
    def __init__(self, min_elements=1, max_elements=5, min_value=1, max_value=20):
        super().__init__()  # 显式调用父类初始化
        self.min_elements = min_elements
        self.max_elements = max_elements
        self.min_value = min_value
        self.max_value = max_value
    
    def case_generator(self):
        # 确保生成的包都是非空的
        n1 = random.randint(max(1, self.min_elements), self.max_elements)
        n2 = random.randint(max(1, self.min_elements), self.max_elements)
        n3 = random.randint(max(1, self.min_elements), self.max_elements)
        
        bag1 = [random.randint(self.min_value, self.max_value) for _ in range(n1)]
        bag2 = [random.randint(self.min_value, self.max_value) for _ in range(n2)]
        bag3 = [random.randint(self.min_value, self.max_value) for _ in range(n3)]
        
        return {
            'bags': [
                sorted(bag1, reverse=True),  # 排序方便后续验证
                sorted(bag2, reverse=True),
                sorted(bag3, reverse=True)
            ]
        }
    
    @staticmethod
    def prompt_func(question_case):
        bags = question_case['bags']
        n_values = [len(bag) for bag in bags]
        input_str = f"{n_values[0]} {n_values[1]} {n_values[2]}\n" 
        input_str += "\n".join(" ".join(map(str, bag)) for bag in bags)
        
        return f"""## Three Bags Puzzle ##
You have three bags containing numbers. In each operation:
1. Choose two non-empty bags
2. Pick one number from each (a from first, b from second)
3. Replace a with a-b in the first bag
4. Remove b from the second bag

Goal: Leave exactly one number in any bag with others empty. Find the MAXIMUM possible final number.

Input format:
Line 1: n1 n2 n3 (element counts)
Next 3 lines: space-separated numbers for each bag

Output format:
Single integer in [answer]...[/answer] tags

Example:
Input:
2 4 1
1 2
6 3 4 5
5
Answer:
[answer]20[/answer]

Now solve this problem:
{input_str}
[answer][/answer]"""

    @staticmethod
    def extract_output(output):
        # 支持负数和科学计数法格式
        pattern = r'\[answer\](.*?)\[/answer\]'
        matches = re.findall(pattern, output, re.DOTALL)
        if not matches:
            return None
        
        try:
            return int(matches[-1].strip())
        except ValueError:
            try:
                return float(matches[-1].strip())
            except:
                return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 获取参考代码中的决策逻辑
        bags = identity['bags']
        
        sums = [sum(bag) for bag in bags]
        mins = [min(bag) for bag in bags]
        
        sorted_sums = sorted(sums)
        sorted_mins = sorted(mins)
        
        # 计算两种可能策略
        strategy1 = sorted_sums[2] + sorted_sums[1] - sorted_sums[0]
        strategy2 = sum(sorted_sums) - 2 * (sorted_mins[0] + sorted_mins[1])
        
        # 允许整数和浮点数比较
        return abs(max(strategy1, strategy2) - solution) < 1e-9
