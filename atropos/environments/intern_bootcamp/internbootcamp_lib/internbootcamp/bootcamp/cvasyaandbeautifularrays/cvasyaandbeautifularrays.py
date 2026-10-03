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
from Queue import * # Queue, LifoQueue, PriorityQueue
from bisect import * #bisect, insort
from datetime import * 
from collections import * #deque, Counter,OrderedDict,defaultdict
import calendar
import heapq
import math
import copy
import itertools

def solver():
    n,k = map(int,raw_input().split())
    num = map(int, raw_input().split())
    num.sort()
    ans = num[0]
    
    while True:
        for i in range(n):
            if num[i]%ans > k:
                ans = num[i] / (num[i]/ans + 1)
                break
        if(i == n -1):
            break

    print ans
        
    
    



if __name__ == \"__main__\":
    solver()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

def compute_max_beauty(n, k, a):
    a_sorted = sorted(a)
    current_gcd = a_sorted[0]
    
    while True:
        update_flag = False
        for num in a_sorted:
            residue = num % current_gcd
            if residue > k:
                # 计算可能的新候选GCD
                new_quotient = (num // current_gcd) + 1
                candidate_gcd = num // new_quotient
                
                # 确保候选GCD不小于1
                current_gcd = max(1, candidate_gcd)
                update_flag = True
                break
        
        if not update_flag:
            break
    
    # 最终验证所有元素符合条件
    for num in a_sorted:
        if num % current_gcd > k:
            return 1  # 安全回退
    
    return current_gcd

class Cvasyaandbeautifularraysbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n_range = params.get('n_range', (3, 10))
        self.k_max = params.get('k_max', 50)
        self.a_range = params.get('a_range', (100, 1000))
        
        # 动态计算有效参数范围
        self.a_min = max(self.a_range[0], self.k_max + 2)
        self.a_max = self.a_range[1]

    def case_generator(self):
        n = random.randint(*self.n_range)
        k = random.randint(1, min(self.k_max, self.a_min-1))
        
        # 生成合法数组
        a = [random.randint(k+1, self.a_max) for _ in range(n)]
        
        # 确保至少有两个不同的元素
        while len(set(a)) < 2:
            a = [random.randint(k+1, self.a_max) for _ in range(n)]
        
        correct_answer = compute_max_beauty(n, k, a)
        return {'n': n, 'k': k, 'a': a, 'correct_answer': correct_answer}

    @staticmethod
    def prompt_func(question_case):
        return (
            f"Given array {question_case['a']} with length {question_case['n']}, "
            f"find maximum possible GCD when decreasing each element by at most {question_case['k']}.\n"
            "Format answer as [answer]N[/answer] where N is the integer result."
        )

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
