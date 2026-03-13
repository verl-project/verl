"""# 

### 谜题描述
There are some rabbits in Singapore Zoo. To feed them, Zookeeper bought n carrots with lengths a_1, a_2, a_3, …, a_n. However, rabbits are very fertile and multiply very quickly. Zookeeper now has k rabbits and does not have enough carrots to feed all of them. To solve this problem, Zookeeper decided to cut the carrots into k pieces. For some reason, all resulting carrot lengths must be positive integers.

Big carrots are very difficult for rabbits to handle and eat, so the time needed to eat a carrot of size x is x^2.

Help Zookeeper split his carrots while minimizing the sum of time taken for rabbits to eat the carrots.

Input

The first line contains two integers n and k (1 ≤ n ≤ k ≤ 10^5): the initial number of carrots and the number of rabbits.

The next line contains n integers a_1, a_2, …, a_n (1 ≤ a_i ≤ 10^6): lengths of carrots.

It is guaranteed that the sum of a_i is at least k.

Output

Output one integer: the minimum sum of time taken for rabbits to eat carrots.

Examples

Input


3 6
5 3 1


Output


15


Input


1 4
19


Output


91

Note

For the first test, the optimal sizes of carrots are \{1,1,1,2,2,2\}. The time taken is 1^2+1^2+1^2+2^2+2^2+2^2=15

For the second test, the optimal sizes of carrots are \{4,5,5,5\}. The time taken is 4^2+5^2+5^2+5^2=91.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from collections import Counter, defaultdict, deque
import bisect
import heapq
from sys import stdin, stdout
from itertools import repeat
import math
import random
 
# sys.stdin = open('input')
 
def mod(x, y, mod):
    re = 1
    now = x
    while y:
        if y&1:
            re *= now
            re %= mod
        y >>= 1
        now = (now*now)%mod
    return re
 
def inp(force_list=False):
    re = map(int, raw_input().split())
    if len(re) == 1 and not force_list:
        return re[0]
    return re
 
def inst():
    return raw_input().strip()
 
def gcd(x, y):
   while(y):
       x, y = y, x % y
   return x
 
def ggcd(x, y):
    if y:
        return ggcd(y, x%y)
    return x
 
MOD = int(1e9+7)
 
 
def my_main():
    T = 1
    for _ in range(T):
        n, k = inp()
        da = inp(True)
        hp = []
        idx = [1 for i in range(n)]
        def ff(p, l):
            w = p/l
            wp = p/l+1
            nwp = p%l
            nw = l-nwp
            return nw*w*w+nwp*wp*wp
        ans = 0

        for i in range(n):
            hp.append((ff(da[i], 2) - ff(da[i], 1), i))
        heapq.heapify(hp)
        for i in range(k-n):
            item, ii = hp[0]
            idx[ii] += 1
            ans += item
            heapq.heapreplace(hp, (ff(da[ii], idx[ii]+1) - ff(da[ii], idx[ii]), ii))
        for i in da:
           ans += i*i
        print ans

            
my_main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from heapq import heappush, heappop
from bootcamp import Basebootcamp

class Ecarrotsforrabbitsbootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_a=1000000, max_k=100000):
        self.max_n = max_n
        self.max_a = max_a
        self.max_k = max_k
    
    def case_generator(self):
        # Generate valid parameters meeting sum(a) >= k
        n = random.randint(1, self.max_n)
        min_k = n
        max_possible_k = min(self.max_k, self.max_n * self.max_a)
        
        # Randomly select target k within valid range
        k = random.randint(min_k, max_possible_k)
        
        # Generate carrot lengths ensuring sum >= k
        base = k // n
        remainder = k % n
        a = [
            base + (1 if i < remainder else 0) + random.randint(0, self.max_a)
            for i in range(n)
        ]
        
        # Recalculate actual sum to ensure validity
        sum_a = sum(a)
        if sum_a < k:  # Fallback for edge cases
            deficit = k - sum_a
            a[-1] += deficit
        
        # Compute correct answer
        correct_output = self.compute_min_time(n, k, a)
        
        return {
            'n': n,
            'k': k,
            'a': a,
            'correct_output': correct_output
        }
    
    @staticmethod
    def prompt_func(question_case):
        return f"""新加坡动物园需要喂养 {question_case['k']} 只兔子。现有 {question_case['n']} 根胡萝卜，长度分别为：{', '.join(map(str, question_case['a']))}。

任务要求：
1. 将胡萝卜切分正好 {question_case['k']} 段
2. 每段长度必须是正整数
3. 最小化总进食时间（时间=段长平方的累加）

输入格式：
第一行：n k
第二行：a_1 a_2 ... a_n

输出：
一个整数表示最小总时间，置于[answer]和[/answer]之间。示例：[answer]91[/answer]"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return int(solution) == identity['correct_output']
        except:
            return False
    
    @staticmethod
    def compute_min_time(n, k, a):
        def calculate_cost(length, splits):
            base, remainder = divmod(length, splits)
            return (splits - remainder) * (base ** 2) + remainder * ((base + 1) ** 2)
        
        heap = []
        current_splits = [1] * n
        total = sum(x**2 for x in a)
        
        # Initialize priority queue with initial split options
        for i in range(n):
            if a[i] > 1:
                cost_diff = calculate_cost(a[i], 2) - (a[i] ** 2)
                heappush(heap, (cost_diff, i))
        
        # Perform required splits
        for _ in range(k - n):
            if not heap:
                break
            
            delta, idx = heappop(heap)
            total += delta
            current_splits[idx] += 1
            
            # Schedule next possible split
            if current_splits[idx] < a[idx]:
                next_splits = current_splits[idx] + 1
                new_delta = calculate_cost(a[idx], next_splits) - calculate_cost(a[idx], current_splits[idx])
                heappush(heap, (new_delta, idx))
        
        return total

# 验证测试
if __name__ == "__main__":
    # 测试官方示例1
    case1 = {'n':3, 'k':6, 'a':[5,3,1], 'correct_output':15}
    assert Ecarrotsforrabbitsbootcamp.compute_min_time(**case1) == 15
    
    # 测试官方示例2
    case2 = {'n':1, 'k':4, 'a':[19], 'correct_output':91}
    assert Ecarrotsforrabbitsbootcamp.compute_min_time(**case2) == 91
    
    # 边界测试：最小分割次数
    edge_case = {'n':2, 'k':2, 'a':[5,5], 'correct_output':50}
    assert Ecarrotsforrabbitsbootcamp.compute_min_time(**edge_case) == 50
    
    # 随机案例验证
    bootcamp = Ecarrotsforrabbitsbootcamp(max_n=5, max_a=20, max_k=20)
    test_case = bootcamp.case_generator()
    calculated = Ecarrotsforrabbitsbootcamp.compute_min_time(
        test_case['n'], 
        test_case['k'], 
        test_case['a']
    )
    print(f"验证案例 {test_case} → 计算结果：{calculated}")
