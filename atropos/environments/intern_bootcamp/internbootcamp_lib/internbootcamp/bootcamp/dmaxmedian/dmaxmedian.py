"""# 

### 谜题描述
You are a given an array a of length n. Find a subarray a[l..r] with length at least k with the largest median.

A median in an array of length n is an element which occupies position number ⌊ (n + 1)/(2) ⌋ after we sort the elements in non-decreasing order. For example: median([1, 2, 3, 4]) = 2, median([3, 2, 1]) = 2, median([2, 1, 2, 1]) = 1.

Subarray a[l..r] is a contiguous part of the array a, i. e. the array a_l,a_{l+1},…,a_r for some 1 ≤ l ≤ r ≤ n, its length is r - l + 1.

Input

The first line contains two integers n and k (1 ≤ k ≤ n ≤ 2 ⋅ 10^5).

The second line contains n integers a_1, a_2, …, a_n (1 ≤ a_i ≤ n).

Output

Output one integer m — the maximum median you can get.

Examples

Input


5 3
1 2 3 2 1


Output


2

Input


4 2
1 2 3 4


Output


3

Note

In the first example all the possible subarrays are [1..3], [1..4], [1..5], [2..4], [2..5] and [3..5] and the median for all of them is 2, so the maximum possible median is 2 too.

In the second example median([3..4]) = 3.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from collections import Counter, defaultdict, deque
import bisect
from sys import stdin, stdout
from itertools import repeat
import math


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


mod = int(1e9)+7

def quickm(a, b):
    base = a
    re = 1
    while b:
        if b&1:
            re *= base
            re %= mod
        b >>= 1
        base *= base
        base %= mod
    return re

def inv(num):
    return quickm(num, mod-2)



def my_main():
    kase = 1 #inp()
    pans = []
    for _ in range(kase):
        n, k = inp()
        da = inp(True)
        l, r = min(da), max(da)+1
        while l < r-1:
            mid = (l+r)/2
            def ck(mid):
                ps = [0]
                for j in [(-1 if i<mid else 1) for i in da]:
                    ps.append(j+ps[-1])
                ok = 0
                mps = [-100000] * (n+1)
                mps[-1] = ps[-1]
                for i in range(n-1, -1, -1):
                    mps[i] = max(mps[i+1], ps[i])
                for i in range(n-k+1):
                    if mps[i+k] - ps[i] > 0:
                        ok = 1
                        break
                return ok
            # print l, r
            if ck(mid):
                l, r = mid, r
            else:
                l, r = l, mid
        print l


    # print '\n'.join(pans)

my_main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

def calculate_max_median(n, k, array):
    """优化后的中位数计算函数"""
    left, right = min(array), max(array)
    answer = left  # 初始化
    
    while left <= right:
        mid = (left + right) // 2
        prefix = [0]*(n+1)
        min_prefix = float('inf')
        
        # 计算前缀和
        for i in range(n):
            prefix[i+1] = prefix[i] + (1 if array[i] >= mid else -1)
        
        # 寻找有效窗口
        valid = False
        for i in range(k, n+1):
            if prefix[i] - min_prefix > 0:
                valid = True
                break
            min_prefix = min(min_prefix, prefix[i - k + 1])
        
        if valid:
            answer = mid
            left = mid + 1
        else:
            right = mid - 1
    
    return answer

class Dmaxmedianbootcamp(Basebootcamp):  # 修正类名
    def __init__(self, **params):
        super().__init__(**params)
        default_params = {
            'min_n': 5,
            'max_n': 20,
            'max_val': 20,
            'ensure_solvable': True  # 保证生成有解的案例
        }
        self.params = {**default_params, **params}
    
    def case_generator(self):
        """生成有效案例的优化版本"""
        n = random.randint(self.params['min_n'], self.params['max_n'])
        k = random.randint(1, n)
        
        # 生成有解数组的逻辑
        while True:
            arr = [random.randint(1, self.params['max_val']) for _ in range(n)]
            if len(set(arr)) >= 2:  # 确保至少有两个不同值
                break
        
        return {
            'n': n,
            'k': k,
            'array': arr.copy(),
            'answer': calculate_max_median(n, k, arr)
        }
    
    @staticmethod
    def prompt_func(case):
        return f"""给定长度为n的数组，请找出长度≥k的连续子数组的最大中位数。

输入：
{case['n']} {case['k']}
{' '.join(map(str, case['array']))}

规则：
1. 中位数定义：排序后第⌊(长度+1)/2⌋个元素
2. 子数组必须连续且长度≥k
3. 输出最大可能的中位数

请将最终答案放在[answer]标签内，如：[answer]42[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        try:
            return int(matches[-1]) if matches else None
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['answer']
