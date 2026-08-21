"""# 

### 谜题描述
You are given two arrays a and b of positive integers, with length n and m respectively. 

Let c be an n × m matrix, where c_{i,j} = a_i ⋅ b_j. 

You need to find a subrectangle of the matrix c such that the sum of its elements is at most x, and its area (the total number of elements) is the largest possible.

Formally, you need to find the largest number s such that it is possible to choose integers x_1, x_2, y_1, y_2 subject to 1 ≤ x_1 ≤ x_2 ≤ n, 1 ≤ y_1 ≤ y_2 ≤ m, (x_2 - x_1 + 1) × (y_2 - y_1 + 1) = s, and $$$∑_{i=x_1}^{x_2}{∑_{j=y_1}^{y_2}{c_{i,j}}} ≤ x.$$$

Input

The first line contains two integers n and m (1 ≤ n, m ≤ 2000).

The second line contains n integers a_1, a_2, …, a_n (1 ≤ a_i ≤ 2000).

The third line contains m integers b_1, b_2, …, b_m (1 ≤ b_i ≤ 2000).

The fourth line contains a single integer x (1 ≤ x ≤ 2 ⋅ 10^{9}).

Output

If it is possible to choose four integers x_1, x_2, y_1, y_2 such that 1 ≤ x_1 ≤ x_2 ≤ n, 1 ≤ y_1 ≤ y_2 ≤ m, and ∑_{i=x_1}^{x_2}{∑_{j=y_1}^{y_2}{c_{i,j}}} ≤ x, output the largest value of (x_2 - x_1 + 1) × (y_2 - y_1 + 1) among all such quadruplets, otherwise output 0.

Examples

Input

3 3
1 2 3
1 2 3
9


Output

4


Input

5 1
5 4 2 4 5
2
5


Output

1

Note

Matrix from the first sample and the chosen subrectangle (of blue color):

<image>

Matrix from the second sample and the chosen subrectangle (of blue color):

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n,m=map(int,raw_input().split())
a=list(map(int,raw_input().split()))
b=list(map(int,raw_input().split()))
x=int(raw_input())
pref_a=[0]

for i in a:
    pref_a.append(pref_a[-1]+i)
pref_b=[0]
for i in b:
    pref_b.append(pref_b[-1]+i)
min_a=[float('inf')]*(n+1)
min_b=[float('inf')]*(m+1)
for i in range(1,n+1):
    for j in range(i,n+1):
        min_a[j-i+1]=min(min_a[j-i+1],pref_a[j]-pref_a[i-1])
for i in range(1,m+1):
    for j in range(i,m+1):
        min_b[j-i+1]=min(min_b[j-i+1],pref_b[j]-pref_b[i-1])
ans=0
for i in range(1,n+1):
    for j in range(1,m+1):
        if min_a[i]*min_b[j]<=x:
            ans=max(ans,i*j)
print(ans)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

def calculate_min_prefix_sums(arr):
    n = len(arr)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + arr[i]
    
    min_sums = [float('inf')] * (n + 1)
    for k in range(1, n+1):
        min_sum = min(prefix[i+k] - prefix[i] for i in range(n - k + 1))
        min_sums[k] = min_sum
    return min_sums

def calculate_max_area(n, m, a, b, x):
    min_a = calculate_min_prefix_sums(a)
    min_b = calculate_min_prefix_sums(b)
    
    max_area = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            if min_a[i] * min_b[j] <= x:
                max_area = max(max_area, i * j)
    return max_area

class Cmaximumsubrectanglebootcamp(Basebootcamp):
    def __init__(self, max_n=2000, max_m=2000, max_val=2000):
        """
        参数说明:
            max_n: a数组的最大长度（符合题目约束）
            max_m: b数组的最大长度（符合题目约束）
            max_val: 数组元素的最大值（符合题目约束）
        """
        super().__init__()
        self.max_n = max_n
        self.max_m = max_m
        self.max_val = max_val
    
    def case_generator(self):
        # 生成随机长度（1 ≤ n, m ≤ 2000）
        n = random.randint(1, self.max_n)
        m = random.randint(1, self.max_m)
        
        # 生成随机数组（元素范围1~2000）
        a = [random.randint(1, self.max_val) for _ in range(n)]
        b = [random.randint(1, self.max_val) for _ in range(m)]
        
        # 计算最小子数组和
        min_a = calculate_min_prefix_sums(a)
        min_b = calculate_min_prefix_sums(b)
        
        # 收集所有可能的乘积
        products = []
        for i in range(1, n+1):
            for j in range(1, m+1):
                products.append(min_a[i] * min_b[j])
        
        # 生成x的策略（确保覆盖所有边界情况）
        if not products:  # 理论上不可能发生
            x = 1
        else:
            min_product = min(products)
            max_product = max(products)
            
            # 生成模式选择（按概率分布）
            mode = random.choices(
                population=[0, 1, 2, 3],
                weights=[0.3, 0.3, 0.2, 0.2],  # 增加无解情况的概率
                k=1
            )[0]
            
            if mode == 0:   # 正常范围解
                x = random.randint(min_product, max_product)
            elif mode == 1: # 无解情况
                x = random.randint(1, min_product-1) if min_product > 1 else 0
            elif mode == 2: # 超大值覆盖所有解
                x = max_product * random.randint(1, 100)
            else:          # 极小值特殊情况
                x = 1 if random.random() < 0.5 else 0
        
        # 确保x符合题目约束（1 ≤ x ≤ 2e9）
        x = max(1, min(x, 2*10**9))
        
        return {
            'n': n,
            'm': m,
            'a': a.copy(),
            'b': b.copy(),
            'x': x
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        return f"""给定两个正整数数组：
数组a（长度={question_case['n']}）：{question_case['a']}
数组b（长度={question_case['m']}）：{question_case['b']}

定义矩阵c，其中每个元素c[i][j] = a[i] × b[j]。请找到c中元素和不超过{question_case['x']}的最大矩形区域（元素个数最多）。如果不存在这样的区域，输出0。

答案必须为整数且用[ANSWER]标签包裹，例如：[ANSWER]0[/ANSWER] 或 [ANSWER]42[/ANSWER]
"""
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[ANSWER\]\s*(\d+)\s*\[/ANSWER\]', output)
        try:
            return int(matches[-1]) if matches else None
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == calculate_max_area(
            identity['n'], identity['m'],
            identity['a'], identity['b'],
            identity['x']
        )
