"""# 

### 谜题描述
There are n cities numbered from 1 to n, and city i has beauty a_i.

A salesman wants to start at city 1, visit every city exactly once, and return to city 1.

For all i≠ j, a flight from city i to city j costs max(c_i,a_j-a_i) dollars, where c_i is the price floor enforced by city i. Note that there is no absolute value. Find the minimum total cost for the salesman to complete his trip.

Input

The first line contains a single integer n (2≤ n≤ 10^5) — the number of cities.

The i-th of the next n lines contains two integers a_i, c_i (0≤ a_i,c_i≤ 10^9) — the beauty and price floor of the i-th city.

Output

Output a single integer — the minimum total cost.

Examples

Input


3
1 9
2 1
4 1


Output


11


Input


6
4 2
8 4
3 0
2 3
7 1
0 1


Output


13

Note

In the first test case, we can travel in order 1→ 3→ 2→ 1. 

  * The flight 1→ 3 costs max(c_1,a_3-a_1)=max(9,4-1)=9. 
  * The flight 3→ 2 costs max(c_3, a_2-a_3)=max(1,2-4)=1. 
  * The flight 2→ 1 costs max(c_2,a_1-a_2)=max(1,1-2)=1. 



The total cost is 11, and we cannot do better.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin
from heapq import heappush, heappop
from bisect import bisect
def main():
    n = int(stdin.readline())
    a = [map(int, stdin.readline().split(), (10, 10)) for _ in xrange(n)]
    a.sort()
    to = [0] * n
    d = [10 ** 18 for _ in xrange(n)]
    q = []
    d[0] = 0
    heappush(q, (0, 0))
    s = 0
    for i, x in enumerate(a):
        x, y = x
        p = bisect(a, [x + y, 1001001001], lo=i)
        to[i] = p - 1
        s += y
    while q:
        x, p = heappop(q)
        if d[p] != x:
            continue
        if p and d[p-1] > x:
            d[p-1] = x
            heappush(q, (x, p - 1))
        y = to[p]
        if d[y] > x:
            d[y] = x
            heappush(q, (x, y))
        y += 1
        if y < n and d[y] > x + a[y][0] - a[p][0] - a[p][1]:
            d[y] = z = x + a[y][0] - a[p][0] - a[p][1]
            heappush(q, (z, y))
    print s + d[-1]
main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from heapq import heappush, heappop
import random
import re

class Ctravellingsalesmanproblembootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=15, a_min=0, a_max=1e9, c_min=0, c_max=1e9):
        self.min_n = min_n
        self.max_n = max_n
        self.a_min = a_min
        self.a_max = a_max
        self.c_min = c_min
        self.c_max = c_max
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        cities = []
        prev_a = -1
        for _ in range(n):
            # 生成保证严格递增的a序列
            new_a = random.randint(max(self.a_min, prev_a + 1), 
                                 max(self.a_max, prev_a + 100))
            new_c = random.randint(self.c_min, self.c_max)
            cities.append((new_a, new_c))
            prev_a = new_a
        
        # 添加随机扰动保证案例多样性
        if random.random() < 0.3:
            cities = sorted(cities, key=lambda x: x[0], reverse=True)[::-1]
        
        correct_answer = self.calculate_min_cost(n, cities)
        return {
            'n': n,
            'cities': cities,
            'correct_answer': correct_answer
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        cities = question_case['cities']
        n = question_case['n']
        input_lines = [f"{ai} {ci}" for ai, ci in cities]
        input_str = '\n'.join([str(n)] + input_lines)
        return f"""城市数据（已按美景值排序）：
{input_str}

请计算最小旅行费用，将最终答案放在[answer]标签内。规则重申：
1. 必须从城市1出发并返回
2. 每个城市访问恰好一次
3. 费用计算规则：max(出发城市c值，目的城市a值 - 出发城市a值)"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\][^\d]*(-?\d+)[^\d]*\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
    
    @staticmethod
    def calculate_min_cost(n, a):
        a.sort()
        to = [0] * n
        d = [float('inf')] * n
        d[0] = 0
        q = [(0, 0)]
        total_c = sum(c for _, c in a)
        
        # 预计算每个城市的可达范围
        for i in range(n):
            x, y = a[i]
            l, r = i, n
            while l < r:
                m = (l + r) // 2
                if a[m][0] <= x + y:
                    l = m + 1
                else:
                    r = m
            to[i] = l - 1
        
        # 动态规划推进
        while q:
            cost, p = heappop(q)
            if cost > d[p]:
                continue
            
            # 向左扩展
            if p > 0 and d[p-1] > cost:
                d[p-1] = cost
                heappush(q, (cost, p-1))
            
            # 向右扩展
            if to[p] < n and d[to[p]] > cost:
                d[to[p]] = cost
                heappush(q, (cost, to[p]))
            
            # 跳跃扩展
            if to[p] + 1 < n:
                new_cost = cost + (a[to[p]+1][0] - a[p][0] - a[p][1])
                if new_cost < d[to[p]+1]:
                    d[to[p]+1] = new_cost
                    heappush(q, (new_cost, to[p]+1))
        
        return total_c + d[-1]
