"""# 

### 谜题描述
Johnny drives a truck and must deliver a package from his hometown to the district center. His hometown is located at point 0 on a number line, and the district center is located at the point d.

Johnny's truck has a gas tank that holds exactly n liters, and his tank is initially full. As he drives, the truck consumes exactly one liter per unit distance traveled. Moreover, there are m gas stations located at various points along the way to the district center. The i-th station is located at the point xi on the number line and sells an unlimited amount of fuel at a price of pi dollars per liter. Find the minimum cost Johnny must pay for fuel to successfully complete the delivery.

Input

The first line of input contains three space separated integers d, n, and m (1 ≤ n ≤ d ≤ 109, 1 ≤ m ≤ 200 000) — the total distance to the district center, the volume of the gas tank, and the number of gas stations, respectively.

Each of the next m lines contains two integers xi, pi (1 ≤ xi ≤ d - 1, 1 ≤ pi ≤ 106) — the position and cost of gas at the i-th gas station. It is guaranteed that the positions of the gas stations are distinct.

Output

Print a single integer — the minimum cost to complete the delivery. If there is no way to complete the delivery, print -1.

Examples

Input

10 4 4
3 5
5 8
6 3
8 4


Output

22


Input

16 5 2
8 2
5 1


Output

-1

Note

In the first sample, Johnny's truck holds 4 liters. He can drive 3 units to the first gas station, buy 2 liters of gas there (bringing the tank to 3 liters total), drive 3 more units to the third gas station, buy 4 liters there to fill up his tank, and then drive straight to the district center. His total cost is 2·5 + 4·3 = 22 dollars.

In the second sample, there is no way for Johnny to make it to the district center, as his tank cannot hold enough gas to take him from the latest gas station to the district center.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin
from itertools import repeat
from heapq import heappush, heappop
def main():
    d, n, m = map(int, stdin.readline().split())
    dat = map(int, stdin.read().split(), repeat(10, 2 * m))
    a = [(d, 0)] + [(dat[i*2], dat[i*2+1]) for i in xrange(m)]
    a.sort(reverse=True)
    if a[-1][0] > n:
        print -1
        return
    v = []
    while a and a[-1][0] <= n:
        t, x = a.pop()
        heappush(v, (x, t))
    ans = 0
    p = n
    for t, x in reversed(a):
        if t - p > n:
            print -1
            return
        while p < t:
            y, s = v[0]
            f = s + n - p
            if f > 0:
                if f > t - p:
                    f = t - p
                ans += y * f
                p += f
            else:
                heappop(v)
        heappush(v, (x, t))
    print ans
main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from heapq import heappush, heappop
import re
from bootcamp import Basebootcamp

class Dpackagedeliverybootcamp(Basebootcamp):
    def __init__(self, **params):
        default_params = {
            'd_min': 10,
            'd_max': 1000,
            'n_min': 1,
            'n_max': 1000,
            'm_min': 1,  # 强制m最小值
            'm_max': 200
        }
        default_params.update(params)
        # 参数合法性校验
        default_params['m_min'] = max(default_params['m_min'], 1)
        default_params['m_max'] = max(default_params['m_max'], default_params['m_min'])
        self.params = default_params
    
    def case_generator(self):
        params = self.params
        d = random.randint(params['d_min'], params['d_max'])
        n = random.randint(
            max(params['n_min'], 1), 
            min(params['n_max'], d)
        )
        m = random.randint(params['m_min'], min(params['m_max'], d-1))
        
        # 生成有效加油站位置
        xi_options = list(range(1, d))
        if len(xi_options) < m:
            m = len(xi_options)
        xi_list = sorted(random.sample(xi_options, m))
        pi_list = [random.randint(1, 10**6) for _ in range(m)]
        stations = list(zip(xi_list, pi_list))
        
        # 正确计算预期结果
        try:
            expected = self._reference_solution(d, n, stations)
        except:
            expected = -1
        
        return {
            'd': d,
            'n': n,
            'm': m,
            'stations': stations,
            'expected_output': expected
        }
    
    def _reference_solution(self, d, n, stations):
        """与题解参考代码完全一致的计算逻辑"""
        a = [(d, 0)] + stations
        a.sort(reverse=True)
        if a[-1][0] > n:
            return -1
        
        heap = []
        while a and a[-1][0] <= n:
            t = a.pop()
            heappush(heap, (t[1], t[0]))
        
        ans = 0
        current = n
        pos = 0
        
        for t, x in reversed(a):
            gap = t - pos
            if gap > n:
                return -1
            
            while current < gap:
                if not heap: return -1
                price, s = heap[0]
                fill = min(gap - current, n - current)
                ans += fill * price
                current += fill
                if current == n:
                    heappop(heap)
            
            current -= gap
            pos = t
            heappush(heap, (x, t))
        
        # 处理最后到达d的步骤
        gap = d - pos
        if gap > n:
            return -1
        
        while current < gap:
            if not heap: return -1
            price, s = heap[0]
            fill = min(gap - current, n - current)
            ans += fill * price
            current += fill
        
        return ans
    
    @staticmethod
    def prompt_func(question_case) -> str:
        return f"""卡车送货问题：
- 总距离：{question_case['d']}公里
- 油箱容量：{question_case['n']}升（初始满油）
- 加油站：{'； '.join(f'位置{pos}公里，价格{price}美元' for pos, price in question_case['stations'])}
要求：计算到达终点的最小油费，无法到达输出-1。答案用[answer]包裹，示例：[answer]100[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](-?\d+)\[/answer\]', output)
        return int(matches[-1]) if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected_output']
