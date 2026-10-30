"""# 

### 谜题描述
Vasya is currently at a car rental service, and he wants to reach cinema. The film he has bought a ticket for starts in t minutes. There is a straight road of length s from the service to the cinema. Let's introduce a coordinate system so that the car rental service is at the point 0, and the cinema is at the point s.

There are k gas stations along the road, and at each of them you can fill a car with any amount of fuel for free! Consider that this operation doesn't take any time, i.e. is carried out instantly.

There are n cars in the rental service, i-th of them is characterized with two integers ci and vi — the price of this car rent and the capacity of its fuel tank in liters. It's not allowed to fuel a car with more fuel than its tank capacity vi. All cars are completely fueled at the car rental service.

Each of the cars can be driven in one of two speed modes: normal or accelerated. In the normal mode a car covers 1 kilometer in 2 minutes, and consumes 1 liter of fuel. In the accelerated mode a car covers 1 kilometer in 1 minutes, but consumes 2 liters of fuel. The driving mode can be changed at any moment and any number of times.

Your task is to choose a car with minimum price such that Vasya can reach the cinema before the show starts, i.e. not later than in t minutes. Assume that all cars are completely fueled initially.

Input

The first line contains four positive integers n, k, s and t (1 ≤ n ≤ 2·105, 1 ≤ k ≤ 2·105, 2 ≤ s ≤ 109, 1 ≤ t ≤ 2·109) — the number of cars at the car rental service, the number of gas stations along the road, the length of the road and the time in which the film starts. 

Each of the next n lines contains two positive integers ci and vi (1 ≤ ci, vi ≤ 109) — the price of the i-th car and its fuel tank capacity.

The next line contains k distinct integers g1, g2, ..., gk (1 ≤ gi ≤ s - 1) — the positions of the gas stations on the road in arbitrary order.

Output

Print the minimum rent price of an appropriate car, i.e. such car that Vasya will be able to reach the cinema before the film starts (not later than in t minutes). If there is no appropriate car, print -1.

Examples

Input

3 1 8 10
10 8
5 7
11 9
3


Output

10


Input

2 2 10 18
10 4
20 6
5 3


Output

20

Note

In the first sample, Vasya can reach the cinema in time using the first or the third cars, but it would be cheaper to choose the first one. Its price is equal to 10, and the capacity of its fuel tank is 8. Then Vasya can drive to the first gas station in the accelerated mode in 3 minutes, spending 6 liters of fuel. After that he can full the tank and cover 2 kilometers in the normal mode in 4 minutes, spending 2 liters of fuel. Finally, he drives in the accelerated mode covering the remaining 3 kilometers in 3 minutes and spending 6 liters of fuel. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin


def fast2():
    import os, sys, atexit
    from cStringIO import StringIO as BytesIO
    # range = xrange
    sys.stdout = BytesIO()
    atexit.register(lambda: os.write(1, sys.stdout.getvalue()))
    return BytesIO(os.read(0, os.fstat(0).st_size)).readline


input = fast2()
rints = lambda: [int(x) for x in input().split()]
rints_2d = lambda n: [tuple(rints()) for _ in range(n)]
n, k, s, t = rints()
a, g = sorted(rints_2d(n), key=lambda x: x[1]), sorted(rints()) + [s]

be, en, ans = 0, n - 1, float('inf')
while be <= en:
    md = (be + en) >> 1
    fuel, point, time = a[md][1], 0, 0

    for i in range(k + 1):
        dist = g[i] - point
        rem = fuel - dist
        if rem < 0:
            time = float('inf')
            break
        else:
            x = min(dist, rem)
            time += x + (dist - x) * 2
        point = g[i]

    if time > t:
        be = md + 1
    else:
        en = md - 1
        ans = min(ans, a[md][1])

if ans == float('inf'):
    print(-1)
else:
    out = float('inf')
    for i, j in a:
        if j >= ans:
            out = min(out, i)
    print(out)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Croadtocinemabootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = params
        
    def case_generator(self):
        # 参数初始化
        s = self.params.get('s', random.randint(10, 1000))
        k = self.params.get('k', random.randint(1, min(10, s-2)))
        n = self.params.get('n', random.randint(3, 10))
        
        # 生成未排序的加油站位置
        gas_stations = []
        while len(gas_stations) < k:
            candidate = random.randint(1, s-1)
            if candidate not in gas_stations:
                gas_stations.append(candidate)
        random.shuffle(gas_stations)  # 模拟输入时的任意顺序

        # 计算最大间距（需排序处理）
        sorted_gas = sorted(gas_stations + [s])
        prev, max_gap = 0, 0
        segments = []
        for pos in sorted_gas:
            seg = pos - prev
            segments.append(seg)
            max_gap = max(max_gap, seg)
            prev = pos

        # 20%概率生成无解案例
        if random.random() < 0.2:
            case_type = random.choice(['capacity', 'time'])
            if case_type == 'capacity':
                # 所有车辆油量不足
                cars = [(random.randint(1, 100), random.randint(1, max_gap-1)) 
                        for _ in range(n)]
                t = random.randint(1, 2*10**9)
            else:
                # 存在油量足够车辆但时间不足
                vi = max_gap + random.randint(0, 100)
                time_needed = sum(seg*2 - min(seg, vi - seg) for seg in segments)
                cars = [(random.randint(1, 100), vi)] + \
                       [(random.randint(101, 200), random.randint(max_gap, max_gap+100)) 
                        for _ in range(n-1)]
                t = random.randint(1, time_needed-1)  # 确保时间不足
            return {
                'n': n, 'k': k, 's': s, 't': t,
                'cars': cars, 'gas_stations': gas_stations.copy(),
                '_sorted_gas': sorted_gas  # 内部记录排序后位置用于验证
            }

        # 生成有解案例
        min_price = random.randint(50, 200)
        valid_vi = max_gap + random.randint(0, 100)
        cars = [(min_price, valid_vi)]
        
        # 生成干扰车辆：价格更低（但油量不足）或价格更高（油量足够）
        for _ in range(n-1):
            if random.random() < 0.5:
                # 油量足够的高价车
                cars.append((min_price + random.randint(10, 100), valid_vi + random.randint(0, 50)))
            else:
                # 油量不足的低价车
                cars.append((random.randint(1, min_price-1), random.randint(1, max_gap-1)))
        random.shuffle(cars)
        
        # 计算合法最小时间
        t = sum(seg * 2 - min(seg, valid_vi - seg) for seg in segments)
        return {
            'n': n, 'k': k, 's': s, 't': t,
            'cars': cars, 'gas_stations': gas_stations.copy(),
            '_sorted_gas': sorted_gas  # 内部记录排序后位置用于验证
        }

    @staticmethod
    def prompt_func(case):
        input_data = '\n'.join([
            f"{case['n']} {case['k']} {case['s']} {case['t']}",
            *[f"{c} {v}" for c, v in case['cars']],
            ' '.join(map(str, case['gas_stations']))
        ])
        return f"""你需要帮助Vasya选择最便宜的汽车按时到达电影院。规则：
1. 油箱容量必须≥最大相邻加油站间距（含起点终点）
2. 两种驾驶模式：
   - 正常模式：1km/2分钟，耗油1L
   - 加速模式：1km/1分钟，耗油2L
3. 输出最低租金，无解输出-1

输入数据：
{input_data}

请将最终答案置于[answer]标签内，如：[answer]50[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        try:
            return int(matches[-1].strip()) if matches else None
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, case):
        try:
            user_ans = int(solution)
        except:
            return False

        # 解析关键参数
        sorted_gas = case['_sorted_gas']
        segments = [sorted_gas[0]] + [sorted_gas[i]-sorted_gas[i-1] for i in range(1, len(sorted_gas))]
        max_gap = max(segments)
        valid_cars = []

        for price, capacity in case['cars']:
            if capacity < max_gap:
                continue
            # 计算最短时间
            total_time = 0
            for seg in segments:
                rem = capacity - seg
                if rem < 0:
                    total_time = float('inf')
                    break
                x = min(seg, rem)
                total_time += x + (seg - x)*2
            if total_time <= case['t']:
                valid_cars.append(price)
        
        correct_ans = min(valid_cars) if valid_cars else -1
        return user_ans == correct_ans
