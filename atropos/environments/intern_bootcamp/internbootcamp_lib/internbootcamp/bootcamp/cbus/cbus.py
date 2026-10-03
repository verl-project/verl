"""# 

### 谜题描述
A bus moves along the coordinate line Ox from the point x = 0 to the point x = a. After starting from the point x = 0, it reaches the point x = a, immediately turns back and then moves to the point x = 0. After returning to the point x = 0 it immediately goes back to the point x = a and so on. Thus, the bus moves from x = 0 to x = a and back. Moving from the point x = 0 to x = a or from the point x = a to x = 0 is called a bus journey. In total, the bus must make k journeys.

The petrol tank of the bus can hold b liters of gasoline. To pass a single unit of distance the bus needs to spend exactly one liter of gasoline. The bus starts its first journey with a full petrol tank.

There is a gas station in point x = f. This point is between points x = 0 and x = a. There are no other gas stations on the bus route. While passing by a gas station in either direction the bus can stop and completely refuel its tank. Thus, after stopping to refuel the tank will contain b liters of gasoline.

What is the minimum number of times the bus needs to refuel at the point x = f to make k journeys? The first journey starts in the point x = 0.

Input

The first line contains four integers a, b, f, k (0 < f < a ≤ 106, 1 ≤ b ≤ 109, 1 ≤ k ≤ 104) — the endpoint of the first bus journey, the capacity of the fuel tank of the bus, the point where the gas station is located, and the required number of journeys.

Output

Print the minimum number of times the bus needs to refuel to make k journeys. If it is impossible for the bus to make k journeys, print -1.

Examples

Input

6 9 2 4


Output

4


Input

6 10 2 4


Output

2


Input

6 5 4 3


Output

-1

Note

In the first example the bus needs to refuel during each journey.

In the second example the bus can pass 10 units of distance without refueling. So the bus makes the whole first journey, passes 4 units of the distance of the second journey and arrives at the point with the gas station. Then it can refuel its tank, finish the second journey and pass 2 units of distance from the third journey. In this case, it will again arrive at the point with the gas station. Further, he can refill the tank up to 10 liters to finish the third journey and ride all the way of the fourth journey. At the end of the journey the tank will be empty. 

In the third example the bus can not make all 3 journeys because if it refuels during the second journey, the tanks will contain only 5 liters of gasoline, but the bus needs to pass 8 units of distance until next refueling.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys

a, b, f, k = map(int, raw_input().split())

direction = 0
tank = b
ans = 0

for i in range(k):
    T = 1 if i == k - 1 else 2
    if direction == 0:
        if tank < f:
            print('-1')
            sys.exit(0)
        tank -= f
        if b < T * (a - f):
            print('-1')
            sys.exit(0)
        if tank < T * (a - f):
            tank = b
            ans += 1
        tank -= (a - f)
    else:
        if tank < a - f:
            print('-1')
            sys.exit(0)
        tank -= a - f
        if b < T * f:
            print('-1')
            sys.exit(0)
        if tank < T * f:
            tank = b
            ans += 1
        tank -= f
    direction ^= 1
            
print(ans)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cbusbootcamp(Basebootcamp):
    def __init__(self, a_min=2, a_max=10**6, b_min=1, b_max=10**9, k_min=1, k_max=10**4, f_min=1):
        self.a_min = max(a_min, f_min + 1)  # 确保最小a满足约束条件
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.k_min = max(k_min, 1)  # 确保k≥1
        self.k_max = k_max
        self.f_min = f_min

    def case_generator(self):
        # 动态计算有效a范围
        a_lower = max(self.a_min, self.f_min + 1)
        if a_lower > self.a_max:
            raise ValueError("Invalid parameter range: a_min must be ≥ f_min + 1 and ≤ a_max")
        
        a = random.randint(a_lower, self.a_max)
        f = random.randint(self.f_min, a-1)
        b = random.randint(self.b_min, self.b_max)
        k = random.randint(self.k_min, self.k_max)
        
        return {'a': a, 'b': b, 'f': f, 'k': k}

    @staticmethod
    def prompt_func(question_case) -> str:
        params = question_case
        return f"""一辆公交车在坐标轴0到{params['a']}之间往返行驶，必须完成{params['k']}次完整行程。油箱容量为{params['b']}升，每单位距离消耗1升油。起点有加油站（x=0），途中在x={params['f']}处有唯一加油站。公交车需要多少次加油才能完成任务？如果不可能完成，输出-1。答案请用[answer][/answer]标签包裹。"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](-?\d+)\[\/answer\]', output)
        return int(matches[-1]) if matches else None

    @staticmethod
    def _validate_case(a, b, f, k):
        """预验证参数有效性"""
        if not (0 < f < a):
            raise ValueError(f"Invalid parameters: 0 < f < a not satisfied (f={f}, a={a})")
        if b <= 0 or k <= 0:
            raise ValueError("b and k must be positive")

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            a = identity['a']
            b = identity['b']
            f_val = identity['f']
            k_val = identity['k']
            cls._validate_case(a, b, f_val, k_val)
            
            # 重新实现核心验证逻辑
            fuel = b
            refuels = 0
            direction = 0  # 0: 0→a, 1: a→0
            
            for journey in range(k_val):
                required = f_val if direction == 0 else (a - f_val)
                if fuel < required:
                    return solution == -1
                fuel -= required
                
                next_trip = (a - f_val) if direction == 0 else f_val
                trips_needed = 1 if journey == k_val - 1 else 2
                
                if fuel < trips_needed * next_trip:
                    if b < trips_needed * next_trip:
                        return solution == -1
                    fuel = b
                    refuels += 1
                
                fuel -= next_trip
                direction ^= 1
            
            return solution == refuels
        except ValueError:
            return solution == -1
