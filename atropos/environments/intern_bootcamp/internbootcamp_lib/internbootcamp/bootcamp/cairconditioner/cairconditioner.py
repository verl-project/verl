"""# 

### 谜题描述
Gildong owns a bulgogi restaurant. The restaurant has a lot of customers, so many of them like to make a reservation before visiting it.

Gildong tries so hard to satisfy the customers that he even memorized all customers' preferred temperature ranges! Looking through the reservation list, he wants to satisfy all customers by controlling the temperature of the restaurant.

The restaurant has an air conditioner that has 3 states: off, heating, and cooling. When it's off, the restaurant's temperature remains the same. When it's heating, the temperature increases by 1 in one minute. Lastly, when it's cooling, the temperature decreases by 1 in one minute. Gildong can change the state as many times as he wants, at any integer minutes. The air conditioner is off initially.

Each customer is characterized by three values: t_i — the time (in minutes) when the i-th customer visits the restaurant, l_i — the lower bound of their preferred temperature range, and h_i — the upper bound of their preferred temperature range.

A customer is satisfied if the temperature is within the preferred range at the instant they visit the restaurant. Formally, the i-th customer is satisfied if and only if the temperature is between l_i and h_i (inclusive) in the t_i-th minute.

Given the initial temperature, the list of reserved customers' visit times and their preferred temperature ranges, you're going to help him find if it's possible to satisfy all customers.

Input

Each test contains one or more test cases. The first line contains the number of test cases q (1 ≤ q ≤ 500). Description of the test cases follows.

The first line of each test case contains two integers n and m (1 ≤ n ≤ 100, -10^9 ≤ m ≤ 10^9), where n is the number of reserved customers and m is the initial temperature of the restaurant.

Next, n lines follow. The i-th line of them contains three integers t_i, l_i, and h_i (1 ≤ t_i ≤ 10^9, -10^9 ≤ l_i ≤ h_i ≤ 10^9), where t_i is the time when the i-th customer visits, l_i is the lower bound of their preferred temperature range, and h_i is the upper bound of their preferred temperature range. The preferred temperature ranges are inclusive.

The customers are given in non-decreasing order of their visit time, and the current time is 0.

Output

For each test case, print \"YES\" if it is possible to satisfy all customers. Otherwise, print \"NO\".

You can print each letter in any case (upper or lower).

Example

Input


4
3 0
5 1 2
7 3 5
10 -1 0
2 12
5 7 10
10 16 20
3 -100
100 0 0
100 -50 50
200 100 100
1 100
99 -100 0


Output


YES
NO
YES
NO

Note

In the first case, Gildong can control the air conditioner to satisfy all customers in the following way:

  * At 0-th minute, change the state to heating (the temperature is 0). 
  * At 2-nd minute, change the state to off (the temperature is 2). 
  * At 5-th minute, change the state to heating (the temperature is 2, the 1-st customer is satisfied). 
  * At 6-th minute, change the state to off (the temperature is 3). 
  * At 7-th minute, change the state to cooling (the temperature is 3, the 2-nd customer is satisfied). 
  * At 10-th minute, the temperature will be 0, which satisfies the last customer. 



In the third case, Gildong can change the state to heating at 0-th minute and leave it be. Then all customers will be satisfied. Note that the 1-st customer's visit time equals the 2-nd customer's visit time.

In the second and the fourth case, Gildong has to make at least one customer unsatisfied.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from collections import Counter, defaultdict, deque
import bisect
from sys import stdin, stdout
from itertools import repeat
import math

# sys.stdin = open('input')

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

mod = 1000000007

def my_main():
    T = inp()
    for _ in range(T):
        n, m = inp()
        da = []
        for i in range(n):
            da.append(inp())
        da.sort()
        l, r = m, m
        lt = 0
        for t, il, ir in da:
            dt = t-lt
            l, r = l-dt, r+dt
            if r<il or l>ir:
                break
            l = max(il, l)
            r = min(r, ir)
            lt = t
        else:
            print \"YES\"
            continue
        print \"NO\"








my_main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import defaultdict
from bootcamp import Basebootcamp

class Cairconditionerbootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=100, m_min=-1000, m_max=1000, time_delta_min=0, time_delta_max=100):
        self.n_min = n_min
        self.n_max = n_max
        self.m_min = m_min
        self.m_max = m_max
        self.time_delta_min = time_delta_min  # 允许0间隔
        self.time_delta_max = time_delta_max
    
    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        m = random.randint(self.m_min, self.m_max)
        prev_t = 0
        t_list = []
        
        # 生成允许重复的时间序列（非递减）
        for _ in range(n):
            delta = random.randint(self.time_delta_min, self.time_delta_max)
            prev_t += delta
            t_list.append(prev_t)
        
        current_l = current_r = m
        previous_time = 0
        customers = []
        
        # 生成基础可行案例
        for t in t_list:
            dt = t - previous_time
            new_l = current_l - dt
            new_r = current_r + dt
            
            # 确保生成有效温度区间
            a = random.randint(new_l, new_r)
            b = random.randint(a, new_r)
            customers.append({'t': t, 'l': a, 'h': b})
            
            current_l = max(new_l, a)
            current_r = min(new_r, b)
            previous_time = t

        # 50%概率转为不可解案例
        if random.choice([True, False]):
            # 计算每个时间点的允许温度范围
            allowed_ranges = []
            sim_l = sim_r = m
            sim_prev = 0
            for c in customers:
                dt = c['t'] - sim_prev
                allowed_l = sim_l - dt
                allowed_r = sim_r + dt
                allowed_ranges.append((allowed_l, allowed_r))
                sim_l = max(allowed_l, c['l'])
                sim_r = min(allowed_r, c['h'])
                sim_prev = c['t']
            
            # 查找可破坏的客户
            candidates = []
            for i, (al, ar) in enumerate(allowed_ranges):
                current_l, current_h = customers[i]['l'], customers[i]['h']
                if current_l > ar or current_h < al:
                    continue  # 已经无法满足的客户不处理
                candidates.append(i)
            
            # 找到可破坏的客户后进行调整
            if candidates:
                idx = random.choice(candidates)
                al, ar = allowed_ranges[idx]
                
                # 确保新区间与允许范围无交集
                if random.random() < 0.5:
                    new_l = ar + 1
                    new_h = new_l + 10  # 确保区间有效性
                else:
                    new_h = al - 1
                    new_l = new_h - 10  # 确保区间有效性
                new_l, new_h = sorted([new_l, new_h])
                
                # 应用破坏
                customers[idx]['l'] = new_l
                customers[idx]['h'] = new_h

        return {
            'initial_temp': m,
            'customers': sorted(customers, key=lambda x: x['t'])  # 确保时间有序
        }
    
    @staticmethod
    def prompt_func(question_case):
        m = question_case['initial_temp']
        customers = question_case['customers']
        
        problem = f"Gildong's restaurant has an initial temperature of {m}°C. Customers will arrive at specific times with preferred temperature ranges:\n"
        problem += "\nThe air conditioner can be in three states: off (maintains temperature), heating (+1°C/min), or cooling (-1°C/min). State changes can occur at any integer minute.\n\n"
        problem += "Customers (sorted by visit time):\n"
        for idx, c in enumerate(customers, 1):
            problem += f"{idx}. Time: {c['t']} min, Range: [{c['l']}, {c['h']}]°C\n"
        problem += "\nDetermine if all customers can be satisfied. Reply with [answer]YES[/answer] or [answer]NO[/answer]."
        return problem
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.IGNORECASE)
        if not matches:
            return None
        answer = matches[-1].strip().upper()
        return answer if answer in {'YES', 'NO'} else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = cls._solve_identity(identity)
        return solution.upper() == expected
    
    @classmethod
    def _solve_identity(cls, identity):
        m = identity['initial_temp']
        customers = identity['customers']
        time_dict = defaultdict(list)
        
        # 合并同时到达的客户
        for c in customers:
            time_dict[c['t']].append((c['l'], c['h']))
        
        merged = []
        for t in sorted(time_dict):
            ls, hs = zip(*time_dict[t]) if time_dict[t] else ([], [])
            merged_l = max(ls) if ls else 0
            merged_h = min(hs) if hs else 0
            if merged_l > merged_h:
                return 'NO'
            merged.append((t, merged_l, merged_h))
        
        current_l = current_r = m
        prev_t = 0
        
        for t, l, h in merged:
            dt = t - prev_t
            new_l = current_l - dt
            new_r = current_r + dt
            
            # 检查是否存在可行区间
            if new_r < l or new_l > h:
                return 'NO'
            
            # 收紧温度范围
            current_l = max(new_l, l)
            current_r = min(new_r, h)
            prev_t = t
        
        return 'YES'
