"""# 

### 谜题描述
Arkady bought an air ticket from a city A to a city C. Unfortunately, there are no direct flights, but there are a lot of flights from A to a city B, and from B to C.

There are n flights from A to B, they depart at time moments a_1, a_2, a_3, ..., a_n and arrive at B t_a moments later.

There are m flights from B to C, they depart at time moments b_1, b_2, b_3, ..., b_m and arrive at C t_b moments later.

The connection time is negligible, so one can use the i-th flight from A to B and the j-th flight from B to C if and only if b_j ≥ a_i + t_a.

You can cancel at most k flights. If you cancel a flight, Arkady can not use it.

Arkady wants to be in C as early as possible, while you want him to be in C as late as possible. Find the earliest time Arkady can arrive at C, if you optimally cancel k flights. If you can cancel k or less flights in such a way that it is not possible to reach C at all, print -1.

Input

The first line contains five integers n, m, t_a, t_b and k (1 ≤ n, m ≤ 2 ⋅ 10^5, 1 ≤ k ≤ n + m, 1 ≤ t_a, t_b ≤ 10^9) — the number of flights from A to B, the number of flights from B to C, the flight time from A to B, the flight time from B to C and the number of flights you can cancel, respectively.

The second line contains n distinct integers in increasing order a_1, a_2, a_3, ..., a_n (1 ≤ a_1 < a_2 < … < a_n ≤ 10^9) — the times the flights from A to B depart.

The third line contains m distinct integers in increasing order b_1, b_2, b_3, ..., b_m (1 ≤ b_1 < b_2 < … < b_m ≤ 10^9) — the times the flights from B to C depart.

Output

If you can cancel k or less flights in such a way that it is not possible to reach C at all, print -1.

Otherwise print the earliest time Arkady can arrive at C if you cancel k flights in such a way that maximizes this time.

Examples

Input


4 5 1 1 2
1 3 5 7
1 2 3 9 10


Output


11


Input


2 2 4 4 2
1 10
10 20


Output


-1


Input


4 3 2 3 1
1 999999998 999999999 1000000000
3 4 1000000000


Output


1000000003

Note

Consider the first example. The flights from A to B depart at time moments 1, 3, 5, and 7 and arrive at B at time moments 2, 4, 6, 8, respectively. The flights from B to C depart at time moments 1, 2, 3, 9, and 10 and arrive at C at time moments 2, 3, 4, 10, 11, respectively. You can cancel at most two flights. The optimal solution is to cancel the first flight from A to B and the fourth flight from B to C. This way Arkady has to take the second flight from A to B, arrive at B at time moment 4, and take the last flight from B to C arriving at C at time moment 11.

In the second example you can simply cancel all flights from A to B and you're done.

In the third example you can cancel only one flight, and the optimal solution is to cancel the first flight from A to B. Note that there is still just enough time to catch the last flight from B to C.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import bisect, collections, heapq, sys

n, m, ta, tb, k = map(int, sys.stdin.readline().rstrip().split(' '))
a = map(int, sys.stdin.readline().rstrip().split(' '))
b = map(int, sys.stdin.readline().rstrip().split(' '))

if k >= n or k >= m:
    print(-1)
    sys.exit()

ret = -1
for i in range(k+1):
    val = a[i] + ta
    numDeleteB = k - i
    idx = bisect.bisect_left(b, val)
    if idx + numDeleteB >= m:
        print(-1)
        sys.exit()
    j = idx + numDeleteB
    ret = max(ret, b[j] + tb)

print(ret)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import bisect
import random
import re
from bootcamp import Basebootcamp


class Bbornthiswaybootcamp(Basebootcamp):
    def __init__(self, **params):
        default_params = {
            'n_min': 1,
            'n_max': 5,
            'm_min': 1,
            'm_max': 5,
            'ta_min': 1,
            'ta_max': 100,
            'tb_min': 1,
            'tb_max': 100,
            'k_min': 1,
            'k_max': None,
        }
        self.params = default_params.copy()
        self.params.update(params)
    
    def case_generator(self):
        n = random.randint(self.params['n_min'], self.params['n_max'])
        m = random.randint(self.params['m_min'], self.params['m_max'])
        ta = random.randint(self.params['ta_min'], self.params['ta_max'])
        tb = random.randint(self.params['tb_min'], self.params['tb_max'])
        max_k = n + m

        # Handle k constraints
        k_min = max(1, self.params['k_min'])
        k_min = min(k_min, max_k)  # Ensure within valid range
        
        k_max_param = self.params['k_max']
        if k_max_param is None:
            effective_k_max = max_k
        else:
            effective_k_max = min(k_max_param, max_k)
        
        effective_k_max = max(k_min, effective_k_max)
        effective_k_max = min(effective_k_max, max_k)  # Final validation
        
        k = random.randint(k_min, effective_k_max) if effective_k_max >= k_min else k_min

        # Generate strictly increasing flight times
        a = sorted(random.sample(range(1, 1000), n))
        b = sorted(random.sample(range(1, 1000), m))

        return {
            'n': n, 'm': m, 'ta': ta, 'tb': tb, 'k': k,
            'a': a, 'b': b
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        a_str = ', '.join(map(str, question_case['a']))
        b_str = ', '.join(map(str, question_case['b']))
        prompt = f"""航空调度谜题：
- 从A到B有{question_case['n']}个航班（出发时间：{a_str}，飞行耗时{question_case['ta']}）
- 从B到C有{question_case['m']}个航班（出发时间：{b_str}，飞行耗时{question_case['tb']}）
- 你可取消最多{question_case['k']}个航班

规则：
1. 能转机的条件：B出发时间 ≥ A航班到达时间（A出发+{question_case['ta']}）
2. 你的目标是让Arkady最晚到达C
3. 若无法到达输出-1

请计算最优解，并将最终答案用[answer]包裹。例如：[answer]11[/answer]"""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        n, m = identity['n'], identity['m']
        ta, tb, k = identity['ta'], identity['tb'], identity['k']
        a, b = identity['a'], identity['b']
        
        # Edge case: impossible to connect
        if k >= n or k >= m:
            return solution == -1
        
        max_time = -1
        valid = False
        for x in range(k+1):
            if x >= n: continue
            
            # Find earliest possible B flight
            arrive_b = a[x] + ta
            idx = bisect.bisect_left(b, arrive_b)
            
            # Check remaining deletions in B flights
            remaining_deletions = k - x
            if idx + remaining_deletions >= m:
                continue
            
            # Calculate arrival time
            candidate = b[idx + remaining_deletions] + tb
            max_time = max(max_time, candidate)
            valid = True
        
        expected = max_time if valid else -1
        return solution == expected
