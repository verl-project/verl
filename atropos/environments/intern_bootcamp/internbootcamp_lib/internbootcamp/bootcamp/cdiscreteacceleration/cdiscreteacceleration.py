"""# 

### 谜题描述
There is a road with length l meters. The start of the road has coordinate 0, the end of the road has coordinate l.

There are two cars, the first standing at the start of the road and the second standing at the end of the road. They will start driving simultaneously. The first car will drive from the start to the end and the second car will drive from the end to the start.

Initially, they will drive with a speed of 1 meter per second. There are n flags at different coordinates a_1, a_2, …, a_n. Each time when any of two cars drives through a flag, the speed of that car increases by 1 meter per second.

Find how long will it take for cars to meet (to reach the same coordinate). 

Input

The first line contains one integer t (1 ≤ t ≤ 10^4): the number of test cases.

The first line of each test case contains two integers n, l (1 ≤ n ≤ 10^5, 1 ≤ l ≤ 10^9): the number of flags and the length of the road.

The second line contains n integers a_1, a_2, …, a_n in the increasing order (1 ≤ a_1 < a_2 < … < a_n < l).

It is guaranteed that the sum of n among all test cases does not exceed 10^5.

Output

For each test case print a single real number: the time required for cars to meet.

Your answer will be considered correct, if its absolute or relative error does not exceed 10^{-6}. More formally, if your answer is a and jury's answer is b, your answer will be considered correct if \frac{|a-b|}{max{(1, b)}} ≤ 10^{-6}.

Example

Input


5
2 10
1 9
1 10
1
5 7
1 2 3 4 6
2 1000000000
413470354 982876160
9 478
1 10 25 33 239 445 453 468 477


Output


3.000000000000000
3.666666666666667
2.047619047619048
329737645.750000000000000
53.700000000000000

Note

In the first test case cars will meet in the coordinate 5.

The first car will be in the coordinate 1 in 1 second and after that its speed will increase by 1 and will be equal to 2 meters per second. After 2 more seconds it will be in the coordinate 5. So, it will be in the coordinate 5 in 3 seconds.

The second car will be in the coordinate 9 in 1 second and after that its speed will increase by 1 and will be equal to 2 meters per second. After 2 more seconds it will be in the coordinate 5. So, it will be in the coordinate 5 in 3 seconds.

In the second test case after 1 second the first car will be in the coordinate 1 and will have the speed equal to 2 meters per second, the second car will be in the coordinate 9 and will have the speed equal to 1 meter per second. So, they will meet after (9-1)/(2+1) = 8/3 seconds. So, the answer is equal to 1 + 8/3 = 11/3.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdout

eps = 0.0000001

out = []
for _ in xrange(input()):
    n, l = map(int, raw_input().strip().split())
    arr = map(int, raw_input().strip().split())

    a = [0 for i in xrange(n + 2)]
    a[n + 1] = l

    for i in xrange(1, n + 1): a[i] = arr[i - 1]

    # times from left
    ltimes = [a[0] for i in xrange(n + 2)]
    for i in xrange(1, n + 2):
        ltimes[i] = ltimes[i - 1] + ((a[i] - a[i - 1]) * 1.0 / i)

    # times from right
    rtimes = [l - a[n + 1] for i in xrange(n + 2)]
    for i in xrange(n, -1, -1):
        rtimes[i] = rtimes[i + 1] + ((a[i + 1] - a[i]) * 1.0 / (n + 1 - i))

    ans = -1
    # will they meet between flags i and i + 1?
    for i in xrange(n + 1):
        # meet between i and i + 1
        lt = ltimes[i]
        rt = rtimes[i + 1]

        # distance to be traversed
        d = a[i + 1] - a[i]

        if (abs(lt - rt) <= eps):
            tot = lt + (d * 1.0 / (n + 2))
            ans = \"{0:.9f}\".format(tot)
            break
        
        elif (lt < rt):
            if abs(lt + (d * 1.0 / (i + 1)) - rt) <= eps:
                ans = \"{0:.9f}\".format(rt)
                break
            
            if lt + (d * 1.0 / (i + 1)) > rt:
                dt = rt - lt

                X = a[i] + dt * (i + 1)
                Y = a[i + 1]

                tot = rt + ((Y - X) * 1.0 / (n + 2))
                ans = \"{0:.9f}\".format(tot)
                break
        else:
            if abs(rt + (d * 1.0 / (n + 1 - i)) - lt) <= eps:
                ans = \"{0:.9f}\".format(lt)
                break
            
            if rt + (d * 1.0 / (n + 1 - i)) > lt:
                dt = lt - rt

                X = a[i]
                Y = a[i + 1] - dt * (n + 1 - i)

                tot = lt + ((Y - X) * 1.0 / (n + 2))
                ans = \"{0:.9f}\".format(tot)
                break            

    out.append(ans)

stdout.write(\"\n\".join(out))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Cdiscreteaccelerationbootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=5, l_min=10, l_max=1000):
        super().__init__()
        # 参数校验和自适应调整
        valid_l_min = max(l_min, 2)
        possible_max_n = valid_l_min - 1
        
        # 自动调整n范围
        effective_n_min = max(1, min(n_min, possible_max_n))
        effective_n_max = max(effective_n_min, min(n_max, possible_max_n))
        
        self.params = {
            'n_min': effective_n_min,
            'n_max': effective_n_max,
            'l_min': valid_l_min,
            'l_max': max(l_max, valid_l_min)
        }

    def case_generator(self):
        # 生成合法道路长度
        l_val = random.randint(self.params['l_min'], self.params['l_max'])
        
        # 计算n的合法范围
        max_n = min(l_val - 1, self.params['n_max'])
        min_n = max(1, self.params['n_min'])
        if min_n > max_n:
            min_n, max_n = 1, max(1, min(l_val-1, self.params['n_max']))
        
        n = random.randint(min_n, max_n)
        
        # 生成严格递增的标志物位置
        candidates = list(range(1, l_val))
        selected = sorted(random.sample(candidates, n))
        
        return {
            'n': n,
            'l': l_val,
            'a': selected
        }

    @staticmethod
    def prompt_func(question_case):
        case = question_case
        prompt = f"""两辆汽车在长{case['l']}米的道路两端同时出发，初始速度1米/秒。当经过标志物时，车辆速度立即+1米/秒。标志物位置为：{', '.join(map(str, case['a']))}米。
请精确计算相遇时间（保留9位小数），答案放入[answer][/answer]。例如：[answer]3.000000000[/answer]"""
        return prompt

    @staticmethod
    def extract_output(output):
        # 增强模式匹配
        pattern = r'\[answer\][\s\n]*(\d+\.?\d*|\.\d+|\d+\.?\d*[eE][+-]?\d+)[\s\n]*\[/answer\]'
        matches = re.findall(pattern, output)
        if not matches:
            return None
        try:
            return float(matches[-1].strip())
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            correct = cls._compute_meeting_time(identity['n'], identity['l'], identity['a'])
            user_ans = float(solution)
        except:
            return False
        
        # 双精度验证机制
        if correct < 1e-10:
            return abs(user_ans) < 1e-6
        return abs(user_ans - correct) <= 1e-6 or abs((user_ans - correct)/correct) <= 1e-6

    @staticmethod
    def _compute_meeting_time(n, l_val, a_list):
        """优化后的精确计算算法"""
        a = [0] + sorted(a_list) + [l_val]
        n_segments = len(a) - 1
        
        # 左车时间计算
        left_time = [0.0]*(n_segments+1)
        speed = 1.0
        for i in range(1, n_segments+1):
            delta = a[i] - a[i-1]
            left_time[i] = left_time[i-1] + delta / speed
            speed += 1
        
        # 右车时间计算
        right_time = [0.0]*(n_segments+1)
        speed = 1.0
        for i in range(n_segments-1, -1, -1):
            delta = a[i+1] - a[i]
            right_time[i] = right_time[i+1] + delta / speed
            speed += 1
        
        # 寻找相遇区间
        eps = 1e-12
        for i in range(n_segments):
            t_left = left_time[i]
            t_right = right_time[i+1]
            seg_length = a[i+1] - a[i]
            v_left = i + 1
            v_right = n_segments - i
            
            if abs(t_left - t_right) < eps:
                return t_left + seg_length / (v_left + v_right)
            
            if t_left < t_right:
                if t_left + seg_length/v_left >= t_right - eps:
                    dt = t_right - t_left
                    meet_pos = a[i] + dt * v_left
                    remain = a[i+1] - meet_pos
                    return t_right + remain / (v_left + v_right)
            else:
                if t_right + seg_length/v_right >= t_left - eps:
                    dt = t_left - t_right
                    meet_pos = a[i+1] - dt * v_right
                    remain = meet_pos - a[i]
                    return t_left + remain / (v_left + v_right)
        
        raise ValueError("No solution found")
