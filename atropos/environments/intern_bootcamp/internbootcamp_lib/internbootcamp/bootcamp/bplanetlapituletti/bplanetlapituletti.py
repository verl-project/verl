"""# 

### 谜题描述
The time on the planet Lapituletti goes the same way it goes on Earth but a day lasts h hours and each hour lasts m minutes. The inhabitants of that planet use digital clocks similar to earth ones. Clocks display time in a format HH:MM (the number of hours in decimal is displayed first, then (after the colon) follows the number of minutes in decimal; the number of minutes and hours is written with leading zeros if needed to form a two-digit number). Hours are numbered from 0 to h-1 and minutes are numbered from 0 to m-1. 

<image>

That's how the digits are displayed on the clock. Please note that digit 1 is placed in the middle of its position. 

A standard mirror is in use on the planet Lapituletti. Inhabitants often look at the reflection of the digital clocks in the mirror and feel happy when what you see on the reflected clocks is a valid time (that means that you see valid digits in the reflection and this time can be seen on the normal clocks at some moment of a day).

The image of the clocks in the mirror is reflected against a vertical axis. 

<image>

The reflection is not a valid time.

<image>

The reflection is a valid time with h=24, m = 60. However, for example, if h=10, m=60, then the reflection is not a valid time. 

An inhabitant of the planet Lapituletti begins to look at a mirrored image of the clocks at some time moment s and wants to know the nearest future time moment (which can possibly happen on the next day), when the reflected clock time is valid.

It can be shown that with any h, m, s such a moment exists. If the reflected time is correct at the moment the inhabitant began to look at the clock, that moment is considered the nearest.

You are asked to solve the problem for several test cases.

Input

The first line contains a single integer T (1 ≤ T ≤ 100) — the number of test cases.

The next 2 ⋅ T lines contain the description of test cases. The description of each test case consists of two lines.

The first line of a test case contains two integers h, m (1 ≤ h, m ≤ 100).

The second line contains the start time s in the described format HH:MM.

Output

For each test case output in a separate line the nearest moment in format HH:MM when the reflected time is correct.

Example

Input


5
24 60
12:21
24 60
23:59
90 80
52:26
1 100
00:01
10 10
04:04


Output


12:21
00:00
52:28
00:00
00:00

Note

In the second test case it is not hard to show that the reflection of 23:59 is incorrect, while the reflection of the moment 00:00 on the next day is correct. 

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
nums = {'0': '0', '1': '1', '2': '5', '5': '2', '8': '8'}
def fmt_num(x):
    return str(x).rjust(2, '0')

def is_valid(hh, mm, h, m):
    hh = fmt_num(hh)
    mm = fmt_num(mm)
    if not all(c in nums for c in hh+mm): return False
    hh, mm = int(nums[mm[1]]+nums[mm[0]]), int(nums[hh[1]]+nums[hh[0]])
    return 0 <= hh < h and 0 <= mm < m
    
def solve(h, m, time):
    hh, mm = map(int, time.split(':'))
    while not is_valid(hh, mm, h, m):
        mm = (mm + 1) % m
        if mm == 0:
            hh = (hh + 1) % h
    return fmt_num(hh) + ':' + fmt_num(mm)

for _ in xrange(int(raw_input())):
    h, m = map(int, raw_input().split())
    time = raw_input().strip()
    print solve(h, m, time)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Bplanetlapitulettibootcamp(Basebootcamp):
    def __init__(self, h_min=1, h_max=100, m_min=1, m_max=100):
        self.h_min = h_min
        self.h_max = h_max
        self.m_min = m_min
        self.m_max = m_max

    def case_generator(self):
        h = random.randint(self.h_min, self.h_max)
        m = random.randint(self.m_min, self.m_max)
        
        # 生成随机合法起始时间逻辑优化
        valid_time = self._find_valid_time(h, m, "00:00")  # 保证存在解
        start_hh, start_mm = map(int, valid_time.split(':'))
        
        # 随机回退步数生成起始时间
        steps_back = random.randint(0, h*m-1)
        for _ in range(steps_back):
            start_mm -= 1
            if start_mm < 0:
                start_mm = m-1
                start_hh -= 1
                if start_hh < 0:
                    start_hh = h-1
        s_time = f"{start_hh:02d}:{start_mm:02d}"
        return {'h': h, 'm': m, 's': s_time}

    @staticmethod
    def prompt_func(question_case):
        h = question_case['h']
        m = question_case['m']
        s = question_case['s']
        prompt = f"""行星Bplanetlapituletti的镜像时间谜题
数字镜像规则：
┌───┬───┬───┬───┬───┐
│ 0 → 0 │ 1 → 1 │ 2 → 5 │
│ 5 → 2 │ 8 → 8 │ 其他 → 无效 │
└───────────────────┘

时间格式要求：
- 小时范围：00 至 {h-1:02d}
- 分钟范围：00 至 {m-1:02d}

验证规则：
1. 原时间所有数字必须可镜像
2. 镜像时间构成方式：
   - 镜像后的小时 = 原分钟数字镜像并反转顺序
   - 镜像后的分钟 = 原小时数字镜像并反转顺序
3. 镜像时间必须满足有效时间范围

当前观测时间：{s}
请计算之后最近的合法镜像时刻（包含当前时间），答案格式：[answer]HH:MM[/answer]"""
        return prompt

    @staticmethod
    def extract_output(output):
        import re
        candidates = re.findall(r'\[answer\](.*?)\[\/answer\]', output, flags=re.I|re.DOTALL)
        for candidate in reversed(candidates):
            candidate = candidate.strip()
            if re.fullmatch(r'\d{2}:\d{2}', candidate):
                return candidate
        return None

    @classmethod
    def _nums(cls):
        return {'0':'0','1':'1','2':'5','5':'2','8':'8'}

    @classmethod
    def _is_valid_time(cls, hh, mm, h, m):
        hh_str = f"{hh:02d}"
        mm_str = f"{mm:02d}"
        nums = cls._nums()
        
        # 检查原始数字有效性
        for c in hh_str + mm_str:
            if c not in nums:
                return False
        
        # 构建镜像时间
        try:
            mirrored_hh = int(nums[mm_str[1]] + nums[mm_str[0]])
            mirrored_mm = int(nums[hh_str[1]] + nums[hh_str[0]])
        except KeyError:
            return False
        
        # 验证镜像时间范围
        return 0 <= mirrored_hh < h and 0 <= mirrored_mm < m

    @classmethod
    def _find_valid_time(cls, h, m, start_time):
        current_hh, current_mm = map(int, start_time.split(':'))
        for _ in range(h * m):
            if cls._is_valid_time(current_hh, current_mm, h, m):
                return f"{current_hh:02d}:{current_mm:02d}"
            
            # 时间递增逻辑
            current_mm += 1
            if current_mm >= m:
                current_mm = 0
                current_hh += 1
                if current_hh >= h:
                    current_hh = 0
        return f"{current_hh:02d}:{current_mm:02d}"  # 理论上不会执行到这

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 格式严格验证
        if not isinstance(solution, str):
            return False
        if len(solution) != 5 or solution[2] != ':':
            return False
        hh_part, mm_part = solution[:2], solution[3:]
        if not (hh_part.isdigit() and mm_part.isdigit()):
            return False
        
        hh, mm = int(hh_part), int(mm_part)
        if not (0 <= hh < identity['h'] and 0 <= mm < identity['m']):
            return False
        
        # 逻辑一致性验证
        return solution == cls._find_valid_time(identity['h'], identity['m'], identity['s'])

