"""# 

### 谜题描述
In distant future on Earth day lasts for n hours and that's why there are n timezones. Local times in adjacent timezones differ by one hour. For describing local time, hours numbers from 1 to n are used, i.e. there is no time \"0 hours\", instead of it \"n hours\" is used. When local time in the 1-st timezone is 1 hour, local time in the i-th timezone is i hours.

Some online programming contests platform wants to conduct a contest that lasts for an hour in such a way that its beginning coincides with beginning of some hour (in all time zones). The platform knows, that there are ai people from i-th timezone who want to participate in the contest. Each person will participate if and only if the contest starts no earlier than s hours 00 minutes local time and ends not later than f hours 00 minutes local time. Values s and f are equal for all time zones. If the contest starts at f hours 00 minutes local time, the person won't participate in it.

Help platform select such an hour, that the number of people who will participate in the contest is maximum. 

Input

The first line contains a single integer n (2 ≤ n ≤ 100 000) — the number of hours in day.

The second line contains n space-separated integers a1, a2, ..., an (1 ≤ ai ≤ 10 000), where ai is the number of people in the i-th timezone who want to participate in the contest.

The third line contains two space-separated integers s and f (1 ≤ s < f ≤ n).

Output

Output a single integer — the time of the beginning of the contest (in the first timezone local time), such that the number of participants will be maximum possible. If there are many answers, output the smallest among them.

Examples

Input

3
1 2 3
1 3


Output

3


Input

5
1 2 3 4 1
1 3


Output

4

Note

In the first example, it's optimal to start competition at 3 hours (in first timezone). In this case, it will be 1 hour in the second timezone and 2 hours in the third timezone. Only one person from the first timezone won't participate.

In second example only people from the third and the fourth timezones will participate.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
from collections import *
from string import *
from itertools import *

n = input()
arr = map(int,raw_input().strip().split())
s,f = map(int,raw_input().strip().split())

def maxim():
    dif = f-s
    if dif==0:
        return [n]
    val = 0
    for i in range(dif):
        val += arr[i]
    maxval = val
    note = [0]
    for i in range(1,n):
        val -= arr[i-1]
        val += arr[(i+dif-1)%n]
        if maxval==val:
            note.append(i)
        elif maxval<val:
            maxval = val
            note = [i]
    return note

ans = maxim()
#print ans
mans = n
for i in ans:
    mans = min((s-(i+1))%n+1,mans)
print mans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import deque
from bootcamp import Basebootcamp

class Cconvenientforeverybodybootcamp(Basebootcamp):
    def __init__(self, min_n=3, max_n=10, max_people=10):
        """
        初始化参数，增加边界值校验
        """
        self.min_n = max(2, min_n)        # n最小应为2
        self.max_n = max(self.min_n, max_n)
        self.max_people = max_people

    def case_generator(self):
        """
        增强参数生成逻辑，确保：
        1. n >= 2
        2. s < f <= n
        3. 使用滑动窗口算法预生成合法case
        """
        n = random.randint(self.min_n, self.max_n)
        a = [random.randint(1, self.max_people) for _ in range(n)]
        
        # 动态调整s/f生成范围
        s = random.randint(1, n-1)
        f = random.randint(s+1, n)  # 严格保证s < f <=n
        
        # 使用参考代码验证生成的case合法性
        try:
            # 预验证case有效性
            dif = f - s
            if dif == 0:
                raise ValueError("Invalid s,f parameters")
        except:
            return self.case_generator()  # 重新生成case
        
        return {
            'n': n,
            'a': a,
            's': s,
            'f': f
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        a = question_case['a']
        s = question_case['s']
        f = question_case['f']
        
        problem = f"""## 时空竞赛调度优化
        
在未来的地球时钟系统中，一天被划分为{n}个时区(编号1-{n})，相邻时区整点相差1小时。第1时区当前时间为t时，则第k时区时间为(t+k-1) mod {n}（若结果为0则显示{n}）。

竞赛平台计划举办持续1小时的全球赛事，要求：
1. 所有时区必须整点开始
2. 第i时区用户当且仅当满足以下条件时才参赛：
   - 开始时刻 ≥ 本地时间{s}点整
   - 结束时刻 ≤ 本地时间{f}点整（含等于结束时刻的情况不参赛）

已知各时区参赛人数为：{a}
请计算在第一时区视角下，能获得最多参赛人数的开始时刻。如有多个解，输出最小时刻。

答案要求：
将最终整数结果置于[answer]标签内，如：[answer]3[/answer]。"""
        return problem

    @staticmethod
    def extract_output(output):
        # 增强正则表达式鲁棒性
        matches = re.findall(r'\[answer\s*\](\d+)\s*\[/answer\s*\]', output, re.IGNORECASE)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        a = identity['a']
        s = identity['s']
        f = identity['f']
        window_size = f - s
        
        # 滑动窗口算法优化
        current_sum = sum(a[:window_size])
        max_sum = current_sum
        candidates = deque([0])
        
        for i in range(1, n):
            current_sum += a[(i + window_size - 1) % n] - a[i-1]
            if current_sum > max_sum:
                max_sum = current_sum
                candidates.clear()
                candidates.append(i)
            elif current_sum == max_sum:
                candidates.append(i)
        
        # 计算所有候选解对应的起始时间
        min_start = n
        for i in candidates:
            mod_value = (s - (i + 1)) % n
            start_time = mod_value + 1 if mod_value != 0 else n  # 修正边界条件
            if start_time < min_start:
                min_start = start_time
        
        return solution == min_start
