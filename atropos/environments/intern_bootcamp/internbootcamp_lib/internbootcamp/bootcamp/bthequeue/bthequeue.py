"""# 

### 谜题描述
Finally! Vasya have come of age and that means he can finally get a passport! To do it, he needs to visit the passport office, but it's not that simple. There's only one receptionist at the passport office and people can queue up long before it actually opens. Vasya wants to visit the passport office tomorrow.

He knows that the receptionist starts working after ts minutes have passed after midnight and closes after tf minutes have passed after midnight (so that (tf - 1) is the last minute when the receptionist is still working). The receptionist spends exactly t minutes on each person in the queue. If the receptionist would stop working within t minutes, he stops serving visitors (other than the one he already serves). 

Vasya also knows that exactly n visitors would come tomorrow. For each visitor Vasya knows the point of time when he would come to the passport office. Each visitor queues up and doesn't leave until he was served. If the receptionist is free when a visitor comes (in particular, if the previous visitor was just served and the queue is empty), the receptionist begins to serve the newcomer immediately.

<image> \"Reception 1\"

For each visitor, the point of time when he would come to the passport office is positive. Vasya can come to the office at the time zero (that is, at midnight) if he needs so, but he can come to the office only at integer points of time. If Vasya arrives at the passport office at the same time with several other visitors, he yields to them and stand in the queue after the last of them.

Vasya wants to come at such point of time that he will be served by the receptionist, and he would spend the minimum possible time in the queue. Help him!

Input

The first line contains three integers: the point of time when the receptionist begins to work ts, the point of time when the receptionist stops working tf and the time the receptionist spends on each visitor t. The second line contains one integer n — the amount of visitors (0 ≤ n ≤ 100 000). The third line contains positive integers in non-decreasing order — the points of time when the visitors arrive to the passport office.

All times are set in minutes and do not exceed 1012; it is guaranteed that ts < tf. It is also guaranteed that Vasya can arrive at the passport office at such a point of time that he would be served by the receptionist.

Output

Print single non-negative integer — the point of time when Vasya should arrive at the passport office. If Vasya arrives at the passport office at the same time with several other visitors, he yields to them and queues up the last. If there are many answers, you can print any of them.

Examples

Input

10 15 2
2
10 13


Output

12

Input

8 17 3
4
3 4 5 8


Output

2

Note

In the first example the first visitor comes exactly at the point of time when the receptionist begins to work, and he is served for two minutes. At 12 minutes after the midnight the receptionist stops serving the first visitor, and if Vasya arrives at this moment, he will be served immediately, because the next visitor would only come at 13 minutes after midnight.

In the second example, Vasya has to come before anyone else to be served. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
# -*- coding: utf-8 -*-
# @Author: HaonanWu
# @Date:   2017-02-26 21:22:14
# @Last Modified by:   HaonanWu
# @Last Modified time: 2017-02-26 21:44:53

a, b, t = map(int, raw_input().split())
n = int(input())
if n:
    arr = map(int,raw_input().split(' '))
else:
    arr = []
ma = 1e12+7
ans = 0L

for k in arr:
    if k+t <= b:
        if k and max(a, k-1)+t <= b and a-k+1 < ma:
            ma = a-k+1
            ans = min(k-1, a)
        a = max(a, k)+t

if a+t<=b:
    ans = a

print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Bthequeuebootcamp(Basebootcamp):
    def __init__(self, **params):
        default_params = {
            'ts_min': 1,
            'ts_max': 1000,
            't_min': 1,
            't_max': 1000,
            'n_max': 100,
            'max_visit_gap': 500
        }
        self.params = {**default_params, **params}
    
    def case_generator(self):
        while True:
            ts = random.randint(self.params['ts_min'], self.params['ts_max'])
            t = random.randint(self.params['t_min'], self.params['t_max'])
            n = random.randint(0, self.params['n_max'])
            
            # 生成符合题目要求的访客到达时间序列
            arr = []
            if n > 0:
                # 生成包含不同时间特征的测试案例
                time_points = [
                    random.randint(1, ts),  # ts之前的案例
                    random.randint(ts, ts*2),  # ts之后的案例
                    random.randint(ts//2, ts*3)  # 混合案例
                ]
                arr = sorted(random.choice(time_points) for _ in range(n))
                
                # 确保严格非递减序列
                for i in range(1, n):
                    if arr[i] < arr[i-1]:
                        arr[i] = arr[i-1]

            # 模拟真实处理流程
            current_end = ts
            service_records = []
            for k in arr:
                start_time = max(current_end, k)
                end_time = start_time + t
                service_records.append((start_time, end_time))
                current_end = end_time

            # 计算所有可能候选时间
            candidates = []
            # 检查最早可能的时间间隙
            if ts > 0:
                candidates.append(0)
                
            # 检查各服务记录之间的间隙
            prev_end = ts
            for (start, end) in service_records:
                if prev_end < start:
                    candidates.append(prev_end)
                    candidates.append(start - 1)
                prev_end = end
            
            # 检查最后时间点
            if prev_end + t <= 2**63-1:  # 保证不越界
                candidates.append(prev_end)

            # 动态确定合法的tf
            valid_tf = max(
                current_end + t + 1,  # 正常结束
                max(candidates, default=0) + t + 1 if candidates else ts + t + 1
            )
            
            # 确保最少存在一个合法解
            if any(x + t <= valid_tf for x in candidates):
                return {
                    'ts': ts,
                    'tf': valid_tf,
                    't': t,
                    'n': n,
                    'arr': arr,
                    '_candidates': [x for x in candidates if x + t <= valid_tf]
                }

    @staticmethod
    def prompt_func(question_case) -> str:
        problem_desc = (
            f"The passport office operates from {question_case['ts']} to {question_case['tf']} minutes.\n"
            f"Service duration: {question_case['t']} minutes per visitor.\n"
            f"Existing visitors arrive at: {question_case['arr'] if question_case['arr'] else 'none'}.\n"
            "Find Vasya's optimal arrival time (integer) to achieve:\n"
            "1. Being served before closing\n"
            "2. Minimize waiting time\n"
            "3. If multiple solutions, choose earliest\n"
            "Format answer as [answer]time[/answer]"
        )
        return problem_desc

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            clean_str = matches[-1].strip().split()[0]  # 处理可能的多余内容
            return int(clean_str)
        except (ValueError, IndexError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 完整复现原始验证逻辑
        ts = identity['ts']
        tf = identity['tf']
        t = identity['t']
        arr = identity['arr']
        
        best_time = None
        min_wait = float('inf')
        
        # 生成所有可能候选时间
        candidates = []
        if ts > 0:
            candidates.append(0)  # 午夜到达的情况
        
        current_end = ts
        prev_end = ts
        for k in arr:
            start = max(current_end, k)
            end = start + t
            if prev_end < start:
                candidates.append(prev_end)  # 前一个服务结束后的间隙开始
                candidates.append(start - 1) # 当前服务开始前的最后时刻
            prev_end = end
            current_end = end
        
        # 最后时间点
        if current_end + t <= tf:
            candidates.append(current_end)
        
        # 筛选合法候选
        valid_candidates = [x for x in candidates if x + t <= tf]
        
        # 特殊情况处理：没有其他访客
        if not arr:
            if ts + t <= tf:
                return solution == ts
            return False
        
        # 寻找最优解
        for candidate in valid_candidates:
            service_start = max(ts, candidate)
            for k in arr:
                if k > service_start:
                    break
                service_start = max(service_start, k) + t
            
            if service_start > tf:
                continue
                
            wait_time = service_start - candidate
            if wait_time < min_wait or (wait_time == min_wait and candidate < best_time):
                min_wait = wait_time
                best_time = candidate
        
        return solution == best_time
