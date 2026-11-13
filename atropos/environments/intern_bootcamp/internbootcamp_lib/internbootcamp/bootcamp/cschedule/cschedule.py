"""# 

### 谜题描述
At the beginning of the new semester there is new schedule in the Berland State University. According to this schedule, n groups have lessons at the room 31. For each group the starting time of the lesson and the finishing time of the lesson are known. It has turned out that it is impossible to hold all lessons, because for some groups periods of their lessons intersect. If at some moment of time one groups finishes it's lesson, and the other group starts the lesson, their lessons don't intersect.

The dean wants to cancel the lesson in one group so that no two time periods of lessons of the remaining groups intersect. You are to find all ways to do that.

Input

The first line contains integer n (1 ≤ n ≤ 5000) — amount of groups, which have lessons in the room 31. Then n lines follow, each of them contains two integers li ri (1 ≤ li < ri ≤ 106) — starting and finishing times of lesson of the i-th group. It is possible that initially no two lessons intersect (see sample 1).

Output

Output integer k — amount of ways to cancel the lesson in exactly one group so that no two time periods of lessons of the remaining groups intersect. In the second line output k numbers — indexes of groups, where it is possible to cancel the lesson. Groups are numbered starting from 1 in the order that they were given in the input. Output the numbers in increasing order.

Examples

Input

3
3 10
20 30
1 3


Output

3
1 2 3 

Input

4
3 10
20 30
1 3
1 39


Output

1
4 

Input

3
1 5
2 6
3 7


Output

0

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
range = xrange
input = raw_input

n = int(input())

L = []
R = []
for _ in range(n):
    l,r = [int(x) for x in input().split()]
    L.append(l)
    R.append(r)

A = []
for i in range(n):
    x = 0
    li = L[i]
    ri = R[i]
    for j in range(n):
        x += i != j and (li <= L[j] < ri or L[j] <= li < R[j])
    A.append(x)

y = sum(A)
B = [i for i in range(n) if 2 * A[i] == y]
print len(B)
print ' '.join(str(i + 1) for i in B)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Cschedulebootcamp(Basebootcamp):
    def __init__(self, min_groups=1, max_groups=10, time_max=10**6):
        self.min_groups = max(1, min_groups)
        self.max_groups = max(self.min_groups, max_groups)
        self.time_max = time_max

    def case_generator(self):
        case_type = random.choices([0, 1, 2, 3], weights=[1, 1, 1, 3], k=1)[0]
        n = random.randint(self.min_groups, self.max_groups)
        intervals = []

        if case_type == 0:  # 全不重叠
            current_end = 0
            max_possible = (self.time_max - 1) // 2
            n = min(n, max_possible) if max_possible > 0 else 1
            intervals = []
            for _ in range(n):
                start = current_end + 1
                if start >= self.time_max:
                    break
                end = random.randint(start + 1, min(self.time_max, start + (self.time_max - start) // (n - _)))
                intervals.append({'l': start, 'r': end})
                current_end = end
            n = len(intervals)
        
        elif case_type == 1:  # 必须删除特定组
            n = max(n, 2)
            base_count = n - 1
            intervals = []
            current_end = 0
            for _ in range(base_count):
                start = current_end + 1
                end = start + 1
                intervals.append({'l': start, 'r': end})
                current_end = end
            conflict_group = {'l': intervals[0]['l'], 'r': current_end + 1}
            intervals.append(conflict_group)
        
        elif case_type == 2:  # 全重叠无解
            common_mid = random.randint(1, self.time_max - 1)
            radius = random.randint(1, min(common_mid, self.time_max - common_mid))
            for _ in range(n):
                l = random.randint(common_mid - radius, common_mid)
                r = random.randint(common_mid + 1, common_mid + radius)
                intervals.append({'l': l, 'r': r})
        
        else:  # 随机案例
            intervals = []
            for _ in range(n):
                li = random.randint(1, self.time_max - 1)
                ri = random.randint(li + 1, self.time_max)
                intervals.append({'l': li, 'r': ri})

        return {'n': len(intervals), 'intervals': intervals}

    @staticmethod
    def prompt_func(question_case) -> str:
        input_str = "\n".join([f"{x['l']} {x['r']}" for x in question_case['intervals']])
        return f"""请解决课程冲突问题。输入：
{question_case['n']}
{input_str}

输出两行：解的数量k和升序排列的索引（用空格分隔），如：
[answer]
3
1 2 3
[/answer]"""

    @staticmethod
    def extract_output(output):
        match = re.search(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not match: return None
        
        lines = [l.strip() for l in match.group(1).strip().split('\n') if l.strip()]
        if not lines: return None
        
        try:
            k = int(lines[0])
            if k == 0 and len(lines) == 1:
                return "0\n"
            if len(lines) < 2: return None
            indices = list(map(int, lines[1].split()))
            if len(indices) != k: return None
            return f"{k}\n{' '.join(map(str, sorted(indices)))}"
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            k = int(solution.split('\n')[0])
            if k == 0: 
                return len(identity['intervals']) == 0 or all(
                    any(a['r'] > b['l'] for a, b in zip(identity['intervals'], identity['intervals'][1:]))
                )
            
            indices = list(map(int, solution.split('\n')[1].split()))
            sorted_intervals = sorted(
                (x for i, x in enumerate(identity['intervals']) if (i+1) not in indices),
                key=lambda x: x['l']
            )
            return all(x['r'] <= y['l'] for x, y in zip(sorted_intervals, sorted_intervals[1:]))
        except:
            return False
