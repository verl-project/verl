"""# 

### 谜题描述
Professor GukiZ is concerned about making his way to school, because massive piles of boxes are blocking his way. 

In total there are n piles of boxes, arranged in a line, from left to right, i-th pile (1 ≤ i ≤ n) containing ai boxes. Luckily, m students are willing to help GukiZ by removing all the boxes from his way. Students are working simultaneously. At time 0, all students are located left of the first pile. It takes one second for every student to move from this position to the first pile, and after that, every student must start performing sequence of two possible operations, each taking one second to complete. Possible operations are:

  1. If i ≠ n, move from pile i to pile i + 1;
  2. If pile located at the position of student is not empty, remove one box from it.



GukiZ's students aren't smart at all, so they need you to tell them how to remove boxes before professor comes (he is very impatient man, and doesn't want to wait). They ask you to calculate minumum time t in seconds for which they can remove all the boxes from GukiZ's way. Note that students can be positioned in any manner after t seconds, but all the boxes must be removed.

Input

The first line contains two integers n and m (1 ≤ n, m ≤ 105), the number of piles of boxes and the number of GukiZ's students. 

The second line contains n integers a1, a2, ... an (0 ≤ ai ≤ 109) where ai represents the number of boxes on i-th pile. It's guaranteed that at least one pile of is non-empty.

Output

In a single line, print one number, minimum time needed to remove all the boxes in seconds.

Examples

Input

2 1
1 1


Output

4


Input

3 2
1 0 2


Output

5


Input

4 100
3 4 5 4


Output

5

Note

First sample: Student will first move to the first pile (1 second), then remove box from first pile (1 second), then move to the second pile (1 second) and finally remove the box from second pile (1 second).

Second sample: One of optimal solutions is to send one student to remove a box from the first pile and a box from the third pile, and send another student to remove a box from the third pile. Overall, 5 seconds.

Third sample: With a lot of available students, send three of them to remove boxes from the first pile, four of them to remove boxes from the second pile, five of them to remove boxes from the third pile, and four of them to remove boxes from the fourth pile. Process will be over in 5 seconds, when removing the boxes from the last pile is finished.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, m = map(int, raw_input().split())
a = map(int, raw_input().split())

def check(t):
    b = a[:]
    idx = len(b) - 1
    for _ in xrange(m):
        while idx >= 0 and b[idx] == 0:
            idx -= 1
        if idx < 0:
            break
        remain = t - (idx + 1)
        if remain <= 0:
            break
        while idx >= 0 and remain > b[idx]:
            remain -= b[idx]
            b[idx] = 0
            idx -= 1
        if idx < 0:
            break
        b[idx] -= remain
    return all(num == 0 for num in b)

ubound = sum(a) + n * 3
l, r = 0, ubound
while l < r:
    mid = l + r >> 1
    if check(mid):
        r = mid
    else:
        l = mid + 1

print l
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cgukizhatesboxesbootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_m=10**3, max_a=10**4):
        """
        参数调整支持更大规模的测试用例生成
        """
        self.max_n = max_n
        self.max_m = max_m
        self.max_a = max_a
    
    def case_generator(self):
        n = random.randint(1, self.max_n)
        m = random.randint(1, self.max_m)
        a = [random.randint(0, self.max_a) for _ in range(n)]
        
        # 确保至少有一个非空堆的非零概率
        if sum(a) == 0:
            a[random.randint(0, n-1)] = random.randint(1, self.max_a)
        
        return {
            'n': n,
            'm': m,
            'a': a
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        a_str = ' '.join(map(str, question_case['a']))
        prompt = (
            "Professor GukiZ's students need to remove all boxes blocking his way. Here's the problem instance:\n\n"
            "Input Format:\n"
            "Two lines:\n"
            "1. Number of piles (n) and students (m)\n"
            "2. Space-separated list of boxes in each pile\n\n"
            "Problem Input:\n"
            f"{n} {m}\n"
            f"{a_str}\n\n"
            "Rules:\n"
            "1. Students start left of first pile and take 1 second to reach it\n"
            "2. Each subsequent operation (move or remove) takes 1 second\n"
            "3. Movement from pile i to i+1 requires pile i < n\n"
            "4. Remove operations can only happen on non-empty piles\n"
            "5. Students work simultaneously\n\n"
            "Output Requirement:\n"
            "The minimal time t in seconds to clear all boxes, put your final answer within [answer] and [/answer], e.g.:\n"
            "[answer]5[/answer]"
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except (ValueError, TypeError):
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            t = int(solution)
        except:
            return False
        
        return t == cls.compute_min_time(
            identity['n'],
            identity['m'],
            identity['a']
        )
    
    @staticmethod
    def compute_min_time(n, m, a):
        def check(t):
            remaining = list(a)
            idx = n - 1
            students = m
            
            while students > 0 and idx >= 0:
                if remaining[idx] == 0:
                    idx -= 1
                    continue
                
                required_time = idx + 1  # 移动到该堆需要的时间
                if t < required_time:
                    break
                
                working_time = t - required_time
                if working_time <= 0:
                    idx -= 1
                    continue
                
                students -= 1
                if working_time >= remaining[idx]:
                    working_time -= remaining[idx]
                    remaining[idx] = 0
                    idx -= 1
                    
                    # 处理剩余时间
                    while working_time > 0 and idx >= 0:
                        if remaining[idx] == 0:
                            idx -= 1
                            continue
                        if working_time >= remaining[idx]:
                            working_time -= remaining[idx]
                            remaining[idx] = 0
                            idx -= 1
                        else:
                            remaining[idx] -= working_time
                            working_time = 0
                else:
                    remaining[idx] -= working_time

            return sum(remaining) == 0

        # 二分查找上下界优化
        left = 0
        right = sum(a) + n + 1  # 更精确的初始上界
        
        while left < right:
            mid = (left + right) // 2
            if check(mid):
                right = mid
            else:
                left = mid + 1

        return left
