"""# 

### 谜题描述
Petya has n integers: 1, 2, 3, ..., n. He wants to split these integers in two non-empty groups in such a way that the absolute difference of sums of integers in each group is as small as possible. 

Help Petya to split the integers. Each of n integers should be exactly in one group.

Input

The first line contains a single integer n (2 ≤ n ≤ 60 000) — the number of integers Petya has.

Output

Print the smallest possible absolute difference in the first line.

In the second line print the size of the first group, followed by the integers in that group. You can print these integers in arbitrary order. If there are multiple answers, print any of them.

Examples

Input

4


Output

0
2 1 4 


Input

2


Output

1
1 1 

Note

In the first example you have to put integers 1 and 4 in the first group, and 2 and 3 in the second. This way the sum in each group is 5, and the absolute difference is 0.

In the second example there are only two integers, and since both groups should be non-empty, you have to put one integer in the first group and one in the second. This way the absolute difference of sums of integers in each group is 1.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n=input()
if n%4==0:
    print 0
    print n/2,
    for i in range(n/4):
        print i+1,n-i,
if n%4==2:
    print 1
    print n/2,n/2,
    for i in range(n/4):
        print i+1,n-i,
if n%4==1:
    print 1
    print (n/2)+1,1,
    for i in range(1,(n/4)+1):
        print i+1,n-i+1,
if n%4==3:
    print 0
    print n/2,
    for i in range(1,(n/4)+1):
        print i+1,n-i+1,
    print (n/2)+2
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import re
import random

class Cdividingthenumbersbootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=60000):
        self.min_n = max(int(min_n), 2)
        self.max_n = min(int(max_n), 60000)
        if self.max_n < self.min_n:
            self.max_n = self.min_n

    def case_generator(self):
        # 强制生成覆盖不同余数情况的测试用例
        candidates = []
        for _ in range(3):  # 每种余数至少生成一个案例
            n = random.randint(self.min_n, self.max_n)
            candidates.append(n)
        # 确保至少包含每个余数类型中的一个案例
        for remainder in [0, 1, 2, 3]:
            base_n = random.randint(3, 15) * 4 + remainder
            if 2 <= base_n <= 60000:
                candidates.append(base_n)
        n = random.choice(candidates)
        
        total_sum = n * (n + 1) // 2
        remainder = n % 4
        correct_diff = 0 if (remainder == 0 or remainder ==3) else 1
        return {
            'n': n,
            'correct_diff': correct_diff,
            'total_sum': total_sum,
            'remainder_class': remainder  # 增加余数分类用于调试
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        return f"""Petya has integers from 1 to {n}. Split them into two non-empty groups to minimize the absolute difference of their sums.

Output format:
[answer]
<min_difference>
<group_size> <element1> <element2> ... <elementK>
[/answer]

Examples:
For n=4:
[answer]
0
2 1 4
[/answer]

For n=2:
[answer]
1
1 1
[/answer]"""

    @staticmethod
    def extract_output(output):
        # 使用更健壮的正则表达式匹配
        answer_pattern = re.compile(
            r'\[answer\][\s\r\n]*(\d+)[\s\r\n]+(\d+[\s\d]*)[\s\r\n]*\[/answer\]', 
            re.DOTALL | re.IGNORECASE
        )
        matches = answer_pattern.findall(output)
        if not matches:
            return None
        
        # 只处理最后一个有效匹配
        last_match = matches[-1]
        try:
            diff = int(last_match[0])
            group_part = list(map(int, last_match[1].split()))
            if len(group_part) < 1:
                return None
            group_size = group_part[0]
            elements = group_part[1:]
            if group_size != len(elements):
                return None
            return {
                'diff': diff,
                'group_size': group_size,
                'elements': elements
            }
        except (ValueError, IndexError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 防御性编程：逐步排除所有可能错误
        if not solution:
            return False
        
        # 必需字段检查
        required_keys = {'diff', 'group_size', 'elements'}
        if any(key not in solution for key in required_keys):
            return False

        # 基础类型校验
        if not isinstance(solution['diff'], int):
            return False
        if not isinstance(solution['group_size'], int):
            return False
        if not isinstance(solution['elements'], (list, tuple)):
            return False

        # 获取验证参数
        n = identity['n']
        correct_diff = identity['correct_diff']
        total_sum = identity['total_sum']
        elements = solution['elements']
        group_size = solution['group_size']

        # 差值验证
        if solution['diff'] != correct_diff:
            return False

        # 分组大小有效性
        if group_size < 1 or group_size >= n:
            return False

        # 元素唯一性检查
        if len(set(elements)) != len(elements):
            return False

        # 元素范围检查
        if any(not (1 <= num <= n) for num in elements):
            return False

        # 数学验证
        sum_group = sum(elements)
        actual_diff = abs(2*sum_group - total_sum)  # 优化计算方式
        return actual_diff == correct_diff
