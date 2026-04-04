"""# 

### 谜题描述
You are given an infinite periodic array a0, a1, ..., an - 1, ... with the period of length n. Formally, <image>. A periodic subarray (l, s) (0 ≤ l < n, 1 ≤ s < n) of array a is an infinite periodic array with a period of length s that is a subsegment of array a, starting with position l.

A periodic subarray (l, s) is superior, if when attaching it to the array a, starting from index l, any element of the subarray is larger than or equal to the corresponding element of array a. An example of attaching is given on the figure (top — infinite array a, bottom — its periodic subarray (l, s)):

<image>

Find the number of distinct pairs (l, s), corresponding to the superior periodic arrays.

Input

The first line contains number n (1 ≤ n ≤ 2·105). The second line contains n numbers a0, a1, ..., an - 1 (1 ≤ ai ≤ 106), separated by a space.

Output

Print a single integer — the sought number of pairs.

Examples

Input

4
7 1 2 3


Output

2


Input

2
2 1


Output

1


Input

3
1 1 1


Output

6

Note

In the first sample the superior subarrays are (0, 1) and (3, 2).

Subarray (0, 1) is superior, as a0 ≥ a0, a0 ≥ a1, a0 ≥ a2, a0 ≥ a3, a0 ≥ a0, ...

Subarray (3, 2) is superior a3 ≥ a3, a0 ≥ a0, a3 ≥ a1, a0 ≥ a2, a3 ≥ a3, ...

In the third sample any pair of (l, s) corresponds to a superior subarray as all the elements of an array are distinct.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
# -*- coding: utf-8 -*-
import fractions
from collections import defaultdict


if __name__ == '__main__':
    n = int(raw_input())
    a = map(int, raw_input().split())
    a *= 2
    inf = min(a) - 1
    a[-1] = inf
    result = 0

    numbers_by_gcd = defaultdict(list)
    for i in xrange(1, n):
        numbers_by_gcd[fractions.gcd(i, n)].append(i)

    for d in xrange(1, n):
        if n % d != 0:
            continue
        m = [inf] * d
        for i in xrange(n):
            if a[i] > m[i % d]:
                m[i % d] = a[i]
        l = 0
        r = 0
        while l < n:
            if a[r] < m[r % d]:
                for i in numbers_by_gcd[d]:
                    if i > r - l:
                        break
                    result += min(r - i, n - 1) - l + 1
                l = r + 1
            r += 1
    print result
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import math
from collections import defaultdict
import random
import re

def solve_puzzle(n, a):
    if n == 1:
        return 0  # s必须≥1且<1，无解

    a_extended = a.copy()
    a_extended.extend(a)
    inf = min(a) - 1
    a_extended[-1] = inf  # 保证最后元素最小
    result = 0

    numbers_by_gcd = defaultdict(list)
    for i in range(1, n):
        current_gcd = math.gcd(i, n)
        numbers_by_gcd[current_gcd].append(i)

    for d in numbers_by_gcd:  # 遍历每个可能的gcd值
        if n % d != 0:
            continue
        
        # 计算每个模位的最大值
        m = [-math.inf] * d
        for i in range(n):
            mod = i % d
            if a_extended[i] > m[mod]:
                m[mod] = a_extended[i]
        
        l = 0
        r = 0
        max_r = len(a_extended) - 1  # 防止越界
        while l < n and r <= max_r:
            if a_extended[r] < m[r % d]:
                # 处理当前有效区间
                sorted_s = sorted(numbers_by_gcd[d])
                for s in sorted_s:
                    if s > (r - l):
                        break
                    # 计算有效区间长度
                    start = l
                    end = min(r - s, n - 1)
                    if start <= end:
                        result += end - start + 1
                l = r + 1
                r = l
            else:
                r += 1
    return result

class Esuperiorperiodicsubarraysbootcamp(Basebootcamp):
    def __init__(self, max_n=20, max_a=10):
        self.max_n = max(max_n, 1)  # 保证n≥1
        self.max_a = max_a
    
    def case_generator(self):
        n = random.randint(1, self.max_n)
        a = [random.randint(1, self.max_a) for _ in range(n)]
        return {
            'n': n,
            'a': a,
            'correct_answer': solve_puzzle(n, a)
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a_str = ' '.join(map(str, question_case['a']))
        if n == 1:
            s_range_hint = "（注意：当n=1时没有有效的s值）"
        else:
            s_range_hint = f"其中1 ≤ s < {n}"
        
        return f"""请解决以下无限周期数组问题：

给定参数：
- 原始数组长度n = {n}
- 数组元素 = [{a_str}]

需要找出所有满足以下条件的(l, s)对：
1. 起始位置l满足0 ≤ l < {n}
2. 周期长度s满足{s_range_hint}
3. 对于所有k ≥ 0，子数组在位置k的元素 ≥ 原数组位置k的元素

规则详解：
- 子数组定义为从l开始取s个元素，并无限重复：元素k的值为a[(l + k) % {n}]
- 比较时，原数组元素k的值为a[k % {n}]

输出要求：
- 输出最终答案的整数形式
- 将答案放在[answer]标签内，如：[answer]0[/answer]

请计算符合条件的总对数："""

    @staticmethod
    def extract_output(output):
        # 匹配最后出现的答案标签
        matches = re.findall(r'\[/?answer\]', output, re.IGNORECASE)
        if len(matches) < 2:
            return None
        
        last_match = re.findall(r'\[answer\][\s]*(\d+)[\s]*\[/answer\]', output, re.IGNORECASE)
        return int(last_match[-1]) if last_match else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
