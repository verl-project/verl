"""# 

### 谜题描述
It's well known that the best way to distract from something is to do one's favourite thing. Job is such a thing for Leha.

So the hacker began to work hard in order to get rid of boredom. It means that Leha began to hack computers all over the world. For such zeal boss gave the hacker a vacation of exactly x days. You know the majority of people prefer to go somewhere for a vacation, so Leha immediately went to the travel agency. There he found out that n vouchers left. i-th voucher is characterized by three integers li, ri, costi — day of departure from Vičkopolis, day of arriving back in Vičkopolis and cost of the voucher correspondingly. The duration of the i-th voucher is a value ri - li + 1.

At the same time Leha wants to split his own vocation into two parts. Besides he wants to spend as little money as possible. Formally Leha wants to choose exactly two vouchers i and j (i ≠ j) so that they don't intersect, sum of their durations is exactly x and their total cost is as minimal as possible. Two vouchers i and j don't intersect if only at least one of the following conditions is fulfilled: ri < lj or rj < li.

Help Leha to choose the necessary vouchers!

Input

The first line contains two integers n and x (2 ≤ n, x ≤ 2·105) — the number of vouchers in the travel agency and the duration of Leha's vacation correspondingly.

Each of the next n lines contains three integers li, ri and costi (1 ≤ li ≤ ri ≤ 2·105, 1 ≤ costi ≤ 109) — description of the voucher.

Output

Print a single integer — a minimal amount of money that Leha will spend, or print  - 1 if it's impossible to choose two disjoint vouchers with the total duration exactly x.

Examples

Input

4 5
1 3 4
1 2 5
5 6 1
1 2 4


Output

5


Input

3 2
4 6 3
2 4 1
3 5 4


Output

-1

Note

In the first sample Leha should choose first and third vouchers. Hereupon the total duration will be equal to (3 - 1 + 1) + (6 - 5 + 1) = 5 and the total cost will be 4 + 1 = 5.

In the second sample the duration of each voucher is 3 therefore it's impossible to choose two vouchers with the total duration equal to 2.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from collections import defaultdict
from operator import itemgetter as it


def build(seg_tree, n):
    for i in reversed(xrange(1, n)):
        if 2 * i > n - 1:
            seg_tree[i] = min(seg_tree[2 * i][2], seg_tree[2 * i + 1][2])
        else:
            seg_tree[i] = min(seg_tree[2 * i], seg_tree[2 * i + 1])


def query(seg_tree, l, r, n):
    #l and r inclusive
    res = float('inf')
    l, r = l + n, r + n
    while l <= r:
        #if l is odd then its parent is not included so we have to include this when visiting left borders
        if l % 2 != 0:
            if l > n - 1: res = min(res, seg_tree[l][2])
            else: res = min(res, seg_tree[l])
            l += 1
        #if r is even then its parent is not included so we have to include this when visiting right borders
        if r % 2 == 0:
            if r > n - 1: res = min(res, seg_tree[r][2])
            else: res = min(res, seg_tree[r])
            r -= 1
        if l == r:
            break
        l, r = l / 2, r / 2
    return res


def bs_floor(arr, val):
    l = -1
    r = len(arr)
    # -1 to n - 1
    while r - l > 1:
        mid = (l + r) / 2
        if arr[mid][0] <= val:
            l = mid
        else:
            r = mid
    return l


n, x = map(int, raw_input().split())
d = defaultdict(list)
d1 = {}
ans = float('inf')
for a0 in xrange(1, n + 1):
    l, r, c = map(int, raw_input().split())
    d[r - l + 1].append((l, r, c))

for key in d:
    d[key].sort(key=it(0))
    m = len(d[key])
    d1[key] = [0] * m + d[key][:]
    build(d1[key], m)

for key in d:
    for (l, r, c) in d[key]:
        key1 = x - (r - l + 1)
        if key1 in d1:
            p = bs_floor(d[key1], r) + 1
            m = len(d[key1])
            if p != m:
                c1 = query(d1[key1], p, len(d[key1]) - 1, len(d[key1]))
                ans = min(ans, c + c1)
if ans == float('inf'):
    print -1
else:
    print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
from collections import defaultdict
import random
import re
import bisect

class Chackerpackyourbagsbootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_x=20, max_duration=200000, max_cost=10**9):
        self.max_n = max_n
        self.max_x = max_x
        self.max_duration = max_duration
        self.max_cost = max_cost
    
    def case_generator(self):
        x = random.randint(2, self.max_x)
        vouchers = []
        has_solution = random.choice([True, False])
        
        if has_solution:
            # 确保存在唯一最优解
            duration1 = random.randint(1, x-1)
            duration2 = x - duration1
            attempts = 0
            
            while True:
                # 生成第一个优惠券
                l1 = random.randint(1, self.max_duration - duration1)
                r1 = l1 + duration1 - 1
                
                # 生成第二个优惠券的位置
                if random.choice([True, False]):
                    # 在第一个之后
                    l2_min = r1 + 1
                    r2_max = min(l2_min + duration2 + 3, self.max_duration)
                    if r2_max - l2_min + 1 < duration2:
                        continue
                    l2 = random.randint(l2_min, r2_max - duration2 + 1)
                    r2 = l2 + duration2 - 1
                else:
                    # 在第一个之前
                    r2_max = l1 - 1
                    if r2_max < 1:
                        continue
                    l2_min = max(1, r2_max - duration2 + 1)
                    r2 = random.randint(l2_min + duration2 - 1, r2_max)
                    l2 = r2 - duration2 + 1
                
                # 验证区间有效性
                if (1 <= l2 <= r2 <= self.max_duration and 
                    (r1 < l2 or r2 < l1) and 
                    (r2 - l2 + 1) == duration2):
                    break
                
                attempts += 1
                if attempts > 100:
                    has_solution = False
                    break
            
            if has_solution:
                # 生成最优解对
                c1 = random.randint(1, self.max_cost//4)
                c2 = random.randint(1, self.max_cost//4)
                vouchers = [(l1, r1, c1), (l2, l2 + duration2 - 1, c2)]
                
                # 添加干扰数据
                for _ in range(random.randint(0, self.max_n-2)):
                    while True:
                        l = random.randint(1, self.max_duration)
                        r = random.randint(l, self.max_duration)
                        d = r - l + 1
                        if d + duration1 != x and d + duration2 != x:
                            break
                    cost = random.randint(max(c1, c2)+1, self.max_cost)
                    vouchers.append((l, r, cost))
        
        if not has_solution:
            # 生成无解情况
            base_duration = x + 1
            min_duration = max(1, x//2 + 1)
            for _ in range(random.randint(2, self.max_n)):
                duration = random.randint(min_duration, x + 5)
                l = random.randint(1, self.max_duration - duration)
                r = l + duration - 1
                cost = random.randint(1, self.max_cost)
                vouchers.append((l, r, cost))
        
        random.shuffle(vouchers)
        return {
            'n': len(vouchers),
            'x': x,
            'vouchers': [{'l': l, 'r': r, 'cost': c} for (l, r, c) in vouchers]
        }
    
    @staticmethod
    def prompt_func(question_case):
        voucher_lines = '\n'.join(
            f"{v['l']} {v['r']} {v['cost']}" 
            for v in question_case['vouchers']
        )
        return f"""Select two NON-OVERLAPPING travel vouchers meeting:
1. Sum of durations EXACTLY {question_case['x']} days
2. Minimum total cost

Available vouchers:
{voucher_lines}

Output format: [answer]number[/answer] where number is the minimal cost or -1"""

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
        vouchers = [(v['l'], v['r'], v['cost']) for v in identity['vouchers']]
        return solution == cls.optimal_solution(vouchers, identity['x'])
    
    @staticmethod
    def optimal_solution(vouchers, x):
        duration_map = defaultdict(list)
        for l, r, cost in vouchers:
            duration = r - l + 1
            duration_map[duration].append((l, cost))
        
        # 预处理每个duration的最小成本
        pre_min = {}
        for d in duration_map:
            sorted_list = sorted(duration_map[d], key=lambda x: x[0])
            min_prefix = []
            current_min = float('inf')
            for l, c in reversed(sorted_list):
                current_min = min(current_min, c)
                min_prefix.append(current_min)
            min_prefix.reverse()
            pre_min[d] = (sorted_list, min_prefix)
        
        min_cost = float('inf')
        
        for l, r, cost in vouchers:
            current_duration = r - l + 1
            need_duration = x - current_duration
            if need_duration not in pre_min:
                continue
            
            sorted_list, min_prefix = pre_min[need_duration]
            idx = bisect.bisect_right(sorted_list, (r, float('inf')))
            
            if idx < len(min_prefix):
                min_cost = min(min_cost, cost + min_prefix[idx])
        
        return min_cost if min_cost != float('inf') else -1
