"""# 

### 谜题描述
As meticulous Gerald sets the table and caring Alexander sends the postcards, Sergey makes snowmen. Each showman should consist of three snowballs: a big one, a medium one and a small one. Sergey's twins help him: they've already made n snowballs with radii equal to r1, r2, ..., rn. To make a snowman, one needs any three snowballs whose radii are pairwise different. For example, the balls with radii 1, 2 and 3 can be used to make a snowman but 2, 2, 3 or 2, 2, 2 cannot. Help Sergey and his twins to determine what maximum number of snowmen they can make from those snowballs.

Input

The first line contains integer n (1 ≤ n ≤ 105) — the number of snowballs. The next line contains n integers — the balls' radii r1, r2, ..., rn (1 ≤ ri ≤ 109). The balls' radii can coincide.

Output

Print on the first line a single number k — the maximum number of the snowmen. Next k lines should contain the snowmen's descriptions. The description of each snowman should consist of three space-separated numbers — the big ball's radius, the medium ball's radius and the small ball's radius. It is allowed to print the snowmen in any order. If there are several solutions, print any of them.

Examples

Input

7
1 2 3 4 5 6 7


Output

2
3 2 1
6 5 4


Input

3
2 2 3


Output

0

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from collections import defaultdict
import heapq


raw_input()
d = defaultdict(int)
for j in [int(i) for i in raw_input().split()]:
  d[j] += 1

h = list([(-1 * v, k) for k, v in d.iteritems()])
heapq.heapify(h)

ret = []


while len(h) > 2:
  a, b, c  = heapq.heappop(h), heapq.heappop(h), heapq.heappop(h)

  ret.append(sorted([a[1], b[1], c[1]], reverse=True))

  for x, xi in [(i+1, j) for i, j in (a, b, c)]:
    if x:
      heapq.heappush(h, (x, xi))

print len(ret)
for a, b, c in ret:
  print a, b, c
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
import heapq
from collections import defaultdict
from bootcamp import Basebootcamp

class Cnewyearsnowmenbootcamp(Basebootcamp):
    def __init__(self, max_n=100000, max_radius=10**9, seed=None):
        self.max_n = max_n
        self.max_radius = max_radius
        self.seed = seed or random.randint(0, 999999)
        random.seed(self.seed)
    
    def case_generator(self):
        # 生成具有三种典型场景的测试案例
        case_type = random.choice(['normal', 'edge', 'invalid'])
        
        if case_type == 'normal':
            # 生成有效雪人案例
            valid_radii = []
            groups = random.randint(1, 5)
            for _ in range(groups):
                r = random.randint(1, self.max_radius)
                count = random.randint(2, 5)
                valid_radii += [r] * count
            n = len(valid_radii)
        elif case_type == 'edge':
            # 极端案例：最多可能雪人
            valid_radii = []
            max_groups = random.randint(3, 10)
            for r in range(1, max_groups+1):
                valid_radii += [r] * 3
            n = len(valid_radii)
        else:  # invalid
            # 无法构成任何雪人的案例
            n = random.randint(3, 100)
            base_r = random.randint(1, self.max_radius)
            valid_radii = [base_r] * n
        
        # 添加随机噪声数据
        noise = [random.randint(1, self.max_radius) for _ in range(random.randint(0, n//2))]
        radii = valid_radii + noise
        random.shuffle(radii)
        
        # 计算正确解
        counts = defaultdict(int)
        for r in radii:
            counts[r] += 1

        heap = [(-cnt, r) for r, cnt in counts.items()]
        heapq.heapify(heap)
        solution = []
        
        while len(heap) >= 3:
            items = [heapq.heappop(heap) for _ in range(3)]
            solution.append(sorted([r for _, r in items], reverse=True))
            
            # 更新计数并重新入堆
            for cnt, r in items:
                new_cnt = -cnt - 1
                if new_cnt > 0:
                    heapq.heappush(heap, (-new_cnt, r))
        
        return {
            'n': len(radii),
            'radii': radii,
            'counts': dict(counts),
            'max_k': len(solution),
            'case_type': case_type  # 用于调试
        }
    
    @staticmethod
    def prompt_func(question_case):
        input_str = f"{question_case['n']}\n{' '.join(map(str, question_case['radii']))}"
        return f"""Sergey needs to build snowmen using triplets of different-sized snowballs. Given the snowball list, find the maximum number of snowmen and their composition.

Input Format:
{input_str}

Output Format:
[answer]
<k>
<r1_1 r1_2 r1_3>
...
<rk_1 rk_2 rk_3>
[/answer]
Requirements:
1. k must be the maximum possible number
2. Each line after k must contain three DESCENDING radii
3. All radii must exist in the input
4. Strictly follow output format"""

    @staticmethod
    def extract_output(output):
        pattern = r'\[\s*answer\s*\](.*?)\[\s*/answer\s*\]'
        matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
        if not matches:
            return None
        content = matches[-1].strip()
        return content if content.count('\n') >= 1 else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            if not solution:
                return False
            
            lines = [l.strip() for l in solution.split('\n') if l.strip()]
            if len(lines) < 1:
                return False
            
            k = int(lines[0])
            if k != identity['max_k'] or len(lines) != k + 1:
                return False

            original_counts = identity['counts'].copy()
            used = defaultdict(int)
            
            for line in lines[1:]:
                parts = list(map(int, line.split()))
                if len(parts) != 3:
                    return False
                a, b, c = parts
                if not (a > b > c):
                    return False
                for r in parts:
                    used[r] += 1
                    if used[r] > original_counts.get(r, 0):
                        return False
            return True
        except Exception as e:
            return False
