"""# 

### 谜题描述
Phoenix has n blocks of height h_1, h_2, ..., h_n, and all h_i don't exceed some value x. He plans to stack all n blocks into m separate towers. The height of a tower is simply the sum of the heights of its blocks. For the towers to look beautiful, no two towers may have a height difference of strictly more than x. 

Please help Phoenix build m towers that look beautiful. Each tower must have at least one block and all blocks must be used.

Input

The input consists of multiple test cases. The first line contains an integer t (1 ≤ t ≤ 1000) — the number of test cases.

The first line of each test case contains three integers n, m, and x (1 ≤ m ≤ n ≤ 10^5; 1 ≤ x ≤ 10^4) — the number of blocks, the number of towers to build, and the maximum acceptable height difference of any two towers, respectively. 

The second line of each test case contains n space-separated integers (1 ≤ h_i ≤ x) — the heights of the blocks. 

It is guaranteed that the sum of n over all the test cases will not exceed 10^5.

Output

For each test case, if Phoenix cannot build m towers that look beautiful, print NO. Otherwise, print YES, followed by n integers y_1, y_2, ..., y_n, where y_i (1 ≤ y_i ≤ m) is the index of the tower that the i-th block is placed in.

If there are multiple solutions, print any of them.

Example

Input


2
5 2 3
1 2 3 1 2
4 3 3
1 1 2 3


Output


YES
1 1 1 2 2
YES
1 2 2 3

Note

In the first test case, the first tower has height 1+2+3=6 and the second tower has height 1+2=3. Their difference is 6-3=3 which doesn't exceed x=3, so the towers are beautiful.

In the second test case, the first tower has height 1, the second tower has height 1+2=3, and the third tower has height 3. The maximum height difference of any two towers is 3-1=2 which doesn't exceed x=3, so the towers are beautiful.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
testing = len(sys.argv) == 4 and sys.argv[3] == \"myTest\"
if testing:
    cmd = sys.stdout
    from time import time
    start_time = int(round(time() * 1000)) 
    readAll = open(sys.argv[1], 'r').read
    sys.stdout = open(sys.argv[2], 'w')
else:
    readAll = sys.stdin.read

# ############ ---- I/O Functions ---- ############

flush = sys.stdout.flush
class InputData:
    def __init__(self):
        self.lines = readAll().split('\n')
        self.n = len(self.lines)
        self.ii = -1
    def input(self):
        self.ii += 1
        assert self.ii < self.n
        return self.lines[self.ii]
inputData = InputData()
input = inputData.input

def intin():
    return(int(input()))
def intlin():
    return(list(map(int,input().split())))
def chrin():
    return(list(input()))
def strin():
    return input()
def lout(l, sep=\"\n\", toStr=True):
    print(sep.join(map(str, l) if toStr else l))
def dout(*args, **kargs):
    if not testing: return
    if args: print(args[0] if len(args)==1 else args)
    if kargs: print([(k,v) for k,v in kargs.items()])
    
# ############ ---- I/O Functions ---- ############

# from math import ceil
# from collections import defaultdict as ddict, Counter
from heapq import *
# from Queue import Queue

def main():
    n,m,x = intlin()
    h = intlin()
    if n==m:
        return range(1,n+1)
    # h.sort()
    ans = [0]*n
    q = [(0,i) for i in xrange(1,m+1)]
    heapify(q)
    for i in xrange(n):
        # dout(q)
        s,j = heappop(q)
        ans[i] = j
        s += h[i]
        heappush(q,(s,j))
    return ans



anss = []
for _ in xrange(intin()):
    ans = main()
    anss.append(\"YES\" if ans else \"NO\")
    if ans:
        anss.append(' '.join(map(str,ans)))
lout(anss)

if testing:
    sys.stdout = cmd
    print(int(round(time() * 1000))  - start_time)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from heapq import heapify, heappop, heappush
import re

class Cphoenixandtowersbootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_m=5, max_x=10):
        self.max_n = max_n      # 最大块数
        self.max_m = max_m      # 最大塔数
        self.max_x = max_x      # 单块最大高度
    
    def case_generator(self):
        """生成保证有解的问题实例"""
        # 先决定测试案例类型
        m = random.randint(1, self.max_m)
        n_options = [
            m,  # 每个塔一个块的简单情形
            random.randint(m+1, self.max_n)  # 常规情形
        ]
        n = random.choice(n_options)
        x = random.randint(1, self.max_x)
        
        # 生成保证有解的块列表
        h = [random.randint(1, x) for _ in range(n)]
        
        # 使用贪心算法生成解（数学保证有效性）
        solution = self._solve_phoenix(n, m, x, h)
        
        return {
            'n': n,
            'm': m,
            'x': x,
            'blocks': h,
            'solution': solution
        }

    def _solve_phoenix(self, n, m, x, h):
        """贪心算法求解器"""
        if n == m:
            return list(range(1, m+1))
        
        ans = [0] * n
        heap = [(0, i+1) for i in range(m)]
        heapify(heap)
        
        for i in range(n):
            s, j = heappop(heap)
            ans[i] = j
            heappush(heap, (s + h[i], j))
        
        return ans

    @staticmethod
    def prompt_func(case) -> str:
        """生成标准问题描述"""
        return (
            "Cphoenixandtowers has {n} blocks with heights {blocks} (each ≤{x}).\n"
            "Build exactly {m} towers where:\n"
            "1. Each tower has ≥1 block\n"
            "2. Max height difference between any two towers ≤{x}\n"
            "Output format:\n"
            "[answer]\n"
            "YES\n"
            "y₁ y₂ ... yₙ (tower indices) OR NO\n"
            "[/answer]"
        ).format(**case)

    @staticmethod
    def extract_output(text):
        """精确答案提取"""
        answer_blocks = re.findall(
            r'\[answer\](.*?)\[/answer\]', 
            text, 
            re.DOTALL
        )
        
        if not answer_blocks:
            return None
            
        last_answer = answer_blocks[-1].strip()
        lines = [l.strip() for l in last_answer.split('\n') if l.strip()]
        
        if not lines:
            return None
            
        if lines[0].upper() == 'NO':
            return 'NO'
            
        if len(lines) > 1 and lines[0].upper() == 'YES':
            try:
                return list(map(int, lines[1].split()))
            except ValueError:
                pass
        return None

    @classmethod
    def _verify_correction(cls, solution, case):
        """严格答案验证"""
        # 所有生成案例都有解，NO回答直接判错
        if isinstance(solution, str) or solution == 'NO':
            return False
            
        n, m, x = case['n'], case['m'], case['x']
        blocks = case['blocks']
        
        # 基本格式检查
        if len(solution) != n:
            return False
        if any(not (1 <= y <= m) for y in solution):
            return False
            
        # 检查每个塔至少一个块
        towers = set(solution)
        if len(towers) != m:
            return False
            
        # 计算各塔高度
        height = [0] * (m+1)  # 1-based索引
        for h, y in zip(blocks, solution):
            height[y] += h
            
        # 检查高度差
        existing_heights = [h for h in height[1:] if h > 0]
        return max(existing_heights) - min(existing_heights) <= x
