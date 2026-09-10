"""# 

### 谜题描述
Given an array a of length n, find another array, b, of length n such that:

  * for each i (1 ≤ i ≤ n) MEX(\\{b_1, b_2, …, b_i\})=a_i. 



The MEX of a set of integers is the smallest non-negative integer that doesn't belong to this set.

If such array doesn't exist, determine this.

Input

The first line contains an integer n (1 ≤ n ≤ 10^5) — the length of the array a.

The second line contains n integers a_1, a_2, …, a_n (0 ≤ a_i ≤ i) — the elements of the array a. It's guaranteed that a_i ≤ a_{i+1} for 1≤ i < n.

Output

If there's no such array, print a single line containing -1.

Otherwise, print a single line containing n integers b_1, b_2, …, b_n (0 ≤ b_i ≤ 10^6)

If there are multiple answers, print any.

Examples

Input


3
1 2 3


Output


0 1 2 

Input


4
0 0 0 2


Output


1 3 4 0 

Input


3
1 1 3


Output


0 2 1 

Note

In the second test case, other answers like [1,1,1,0], for example, are valid.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
if sys.subversion[0] == \"PyPy\":
    import io, atexit
    sys.stdout = io.BytesIO()
    atexit.register(lambda: sys.__stdout__.write(sys.stdout.getvalue()))
    
    sys.stdin = io.BytesIO(sys.stdin.read())
    raw_input = lambda: sys.stdin.readline().rstrip()

RS = raw_input
RI = lambda : map(int,RS().split())
RN = lambda : int(RS())
''' ...................................................................... '''
import heapq as hp

def solve():
    n = RN()
    arr = RI()

    for i in xrange(n):
        if arr[i]>i+1:
            print -1
            return

    brr = [-1]*n
    left = []

    for i in xrange(n-1,0,-1):        
        x = arr[i-1]
        y = arr[i]

        for j in xrange(x,y):
            hp.heappush(left,j)

        if left == []:
            brr[i] = 10**6
        else:
            brr[i] = hp.heappop(left)

    for j in xrange(0,arr[0]):
        hp.heappush(left,j)

    if left == []:
        brr[0] = 10**6
    else:
        brr[0] = hp.heappop(left)

    if left != []:
        print -1
        return

    print ' '.join(map(str,brr))

solve()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import heapq
import random
from collections import deque
from bootcamp import Basebootcamp

def calculate_mex(arr):
    s = set(arr)
    mex = 0
    while mex in s:
        mex += 1
    return mex

def solve_case(n, a):
    for i in range(n):
        if a[i] > i + 1:
            return -1
    
    brr = [-1] * n
    heap = []
    
    for i in range(n-1, 0, -1):
        x = a[i-1]
        y = a[i]
        for num in range(x, y):
            heapq.heappush(heap, num)
        
        if heap:
            brr[i] = heapq.heappop(heap)
        else:
            brr[i] = 10**6
    
    x = a[0]
    for num in range(x):
        heapq.heappush(heap, num)
    
    if heap:
        brr[0] = heapq.heappop(heap)
    else:
        brr[0] = 10**6
    
    if heap:
        return -1
    return brr

class Cehabandprefixmexsbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10, allow_unsolvable=True):
        self.min_n = min_n
        self.max_n = max_n
        self.allow_unsolvable = allow_unsolvable
    
    def case_generator(self):
        # 生成合法且有解的案例
        n = random.randint(self.min_n, self.max_n)
        
        # 尝试生成有解的案例
        for _ in range(100):
            b = [random.randint(0, 10**6) for _ in range(n)]
            a = []
            prev_mex = 0
            for i in range(1, n+1):
                current = b[:i]
                mex = calculate_mex(current)
                a.append(mex)
                prev_mex = mex
            
            # 检查是否符合输入约束
            valid = all(a_i <= i for i, a_i in zip(range(1,n+1), a)) 
            valid &= all(a[i] >= a[i-1] for i in range(1,n))
            
            if valid:
                return {'n': n, 'a': a, 'solution': b}
        
        # 生成无解案例 (当允许时)
        if self.allow_unsolvable:
            while True:
                n = random.randint(self.min_n, self.max_n)
                a = []
                prev = 0
                for i in range(1, n+1):
                    lower = prev
                    upper = i
                    if random.random() < 0.2 and i > 1:
                        # 故意创建矛盾条件
                        upper = min(upper, prev + 2)
                    a_i = random.randint(lower, upper)
                    a.append(a_i)
                    prev = a_i
                
                if all(a[i] <= (i+1) for i in range(len(a))):
                    return {'n': n, 'a': a}
        
        # 保底返回简单案例
        return {'n': 3, 'a': [1, 2, 3], 'solution': [0, 1, 2]}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = question_case['a']
        return (
            f"Given an array a of length {n} where a = [{', '.join(map(str,a))}],\n"
            "find an array b such that for each 1 ≤ i ≤ n, the Cehabandprefixmexs of the first i elements of b equals a_i.\n"
            "The Cehabandprefixmexs is the smallest non-negative integer not present in the set.\n"
            "If impossible, output -1. Put your final answer between [answer] and [/answer]."
        )
    
    @staticmethod
    def extract_output(output):
        import re
        answers = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not answers:
            return None
        last = answers[-1].strip()
        try:
            if last == '-1':
                return -1
            return list(map(int, last.split()))
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution == -1:
            # 需要严格验证无解案例
            result = solve_case(identity['n'], identity['a'])
            return result == -1
        else:
            if not isinstance(solution, list) or len(solution) != identity['n']:
                return False
            for i in range(1, identity['n']+1):
                current = solution[:i]
                if calculate_mex(current) != identity['a'][i-1]:
                    return False
            return True
