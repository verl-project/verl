"""# 

### 谜题描述
Vasya likes taking part in Codeforces contests. When a round is over, Vasya follows all submissions in the system testing tab.

There are n solutions, the i-th of them should be tested on a_i tests, testing one solution on one test takes 1 second. The solutions are judged in the order from 1 to n. There are k testing processes which test solutions simultaneously. Each of them can test at most one solution at a time.

At any time moment t when some testing process is not judging any solution, it takes the first solution from the queue and tests it on each test in increasing order of the test ids. Let this solution have id i, then it is being tested on the first test from time moment t till time moment t + 1, then on the second test till time moment t + 2 and so on. This solution is fully tested at time moment t + a_i, and after that the testing process immediately starts testing another solution.

Consider some time moment, let there be exactly m fully tested solutions by this moment. There is a caption \"System testing: d%\" on the page with solutions, where d is calculated as

$$$d = round\left(100⋅m/n\right),$$$

where round(x) = ⌊{x + 0.5}⌋ is a function which maps every real to the nearest integer.

Vasya calls a submission interesting if there is a time moment (possibly, non-integer) when the solution is being tested on some test q, and the caption says \"System testing: q%\". Find the number of interesting solutions.

Please note that in case when multiple processes attempt to take the first submission from the queue at the same moment (for instance, at the initial moment), the order they take the solutions does not matter.

Input

The first line contains two positive integers n and k (1 ≤ n ≤ 1000, 1 ≤ k ≤ 100) standing for the number of submissions and the number of testing processes respectively.

The second line contains n positive integers a_1, a_2, …, a_n (1 ≤ a_i ≤ 150), where a_i is equal to the number of tests the i-th submission is to be run on.

Output

Output the only integer — the number of interesting submissions.

Examples

Input


2 2
49 100


Output


1


Input


4 2
32 100 33 1


Output


2


Input


14 5
48 19 6 9 50 20 3 42 38 43 36 21 44 6


Output


5

Note

Consider the first example. At time moment 0 both solutions start testing. At time moment 49 the first solution is fully tested, so at time moment 49.5 the second solution is being tested on the test 50, and the caption says \"System testing: 50%\" (because there is one fully tested solution out of two). So, the second solution is interesting.

Consider the second example. At time moment 0 the first and the second solutions start testing. At time moment 32 the first solution is fully tested, the third solution starts testing, the caption says \"System testing: 25%\". At time moment 32 + 24.5 = 56.5 the third solutions is being tested on test 25, the caption is still the same, thus this solution is interesting. After that the third solution is fully tested at time moment 32 + 33 = 65, the fourth solution is fully tested at time moment 65 + 1 = 66. The captions becomes \"System testing: 75%\", and at time moment 74.5 the second solution is being tested on test 75. So, this solution is also interesting. Overall, there are two interesting solutions.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys,math
from collections import defaultdict
from io import BytesIO

sys.stdin = BytesIO(sys.stdin.read())
input = lambda: sys.stdin.readline().rstrip('\r\n')
#n = int(input())
n,k = map(int, input().split(' '))
tsk = [int(x) for x in input().split(' ')]

#arrs = [int(x) for x in input().split(' ')]
#arrk = [int(x)-1 for x in input().split(' ')]
sr = set()
rou = 0
fin = 0
cur = 0
res = 0
d = 0
arr = [[0,0,0]] * k
x = set([_ for _ in range(k)])
while cur < n or len(x) < k:
    while cur < n and len(x):
        a = x.pop()
        arr[a] = [0,tsk[cur],cur]
        cur += 1
    fl = 1
    #print(x,cur,arr)
    while fl:
        rou += 1
        for i in range(len(arr)):
            if arr[i][0] == d - 1 and d <= 100 and arr[i][0] < arr[i][1]:
                sr.add(arr[i][2])
        for i in range(len(arr)):
            arr[i][0] += 1
            if arr[i][0] == arr[i][1]:
                x.add(i)
                fin += 1
                d = (100 * fin + n * 0.5) // n
                fl = 0

print(len(sr))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import math
from collections import deque

class Csystemtestingbootcamp(Basebootcamp):
    def __init__(self, n_range=(1, 1000), k_range=(1, 100), a_range=(1, 150)):
        self.n_range = (max(1, n_range[0]), min(1000, n_range[1]))
        self.k_range = (max(1, k_range[0]), min(100, k_range[1]))
        self.a_range = (max(1, a_range[0]), min(150, a_range[1]))
    
    def case_generator(self):
        n = random.randint(*self.n_range)
        k = random.randint(*self.k_range)
        a = [random.randint(*self.a_range) for _ in range(n)]
        return {'n': n, 'k': k, 'a': a}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k = question_case['k']
        a_str = ' '.join(map(str, question_case['a']))
        return f"""编程竞赛系统测试中有{n}个提交，第i个提交需要测试a_i次，使用{k}个并行测试进程。当系统显示d%时(d=round(100*m/{n})，m是已完成数)，若某提交正在执行第q个测试时出现q=d，则该提交为有趣。\n输入：\n{n} {k}\n{a_str}\n答案放在[answer][/answer]中。"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except (ValueError, TypeError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == cls.calculate_interesting(**identity)

    @staticmethod
    def calculate_interesting(n, k, a):
        # 正确实现参考原题解法的修正版本
        processes = [{'current': 0, 'total': 0, 'index': -1} for _ in range(k)]
        queue = deque(range(n))
        completed = 0
        interesting = set()
        d = 0
        
        # 初始分配任务
        for proc in processes:
            if queue:
                sub_id = queue.popleft()
                proc['current'] = 0
                proc['total'] = a[sub_id]
                proc['index'] = sub_id
        
        while True:
            # 检查当前所有进程是否满足current_step == d-1
            for proc in processes:
                if proc['index'] == -1:
                    continue
                if d <= 100 and proc['current'] == d - 1 and proc['current'] < proc['total']:
                    interesting.add(proc['index'])
            
            # 推进所有进程
            has_completion = False
            for proc in processes:
                if proc['index'] != -1:
                    proc['current'] += 1
                    if proc['current'] == proc['total']:
                        completed += 1
                        proc['index'] = -1
                        has_completion = True
            
            # 更新d值
            if has_completion:
                d = int((100 * completed + 0.5 * n) // n)
                # 重新分配任务
                for proc in processes:
                    if proc['index'] == -1 and queue:
                        sub_id = queue.popleft()
                        proc['current'] = 0
                        proc['total'] = a[sub_id]
                        proc['index'] = sub_id
            else:
                # 检查是否所有进程空闲且队列为空
                if all(p['index'] == -1 for p in processes) and not queue:
                    break
        
        return len(interesting)
