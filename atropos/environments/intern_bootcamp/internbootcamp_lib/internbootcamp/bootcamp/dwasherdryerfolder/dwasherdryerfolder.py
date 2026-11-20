"""# 

### 谜题描述
You have k pieces of laundry, each of which you want to wash, dry and fold. You are at a laundromat that has n1 washing machines, n2 drying machines and n3 folding machines. Each machine can process only one piece of laundry at a time. You can't dry a piece of laundry before it is washed, and you can't fold it before it is dried. Moreover, after a piece of laundry is washed, it needs to be immediately moved into a drying machine, and after it is dried, it needs to be immediately moved into a folding machine.

It takes t1 minutes to wash one piece of laundry in a washing machine, t2 minutes to dry it in a drying machine, and t3 minutes to fold it in a folding machine. Find the smallest number of minutes that is enough to wash, dry and fold all the laundry you have.

Input

The only line of the input contains seven integers: k, n1, n2, n3, t1, t2, t3 (1 ≤ k ≤ 104; 1 ≤ n1, n2, n3, t1, t2, t3 ≤ 1000).

Output

Print one integer — smallest number of minutes to do all your laundry.

Examples

Input

1 1 1 1 5 5 5


Output

15


Input

8 4 3 2 10 5 2


Output

32

Note

In the first example there's one instance of each machine, each taking 5 minutes to complete. You have only one piece of laundry, so it takes 15 minutes to process it.

In the second example you start washing first two pieces at moment 0. If you start the third piece of laundry immediately, then by the time it is dried, there will be no folding machine available, so you have to wait, and start washing third piece at moment 2. Similarly, you can't start washing next piece until moment 5, since otherwise there will be no dryer available, when it is washed. Start time for each of the eight pieces of laundry is 0, 0, 2, 5, 10, 10, 12 and 15 minutes respectively. The last piece of laundry will be ready after 15 + 10 + 5 + 2 = 32 minutes.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
# while True:
#     try:
#         # n = int(raw_input())

#     except (EOFError):
#         break
# b = map(int, s2.split())
import heapq
s1 = raw_input()
knt = map(int, s1.split(' '))
k = knt[0];
n = []
for i in xrange(1, 4):
    n.append(knt[i])
t = []
for i in xrange(4, 7):
    t.append(knt[i])
time = [[] for i in xrange(3)]
while n[0] > 0:
    time[0].append(0)
    n[0] -= 1
while n[1] > 0:
    time[1].append(t[0])
    n[1] -= 1
while n[2] > 0:
    time[2].append(t[0] + t[1])
    n[2] -= 1
while k > 0:
    temp1 = min(time[0])
    time[0].remove(temp1)
    temp2 = min(time[1])
    time[1].remove(temp2)
    temp3 = min(time[2])
    time[2].remove(temp3)
    re = max(temp1, temp2 - t[0], temp3 - t[0] - t[1])
    time[0].append(re + t[0])
    time[1].append(re + t[0] + t[1])
    time[2].append(re + t[0] + t[1] + t[2])
    k -= 1
print re + sum(t)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Dwasherdryerfolderbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__(**params)  # 调用父类构造函数
        self.k_range = (
            params.get('k_min', 1),
            params.get('k_max', 10**4)
        )
        self.n_range = (params.get('n_min', 1), params.get('n_max', 1000))
        self.t_range = (params.get('t_min', 1), params.get('t_max', 1000))
    
    def case_generator(self):
        k = random.randint(*self.k_range)
        n1 = random.randint(*self.n_range)
        n2 = random.randint(*self.n_range)
        n3 = random.randint(*self.n_range)
        t1 = random.randint(*self.t_range)
        t2 = random.randint(*self.t_range)
        t3 = random.randint(*self.t_range)
        return {
            'k': k, 'n1': n1, 'n2': n2, 'n3': n3,
            't1': t1, 't2': t2, 't3': t3
        }
    
    @staticmethod
    def prompt_func(question_case):
        return f"""你需要处理{question_case['k']}件衣物的清洗流程。洗衣店有：
- {question_case['n1']}台洗衣机（每件需{question_case['t1']}分钟）
- {question_case['n2']}台烘干机（每件需{question_case['t2']}分钟）
- {question_case['n3']}台折叠机（每件需{question_case['t3']}分钟）

规则：
1. 每台机器一次只能处理一件衣物
2. 必须立即转移至下一阶段（洗→烘→折）
3. 不可中断处理流程

请计算完成所有衣物处理的最短时间，并将最终答案放在[answer]标签内，例如：[answer]32[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return int(matches[-1].strip()) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 修正初始化逻辑
        wash = [0] * identity['n1']
        dry = [0] * identity['n2']  # 正确初始化
        fold = [0] * identity['n3']  # 正确初始化
        
        for _ in range(identity['k']):
            # 计算各阶段最早开始时间
            wash_time = min(wash)
            dry_start = min(dry)
            fold_start = min(fold)
            
            # 确定实际开始时间
            start = max(wash_time, dry_start - identity['t1'], fold_start - identity['t1'] - identity['t2'])
            
            # 更新各机器时间
            wash[wash.index(min(wash))] = start + identity['t1']
            dry[dry.index(min(dry))] = start + identity['t1'] + identity['t2']
            fold[fold.index(min(fold))] = start + identity['t1'] + identity['t2'] + identity['t3']
        
        return solution == max(fold)
