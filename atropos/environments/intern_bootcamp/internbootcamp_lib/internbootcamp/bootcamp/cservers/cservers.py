"""# 

### 谜题描述
There are n servers in a laboratory, each of them can perform tasks. Each server has a unique id — integer from 1 to n.

It is known that during the day q tasks will come, the i-th of them is characterized with three integers: ti — the moment in seconds in which the task will come, ki — the number of servers needed to perform it, and di — the time needed to perform this task in seconds. All ti are distinct.

To perform the i-th task you need ki servers which are unoccupied in the second ti. After the servers begin to perform the task, each of them will be busy over the next di seconds. Thus, they will be busy in seconds ti, ti + 1, ..., ti + di - 1. For performing the task, ki servers with the smallest ids will be chosen from all the unoccupied servers. If in the second ti there are not enough unoccupied servers, the task is ignored.

Write the program that determines which tasks will be performed and which will be ignored.

Input

The first line contains two positive integers n and q (1 ≤ n ≤ 100, 1 ≤ q ≤ 105) — the number of servers and the number of tasks. 

Next q lines contains three integers each, the i-th line contains integers ti, ki and di (1 ≤ ti ≤ 106, 1 ≤ ki ≤ n, 1 ≤ di ≤ 1000) — the moment in seconds in which the i-th task will come, the number of servers needed to perform it, and the time needed to perform this task in seconds. The tasks are given in a chronological order and they will come in distinct seconds. 

Output

Print q lines. If the i-th task will be performed by the servers, print in the i-th line the sum of servers' ids on which this task will be performed. Otherwise, print -1.

Examples

Input

4 3
1 3 2
2 2 1
3 4 3


Output

6
-1
10


Input

3 2
3 2 3
5 1 2


Output

3
3


Input

8 6
1 3 20
4 2 1
6 5 5
10 1 1
15 3 6
21 8 8


Output

6
9
30
-1
15
36

Note

In the first example in the second 1 the first task will come, it will be performed on the servers with ids 1, 2 and 3 (the sum of the ids equals 6) during two seconds. In the second 2 the second task will come, it will be ignored, because only the server 4 will be unoccupied at that second. In the second 3 the third task will come. By this time, servers with the ids 1, 2 and 3 will be unoccupied again, so the third task will be done on all the servers with the ids 1, 2, 3 and 4 (the sum of the ids is 10).

In the second example in the second 3 the first task will come, it will be performed on the servers with ids 1 and 2 (the sum of the ids is 3) during three seconds. In the second 5 the second task will come, it will be performed on the server 3, because the first two servers will be busy performing the first task.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from itertools import imap

def ri():
    return imap(int, raw_input().split())

n, q = ri()

s = [0 for i in xrange(n+1)]
op = [-1]
ss = s[:]
for i in xrange(1,q + 1):
    t, k, d = ri()
    opp = 0
    for j in xrange(1, n+1):
        if k == 0:
            break
        if ss[j] < t:
            ss[j] = t+d-1
            k -= 1
            opp += j

    if k == 0:
        s = ss[:]
        op.append(opp)
    else:
        ss = s[:]
        op.append(-1)

for i in xrange(1, q+1):
    print op[i]
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cserversbootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=100, q_min=1, q_max=10, t_min=1, t_max=10**6, d_min=1, d_max=1000):
        # 参数合法性校验
        if n_min < 1 or n_max > 100 or q_min < 1 or q_max > 10**5 or t_min < 1:
            raise ValueError("Invalid parameter constraints")
        
        self.n_min = max(n_min, 1)
        self.n_max = min(n_max, 100)
        self.q_min = max(q_min, 1)
        self.q_max = min(q_max, 10**5)
        self.t_min = max(t_min, 1)
        self.t_max = max(t_max, self.t_min + 1)
        self.d_min = max(d_min, 1)
        self.d_max = min(d_max, 1000)

    def case_generator(self):
        """改进后的测试用例生成逻辑"""
        n = random.randint(self.n_min, self.n_max)
        
        # 参数合法性二次校验
        max_possible_ti = self.t_max - self.t_min + 1
        q_min = max(self.q_min, 1)
        q_max = min(self.q_max, max_possible_ti)
        q_max = max(q_max, q_min)  # 确保q_max >= q_min
        
        q = random.randint(q_min, q_max) if q_max >= q_min else 0
        if q == 0:
            return {'n': n, 'q': 0, 'tasks': []}
        
        # 使用数学方法生成有序唯一时间序列
        ti_step = (self.t_max - self.t_min) // max(q, 1)
        ti_list = sorted(random.sample(
            range(self.t_min, self.t_max + 1), 
            k=q
        ))
        
        tasks = []
        for ti in ti_list:
            ki = random.randint(1, n)
            di = random.randint(self.d_min, self.d_max)
            tasks.append({'t': ti, 'ki': ki, 'di': di})
        
        return {'n': n, 'q': q, 'tasks': tasks}

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        q = question_case['q']
        tasks = question_case['tasks']
        task_desc = '\n'.join(
            f"任务{i+1}：到达时间t={task['t']}秒，需要k={task['ki']}台服务器，执行时间d={task['di']}秒。"
            for i, task in enumerate(tasks)
        )
        return f"""实验室服务器调度系统需要处理{q}个任务，当前可用服务器共{n}台（编号1~{n}）。每个任务需要立即分配指定数量的空闲服务器，若不足则任务失败。

任务规则详解：
1. 时间离散性：时间按整秒计算，服务器状态在每个整秒瞬间更新
2. 抢占规则：任务在t秒到达时立即检查服务器状态，选择最小的可用ID
3. 占用时段：服务器将从t秒开始持续占用di秒，即t ≤ s ≤ t+di-1
4. 输出要求：成功时输出所选服务器ID之和，失败输出-1

任务序列（按到达时间排序）：
{task_desc}

请严格按照时间顺序处理任务，输出{q}个结果，每个结果独占一行，包裹在[answer]标签中。例如：
[answer]
15
-1
[/answer]"""

    @staticmethod
    def extract_output(output):
        # 增加对多格式错误的容错处理
        answer_blocks = re.findall(r'(?i)\[answer\]\s*(.*?)\s*\[\/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        
        answers = []
        for line in answer_blocks[-1].split('\n'):
            line = line.strip()
            if not line: 
                continue
            if line.startswith('-') and line[1:].isdigit():
                answers.append(int(line))
            elif line.isdigit() or (len(line) > 1 and line[0] == '-' and line[1:].isdigit()):
                answers.append(int(line))
        return answers if len(answers) == len(answer_blocks[-1].split('\n')) else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 增强验证逻辑的鲁棒性
        if not identity['tasks']:
            return solution == [] if identity['q'] ==0 else False
        
        if not solution or len(solution) != identity['q']:
            return False
        
        server_pool = {i:0 for i in range(1, identity['n']+1)}
        for idx, task in enumerate(identity['tasks']):
            t = task['t']
            k = task['ki']
            d = task['di']
            available = sorted([sid for sid, end in server_pool.items() if end < t])
            
            if len(available) < k:
                if solution[idx] != -1:
                    return False
                continue
                
            selected = available[:k]
            if sum(selected) != solution[idx]:
                return False
                
            new_end = t + d -1
            for sid in selected:
                server_pool[sid] = new_end
        
        return True
