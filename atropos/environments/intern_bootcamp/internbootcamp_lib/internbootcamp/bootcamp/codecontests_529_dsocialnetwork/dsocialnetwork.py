"""# 

### 谜题描述
Polycarpus got an internship in one well-known social network. His test task is to count the number of unique users who have visited a social network during the day. Polycarpus was provided with information on all user requests for this time period. For each query, we know its time... and nothing else, because Polycarpus has already accidentally removed the user IDs corresponding to the requests from the database. Thus, it is now impossible to determine whether any two requests are made by the same person or by different people.

But wait, something is still known, because that day a record was achieved — M simultaneous users online! In addition, Polycarpus believes that if a user made a request at second s, then he was online for T seconds after that, that is, at seconds s, s + 1, s + 2, ..., s + T - 1. So, the user's time online can be calculated as the union of time intervals of the form [s, s + T - 1] over all times s of requests from him.

Guided by these thoughts, Polycarpus wants to assign a user ID to each request so that:

  * the number of different users online did not exceed M at any moment, 
  * at some second the number of distinct users online reached value M, 
  * the total number of users (the number of distinct identifiers) was as much as possible. 



Help Polycarpus cope with the test.

Input

The first line contains three integers n, M and T (1 ≤ n, M ≤ 20 000, 1 ≤ T ≤ 86400) — the number of queries, the record number of online users and the time when the user was online after a query was sent. Next n lines contain the times of the queries in the format \"hh:mm:ss\", where hh are hours, mm are minutes, ss are seconds. The times of the queries follow in the non-decreasing order, some of them can coincide. It is guaranteed that all the times and even all the segments of type [s, s + T - 1] are within one 24-hour range (from 00:00:00 to 23:59:59). 

Output

In the first line print number R — the largest possible number of distinct users. The following n lines should contain the user IDs for requests in the same order in which the requests are given in the input. User IDs must be integers from 1 to R. The requests of the same user must correspond to the same identifiers, the requests of distinct users must correspond to distinct identifiers. If there are multiple solutions, print any of them. If there is no solution, print \"No solution\" (without the quotes).

Examples

Input

4 2 10
17:05:53
17:05:58
17:06:01
22:39:47


Output

3
1
2
2
3


Input

1 2 86400
00:00:00


Output

No solution

Note

Consider the first sample. The user who sent the first request was online from 17:05:53 to 17:06:02, the user who sent the second request was online from 17:05:58 to 17:06:07, the user who sent the third request, was online from 17:06:01 to 17:06:10. Thus, these IDs cannot belong to three distinct users, because in that case all these users would be online, for example, at 17:06:01. That is impossible, because M = 2. That means that some two of these queries belonged to the same user. One of the correct variants is given in the answer to the sample. For it user 1 was online from 17:05:53 to 17:06:02, user 2 — from 17:05:58 to 17:06:10 (he sent the second and third queries), user 3 — from 22:39:47 to 22:39:56.

In the second sample there is only one query. So, only one user visited the network within the 24-hour period and there couldn't be two users online on the network simultaneously. (The time the user spent online is the union of time intervals for requests, so users who didn't send requests could not be online in the network.) 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, M, T = map(int, raw_input().split())
q = []
for s in xrange(n):
    h, m, s = map(int, raw_input().split(':'))
    ft = s + m * 60 + h * 60 * 60
    q.append(ft)
    
events = []
for i, ft in enumerate(q):
    events.append((ft, 1, i))
    events.append((ft + T, 0, i))
events.sort()
last = -1
was_max = False
onlines = dict()
users = [-1] * len(q)
R = 0
for tm, event_type, query_index in events:
    if event_type == 0:
        user_id = users[query_index]
        onlines[user_id] -= 1
        if onlines[user_id] == 0:
            onlines.pop(user_id)
    else:
        if len(onlines) < M:
            R += 1
            user_id = R
        else:
            user_id = last
        users[query_index] = user_id
        onlines[user_id] = onlines.get(user_id, 0) + 1
        last = user_id
        if len(onlines) == M:
            was_max = True

#    print (tm, event_type, query_index), onlines, len(onlines), m 

if not was_max:
    print \"No solution\"
else:
    print R
    for u in users:
        print u
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from collections import defaultdict
from bootcamp import Basebootcamp

class Dsocialnetworkbootcamp(Basebootcamp):
    def __init__(self, n=5, M=2, T=10):
        if M > n:
            raise ValueError("M cannot be greater than n")
        if T < 1 or T > 86400:
            raise ValueError("T must be between 1 and 86400")
        self.n = n
        self.M = M
        self.T = T

    def case_generator(self):
        while True:
            times_sec = self._generate_times()
            times = [self._sec_to_hhmmss(sec) for sec in times_sec]
            solution = self._solve(times_sec, self.M, self.T)
            if solution is not None:
                return {
                    "n": self.n,
                    "M": self.M,
                    "T": self.T,
                    "times": times,
                }

    def _generate_times(self):
        remaining = 86400 - self.T
        times_sec = []
        current_time = random.randint(0, remaining)
        times_sec.append(current_time)
        for _ in range(self.n - 1):
            max_step = remaining - current_time
            if max_step <= 0:
                next_time = current_time
            else:
                step = random.randint(0, min(100, max_step))
                next_time = current_time + step
                if next_time > remaining:
                    next_time = remaining
            times_sec.append(next_time)
            current_time = next_time
        return times_sec

    @staticmethod
    def _sec_to_hhmmss(sec):
        h = sec // 3600
        remainder = sec % 3600
        m = remainder // 60
        s = remainder % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    @classmethod
    def _solve(cls, times_sec, M, T):
        events = []
        for i, ft in enumerate(times_sec):
            events.append((ft, 1, i))
            events.append((ft + T, 0, i))
        events.sort()
        last = -1
        was_max = False
        onlines = dict()
        users = [-1] * len(times_sec)
        R = 0
        for tm, event_type, query_index in events:
            if event_type == 0:
                user_id = users[query_index]
                onlines[user_id] -= 1
                if onlines[user_id] == 0:
                    onlines.pop(user_id)
            else:
                if len(onlines) < M:
                    R += 1
                    user_id = R
                else:
                    user_id = last
                users[query_index] = user_id
                onlines[user_id] = onlines.get(user_id, 0) + 1
                last = user_id
                if len(onlines) == M:
                    was_max = True
        return (R, users) if was_max else None

    @staticmethod
    def prompt_func(question_case):
        prompt = f"""你是一位社交网络的实习生，需要根据请求时间分配用户ID。每个请求的时间后，该用户将在线{question_case['T']}秒。规则如下：

- 任何时候同时在线的用户数不得超过{question_case['M']}。
- 必须至少有一个时刻，同时在线的用户数正好是{question_case['M']}。
- 需要分配尽可能多的不同用户ID（即最大化R）。

输入数据包括：
- 第一行是三个整数n={question_case['n']}、M={question_case['M']}、T={question_case['T']}。
- 接下来的n行是请求的时间，格式为hh:mm:ss：

"""
        for t in question_case['times']:
            prompt += f"{t}\n"
        prompt += "\n请输出你的答案。第一行是R，随后n行是用户ID，或输出“No solution”。答案放在[answer]和[/answer]之间。"
        return prompt

    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\]\s*(.*?)\s*\[/answer\]', output, re.IGNORECASE | re.DOTALL)
        if not answer_blocks:
            return None
        answer_block = answer_blocks[-1].strip()
        lines = [line.strip() for line in answer_block.split('\n') if line.strip()]
        if not lines:
            return None
        first_line = lines[0]
        if first_line.lower() == 'no solution':
            return 'No solution'
        try:
            R = int(first_line)
            user_ids = []
            for line in lines[1:]:
                user_ids.append(int(line))
            return {'R': R, 'user_ids': user_ids}
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        M = identity['M']
        T = identity['T']
        times = identity['times']
        times_sec = [sum(int(x) * 60**i for i, x in enumerate(reversed(t.split(':')))) for t in times]
        correct_solution = cls._solve(times_sec, M, T)
        if correct_solution is None:
            return solution == 'No solution'
        if solution == 'No solution':
            return False
        correct_R, _ = correct_solution
        user_R = solution.get('R', -1)
        if user_R != correct_R:
            return False
        user_ids = solution.get('user_ids', [])
        if len(user_ids) != n:
            return False
        for uid in user_ids:
            if not 1 <= uid <= user_R:
                return False
        events = []
        for i in range(n):
            s = times_sec[i]
            uid = user_ids[i]
            events.append((s, 'start', uid))
            events.append((s + T, 'end', uid))
        sorted_events = []
        for time, typ, uid in events:
            sorted_events.append((time, 0 if typ == 'end' else 1, uid))
        sorted_events.sort()
        online = defaultdict(int)
        current = set()
        max_reached = False
        for time, typ, uid in sorted_events:
            if typ == 0:
                online[uid] -= 1
                if online[uid] == 0:
                    del online[uid]
                current.discard(uid)
            else:
                online[uid] += 1
                if online[uid] == 1:
                    current.add(uid)
                if len(current) > M:
                    return False
                if len(current) == M:
                    max_reached = True
        return max_reached
