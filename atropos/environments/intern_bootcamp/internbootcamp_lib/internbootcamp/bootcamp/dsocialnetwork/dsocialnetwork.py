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
#include <bits/stdc++.h>
using namespace std;
const int MAXN = 20020;
const int MAXT = 86400;
int n, m, t, a[MAXN], ans[MAXN], r, real_m, cnt[MAXN], last[MAXN];
queue<pair<int, int> > q;
set<pair<int, int> > cur_s;
int main() {
  cin >> n >> m >> t;
  for (int i = 0; i < n; ++i) {
    int hh, mm, ss;
    scanf(\"%d:%d:%d\", &hh, &mm, &ss);
    int s = hh * 60 * 60 + mm * 60 + ss;
    while (!q.empty() && q.front().first <= s) {
      int id = q.front().second;
      q.pop();
      --cnt[id];
      if (cnt[id] == 0) cur_s.erase(make_pair(last[id], id));
    }
    if (cur_s.size() >= m)
      ans[i] = cur_s.rbegin()->second;
    else
      ans[i] = ++r;
    q.push(make_pair(s + t, ans[i]));
    ++cnt[ans[i]];
    if (cnt[ans[i]] != 1) cur_s.erase(make_pair(last[ans[i]], ans[i]));
    last[ans[i]] = s;
    cur_s.insert(make_pair(last[ans[i]], ans[i]));
    real_m = max(real_m, (int)cur_s.size());
  }
  if (real_m < m)
    cout << \"No solution\" << endl;
  else {
    cout << r << endl;
    for (int i = 0; i < n; ++i) cout << ans[i] << endl;
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
from collections import deque, defaultdict
import random
import re
import bisect

class Dsocialnetworkbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = params
        self.default_params = {
            'min_n': 4,
            'max_n': 10,
            'min_M': 2,
            'max_M': 5,
            'min_T': 10,
            'max_T': 100,
        }
        self.default_params.update(params)
    
    def case_generator(self):
        def _solve_case(times, M, T):
            n = len(times)
            times_sec = [self._time_str_to_seconds(t) for t in times]
            ans = [0] * n
            r = 0
            real_m = 0
            cnt = defaultdict(int)
            last = defaultdict(int)
            cur_s = []  # Maintained as sorted list (descending by last time)
            q = deque()

            for i in range(n):
                s = times_sec[i]
                # Remove expired users
                while q and q[0][0] <= s:
                    expire_time, user_id = q.popleft()
                    cnt[user_id] -= 1
                    if cnt[user_id] == 0:
                        # Find and remove from cur_s
                        index = bisect.bisect_left(cur_s, (-last[user_id], -user_id))
                        if index < len(cur_s) and cur_s[index] == (-last[user_id], -user_id):
                            del cur_s[index]

                # Assign user ID
                if len(cur_s) >= M:
                    # Select user with largest last time (first in sorted cur_s)
                    selected_user = -cur_s[0][1]
                    ans[i] = selected_user
                else:
                    r += 1
                    ans[i] = r

                # Update user's expiration and counters
                user_id = ans[i]
                expire_time = s + T
                q.append((expire_time, user_id))
                cnt[user_id] += 1

                # Update last and cur_s
                prev_last = last.get(user_id, 0)
                if prev_last != 0:
                    # Remove previous entry
                    prev_entry = (-prev_last, -user_id)
                    index = bisect.bisect_left(cur_s, prev_entry)
                    if index < len(cur_s) and cur_s[index] == prev_entry:
                        del cur_s[index]
                last[user_id] = s
                new_entry = (-s, -user_id)  # Use negative for descending sort
                bisect.insort(cur_s, new_entry)

                # Update real_m
                current_online = len(cur_s)
                if current_online > real_m:
                    real_m = current_online

            # Check if reached M
            if real_m >= M:
                return r, ans
            else:
                return None, None

        max_attempts = 100
        for _ in range(max_attempts):
            # Generate parameters with M <= possible maximum users
            n = random.randint(self.default_params['min_n'], self.default_params['max_n'])
            max_possible_M = min(n, self.default_params['max_M'])
            M = random.randint(self.default_params['min_M'], max_possible_M)
            T = random.randint(self.default_params['min_T'], self.default_params['max_T'])

            # Generate overlapping times to increase valid cases
            base_time = random.randint(0, 86400 - T)
            times_sec = [base_time + random.randint(0, T//2) for _ in range(n//2)]
            # Add some non-overlapping times
            if n > len(times_sec):
                non_overlap_start = base_time + T + random.randint(1, 100)
                times_sec.extend([non_overlap_start + i*T for i in range(n - len(times_sec))])
            times_sec = sorted(times_sec)
            # Trim times to 86400 - T
            times_sec = [min(t, 86400 - T - 1) for t in times_sec]

            # Format times
            times = []
            for s in times_sec:
                hh, rem = divmod(s, 3600)
                mm, ss = divmod(rem, 60)
                times.append(f"{hh:02d}:{mm:02d}:{ss:02d}")

            # Solve case
            r, ans = _solve_case(times, M, T)
            if r is not None:
                return {
                    'times': times,
                    'M': M,
                    'T': T,
                    'correct_r': r,
                    'correct_ans': ans
                }
        # Fallback example
        return {
            'times': ['17:05:53', '17:05:58', '17:06:01', '22:39:47'],
            'M': 2,
            'T': 10,
            'correct_r': 3,
            'correct_ans': [1, 2, 2, 3]
        }

    @staticmethod
    def prompt_func(question_case):
        case = question_case
        times_str = '\n'.join(case['times'])
        n = len(case['times'])
        M = case['M']
        T = case['T']
        return f"""你是某社交网络的实习生，需要确定在24小时内访问网络的唯一用户最大数目。给定{n}个请求时间，每个请求后用户在线{T}秒。已知该日同时在线用户数达到{M}。请分配用户ID满足：
1. 任何时刻在线用户数≤{M}。
2. 至少有一个时刻在线用户数恰为{M}。
3. 用户总数尽可能多。

输入：
n = {n}
M = {M}
T = {T}

请求时间：
{times_str}

输出要求：
- 首行为用户总数R，随后{n}行为各请求的用户ID（1~R），无解输出"No solution"。

将答案置于[answer]和[/answer]之间。例如：
[answer]
3
1
2
2
3
[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        content = matches[-1].strip()
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return None
        if lines[0].lower() == 'no solution':
            return 'No solution'
        try:
            R = int(lines[0])
            user_ids = list(map(int, lines[1:]))
            return {'R': R, 'user_ids': user_ids}
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if identity.get('correct_r') is None:
            return solution == 'No solution'
        if solution == 'No solution':
            return False
        if not isinstance(solution, dict):
            return False
        user_r = solution['R']
        user_ids = solution['user_ids']
        n = len(identity['times'])
        M = identity['M']
        T = identity['T']
        if len(user_ids) != n:
            return False

        # Verify user ID consistency
        unique_ids = set(user_ids)
        if len(unique_ids) != user_r or min(unique_ids) < 1 or max(unique_ids) > user_r:
            return False

        # Convert times to seconds
        times_sec = [cls._time_str_to_seconds(t) for t in identity['times']]

        # Simulate online periods
        events = []
        for uid, s in zip(user_ids, times_sec):
            start = s
            end = s + T - 1
            events.append((start, 'login', uid))
            events.append((end + 1, 'logout', uid))  # Event after online period

        events.sort(key=lambda x: (x[0], x[1] == 'logout'))

        current_online = set()
        max_online = 0
        reached_M = False
        for time, action, uid in events:
            if action == 'login':
                current_online.add(uid)
            else:
                current_online.discard(uid)
            
            current_count = len(current_online)
            if current_count > M:
                return False
            if current_count > max_online:
                max_online = current_count
            if current_count == M:
                reached_M = True

        return reached_M and max_online >= M

    @staticmethod
    def _time_str_to_seconds(time_str):
        hh, mm, ss = map(int, time_str.split(':'))
        return hh * 3600 + mm * 60 + ss
