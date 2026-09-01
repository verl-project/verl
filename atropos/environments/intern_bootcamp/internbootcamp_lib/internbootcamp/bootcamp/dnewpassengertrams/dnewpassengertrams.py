"""# 

### 谜题描述
There are many freight trains departing from Kirnes planet every day. One day on that planet consists of h hours, and each hour consists of m minutes, where m is an even number. Currently, there are n freight trains, and they depart every day at the same time: i-th train departs at h_i hours and m_i minutes.

The government decided to add passenger trams as well: they plan to add a regular tram service with half-hour intervals. It means that the first tram of the day must depart at 0 hours and t minutes, where 0 ≤ t < {m \over 2}, the second tram departs m \over 2 minutes after the first one and so on. This schedule allows exactly two passenger trams per hour, which is a great improvement.

To allow passengers to board the tram safely, the tram must arrive k minutes before. During the time when passengers are boarding the tram, no freight train can depart from the planet. However, freight trains are allowed to depart at the very moment when the boarding starts, as well as at the moment when the passenger tram departs. Note that, if the first passenger tram departs at 0 hours and t minutes, where t < k, then the freight trains can not depart during the last k - t minutes of the day.

<image> A schematic picture of the correct way to run passenger trams. Here h=2 (therefore, the number of passenger trams is 2h=4), the number of freight trains is n=6. The passenger trams are marked in red (note that the spaces between them are the same). The freight trains are marked in blue. Time segments of length k before each passenger tram are highlighted in red. Note that there are no freight trains inside these segments.

Unfortunately, it might not be possible to satisfy the requirements of the government without canceling some of the freight trains. Please help the government find the optimal value of t to minimize the number of canceled freight trains in case all passenger trams depart according to schedule.

Input

The first line of input contains four integers n, h, m, k (1 ≤ n ≤ 100 000, 1 ≤ h ≤ 10^9, 2 ≤ m ≤ 10^9, m is even, 1 ≤ k ≤ {m \over 2}) — the number of freight trains per day, the number of hours and minutes on the planet, and the boarding time for each passenger tram.

n lines follow, each contains two integers h_i and m_i (0 ≤ h_i < h, 0 ≤ m_i < m) — the time when i-th freight train departs. It is guaranteed that no freight trains depart at the same time.

Output

The first line of output should contain two integers: the minimum number of trains that need to be canceled, and the optimal starting time t. Second line of output should contain freight trains that need to be canceled.

Examples

Input


2 24 60 15
16 0
17 15


Output


0 0



Input


2 24 60 16
16 0
17 15


Output


1 0
2 

Note

In the first test case of the example the first tram can depart at 0 hours and 0 minutes. Then the freight train at 16 hours and 0 minutes can depart at the same time as the passenger tram, and the freight train at 17 hours and 15 minutes can depart at the same time as the boarding starts for the upcoming passenger tram.

In the second test case of the example it is not possible to design the passenger tram schedule without cancelling any of the freight trains: if t ∈ [1, 15], then the freight train at 16 hours and 0 minutes is not able to depart (since boarding time is 16 minutes). If t = 0 or t ∈ [16, 29], then the freight train departing at 17 hours 15 minutes is not able to depart. However, if the second freight train is canceled, one can choose t = 0. Another possible option is to cancel the first train and choose t = 13.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
struct node {
  int x, t, id;
} a[200002];
int n, h, m, k, i, j;
int read() {
  char c = getchar();
  int w = 0;
  while (c < '0' || c > '9') c = getchar();
  while (c <= '9' && c >= '0') {
    w = w * 10 + c - '0';
    c = getchar();
  }
  return w;
}
int cmp1(const node &a, const node &b) {
  if (a.x == b.x) return a.t < b.t;
  return a.x < b.x;
}
int cmp2(const node &a, const node &b) { return a.t < b.t; }
int main() {
  n = read();
  h = read();
  m = read();
  k = read();
  for (i = 1; i <= n; i++) {
    a[i].x = read();
    a[i].t = read() % (m / 2);
    a[i].id = i;
  }
  sort(a + 1, a + n + 1, cmp2);
  for (i = 1; i <= n; i++) a[n + i] = (node){0, a[i].t + m / 2, a[i].id};
  int ans = 1 << 30, tim, l, r;
  for (i = n + 1, j = 1; i <= 2 * n; i++) {
    while (a[i].t - a[j].t >= k && j <= 2 * n) j++;
    if (i - j < ans) ans = i - j, l = j, r = i - 1, tim = a[i].t;
  }
  printf(\"%d %d\n\", ans, tim % (m / 2));
  if (ans != 0) {
    for (i = l; i <= r; i++) printf(\"%d \", a[i].id);
    puts(\"\");
  }
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Dnewpassengertramsbootcamp(Basebootcamp):
    def __init__(self, **params):
        """
        初始化参数，设置默认值
        """
        self.params = {
            'h': 24,
            'm': 60,
            'k': 15,
            'n': 5
        }
        self.params.update(params)

    def case_generator(self):
        """
        生成谜题实例，包含参数和货运列车时间
        """
        params = self.params
        h = params['h']
        m = params['m']
        k = params['k']
        n = params['n']

        m_half = m // 2
        trains = []
        existing_times = set()

        for idx in range(1, n+1):
            while True:
                h_i = random.randint(0, h-1)
                m_i = random.randint(0, m-1)
                total = h_i * m + m_i
                if total not in existing_times:
                    existing_times.add(total)
                    trains.append({'h': h_i, 'm': m_i, 'id': idx})
                    break

        return {
            'n': n,
            'h': h,
            'm': m,
            'k': k,
            'trains': trains
        }

    @staticmethod
    def prompt_func(question_case):
        """
        生成问题描述文本
        """
        h = question_case['h']
        m = question_case['m']
        k = question_case['k']
        n = question_case['n']
        trains = question_case['trains']
        m_half = m // 2

        trains_str = "\n".join([f"ID {train['id']} 出发于 {train['h']}小时 {train['m']}分钟。" for train in trains])

        return f"""你是Kirnes星球的交通调度员，需要安排客运电车时刻表。当前有{n}趟货运列车，出发时间如下：
{trains_str}

规则：
1. 一天有{h}小时，每小时{m}分钟（m是偶数）
2. 客运电车每{m_half}分钟一班，首班在0小时t分钟出发（0 ≤ t < {m_half}）
3. 每班电车需要提前{k}分钟登车，期间禁止货运列车出发（允许在登车开始和发车瞬间出发）

请确定t值，使得需取消的货运列车最少。输出格式：
第一行：[被取消数量] [t值]
第二行（如有取消）：[被取消ID列表]

将最终答案放在[answer]标签内，例如：
[answer]
1 0
2
[/answer]"""

    @staticmethod
    def extract_output(output):
        """
        从模型输出中提取答案
        """
        matches = re.findall(r'(\d+)\s+(\d+)(?:\s+((?:\d+\s*)+))?', output)
        if not matches:
            return None

        last_match = matches[-1]
        min_cancel = int(last_match[0])
        t = int(last_match[1])
        cancel_ids = list(map(int, last_match[2].split())) if len(last_match) > 2 and last_match[2] else []

        return {
            'min_cancel': min_cancel,
            't': t,
            'cancel_ids': cancel_ids
        }

    @classmethod
    def _verify_correction(cls, solution, identity):
        """
        验证答案是否正确
        """
        if not solution or 'min_cancel' not in solution or 't' not in solution or 'cancel_ids' not in solution:
            return False

        user_min = solution['min_cancel']
        user_t = solution['t']
        user_ids = set(solution['cancel_ids'])
        n = identity['n']
        h = identity['h']
        m = identity['m']
        k = identity['k']
        trains = identity['trains']
        m_half = m // 2

        if user_t < 0 or user_t >= m_half:
            return False

        # 计算正确的最小取消数
        trains_data = [{'t': train['m'] % m_half, 'id': train['id']} for train in trains]
        sorted_trains = sorted(trains_data, key=lambda x: x['t'])
        extended = []
        for t in sorted_trains:
            extended.append({'t': t['t'], 'id': t['id']})
            extended.append({'t': t['t'] + m_half, 'id': t['id']})
        extended_sorted = sorted(extended, key=lambda x: x['t'])

        min_cancel = float('inf')
        j = 0
        best_j = best_i = 0
        for i in range(len(extended_sorted)):
            while j <= i and extended_sorted[i]['t'] - extended_sorted[j]['t'] >= k:
                j += 1
            if i - j < min_cancel:
                min_cancel = i - j
                best_j, best_i = j, i

        correct_ids = set()
        for idx in range(best_j, best_i):
            correct_ids.add(extended_sorted[idx]['id'])
        correct_min = len(correct_ids)

        if user_min != correct_min:
            return False

        # 验证用户提供的t对应的取消列车
        a = (user_t - k) % m_half
        b = user_t % m_half
        expected_ids = set()
        for train in trains:
            t_i = train['m'] % m_half
            if a < b:
                if a < t_i < b:
                    expected_ids.add(train['id'])
            else:
                if t_i > a or t_i < b:
                    expected_ids.add(train['id'])

        return user_min == len(expected_ids) and user_ids == expected_ids and user_min == correct_min
