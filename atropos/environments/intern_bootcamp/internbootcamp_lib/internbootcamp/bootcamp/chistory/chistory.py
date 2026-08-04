"""# 

### 谜题描述
Polycarpus likes studying at school a lot and he is always diligent about his homework. Polycarpus has never had any problems with natural sciences as his great-great-grandfather was the great physicist Seinstein. On the other hand though, Polycarpus has never had an easy time with history.

Everybody knows that the World history encompasses exactly n events: the i-th event had continued from the year ai to the year bi inclusive (ai < bi). Polycarpus easily learned the dates when each of n events started and ended (Polycarpus inherited excellent memory from his great-great-granddad). But the teacher gave him a more complicated task: Polycaprus should know when all events began and ended and he should also find out for each event whether it includes another event. Polycarpus' teacher thinks that an event j includes an event i if aj < ai and bi < bj. Your task is simpler: find the number of events that are included in some other event.

Input

The first input line contains integer n (1 ≤ n ≤ 105) which represents the number of events. Next n lines contain descriptions of the historical events, one event per line. The i + 1 line contains two integers ai and bi (1 ≤ ai < bi ≤ 109) — the beginning and the end of the i-th event. No two events start or finish in the same year, that is, ai ≠ aj, ai ≠ bj, bi ≠ aj, bi ≠ bj for all i, j (where i ≠ j). Events are given in arbitrary order.

Output

Print the only integer — the answer to the problem.

Examples

Input

5
1 10
2 9
3 8
4 7
5 6


Output

4


Input

5
1 100
2 50
51 99
52 98
10 60


Output

4


Input

1
1 1000000000


Output

0

Note

In the first example the fifth event is contained in the fourth. Similarly, the fourth event is contained in the third, the third — in the second and the second — in the first.

In the second example all events except the first one are contained in the first.

In the third example only one event, so the answer is 0.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n = int(input())

intervals = []

for _ in range(n) :
	a, b = map(int, raw_input().split())
	intervals.append([a, b])

def getStart(x) :
	return x[0]

intervals.sort(key=getStart)

maxEnd = intervals[0][1]

cnt = 0
for i in range(1, n, 1) :
	if intervals[i][1] > maxEnd :
		maxEnd = intervals[i][1]
	else :
		cnt += 1
print cnt
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Chistorybootcamp(Basebootcamp):
    def __init__(self, n=5):
        super().__init__()
        self.n = n
    
    def case_generator(self):
        events = []
        a_start = 1
        b_start = a_start + 2 * (self.n - 1) + 1
        
        for i in range(self.n):
            a = a_start + i
            b = b_start - i
            events.append([a, b])
        
        random.shuffle(events)
        
        return {
            'n': self.n,
            'events': events
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        events = question_case['events']
        input_lines = [f"{a} {b}" for a, b in events]
        input_str = f"{n}\n" + "\n".join(input_lines)
        prompt = f"""Polycarpus正在学习历史，需要帮助解决一个事件包含的问题。事件的规则如下：

- 世界历史包含n个事件，每个事件i从年份a_i开始，到年份b_i结束（a_i < b_i）。
- 任何两个事件的起始年和结束年互不相同。即，所有a_i和b_i都是唯一的。
- 我们称事件j包含事件i，当且仅当a_j < a_i且b_i < b_j。
- 你的任务是找出有多少个事件被其他事件包含。

输入格式：
第一行是一个整数n，表示事件的数量。
接下来n行每行包含两个整数a_i和b_i，表示每个事件的起始和结束年份。

输出格式：
输出一个整数，表示被其他事件包含的事件数量。

现在，你需要解决以下具体案例：

输入：
{input_str}

请仔细思考，按照正确的格式输出答案，并将最终答案放在[answer]标签内，例如：[answer]42[/answer]。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        events = identity['events']
        if n == 0:
            return solution == 0
        sorted_events = sorted(events, key=lambda x: x[0])
        max_end = sorted_events[0][1]
        cnt = 0
        for i in range(1, n):
            current_end = sorted_events[i][1]
            if current_end > max_end:
                max_end = current_end
            else:
                cnt += 1
        return solution == cnt
