"""# 

### 谜题描述
A tourist hiked along the mountain range. The hike lasted for n days, during each day the tourist noted height above the sea level. On the i-th day height was equal to some integer hi. The tourist pick smooth enough route for his hike, meaning that the between any two consecutive days height changes by at most 1, i.e. for all i's from 1 to n - 1 the inequality |hi - hi + 1| ≤ 1 holds.

At the end of the route the tourist rafted down a mountain river and some notes in the journal were washed away. Moreover, the numbers in the notes could have been distorted. Now the tourist wonders what could be the maximum height during his hike. Help him restore the maximum possible value of the maximum height throughout the hike or determine that the notes were so much distorted that they do not represent any possible height values that meet limits |hi - hi + 1| ≤ 1.

Input

The first line contains two space-separated numbers, n and m (1 ≤ n ≤ 108, 1 ≤ m ≤ 105) — the number of days of the hike and the number of notes left in the journal.

Next m lines contain two space-separated integers di and hdi (1 ≤ di ≤ n, 0 ≤ hdi ≤ 108) — the number of the day when the i-th note was made and height on the di-th day. It is guaranteed that the notes are given in the chronological order, i.e. for all i from 1 to m - 1 the following condition holds: di < di + 1.

Output

If the notes aren't contradictory, print a single integer — the maximum possible height value throughout the whole route.

If the notes do not correspond to any set of heights, print a single word 'IMPOSSIBLE' (without the quotes).

Examples

Input

8 2
2 0
7 0


Output

2


Input

8 3
2 0
7 0
8 3


Output

IMPOSSIBLE

Note

For the first sample, an example of a correct height sequence with a maximum of 2: (0, 0, 1, 2, 1, 1, 0, 1).

In the second sample the inequality between h7 and h8 does not hold, thus the information is inconsistent.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, m = map(int, raw_input().split())

arr = []
for _ in xrange(m):
    d, h = map(int, raw_input().split())
    arr.append((d - 1, h))

flag = True

ans = 0
ans = max(ans, arr[0][1] + arr[0][0])
ans = max(ans, arr[m - 1][1] + n - arr[m - 1][0] - 1)
for i in xrange(m - 1):
    if abs(arr[i][0] - arr[i + 1][0]) < abs(arr[i][1] - arr[i + 1][1]):
        flag = False
        break
    a = -arr[i][0] + arr[i][1]
    b = arr[i + 1][0] + arr[i + 1][1]
    ans = max(ans, (a + b) / 2)

print ans if flag else 'IMPOSSIBLE'
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Ctouristsnotesbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=100, min_m=1, max_m=5, invalid_prob=0.3):
        self.min_n = min_n
        self.max_n = max_n
        self.min_m = min_m
        self.max_m = max_m
        self.invalid_prob = invalid_prob
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        m = random.randint(self.min_m, min(self.max_m, n))
        days = sorted(random.sample(range(1, n + 1), m))
        generate_invalid = random.random() < self.invalid_prob and m >= 2
        notes = []
        h = random.randint(0, 10)
        notes.append((days[0], h))
        conflict_pos = random.randint(0, m - 2) if generate_invalid and m >= 2 else -1

        for i in range(1, m):
            prev_day = days[i-1]
            curr_day = days[i]
            delta_day = curr_day - prev_day
            prev_h = notes[-1][1]
            
            if generate_invalid and (i-1) == conflict_pos:
                possible_high = prev_h + delta_day + 1
                possible_low = prev_h - (delta_day + 1)
                if possible_low >= 0 and random.choice([True, False]):
                    curr_h = possible_low
                else:
                    curr_h = possible_high
                notes.append((curr_day, curr_h))
            else:
                min_h = max(prev_h - delta_day, 0)
                max_h = prev_h + delta_day
                curr_h = random.randint(min_h, max_h)
                notes.append((curr_day, curr_h))
        
        case = {
            'n': n,
            'm': m,
            'notes': notes,
        }
        case['correct_answer'] = self.calculate_answer(case)
        return case
    
    def calculate_answer(self, case):
        n = case['n']
        m = case['m']
        notes = case['notes']
        if m == 0:
            return 'IMPOSSIBLE'
        
        arr = [(d-1, h) for d, h in notes]
        flag = True
        ans = 0
        
        ans = max(arr[0][1] + arr[0][0], arr[-1][1] + (n - arr[-1][0] - 1))
        
        for i in range(m-1):
            delta_d = arr[i+1][0] - arr[i][0]
            delta_h = abs(arr[i][1] - arr[i+1][1])
            if delta_h > delta_d:
                flag = False
                break  # Early termination on conflict
            a = -arr[i][0] + arr[i][1]
            b = arr[i+1][0] + arr[i+1][1]
            current_max = (a + b) // 2
            ans = max(ans, current_max)
        
        return ans if flag else 'IMPOSSIBLE'
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        m = question_case['m']
        notes = question_case['notes']
        prompt = f"""你是一位徒步旅行者，正在查阅旅行日记中的记录。你的旅行持续了{n}天，但只有{m}天的记录保留下来。每个记录包含当天数和当天的海拔高度。你需要根据这些记录推断出旅行过程中可能出现的最大海拔高度，或者判断这些记录是否自相矛盾。

旅行路线的高度变化规则是：相邻两天的海拔高度变化最多为1米。也就是说，如果某天的高度是h，那么第二天的高度只能是h-1、h或h+1（当然不能为负数，但允许任何非负整数）。

日记中的记录如下：

第1行包含两个整数n和m：{n} {m}
"""
        for d, h in notes:
            prompt += f"{d} {h}\n"
        prompt += """
请仔细分析可能的情况，并按照以下要求给出答案：

- 如果存在符合条件的高度序列，输出一个整数，表示可能的最大海拔峰值。
- 如果记录自相矛盾，输出大写的IMPOSSIBLE。
- 将你的最终答案放在[answer]标签内，例如：[answer]2[/answer] 或 [answer]IMPOSSIBLE[/answer]。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip().upper()  # Unified case handling
        if last_match == 'IMPOSSIBLE':
            return 'IMPOSSIBLE'
        try:
            return int(last_match)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        correct = identity['correct_answer']
        if isinstance(correct, int):
            return isinstance(solution, int) and solution == correct
        else:
            return solution == 'IMPOSSIBLE'
