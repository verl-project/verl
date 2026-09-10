"""# 

### 谜题描述
One warm and sunny day king Copa decided to visit the shooting gallery, located at the Central Park, and try to win the main prize — big pink plush panda. The king is not good at shooting, so he invited you to help him.

The shooting gallery is an infinite vertical plane with Cartesian coordinate system on it. The targets are points on this plane. Each target is described by it's coordinates xi, and yi, by the time of it's appearance ti and by the number pi, which gives the probability that Copa hits this target if he aims at it.

A target appears and disappears instantly, so Copa can hit the target only if at the moment ti his gun sight aimed at (xi, yi). Speed of movement of the gun sight on the plane is equal to 1. Copa knows all the information about the targets beforehand (remember, he is a king!). He wants to play in the optimal way, which maximizes the expected value of the amount of hit targets. He can aim at any target at the moment 0.

Input

The first line contains integer n (1 ≤ n ≤ 1000) — amount of targets in the shooting gallery. Then n lines follow, each describing one target. Each description consists of four numbers xi, yi, ti, pi (where xi, yi, ti — integers,  - 1000 ≤ xi, yi ≤ 1000, 0 ≤ ti ≤ 109, real number pi is given with no more than 6 digits after the decimal point, 0 ≤ pi ≤ 1). No two targets may be at the same point.

Output

Output the maximum expected value of the amount of targets that was shot by the king. Your answer will be accepted if it differs from the correct answer by not more than 10 - 6.

Examples

Input

1
0 0 0 0.5


Output

0.5000000000


Input

2
0 0 0 0.6
5 0 5 0.7


Output

1.3000000000

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import math

def dis(x1,y1,x2,y2):
    return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))

n = int(raw_input())
data = []
dp = []
for i in xrange(0,n):
    data.append(tuple(map(float, raw_input().split())))
data.sort(key=lambda x: x[2])
for i in data:
    dp.append(i[3])

for i in xrange(0,n):
    for j in xrange(i+1,n):
        if dis(data[i][0],data[i][1],data[j][0], data[j][1]) <= data[j][2]-data[i][2]:
            dp[j] = max(dp[j], dp[i] + data[j][3])
print max(dp)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import random
from bootcamp import Basebootcamp

class Cshootinggallerybootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=5, coord_min=-10, coord_max=10, time_min=0, time_max=100):
        self.min_n = min_n
        self.max_n = max_n
        self.coord_min = coord_min
        self.coord_max = coord_max
        self.time_min = time_min
        self.time_max = time_max

    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        generated_coords = set()
        targets = []
        
        # Generate targets with temporal continuity
        current_time = 0
        for _ in range(n):
            while True:
                x = random.randint(self.coord_min, self.coord_max)
                y = random.randint(self.coord_min, self.coord_max)
                if (x, y) not in generated_coords:
                    generated_coords.add((x, y))
                    break
            
            # Ensure temporal progression with possible overlaps
            ti = random.randint(current_time, max(current_time, self.time_max))
            current_time = ti  # Allow overlapping times for different coordinates
            pi = round(random.uniform(0, 1), 6)
            targets.append([x, y, ti, float(pi)])  # 使用列表保证序列化

        # Shuffle to test temporal ordering logic
        random.shuffle(targets)
        
        return {
            'n': len(targets),
            'targets': targets,
            'correct_answer': self._calculate_correct_answer(targets)
        }

    @staticmethod
    def _calculate_correct_answer(targets):
        data = sorted(targets, key=lambda x: x[2])  # 按时间排序
        n = len(data)
        dp = [p[3] for p in data]

        for i in range(n):
            xi, yi, ti, _ = data[i]
            for j in range(i+1, n):
                xj, yj, tj, pj = data[j]
                distance = math.hypot(xj-xi, yj-yi)
                if distance <= (tj - ti):
                    dp[j] = max(dp[j], dp[i] + pj)
        return max(dp) if dp else 0.0

    @staticmethod
    def prompt_func(question_case) -> str:
        problem = (
            "As King Copa's advisor, calculate the maximum expected hits in a shooting gallery with these rules:\n"
            "1. Targets appear at (x,y) coordinates at specific times\n"
            "2. Gun moves at 1 unit/sec from any starting point\n"
            "3. Hit probability is given for each target\n\n"
            f"Targets (n={question_case['n']}):\n"
        )
        problem += "\n".join([f"{x} {y} {t} {p:.6f}" for x, y, t, p in question_case['targets']])
        
        problem += (
            "\n\nProvide the maximum expected value with exactly 10 decimal places, enclosed in [answer] tags.\n"
            "Example: [answer]1.2345678901[/answer]"
        )
        return problem

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.IGNORECASE)
        if not matches:
            return None
        
        try:
            # 处理科学计数法和多余空格
            value_str = matches[-1].strip().replace(' ', '').lower()
            if 'e' in value_str:
                return round(float(value_str), 10)
            return float(value_str)
        except (ValueError, IndexError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False
        expected = identity['correct_answer']
        return abs(solution - expected) <= 1e-6 + 1e-10  # 双精度容差
