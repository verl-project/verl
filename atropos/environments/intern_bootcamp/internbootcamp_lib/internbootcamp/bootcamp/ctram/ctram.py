"""# 

### 谜题描述
The tram in Berland goes along a straight line from the point 0 to the point s and back, passing 1 meter per t1 seconds in both directions. It means that the tram is always in the state of uniform rectilinear motion, instantly turning around at points x = 0 and x = s.

Igor is at the point x1. He should reach the point x2. Igor passes 1 meter per t2 seconds. 

Your task is to determine the minimum time Igor needs to get from the point x1 to the point x2, if it is known where the tram is and in what direction it goes at the moment Igor comes to the point x1.

Igor can enter the tram unlimited number of times at any moment when his and the tram's positions coincide. It is not obligatory that points in which Igor enter and exit the tram are integers. Assume that any boarding and unboarding happens instantly. Igor can move arbitrary along the line (but not faster than 1 meter per t2 seconds). He can also stand at some point for some time.

Input

The first line contains three integers s, x1 and x2 (2 ≤ s ≤ 1000, 0 ≤ x1, x2 ≤ s, x1 ≠ x2) — the maximum coordinate of the point to which the tram goes, the point Igor is at, and the point he should come to.

The second line contains two integers t1 and t2 (1 ≤ t1, t2 ≤ 1000) — the time in seconds in which the tram passes 1 meter and the time in seconds in which Igor passes 1 meter.

The third line contains two integers p and d (1 ≤ p ≤ s - 1, d is either 1 or <image>) — the position of the tram in the moment Igor came to the point x1 and the direction of the tram at this moment. If <image>, the tram goes in the direction from the point s to the point 0. If d = 1, the tram goes in the direction from the point 0 to the point s.

Output

Print the minimum time in seconds which Igor needs to get from the point x1 to the point x2.

Examples

Input

4 2 4
3 4
1 1


Output

8


Input

5 4 0
1 2
3 1


Output

7

Note

In the first example it is profitable for Igor to go by foot and not to wait the tram. Thus, he has to pass 2 meters and it takes 8 seconds in total, because he passes 1 meter per 4 seconds. 

In the second example Igor can, for example, go towards the point x2 and get to the point 1 in 6 seconds (because he has to pass 3 meters, but he passes 1 meters per 2 seconds). At that moment the tram will be at the point 1, so Igor can enter the tram and pass 1 meter in 1 second. Thus, Igor will reach the point x2 in 7 seconds in total.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
s,x1,x2 = map(int,raw_input().split())
t1,t2 = map(int,raw_input().split())
p,d = map(int,raw_input().split())

if(t1  >=  t2):
	print abs(x2 - x1)*t2
else:
	if d*(x2-x1) > 0:
		if d > 0:
			if p <= x1:
				t = abs(p-x1)*t1;
			else:
				t = (s-p + s + x1)*t1;
		else:
			if p >= x1 :
				t = abs(p-x1)*t1;
			else:
				t = (p+s + s - x1)*t1;
	else:
		if d > 0:
			t = abs(s - p + s - x1)*t1;
		else:
			t = abs(p + x1)*t1;

	print min(t + abs(x2-x1)*t1 , abs(x2-x1)*t2)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Ctrambootcamp(Basebootcamp):
    def __init__(self, s_min=2, s_max=1000, t_min=1, t_max=1000):
        self.s_min = s_min
        self.s_max = s_max
        self.t_min = t_min
        self.t_max = t_max
        super().__init__()

    def case_generator(self):
        s = random.randint(self.s_min, self.s_max)
        x1 = random.randint(0, s)
        x2 = random.randint(0, s)
        while x2 == x1:
            x2 = random.randint(0, s)
        t1 = random.randint(self.t_min, self.t_max)
        t2 = random.randint(self.t_min, self.t_max)
        p = random.randint(1, s-1)
        d = random.choice([1, -1])
        return {
            's': s,
            'x1': x1,
            'x2': x2,
            't1': t1,
            't2': t2,
            'p': p,
            'd': d
        }

    @staticmethod
    def prompt_func(question_case):
        s = question_case['s']
        x1 = question_case['x1']
        x2 = question_case['x2']
        t1 = question_case['t1']
        t2 = question_case['t2']
        p = question_case['p']
        d = question_case['d']
        direction_desc = "正向（从0到s）" if d == 1 else "负向（从s到0）"
        prompt = f"""Igor需要从坐标{x1}前往坐标{x2}。有轨电车在0到{s}米的直线上往返行驶，每移动1米需要{t1}秒。当前时刻，电车位于坐标{p}，并正在以{direction_desc}行驶。电车在到达0或{s}后会立即掉头，保持匀速运动。

Igor的步行速度为每米{t2}秒。他可以在任何与电车位置重合的时刻上下车，上下车时间不计。请计算Igor到达目的地所需的最短时间（单位：秒）。

请将最终答案放入[answer]标签内，例如：[answer]8[/answer]。"""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*([0-9.]+)\s*\[/answer\]', output)
        if not matches:
            return None
        try:
            return float(matches[-1].strip())
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        s = identity['s']
        x1 = identity['x1']
        x2 = identity['x2']
        t1 = identity['t1']
        t2 = identity['t2']
        p = identity['p']
        d = identity['d']
        
        # 计算正确答案
        walk_time = abs(x2 - x1) * t2
        if t1 >= t2:
            correct_time = walk_time
        else:
            # 计算电车首次到达x1的时间和方向
            if d == 1:
                if p <= x1:
                    t_wait = (x1 - p) * t1
                    dir_after = 1 if x1 < s else -1
                else:
                    t_wait = (s - p + s - x1) * t1
                    dir_after = -1
            else:  # d == -1
                if p >= x1:
                    t_wait = (p - x1) * t1
                    dir_after = -1 if x1 > 0 else 1
                else:
                    t_wait = (p + x1) * t1
                    dir_after = 1
            
            # 计算从x1到x2的行驶时间
            if dir_after == 1:
                if x2 >= x1:
                    t_ride = (x2 - x1) * t1
                else:
                    t_ride = (s - x1 + s - x2) * t1
            else:  # dir_after == -1
                if x2 <= x1:
                    t_ride = (x1 - x2) * t1
                else:
                    t_ride = (x1 + x2) * t1
            
            tram_time = t_wait + t_ride
            correct_time = min(tram_time, walk_time)
        
        # 验证答案
        try:
            return abs(solution - correct_time) < 1e-6
        except:
            return False
