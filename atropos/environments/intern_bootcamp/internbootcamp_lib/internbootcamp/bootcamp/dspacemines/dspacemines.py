"""# 

### 谜题描述
Once upon a time in the galaxy of far, far away...

Darth Wader found out the location of a rebels' base. Now he is going to destroy the base (and the whole planet that the base is located at), using the Death Star.

When the rebels learnt that the Death Star was coming, they decided to use their new secret weapon — space mines. Let's describe a space mine's build.

Each space mine is shaped like a ball (we'll call it the mine body) of a certain radius r with the center in the point O. Several spikes protrude from the center. Each spike can be represented as a segment, connecting the center of the mine with some point P, such that <image> (transporting long-spiked mines is problematic), where |OP| is the length of the segment connecting O and P. It is convenient to describe the point P by a vector p such that P = O + p.

The Death Star is shaped like a ball with the radius of R (R exceeds any mine's radius). It moves at a constant speed along the v vector at the speed equal to |v|. At the moment the rebels noticed the Star of Death, it was located in the point A.

The rebels located n space mines along the Death Star's way. You may regard the mines as being idle. The Death Star does not know about the mines' existence and cannot notice them, which is why it doesn't change the direction of its movement. As soon as the Star of Death touched the mine (its body or one of the spikes), the mine bursts and destroys the Star of Death. A touching is the situation when there is a point in space which belongs both to the mine and to the Death Star. It is considered that Death Star will not be destroyed if it can move infinitely long time without touching the mines.

Help the rebels determine whether they will succeed in destroying the Death Star using space mines or not. If they will succeed, determine the moment of time when it will happen (starting from the moment the Death Star was noticed).

Input

The first input data line contains 7 integers Ax, Ay, Az, vx, vy, vz, R. They are the Death Star's initial position, the direction of its movement, and its radius ( - 10 ≤ vx, vy, vz ≤ 10, |v| > 0, 0 < R ≤ 100).

The second line contains an integer n, which is the number of mines (1 ≤ n ≤ 100). Then follow n data blocks, the i-th of them describes the i-th mine.

The first line of each block contains 5 integers Oix, Oiy, Oiz, ri, mi, which are the coordinates of the mine centre, the radius of its body and the number of spikes (0 < ri < 100, 0 ≤ mi ≤ 10). Then follow mi lines, describing the spikes of the i-th mine, where the j-th of them describes the i-th spike and contains 3 integers pijx, pijy, pijz — the coordinates of the vector where the given spike is directed (<image>).

The coordinates of the mines' centers and the center of the Death Star are integers, their absolute value does not exceed 10000. It is guaranteed that R > ri for any 1 ≤ i ≤ n. For any mines i ≠ j the following inequality if fulfilled: <image>. Initially the Death Star and the mines do not have common points.

Output

If the rebels will succeed in stopping the Death Star using space mines, print the time from the moment the Death Star was noticed to the blast.

If the Death Star will not touch a mine, print \"-1\" (without quotes).

For the answer the absolute or relative error of 10 - 6 is acceptable.

Examples

Input

0 0 0 1 0 0 5
2
10 8 0 2 2
0 -3 0
2 2 0
20 0 0 4 3
2 4 0
-4 3 0
1 -5 0


Output

10.0000000000

Input

8 8 4 4 4 2 6
1
-2 -2 -1 3 0


Output

-1

Input

30 30 2 1 2 1 20
3
0 0 40 5 1
1 4 4
-10 -40 -5 7 0
100 200 95 8 1
-10 0 0


Output

74.6757620881

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import math

inf = float('inf')

ax,ay,az,vx,vy,vz,R=map(int,raw_input().split())
n=input()
t=inf

def check(ox,oy,oz,r):
	x,y,z=ax-ox,ay-oy,az-oz
	
	a=vx**2+vy**2+vz**2
	b=2*(x*vx+y*vy+z*vz)
	c=x**2+y**2+z**2-r**2
	d=b*b-4*a*c
	if d<0: return

	x1=(-b+d**0.5)/a/2
	x2=(-b-d**0.5)/a/2

	global t
	if x1>=0: t=min(t,x1)
	if x2>=0: t=min(t,x2)

for i in xrange(n):
	ox,oy,oz,r,m = map(int,raw_input().split())
	check(ox,oy,oz,r+R)
	for j in xrange(m):
		rx,ry,rz = map(int,raw_input().split())
		check(rx+ox,ry+oy,rz+oz,R)
print -1 if t==inf else \"%.20f\"%t
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import random
from bootcamp import Basebootcamp

class Dspaceminesbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = {
            'A': params.get('A', self.generate_random_A()),
            'v': params.get('v', self.generate_random_v()),
            'R': params.get('R', random.randint(1, 100)),
            'min_mines': params.get('min_mines', 1),
            'max_mines': params.get('max_mines', 3),
        }

    def generate_random_A(self):
        return (
            random.randint(-10000, 10000),
            random.randint(-10000, 10000),
            random.randint(-10000, 10000)
        )

    def generate_random_v(self):
        while True:
            v = (random.randint(-10, 10), random.randint(-10, 10), random.randint(-10, 10))
            if any(v):
                return v

    def generate_mine(self, A, R, existing_mines):
        max_attempts = 1000
        for _ in range(max_attempts):
            # 生成随机方向和距离
            theta = random.uniform(0, math.pi)
            phi = random.uniform(0, 2*math.pi)
            dx = math.sin(theta)*math.cos(phi)
            dy = math.sin(theta)*math.sin(phi)
            dz = math.cos(theta)
            
            r_i = random.randint(1, R-1)
            min_dist = R + r_i + 1
            distance = random.uniform(min_dist, 2*min_dist)  # 生成适中距离
            
            ox = A[0] + dx*distance
            oy = A[1] + dy*distance
            oz = A[2] + dz*distance
            ox, oy, oz = int(round(ox)), int(round(oy)), int(round(oz))

            # 检查与已有地雷的间距
            valid = True
            for mine in existing_mines:
                mo = mine['O']
                mr = mine['r']
                dist_sq = (ox-mo[0])**2 + (oy-mo[1])**2 + (oz-mo[2])**2
                if dist_sq < (r_i + mr)**2:
                    valid = False
                    break
            if valid:
                return {
                    'O': [ox, oy, oz],
                    'r': r_i,
                    'm': random.randint(0, 10),
                    'spikes': [[random.randint(-10,10) for _ in range(3)] 
                              for _ in range(random.randint(0, 10))]
                }
        return None

    def case_generator(self):
        while True:
            A = self.params['A']
            v = self.params['v']
            R = self.params['R']
            n = random.randint(self.params['min_mines'], self.params['max_mines'])
            
            mines = []
            existing = []
            for _ in range(n):
                mine = self.generate_mine(A, R, existing)
                if mine:
                    mines.append(mine)
                    existing.append(mine)
            if mines:
                break  # 成功生成至少一个地雷时退出循环

        # 计算正确解
        def compute_collision_time():
            t = float('inf')
            ax, ay, az = A
            vx, vy, vz = v
            
            def check(ox, oy, oz, r_check):
                nonlocal t
                x = ax - ox
                y = ay - oy
                z = az - oz
                
                a = vx**2 + vy**2 + vz**2
                if a == 0: return
                b = 2*(x*vx + y*vy + z*vz)
                c = x**2 + y**2 + z**2 - r_check**2
                
                disc = b**2 - 4*a*c
                if disc < 0: return
                
                sqrt_d = math.sqrt(disc)
                t1 = (-b + sqrt_d)/(2*a)
                t2 = (-b - sqrt_d)/(2*a)
                
                if t1 >= 0: t = min(t, t1)
                if t2 >= 0: t = min(t, t2)

            for mine in mines:
                # 检查本体碰撞
                ox, oy, oz = mine['O']
                check(ox, oy, oz, mine['r'] + R)
                # 检查尖刺碰撞
                for (px, py, pz) in mine['spikes']:
                    check(ox+px, oy+py, oz+pz, R)
            
            return t if t != float('inf') else -1.0

        return {
            'death_star': {'A': list(A), 'v': list(v), 'R': R},
            'mines': mines,
            'correct_t': compute_collision_time()
        }

    @staticmethod
    def prompt_func(case):
        prompt = "Rebel Commander Analysis Task\n\nDeath Star Parameters:\n"
        prompt += f"- Initial Position: {case['death_star']['A']}\n"
        prompt += f"- Velocity Vector: {case['death_star']['v']}\n"
        prompt += f"- Radius: {case['death_star']['R']}\n\n"
        prompt += f"Minefield Details ({len(case['mines'])} mines):\n"
        
        for i, mine in enumerate(case['mines'], 1):
            prompt += f"\nMine {i}:\n"
            prompt += f"- Center: {mine['O']}\n"
            prompt += f"- Body Radius: {mine['r']}\n"
            prompt += f"- Spikes: {len(mine['spikes'])}\n"
            if mine['spikes']:
                prompt += "  Spike Vectors:\n"
                for vec in mine['spikes']:
                    prompt += f"  {vec}\n"
        
        prompt += "\nCalculate the earliest collision time (precision 1e-6) or -1.\n"
        prompt += "Enclose your final answer within [answer]...[/answer] tags."
        return prompt

    @staticmethod
    def extract_output(output):
        import re
        answers = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        return answers[-1].strip() if answers else None

    @classmethod
    def _verify_correction(cls, solution, case):
        try:
            if solution.strip() == '-1':
                return case['correct_t'] == -1.0
            
            submitted = float(solution)
            correct = case['correct_t']
            if correct == -1.0:
                return False
            return abs(submitted - correct) < 1e-6 or abs(submitted - correct)/correct < 1e-6
        except:
            return False
