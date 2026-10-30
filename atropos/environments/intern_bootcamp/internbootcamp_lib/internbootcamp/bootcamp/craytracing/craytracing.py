"""# 

### 谜题描述
There are k sensors located in the rectangular room of size n × m meters. The i-th sensor is located at point (xi, yi). All sensors are located at distinct points strictly inside the rectangle. 

Opposite corners of the room are located at points (0, 0) and (n, m). Walls of the room are parallel to coordinate axes.

At the moment 0, from the point (0, 0) the laser ray is released in the direction of point (1, 1). The ray travels with a speed of <image> meters per second. Thus, the ray will reach the point (1, 1) in exactly one second after the start.

When the ray meets the wall it's reflected by the rule that the angle of incidence is equal to the angle of reflection. If the ray reaches any of the four corners, it immediately stops.

For each sensor you have to determine the first moment of time when the ray will pass through the point where this sensor is located. If the ray will never pass through this point, print  - 1 for such sensors.

Input

The first line of the input contains three integers n, m and k (2 ≤ n, m ≤ 100 000, 1 ≤ k ≤ 100 000) — lengths of the room's walls and the number of sensors.

Each of the following k lines contains two integers xi and yi (1 ≤ xi ≤ n - 1, 1 ≤ yi ≤ m - 1) — coordinates of the sensors. It's guaranteed that no two sensors are located at the same point.

Output

Print k integers. The i-th of them should be equal to the number of seconds when the ray first passes through the point where the i-th sensor is located, or  - 1 if this will never happen. 

Examples

Input

3 3 4
1 1
1 2
2 1
2 2


Output

1
-1
-1
2


Input

3 4 6
1 1
2 1
1 2
2 2
1 3
2 3


Output

1
-1
-1
2
5
-1


Input

7 4 5
1 3
2 2
5 1
5 3
4 3


Output

13
2
9
5
-1

Note

In the first sample, the ray will consequently pass through the points (0, 0), (1, 1), (2, 2), (3, 3). Thus, it will stop at the point (3, 3) after 3 seconds.

<image>

In the second sample, the ray will consequently pass through the following points: (0, 0), (1, 1), (2, 2), (3, 3), (2, 4), (1, 3), (0, 2), (1, 1), (2, 0), (3, 1), (2, 2), (1, 3), (0, 4). The ray will stop at the point (0, 4) after 12 seconds. It will reflect at the points (3, 3), (2, 4), (0, 2), (2, 0) and (3, 1).

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/usr/bin/env python
# coding: utf-8

if __name__ == '__main__':
    import sys
    f = sys.stdin
    #f = open('/home/ilya/opt/programming/tasks/724C_10.txt')

    def read_int_line():
        lst = next(f).split()
        return (int(v) for v in lst)

    w, h, n = read_int_line()
    ca = w
    cb = h
    maw = 1
    mah = 0
    mbw = 0
    mbh = 1
    while cb != 0:
        q = ca / cb
        ca = ca % cb
        maw -= q * mbw
        mah -= q * mbh

        t = ca
        ca = cb
        cb = t

        t = maw 
        maw = mbw
        mbw = t

        t = mah
        mah = mbh
        mbh = t

    # ca == gcd(w, h) == maw*w + mah*h
    sw = w / ca
    sh = h / ca

    for i in xrange(n):
        x, y = read_int_line()
        time = -1

        for d in xrange(4):
            # 2*w*i + x == 2*h*j + y
            if (y - x) % (2 * ca) == 0:

                m = (y - x) / (2 * ca)
                mw = m * maw
                mh = -m * mah
                q = min(mw / sh, mh / sw)
                mw -= sh * q
                mh -= sw * q
                if mw < 0 or mh < 0:
                    mw += sh
                    mh += sw

                if mw < 0 or mh < 0 or 2 * w * mw + x != 2 * h * mh + y:
                    assert False

                value = 2 * w * mw + x
                if time == -1 or (time > value):
                    time = value

            x = 2 * w - x
            if d == 1:
                y = 2 * h - y

        print time

    #import fractions
    #a = fractions.gcd(n, m)
    #a = (n / a) * m

    #def mirror(x, n):
        #k = x // n

        #if k % 2 == 0:
            #x = x - k*n
        #else:
            #x = (k+1)*n - x

        #return x
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import random
import re
from bootcamp import Basebootcamp

def extended_gcd(a, b):
    if b == 0:
        return (a, 1, 0)
    else:
        g, x, y = extended_gcd(b, a % b)
        return (g, y, x - (a // b) * y)

def compute_sensor_time(n, m, x, y):
    g = math.gcd(n, m)
    if g == 0:
        return -1
    gcd_nm = g
    sw = n // gcd_nm
    sh = m // gcd_nm

    g_bezu, maw, mah = extended_gcd(n, m)
    assert g_bezu == gcd_nm

    min_time = -1
    corner_time = (n + m) // g  # 激光到达角落的总时间

    for d in range(4):
        if d == 0:
            cx, cy = x, y
        elif d == 1:
            cx = 2 * n - x
            cy = y
        elif d == 2:
            cx = x
            cy = 2 * m - y
        else:
            cx = 2 * n - x
            cy = 2 * m - y

        delta = cy - cx
        if delta % (2 * gcd_nm) != 0:
            continue

        m_val = delta // (2 * gcd_nm)
        mw = m_val * maw
        mh = -m_val * mah

        # 调整解到非负范围
        t = max(-mw // sh if sh else 0, -mh // sw if sw else 0)
        mw += sh * (t + 1)
        mh += sw * (t + 1)

        lhs = 2 * n * mw + cx
        rhs = 2 * m * mh + cy
        if lhs != rhs or lhs < 0:
            continue

        # 检查是否在激光停止前
        if lhs >= corner_time:
            continue

        if (min_time == -1) or (lhs < min_time):
            min_time = lhs

    return min_time if min_time != -1 else -1

class Craytracingbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', 7)
        self.m = params.get('m', 4)
        self.k = params.get('k', 5)

        if self.n < 2 or self.m < 2:
            raise ValueError("n and m must be at least 2")
        max_sensors = (self.n - 1) * (self.m - 1)
        if self.k > max_sensors:
            raise ValueError(f"k cannot exceed (n-1)*(m-1) = {max_sensors} for n={self.n}, m={self.m}")

    def case_generator(self):
        sensors = set()
        max_attempts = self.k * 10
        attempts = 0
        while len(sensors) < self.k and attempts < max_attempts:
            x = random.randint(1, self.n - 1)
            y = random.randint(1, self.m - 1)
            sensors.add((x, y))
            attempts += 1
        if len(sensors) < self.k:
            raise ValueError(f"Could not generate {self.k} unique sensors in {max_attempts} attempts")
        return {
            'n': self.n,
            'm': self.m,
            'k': self.k,
            'sensors': list(sensors)
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        k = question_case['k']
        sensors = question_case['sensors']
        sensors_desc = "\n".join([f"{x} {y}" for x, y in sensors])
        prompt = f"""You are an engineer responsible for monitoring sensors in a rectangular room. The room has dimensions {n}×{m} meters. A laser is fired from the corner (0,0) at time 0, moving towards (1,1) at a speed of √2 meters per second. When the laser hits a wall, it reflects such that the angle of incidence equals the angle of reflection. The laser stops if it reaches any of the four corners of the room.

There are {k} sensors located inside the room at the following coordinates (each strictly inside the room, not on the walls):

{sensors_desc}

Your task is to determine the first time each sensor is passed by the laser. If a sensor is never passed, output -1 for it.

Input format:
The first line contains n, m, and k. The next k lines contain the coordinates of each sensor.

Output format:
Output k integers, each being the earliest time the laser passes the sensor or -1, in the order of the input sensors.

Present your final answer as a space-separated list within [answer] tags. For example:

[answer]
1 -1 2 5
[/answer]"""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            numbers = list(map(int, last_match.split()))
            return numbers
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False
        n = identity['n']
        m = identity['m']
        sensors = identity['sensors']
        k = identity['k']
        if len(solution) != k:
            return False

        # 计算激光到达角落的总时间
        g = math.gcd(n, m)
        corner_time = (n + m) // g

        for i in range(k):
            x, y = sensors[i]
            correct_time = compute_sensor_time(n, m, x, y)
            # 处理边界：如果时间超过到达角落的时间则无效
            if correct_time >= corner_time:
                correct_time = -1
            if solution[i] != correct_time:
                return False
        return True
