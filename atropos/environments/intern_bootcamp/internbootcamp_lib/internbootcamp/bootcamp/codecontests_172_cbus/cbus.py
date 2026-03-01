"""# 

### 谜题描述
There is a bus stop near the university. The lessons are over, and n students come to the stop. The i-th student will appear at the bus stop at time ti (all ti's are distinct).

We shall assume that the stop is located on the coordinate axis Ox, at point x = 0, and the bus goes along the ray Ox, that is, towards the positive direction of the coordinate axis, and back. The i-th student needs to get to the point with coordinate xi (xi > 0).

The bus moves by the following algorithm. Initially it is at point 0. The students consistently come to the stop and get on it. The bus has a seating capacity which is equal to m passengers. At the moment when m students get on the bus, it starts moving in the positive direction of the coordinate axis. Also it starts moving when the last (n-th) student gets on the bus. The bus is moving at a speed of 1 unit of distance per 1 unit of time, i.e. it covers distance y in time y.

Every time the bus passes the point at which at least one student needs to get off, it stops and these students get off the bus. The students need 1 + [k / 2] units of time to get off the bus, where k is the number of students who leave at this point. Expression [k / 2] denotes rounded down k / 2. As soon as the last student leaves the bus, the bus turns around and goes back to the point x = 0. It doesn't make any stops until it reaches the point. At the given point the bus fills with students once more, and everything is repeated.

If students come to the stop when there's no bus, they form a line (queue) and get on the bus in the order in which they came. Any number of students get on the bus in negligible time, you should assume that it doesn't take any time. Any other actions also take no time. The bus has no other passengers apart from the students.

Write a program that will determine for each student the time when he got off the bus. The moment a student got off the bus is the moment the bus stopped at the student's destination stop (despite the fact that the group of students need some time to get off).

Input

The first line contains two space-separated integers n, m (1 ≤ n, m ≤ 105) — the number of students and the number of passengers the bus can transport, correspondingly. Next n lines contain descriptions of the students, one per line. Each line contains a pair of integers ti, xi (1 ≤ ti ≤ 105, 1 ≤ xi ≤ 104). The lines are given in the order of strict increasing of ti. Values of xi can coincide.

Output

Print n numbers w1, w2, ..., wn, wi — the moment of time when the i-th student got off the bus. Print the numbers on one line and separate them with single spaces.

Examples

Input

1 10
3 5


Output

8


Input

2 1
3 5
4 5


Output

8 19


Input

5 4
3 5
4 5
5 5
6 5
7 1


Output

11 11 11 11 20


Input

20 4
28 13
31 13
35 6
36 4
52 6
53 4
83 2
84 4
87 1
93 6
108 4
113 6
116 1
125 2
130 2
136 13
162 2
166 4
184 1
192 2


Output

51 51 43 40 93 89 86 89 114 121 118 121 137 139 139 152 195 199 193 195

Note

In the first sample the bus waits for the first student for 3 units of time and drives him to his destination in additional 5 units of time. So the student leaves the bus at the moment of time 3 + 5 = 8.

In the second sample the capacity of the bus equals 1, that's why it will drive the first student alone. This student is the same as the student from the first sample. So the bus arrives to his destination at the moment of time 8, spends 1 + [1 / 2] = 1 units of time on getting him off, and returns back to 0 in additional 5 units of time. That is, the bus returns to the bus stop at the moment of time 14. By this moment the second student has already came to the bus stop. So he immediately gets in the bus, and is driven to his destination in additional 5 units of time. He gets there at the moment 14 + 5 = 19. 

In the third sample the bus waits for the fourth student for 6 units of time, then drives for 5 units of time, then gets the passengers off for 1 + [4 / 2] = 3 units of time, then returns for 5 units of time, and then drives the fifth student for 1 unit of time.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, m = map(int, raw_input().split())
cnt, now = 0, 0
ans = [0]*n
while cnt < n:
	l = []
	p = 0
	while p < m and p + cnt < n:
		t, x = map(int, raw_input().split())
		l.append((x, cnt+p))
		p += 1
	if now < t: now = t
	l.sort()
	cnt += p
	x, last, k, i = 0, 0, 0, 0
	while i <= len(l):
		if i == len(l) or last != l[i][0]:
			if last != 0: now += 1 + k/2
			if i != len(l):
				now += l[i][0] - last
				last = l[i][0]
			k = 1
		else:
			k += 1
		if i < len(l): 
			#print l[i][1]
			ans[l[i][1]] = now
		i += 1
	#print now
	now += l[-1][0]
for x in ans:
	print x,
print ''
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

def simulate_bus(n, m, students):
    ans = [0] * n
    cnt = 0
    now = 0
    students = sorted(students, key=lambda x: x[0])  # 按到达时间确保顺序
    
    while cnt < n:
        # 当前批次学生
        current_batch_size = min(m, n - cnt)
        batch = []
        max_t = 0
        for i in range(cnt, cnt + current_batch_size):
            t, x = students[i]
            batch.append( (x, i) )  # (目标坐标, 原始索引)
            if t > max_t:
                max_t = t
        
        # 同步时间起点
        now = max(now, max_t)
        
        # 按坐标排序处理
        batch.sort()
        current_pos = 0
        
        # 处理行程
        i = 0
        while i < len(batch):
            target_x = batch[i][0]
            # 行驶时间
            drive_time = target_x - current_pos
            now += drive_time
            current_pos = target_x
            
            # 统计同坐标学生
            j = i
            while j < len(batch) and batch[j][0] == target_x:
                j += 1
            k = j - i
            
            # 记录下车时间（立即记录）
            for idx in range(i, j):
                student_idx = batch[idx][1]
                ans[student_idx] = now
            
            # 下车耗时
            now += 1 + (k // 2)
            i = j
        
        # 返回原点
        return_time = current_pos
        now += return_time
        current_pos = 0  # 重置坐标
        
        cnt += current_batch_size
    
    return ans

class Cbusbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = {
            'max_n': 100,
            'min_n': 1,
            'max_m': 100,
            'max_t_step': 10,
            'max_x': 1000
        }
        self.params.update(params)
    
    def case_generator(self):
        params = self.params
        n = random.randint(params['min_n'], params['max_n'])
        m = random.randint(1, params['max_m'])
        
        # 生成严格递增的到达时间
        ti = []
        current_t = 0
        for _ in range(n):
            current_t += random.randint(1, params['max_t_step'])
            ti.append(current_t)
        
        # 生成目标坐标（允许重复）
        xi = [random.randint(1, params['max_x']) for _ in range(n)]
        
        return {
            'n': n,
            'm': m,
            'students': [ {'t': t, 'x': x} for t, x in zip(ti, xi) ]
        }
    
    @staticmethod
    def prompt_func(question_case):
        students = "\n".join(
            f"到达时间 {s['t']} → 坐标 {s['x']}" 
            for s in question_case['students']
        )
        return f"""## 公交接送问题

### 规则说明
{question_case['n']} 名学生按到达时间递增来到车站，公交车容量 {question_case['m']} 人。规则要点：
1. 发车条件：满载或最后一人上车时立刻发车
2. 行驶逻辑：
   - 每次从原点出发，单程只向正方向行驶
   - 到达每个下车点时立即停车（耗时=1+[下车人数/2]）
   - 所有学生下车后立即空车返回原点（返回耗时=最后下车点坐标）
3. 时间计算：
   - 学生的下车时间=公交车到达其坐标的瞬间
   - 示例：发车时间t0，行驶到坐标x需x时间 → 下车时间=t0+x

### 学生数据
{students}

### 输出要求
按输入顺序输出每个学生的下车时间，用[answer]包裹答案，如：
[answer]8 19 21[/answer]"""

    @staticmethod
    def extract_output(output):
        answer_match = re.findall(r'\[answer\]\s*(.*?)\s*\[/answer\]', output, re.DOTALL)
        if not answer_match:
            return None
        try:
            return list(map(int, answer_match[-1].strip().split()))
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or len(solution)!=identity['n']:
            return False
        
        # 转换数据结构
        student_tuples = [ (s['t'], s['x']) for s in identity['students'] ]
        try:
            correct = simulate_bus(identity['n'], identity['m'], student_tuples)
            return solution == correct
        except Exception as e:
            print(f"Verification error: {e}")
            return False
