"""# 

### 谜题描述
Iahub got lost in a very big desert. The desert can be represented as a n × n square matrix, where each cell is a zone of the desert. The cell (i, j) represents the cell at row i and column j (1 ≤ i, j ≤ n). Iahub can go from one cell (i, j) only down or right, that is to cells (i + 1, j) or (i, j + 1). 

Also, there are m cells that are occupied by volcanoes, which Iahub cannot enter. 

Iahub is initially at cell (1, 1) and he needs to travel to cell (n, n). Knowing that Iahub needs 1 second to travel from one cell to another, find the minimum time in which he can arrive in cell (n, n).

Input

The first line contains two integers n (1 ≤ n ≤ 109) and m (1 ≤ m ≤ 105). Each of the next m lines contains a pair of integers, x and y (1 ≤ x, y ≤ n), representing the coordinates of the volcanoes.

Consider matrix rows are numbered from 1 to n from top to bottom, and matrix columns are numbered from 1 to n from left to right. There is no volcano in cell (1, 1). No two volcanoes occupy the same location. 

Output

Print one integer, the minimum time in which Iahub can arrive at cell (n, n). If no solution exists (there is no path to the final cell), print -1.

Examples

Input

4 2
1 3
1 4


Output

6


Input

7 8
1 6
2 6
3 5
3 6
4 3
5 1
5 2
5 3


Output

12


Input

2 2
1 2
2 1


Output

-1

Note

Consider the first sample. A possible road is: (1, 1) →  (1, 2) →  (2, 2) →  (2, 3) →  (3, 3) →  (3, 4) →  (4, 4).

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys

(n, m) = map(int, raw_input().split())
x = {}
times = set([1,n])
for i in xrange(m):
	(a, b) = map(int, raw_input().split())
	if a == n and b == n:
		print '-1'
		sys.exit(0);
	times.add(a)
	if a < n:
		times.add(a+1)
	if a in x:
		x[a].add(b)
	else:
		x[a] = set([b])

rays = []
timelist = list(times)
timelist.sort()
for time in timelist:
	if time == 1:
		if 1 in x:
			rays = [(1, list(x[1])[0] - 1)]
		else:
			rays = [(1, n)]
	else:
		y = []
		if time in x:
			y = list(x[time])
			y.sort()
		y.append(n + 1)
		newrays = []
		lastray = None
		j = 0
		for ray in rays:
			thisray = None
			if lastray is None:
				thisray = ray
			elif lastray[1] >= ray[0]:
				thisray = (lastray[0], ray[1])
			else:
				newrays.append(lastray)
				thisray = ray
			while y[j] <= thisray[1]:
				if y[j] >= thisray[0]:
					if thisray[0] <= y[j] - 1:
						newrays.append((thisray[0], y[j] - 1))
					thisray = (y[j] + 1, thisray[1])
				j += 1
			lastray = None
			if thisray[0] <= thisray[1]:
				lastray = (thisray[0], y[j]-1)
		if lastray != None:
			newrays.append(lastray)
		rays = newrays

if len(rays) >= 1 and rays[-1][1] == n:
	print (n-1)*2
else:
	print '-1'
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random

from bootcamp import Basebootcamp

class Dvolcanoesbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', 4)
        self.m = params.get('m', 2)
        self.volcanoes = params.get('volcanoes', [])

    def case_generator(self):
        n = self.n
        m = self.m
        volcanoes = self.volcanoes.copy()
        # 确保起点和终点没有火山
        if (1, 1) in volcanoes:
            volcanoes.remove((1, 1))
        if (n, n) in volcanoes:
            volcanoes.remove((n, n))
        # 生成足够的火山位置
        while len(volcanoes) < m:
            x = random.randint(1, n)
            y = random.randint(1, n)
            if (x, y) not in volcanoes and (x, y) != (1, 1) and (x, y) != (n, n):
                volcanoes.append((x, y))
        # 随机打乱火山的位置
        random.shuffle(volcanoes)
        # 返回案例
        return {
            'n': n,
            'm': m,
            'volcanoes': volcanoes[:m]
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        volcanoes = question_case['volcanoes']
        volcanoes_str = ', '.join(map(str, volcanoes))
        prompt = f"你是一名沙漠探险者，被困在一个{n}×{n}的沙漠中。你需要从起点(1,1)移动到终点({n},{n})，但只能向右或向下移动。某些格子有火山，无法进入。你的任务是找到从起点到终点的最短时间（每步1秒）。如果没有路径，输出-1。\n"
        prompt += f"输入的沙漠大小是{n}，有{m}个火山，分别位于：{volcanoes_str}。\n"
        prompt += "请输出从起点到终点所需的最短时间，或者-1表示没有路径。\n"
        prompt += "将答案放在[answer]标签中，例如：\n"
        prompt += "[answer]6[/answer]\n"
        return prompt

    @staticmethod
    def extract_output(output):
        pattern = r'\[answer\](.*?)\[/answer\]'
        matches = re.findall(pattern, output)
        if matches:
            return matches[-1].strip()
        else:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        m = identity['m']
        volcanoes = identity['volcanoes']
        # 检查终点是否是火山
        if (n, n) in volcanoes:
            correct = -1
        else:
            # 使用优化后的逻辑计算正确的解
            x = {}
            times = set()
            times.add(1)
            times.add(n)
            valid = True
            for a, b in volcanoes:
                if a == n and b == n:
                    valid = False
                    correct = -1
                    break
                times.add(a)
                if a < n:
                    times.add(a + 1)
                if a in x:
                    x[a].add(b)
                else:
                    x[a] = {b}
            if not valid:
                correct = -1
            else:
                timelist = sorted(times)
                rays = []
                for time in timelist:
                    if time == 1:
                        if 1 in x:
                            y_list = sorted(x[1])
                            y_list.append(n + 1)
                            j = 0
                            current_rays = []
                            lastray = None
                            for y in y_list:
                                if j == 0:
                                    current_start = 1
                                    current_end = y - 1
                                else:
                                    current_start = y_prev + 1
                                    current_end = y - 1
                                if current_start <= current_end:
                                    if lastray is None:
                                        lastray = (current_start, current_end)
                                    else:
                                        if current_start <= lastray[1]:
                                            lastray = (lastray[0], current_end)
                                        else:
                                            current_rays.append(lastray)
                                            lastray = (current_start, current_end)
                                y_prev = y
                            if lastray is not None:
                                current_rays.append(lastray)
                            rays = current_rays
                        else:
                            rays = [(1, n)]
                    else:
                        y_list = []
                        if time in x:
                            y_list = sorted(x[time])
                        y_list.append(n + 1)
                        new_rays = []
                        lastray = None
                        for ray in rays:
                            thisray = ray
                            j = 0
                            while j < len(y_list) and y_list[j] <= thisray[1]:
                                if y_list[j] >= thisray[0]:
                                    if thisray[0] <= y_list[j] - 1:
                                        new_rays.append((thisray[0], y_list[j] - 1))
                                    thisray = (y_list[j] + 1, thisray[1])
                                    if thisray[0] > thisray[1]:
                                        break
                                j += 1
                            if thisray[0] <= thisray[1]:
                                if lastray is None:
                                    lastray = thisray
                                else:
                                    if thisray[0] <= lastray[1]:
                                        lastray = (lastray[0], thisray[1])
                                    else:
                                        new_rays.append(lastray)
                                        lastray = thisray
                            else:
                                lastray = None
                        if lastray is not None:
                            new_rays.append(lastray)
                        rays = new_rays
                    if not rays:
                        break
                if rays and rays[-1][1] == n:
                    correct = (n - 1) * 2
                else:
                    correct = -1
        # 解析用户答案
        try:
            solution_int = int(solution)
        except ValueError:
            return False
        # 比较
        return solution_int == correct

