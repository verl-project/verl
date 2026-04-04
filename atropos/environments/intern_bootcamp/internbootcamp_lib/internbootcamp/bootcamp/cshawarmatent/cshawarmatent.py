"""# 

### 谜题描述
The map of the capital of Berland can be viewed on the infinite coordinate plane. Each point with integer coordinates contains a building, and there are streets connecting every building to four neighbouring buildings. All streets are parallel to the coordinate axes.

The main school of the capital is located in (s_x, s_y). There are n students attending this school, the i-th of them lives in the house located in (x_i, y_i). It is possible that some students live in the same house, but no student lives in (s_x, s_y).

After classes end, each student walks from the school to his house along one of the shortest paths. So the distance the i-th student goes from the school to his house is |s_x - x_i| + |s_y - y_i|.

The Provision Department of Berland has decided to open a shawarma tent somewhere in the capital (at some point with integer coordinates). It is considered that the i-th student will buy a shawarma if at least one of the shortest paths from the school to the i-th student's house goes through the point where the shawarma tent is located. It is forbidden to place the shawarma tent at the point where the school is located, but the coordinates of the shawarma tent may coincide with the coordinates of the house of some student (or even multiple students).

You want to find the maximum possible number of students buying shawarma and the optimal location for the tent itself.

Input

The first line contains three integers n, s_x, s_y (1 ≤ n ≤ 200 000, 0 ≤ s_x, s_y ≤ 10^{9}) — the number of students and the coordinates of the school, respectively.

Then n lines follow. The i-th of them contains two integers x_i, y_i (0 ≤ x_i, y_i ≤ 10^{9}) — the location of the house where the i-th student lives. Some locations of houses may coincide, but no student lives in the same location where the school is situated.

Output

The output should consist of two lines. The first of them should contain one integer c — the maximum number of students that will buy shawarmas at the tent. 

The second line should contain two integers p_x and p_y — the coordinates where the tent should be located. If there are multiple answers, print any of them. Note that each of p_x and p_y should be not less than 0 and not greater than 10^{9}.

Examples

Input


4 3 2
1 3
4 2
5 1
4 1


Output


3
4 2


Input


3 100 100
0 0
0 0
100 200


Output


2
99 100


Input


7 10 12
5 6
20 23
15 4
16 5
4 54
12 1
4 15


Output


4
10 11

Note

In the first example, If we build the shawarma tent in (4, 2), then the students living in (4, 2), (4, 1) and (5, 1) will visit it.

In the second example, it is possible to build the shawarma tent in (1, 1), then both students living in (0, 0) will visit it.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division, print_function
 
import os
import sys
from io import BytesIO, IOBase
 
if sys.version_info[0] < 3:
    from __builtin__ import xrange as range
    from future_builtins import ascii, filter, hex, map, oct, zip


def main():
	n, sx, sy = map(int, input().split())
	first, second, third, fourth, negx, negy, x, y = 0, 0, 0, 0, 0, 0, 0, 0
	for i in range(n):
		a, b = map(int, input().split())
		a -= sx
		b -= sy
		if a!=0 and b!=0:
			if a>0 and b>0:
				first += 1
			elif a<0 and b>0:
				second += 1
			elif a<0 and b<0:
				third += 1
			else:
				fourth += 1
		else:
			if a == 0:
				if b > 0:
					y += 1
				else:
					negy += 1
			else:
				if a > 0:
					x += 1
				else:
					negx += 1
	x += fourth + first
	y += first + second
	negx += second + third
	negy += third + fourth
	ans = max(x, y, negx, negy)
	print (ans)
	if ans == x:
		print (sx+1, sy)
	elif ans == negx:
		print (sx-1, sy)
	elif ans == y:
		print (sx, sy+1)
	else:
		print (sx, sy-1)


 
# region fastio
 
BUFSIZE = 8192
 
 
class FastIO(IOBase):
    newlines = 0
 
    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = \"x\" in file.mode or \"r\" not in file.mode
        self.write = self.buffer.write if self.writable else None
 
    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()
 
    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b\"\n\") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()
 
    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)
 
 
class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode(\"ascii\"))
        self.read = lambda: self.buffer.read().decode(\"ascii\")
        self.readline = lambda: self.buffer.readline().decode(\"ascii\")
 
 
def print(*args, **kwargs):
    \"\"\"Prints the values to a stream, or to sys.stdout by default.\"\"\"
    sep, file = kwargs.pop(\"sep\", \" \"), kwargs.pop(\"file\", sys.stdout)
    at_start = True
    for x in args:
        if not at_start:
            file.write(sep)
        file.write(str(x))
        at_start = False
    file.write(kwargs.pop(\"end\", \"\n\"))
    if kwargs.pop(\"flush\", False):
        file.flush()
 
 
if sys.version_info[0] < 3:
    sys.stdin, sys.stdout = FastIO(sys.stdin), FastIO(sys.stdout)
else:
    sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
 
input = lambda: sys.stdin.readline().rstrip(\"\r\n\")
 
# endregion
 
if __name__ == \"__main__\":
    main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cshawarmatentbootcamp(Basebootcamp):
    def __init__(self, student_count_range=(1, 20), school_coord_range=(0, 10**9), student_coord_range=(0, 10**9)):
        self.student_count_range = student_count_range
        self.school_coord_range = school_coord_range
        self.student_coord_range = student_coord_range
    
    def case_generator(self):
        sx = random.randint(*self.school_coord_range)
        sy = random.randint(*self.school_coord_range)
        n = random.randint(*self.student_count_range)
        
        students = []
        first = second = third = fourth = 0
        x_pos = x_neg = y_pos = y_neg = 0
        
        for _ in range(n):
            while True:
                x = random.randint(*self.student_coord_range)
                y = random.randint(*self.student_coord_range)
                if (x, y) != (sx, sy):
                    break
            students.append([x, y])
            a = x - sx
            b = y - sy
            
            if a == 0 or b == 0:
                if a == 0:
                    if b > 0:
                        y_pos += 1
                    else:
                        y_neg += 1
                else:
                    if a > 0:
                        x_pos += 1
                    else:
                        x_neg += 1
            else:
                if a > 0 and b > 0:
                    first += 1
                elif a < 0 and b > 0:
                    second += 1
                elif a < 0 and b < 0:
                    third += 1
                else:
                    fourth += 1
        
        east_count = x_pos + fourth + first
        west_count = x_neg + second + third
        north_count = y_pos + first + second
        south_count = y_neg + third + fourth
        
        candidates = []
        # 检查东方向坐标有效性
        if (sx + 1) <= 10**9:
            candidates.append((east_count, (sx + 1, sy)))
        # 检查西方向坐标有效性
        if (sx - 1) >= 0:
            candidates.append((west_count, (sx - 1, sy)))
        # 检查北方向坐标有效性
        if (sy + 1) <= 10**9:
            candidates.append((north_count, (sx, sy + 1)))
        # 检查南方向坐标有效性
        if (sy - 1) >= 0:
            candidates.append((south_count, (sx, sy - 1)))
        
        max_count = 0
        valid_positions = []
        if candidates:
            max_count = max(c[0] for c in candidates)
            valid_positions = [c[1] for c in candidates if c[0] == max_count]
        
        return {
            "sx": sx,
            "sy": sy,
            "students": students,
            "max_count": max_count,
            "valid_positions": valid_positions
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        input_lines = [f"{len(question_case['students'])} {question_case['sx']} {question_case['sy']}"]
        for student in question_case['students']:
            input_lines.append(f"{student[0]} {student[1]}")
        input_str = '\n'.join(input_lines)
        
        prompt = f"""你是Berland首都的城市规划师，需要确定放置沙威玛帐篷的最佳位置。学校位于坐标({question_case['sx']}, {question_case['sy']})，现有{len(question_case['students'])}名学生需要从学校回家。每个学生回家的最短路径可能经过不同的网格点。

任务要求：
- 帐篷不能放置在学校的坐标位置
- 学生购买沙威玛的条件是其至少存在一条最短路径经过帐篷所在点
- 你需要找出能吸引最多学生经过的帐篷位置。若有多个最优解，输出任意一个

输入格式：
首行三个整数n, s_x, s_y，分别表示学生数和学校坐标
随后n行每行两个整数，表示学生的住所坐标

输出格式：
第一行输出最大学生数c
第二行输出帐篷坐标p_x和p_y（0 ≤ p_x,p_y ≤ 1e9）

请将答案按以下格式包裹在[answer]标签中：

[answer]
c
p_x p_y
[/answer]

输入数据：
{input_str}

请仔细分析并给出答案。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        answer_block = answer_blocks[-1].strip()
        lines = [line.strip() for line in answer_block.split('\n') if line.strip()]
        if len(lines) < 2:
            return None
        try:
            c = int(lines[0])
            px, py = map(int, lines[1].split())
            return {'c': c, 'px': px, 'py': py}
        except (ValueError, IndexError):
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or 'c' not in solution or 'px' not in solution or 'py' not in solution:
            return False
        if solution['c'] != identity['max_count']:
            return False
        target_pos = (solution['px'], solution['py'])
        return any(target_pos == (pos[0], pos[1]) for pos in identity['valid_positions'])
