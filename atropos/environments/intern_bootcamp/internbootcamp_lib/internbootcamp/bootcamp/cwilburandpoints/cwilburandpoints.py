"""# 

### 谜题描述
Wilbur is playing with a set of n points on the coordinate plane. All points have non-negative integer coordinates. Moreover, if some point (x, y) belongs to the set, then all points (x', y'), such that 0 ≤ x' ≤ x and 0 ≤ y' ≤ y also belong to this set.

Now Wilbur wants to number the points in the set he has, that is assign them distinct integer numbers from 1 to n. In order to make the numbering aesthetically pleasing, Wilbur imposes the condition that if some point (x, y) gets number i, then all (x',y') from the set, such that x' ≥ x and y' ≥ y must be assigned a number not less than i. For example, for a set of four points (0, 0), (0, 1), (1, 0) and (1, 1), there are two aesthetically pleasing numberings. One is 1, 2, 3, 4 and another one is 1, 3, 2, 4.

Wilbur's friend comes along and challenges Wilbur. For any point he defines it's special value as s(x, y) = y - x. Now he gives Wilbur some w1, w2,..., wn, and asks him to find an aesthetically pleasing numbering of the points in the set, such that the point that gets number i has it's special value equal to wi, that is s(xi, yi) = yi - xi = wi.

Now Wilbur asks you to help him with this challenge.

Input

The first line of the input consists of a single integer n (1 ≤ n ≤ 100 000) — the number of points in the set Wilbur is playing with.

Next follow n lines with points descriptions. Each line contains two integers x and y (0 ≤ x, y ≤ 100 000), that give one point in Wilbur's set. It's guaranteed that all points are distinct. Also, it is guaranteed that if some point (x, y) is present in the input, then all points (x', y'), such that 0 ≤ x' ≤ x and 0 ≤ y' ≤ y, are also present in the input.

The last line of the input contains n integers. The i-th of them is wi ( - 100 000 ≤ wi ≤ 100 000) — the required special value of the point that gets number i in any aesthetically pleasing numbering.

Output

If there exists an aesthetically pleasant numbering of points in the set, such that s(xi, yi) = yi - xi = wi, then print \"YES\" on the first line of the output. Otherwise, print \"NO\".

If a solution exists, proceed output with n lines. On the i-th of these lines print the point of the set that gets number i. If there are multiple solutions, print any of them.

Examples

Input

5
2 0
0 0
1 0
1 1
0 1
0 -1 -2 1 0


Output

YES
0 0
1 0
2 0
0 1
1 1


Input

3
1 0
0 0
2 0
0 1 2


Output

NO

Note

In the first sample, point (2, 0) gets number 3, point (0, 0) gets number one, point (1, 0) gets number 2, point (1, 1) gets number 5 and point (0, 1) gets number 4. One can easily check that this numbering is aesthetically pleasing and yi - xi = wi.

In the second sample, the special values of the points in the set are 0,  - 1, and  - 2 while the sequence that the friend gives to Wilbur is 0, 1, 2. Therefore, the answer does not exist.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def solve(n, points, specialVals):
	pointOrder = []
	specialToPoint = {}
	for i in specialVals:
		specialToPoint[i] = []

	for i in points:
		special = i[1] - i[0]
		if special in specialToPoint:
			specialToPoint[special].append(i)
		else:
			return \"NO\"

	for i in specialToPoint:
		xs = [specialToPoint[i][x][0] for x in range(len(specialToPoint[i]))]
		xs.sort()
		newPoints = []
		for x in xs:
			newPoints.append([x, x + i])

		specialToPoint[i] = newPoints

	for i in range(len(specialVals)):
		if len(specialToPoint[specialVals[i]]) > 0:
			pointOrder.append(specialToPoint[specialVals[i]][0])
			specialToPoint[specialVals[i]].pop(0)
		else:
			return \"NO\"

	for i in range(len(pointOrder) - 1):
		if isP1GreaterThanP2(pointOrder[i], pointOrder[i + 1]):
			return \"NO\"


	return pointOrder

def isP1GreaterThanP2(p1, p2):
	if p1[0] >= p2[0] and p1[1] >= p2[1]:
		return True
	return False




n = int(raw_input())
points = [map(int, raw_input().split()) for i in range(n)]
specialVals = map(int, raw_input().split())

ans = solve(n, points, specialVals)
if ans != \"NO\":
	print \"YES\"
	for i in range(len(ans)):
		print \" \".join(map(str, ans[i]))
else:
	print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
from collections import defaultdict

class Cwilburandpointsbootcamp(Basebootcamp):
    def __init__(self, max_size=5):
        """
        初始化训练场参数，max_size控制生成点集的最大规模
        """
        self.max_size = max_size
    
    def case_generator(self):
        # 生成两种类型的案例：简单案例（所有点在y=0）和复杂案例
        if random.random() < 0.5:
            # 简单案例：所有点在y=0的x轴上
            x_max = random.randint(2, self.max_size)
            points = [(x, 0) for x in range(x_max+1)]
            w = [0-x for x in range(x_max+1)]
        else:
            # 复杂案例：包含不同y值的点（类似示例1结构）
            points = [
                (0,0), (1,0), (2,0),
                (0,1), (1,1),          # 基础结构
                (2,1) if random.random() < 0.5 else (0,2)  # 添加随机扩展
            ][:random.randint(4,6)]    # 根据n调整
            
            # 确保闭合条件
            closure = set()
            for x, y in points:
                for xp in range(x+1):
                    for yp in range(y+1):
                        closure.add((xp, yp))
            points = list(closure)
            valid_w = [y-x for (x,y) in points]
            
            # 生成有效w序列
            try:
                # 使用参考解法验证有效性
                temp_solution = self.solve(len(points), points, valid_w)
                if temp_solution == "NO":
                    raise ValueError
            except:
                # Fallback到简单案例
                return self.case_generator()
            
            w = valid_w
        
        # 打乱点顺序但保持闭包
        final_points = self.ensure_closure(points)
        random.shuffle(final_points)
        
        return {
            'n': len(final_points),
            'points': final_points,
            'w': w[:len(final_points)]
        }

    def ensure_closure(self, points):
        """保证点集满足闭合条件"""
        closure = set()
        for x, y in points:
            for xp in range(x+1):
                for yp in range(y+1):
                    closure.add((xp, yp))
        return list(closure)

    @staticmethod
    def solve(n, points, specialVals):
        """参考解法验证有效性"""
        specialToPoint = defaultdict(list)
        for x, y in points:
            special = y - x
            specialToPoint[special].append((x, y))
        
        for val in specialVals:
            if val not in specialToPoint:
                return "NO"
        
        for s in specialToPoint:
            # 对每个s的点按x升序排序
            sorted_points = sorted(specialToPoint[s], key=lambda p: p[0])
            specialToPoint[s] = sorted_points
        
        solution = []
        for val in specialVals:
            if not specialToPoint[val]:
                return "NO"
            solution.append(specialToPoint[val].pop(0))
        
        for i in range(len(solution)-1):
            x1, y1 = solution[i]
            x2, y2 = solution[i+1]
            if x2 >= x1 and y2 >= y1 and i+1 < i:
                return "NO"
        
        return solution
    
    @staticmethod
    def prompt_func(question_case):
        # 保持原有prompt格式
        points = '\n'.join(f"{x} {y}" for x, y in question_case['points'])
        return f"""Wilbur's puzzle challenge. Determine if an aesthetically pleasing numbering exists with given special values.
        
Input:
n = {question_case['n']}
Points:
{points}
Special values: {' '.join(map(str, question_case['w']))}

Format your answer within [answer] tags as:
[answer]
YES
0 0
1 0
...
[/answer]"""
    
    @staticmethod
    def extract_output(output):
        # 保持原有抽取逻辑
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        
        content = matches[-1].strip().split('\n')
        if not content:
            return None
        
        if content[0].strip().upper() == 'NO':
            return 'NO'
        
        try:
            return [tuple(map(int, line.strip().split())) for line in content[1:]]
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 保持原有验证逻辑
        if solution == 'NO':
            return False
        
        points_set = {tuple(p) for p in identity['points']}
        w = identity['w']
        
        # 检查长度匹配
        if len(solution) != len(w):
            return False
        
        # 验证特殊值匹配
        for (x, y), expected in zip(solution, w):
            if y - x != expected:
                return False
        
        # 验证点存在且唯一
        seen = set()
        for p in solution:
            pt = tuple(p)
            if pt not in points_set or pt in seen:
                return False
            seen.add(pt)
        
        # 验证美学条件
        for i in range(len(solution)):
            xi, yi = solution[i]
            for j in range(i):
                xj, yj = solution[j]
                if xj >= xi and yj >= yi:
                    return False
        
        return True
