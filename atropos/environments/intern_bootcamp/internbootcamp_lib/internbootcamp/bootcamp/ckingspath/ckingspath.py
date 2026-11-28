"""# 

### 谜题描述
The black king is standing on a chess field consisting of 109 rows and 109 columns. We will consider the rows of the field numbered with integers from 1 to 109 from top to bottom. The columns are similarly numbered with integers from 1 to 109 from left to right. We will denote a cell of the field that is located in the i-th row and j-th column as (i, j).

You know that some squares of the given chess field are allowed. All allowed cells of the chess field are given as n segments. Each segment is described by three integers ri, ai, bi (ai ≤ bi), denoting that cells in columns from number ai to number bi inclusive in the ri-th row are allowed.

Your task is to find the minimum number of moves the king needs to get from square (x0, y0) to square (x1, y1), provided that he only moves along the allowed cells. In other words, the king can be located only on allowed cells on his way.

Let us remind you that a chess king can move to any of the neighboring cells in one move. Two cells of a chess field are considered neighboring if they share at least one point.

Input

The first line contains four space-separated integers x0, y0, x1, y1 (1 ≤ x0, y0, x1, y1 ≤ 109), denoting the initial and the final positions of the king.

The second line contains a single integer n (1 ≤ n ≤ 105), denoting the number of segments of allowed cells. Next n lines contain the descriptions of these segments. The i-th line contains three space-separated integers ri, ai, bi (1 ≤ ri, ai, bi ≤ 109, ai ≤ bi), denoting that cells in columns from number ai to number bi inclusive in the ri-th row are allowed. Note that the segments of the allowed cells can intersect and embed arbitrarily.

It is guaranteed that the king's initial and final position are allowed cells. It is guaranteed that the king's initial and the final positions do not coincide. It is guaranteed that the total length of all given segments doesn't exceed 105.

Output

If there is no path between the initial and final position along allowed cells, print -1.

Otherwise print a single integer — the minimum number of moves the king needs to get from the initial position to the final one.

Examples

Input

5 7 6 11
3
5 3 8
6 7 11
5 2 5


Output

4


Input

3 4 3 10
3
3 1 4
4 5 9
3 10 10


Output

6


Input

1 1 2 10
2
1 1 3
2 6 10


Output

-1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from collections import deque
R=lambda:map(int,raw_input().split())
x1,y1,x2,y2=R()
n=input()
inf=10**9
t={}
for _ in xrange(n):
  r,a,b=R()
  while a<=b:
    t[(r,a)]=inf
    a+=1
if (x1,y1) in t and (x2,y2) in t:
  t[(x1,y1)]=0
  q=deque(((x1,y1,0),))
  while q and t[(x2,y2)]>=inf:
    x,y,z=q.popleft()
    for u in xrange(x-1,x+2):
      for v in xrange(y-1,y+2):
        if (u,v) in t and t[(u,v)]>=inf:
          t[(u,v)]=z+1
          q.append((u,v,z+1))
  print t[(x2,y2)] if t[(x2,y2)]<inf else -1
else:
  print -1
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from collections import deque, defaultdict
from bootcamp import Basebootcamp

class Ckingspathbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.max_coord = params.get('max_coord', 100)  # 限制坐标范围便于测试
        self.min_segments = params.get('min_segments', 1)
        self.max_segments = params.get('max_segments', 5)
        super().__init__(**params)

    def case_generator(self):
        # 生成唯一坐标对
        def rand_point():
            return (random.randint(1, self.max_coord), 
                    random.randint(1, self.max_coord))
        
        start = rand_point()
        end = rand_point()
        while start == end:
            end = rand_point()

        # 生成有效路径或隔离区域
        if random.random() < 0.7:  # 70%有解案例
            path = self._generate_king_path(start, end)
            segments = self._merge_segments(path)
        else:  # 30%无解案例
            segments = self._generate_disjoint_segments(start, end)
        
        return {
            'start': start,
            'end': end,
            'segments': segments
        }

    def _generate_king_path(self, start, end):
        """生成国王移动的合法路径"""
        path = [start]
        x, y = start
        tx, ty = end
        
        while (x, y) != (tx, ty):
            dx = 0 if x == tx else (1 if tx > x else -1)
            dy = 0 if y == ty else (1 if ty > y else -1)
            x += dx
            y += dy
            path.append((x, y))
        return path

    def _merge_segments(self, path):
        """合并连续列形成线段"""
        row_dict = defaultdict(list)
        for x, y in path:
            row_dict[x].append(y)
        
        segments = []
        for row in row_dict:
            cols = sorted(row_dict[row])
            start = cols[0]
            for i in range(1, len(cols)):
                if cols[i] > cols[i-1] + 1:
                    segments.append([row, start, cols[i-1]])
                    start = cols[i]
            segments.append([row, start, cols[-1]])
        return segments

    def _generate_disjoint_segments(self, start, end):
        """生成隔离区域确保无解"""
        segments = []
        # 单独包裹起点
        segments.append([start[0], start[1]-1, start[1]+1])
        # 单独包裹终点（不同行）
        segments.append([end[0]+2, end[1]-1, end[1]+1])
        # 添加干扰线段
        for _ in range(random.randint(1,3)):
            r = random.randint(1, self.max_coord)
            a = random.randint(1, self.max_coord//2)
            b = random.randint(a+2, self.max_coord)
            segments.append([r, a, b])
        return segments

    @staticmethod
    def prompt_func(question_case):
        case = question_case
        # 构建输入内容
        input_lines = [
            f"{case['start'][0]} {case['start'][1]} {case['end'][0]} {case['end'][1]}",
            str(len(case['segments']))
        ]
        input_lines += [f"{r} {a} {b}" for r, a, b in case['segments']]
        
        # 正确构建多行字符串
        input_str = '\n'.join(input_lines)
        
        return (
            f"国际象棋国王导航问题\n"
            f"起点坐标：({case['start'][0]}, {case['start'][1]})\n"
            f"终点坐标：({case['end'][0]}, {case['end'][1]})\n"
            f"允许区域描述：\n{input_str}\n\n"
            f"请计算最少移动步数（国王每次可向周围8格移动），若无解返回-1。\n"
            f"答案格式：[answer]你的答案[/answer]，例如：[answer]4[/answer]"
        )

    @staticmethod
    def extract_output(output):
        # 允许数字前后有空格
        pattern = r'\[answer\]\s*(-?\d+)\s*\[/answer\]'
        matches = re.findall(pattern, output, re.IGNORECASE)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 构建允许区域字典
        allowed = defaultdict(set)
        for r, a, b in identity['segments']:
            allowed[r].update(range(a, b+1))
        
        start = identity['start']
        end = identity['end']
        
        # 快速检查起点终点合法性
        if start[0] not in allowed or start[1] not in allowed[start[0]]:
            return solution == -1
        if end[0] not in allowed or end[1] not in allowed[end[0]]:
            return solution == -1
        
        # BFS优化实现
        visited = {}
        queue = deque()
        queue.append((start[0], start[1], 0))
        visited[(start[0], start[1])] = 0
        
        while queue:
            x, y, steps = queue.popleft()
            
            # 到达终点立即返回
            if (x, y) == end:
                return solution == steps
            
            # 生成8个方向
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x+dx, y+dy
                    # 快速有效性检查
                    if nx not in allowed or ny not in allowed[nx]:
                        continue
                    # 更新最短路径
                    if (nx, ny) not in visited or steps+1 < visited[(nx, ny)]:
                        visited[(nx, ny)] = steps + 1
                        queue.append((nx, ny, steps+1))
        
        # 未找到路径
        return solution == -1
