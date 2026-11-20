"""# 

### 谜题描述
Alice lives on a flat planet that can be modeled as a square grid of size n × n, with rows and columns enumerated from 1 to n. We represent the cell at the intersection of row r and column c with ordered pair (r, c). Each cell in the grid is either land or water.

<image> An example planet with n = 5. It also appears in the first sample test.

Alice resides in land cell (r_1, c_1). She wishes to travel to land cell (r_2, c_2). At any moment, she may move to one of the cells adjacent to where she is—in one of the four directions (i.e., up, down, left, or right).

Unfortunately, Alice cannot swim, and there is no viable transportation means other than by foot (i.e., she can walk only on land). As a result, Alice's trip may be impossible.

To help Alice, you plan to create at most one tunnel between some two land cells. The tunnel will allow Alice to freely travel between the two endpoints. Indeed, creating a tunnel is a lot of effort: the cost of creating a tunnel between cells (r_s, c_s) and (r_t, c_t) is (r_s-r_t)^2 + (c_s-c_t)^2.

For now, your task is to find the minimum possible cost of creating at most one tunnel so that Alice could travel from (r_1, c_1) to (r_2, c_2). If no tunnel needs to be created, the cost is 0.

Input

The first line contains one integer n (1 ≤ n ≤ 50) — the width of the square grid.

The second line contains two space-separated integers r_1 and c_1 (1 ≤ r_1, c_1 ≤ n) — denoting the cell where Alice resides.

The third line contains two space-separated integers r_2 and c_2 (1 ≤ r_2, c_2 ≤ n) — denoting the cell to which Alice wishes to travel.

Each of the following n lines contains a string of n characters. The j-th character of the i-th such line (1 ≤ i, j ≤ n) is 0 if (i, j) is land or 1 if (i, j) is water.

It is guaranteed that (r_1, c_1) and (r_2, c_2) are land.

Output

Print an integer that is the minimum possible cost of creating at most one tunnel so that Alice could travel from (r_1, c_1) to (r_2, c_2).

Examples

Input


5
1 1
5 5
00001
11111
00111
00110
00110


Output


10


Input


3
1 3
3 1
010
101
010


Output


8

Note

In the first sample, a tunnel between cells (1, 4) and (4, 5) should be created. The cost of doing so is (1-4)^2 + (4-5)^2 = 10, which is optimal. This way, Alice could walk from (1, 1) to (1, 4), use the tunnel from (1, 4) to (4, 5), and lastly walk from (4, 5) to (5, 5).

In the second sample, clearly a tunnel between cells (1, 3) and (3, 1) needs to be created. The cost of doing so is (1-3)^2 + (3-1)^2 = 8.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def read_words(func = eval, sp = ' '):
    return [func(i) for i in raw_input().split(sp)]


n = input()
r1, c1 = read_words(int)
r2, c2 = read_words(int)
data = [raw_input() for i in xrange(n)]
belong_to = {}
land_grid = {}


def find_root(key):
    if key == belong_to[key]:
        return key
    belong_to[key] = find_root(belong_to[key])
    return belong_to[key]


def make_union(key1, key2):
    root1 = find_root(key1)
    root2 = find_root(key2)
    if root1 != root2:
        belong_to[root2] = root1


def create_land(i, j):
    key = (i, j)
    belong_to[key] = key


for i in xrange(n):
    for j in xrange(n):
        if data[i][j] == '0':
            create_land(i, j)
            if i > 0 and data[i-1][j] == '0':
                make_union((i-1, j), (i, j))
            if j > 0 and data[i][j-1] == '0':
                make_union((i, j-1), (i, j))

key1 = find_root((r1-1, c1 -1))
key2 = find_root((r2-1, c2 -1))
if key1 == key2:
    print 0
else:
    land_grid[key1] = []
    land_grid[key2] = []
    for i in xrange(n):
        for j in xrange(n):
            if data[i][j] == '0':
                key = (i, j)
                if find_root(key) == key1:
                    land_grid[key1].append(key)
                if find_root(key) == key2:
                    land_grid[key2].append(key)

    result = 1 << 30
    for k1 in land_grid[key1]:
        for k2 in land_grid[key2]:
            result = min(result, (k1[0]-k2[0])**2 + (k1[1]-k2[1])**2)
    print result
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Cconnectbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', 5)  # 默认网格大小为5x5

    def case_generator(self):
        n = self.n
        # 随机生成起点和终点，确保它们不同
        r1, c1 = random.randint(1, n), random.randint(1, n)
        r2, c2 = random.randint(1, n), random.randint(1, n)
        while (r1, c1) == (r2, c2):
            r2, c2 = random.randint(1, n), random.randint(1, n)
        
        # 生成网格
        grid = []
        for _ in range(n):
            row = []
            for _ in range(n):
                row.append('0' if random.random() < 0.7 else '1')  # 调整概率，增加陆地数量
            grid.append(''.join(row))
        
        # 确保起点和终点是陆地
        grid[r1-1] = grid[r1-1][:c1-1] + '0' + grid[r1-1][c1:]
        grid[r2-1] = grid[r2-1][:c2-1] + '0' + grid[r2-1][c2:]
        
        grid_list = [list(row) for row in grid]
        
        # 计算连通区域
        start = (r1-1, c1-1)
        end = (r2-1, c2-1)
        
        parent = {}
        for i in range(n):
            for j in range(n):
                if grid_list[i][j] == '0':
                    parent[(i, j)] = (i, j)
        
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]  # 路径压缩
                u = parent[u]
            return u
        
        def union(u, v):
            pu = find(u)
            pv = find(v)
            if pu != pv:
                parent[pv] = pu
        
        # 构建并查集
        for i in range(n):
            for j in range(n):
                if grid_list[i][j] == '0':
                    if i < n-1 and grid_list[i+1][j] == '0':
                        union((i, j), (i+1, j))
                    if j < n-1 and grid_list[i][j+1] == '0':
                        union((i, j), (i, j+1))
        
        start_root = find(start)
        end_root = find(end)
        
        if start_root == end_root:
            min_cost = 0
        else:
            region_a = [ (i, j) for i in range(n) for j in range(n) if grid_list[i][j] == '0' and find((i,j)) == start_root ]
            region_b = [ (i, j) for i in range(n) for j in range(n) if grid_list[i][j] == '0' and find((i,j)) == end_root ]
            
            min_cost = float('inf')
            for a in region_a:
                for b in region_b:
                    cost = (a[0] - b[0])**2 + (a[1] - b[1])**2
                    if cost < min_cost:
                        min_cost = cost
        
        # 将grid转换为字符串列表
        grid = [''.join(row) for row in grid_list]
        
        return {
            'n': n,
            'start': (r1, c1),
            'end': (r2, c2),
            'grid': grid,
            'answer': min_cost
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        start = question_case['start']
        end = question_case['end']
        grid = question_case['grid']
        
        grid_str = '\n'.join(grid)
        
        prompt = f"在一个{n}×{n}的网格中，Alice住在位置({start[0]}, {start[1]})，想要移动到({end[0]}, {end[1]})。每个格子是陆地（0）或水（1）。Alice只能在陆地上移动，无法游泳。她可以创建最多一个隧道，连接两个陆地格子，费用是两个格子坐标差的平方和。如果不需要隧道，费用是0。网格如下：\n\n{grid_str}\n\n请计算最小费用，并将答案放在[answer]标签中。例如：[answer]10[/answer]"
        return prompt
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](\d+)\[/answer\]', output)
        if matches:
            return int(matches[-1])
        else:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False
        return solution == identity['answer']
