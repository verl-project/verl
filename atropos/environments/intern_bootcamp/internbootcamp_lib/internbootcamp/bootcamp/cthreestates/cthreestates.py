"""# 

### 谜题描述
The famous global economic crisis is approaching rapidly, so the states of Berman, Berance and Bertaly formed an alliance and allowed the residents of all member states to freely pass through the territory of any of them. In addition, it was decided that a road between the states should be built to guarantee so that one could any point of any country can be reached from any point of any other State.

Since roads are always expensive, the governments of the states of the newly formed alliance asked you to help them assess the costs. To do this, you have been issued a map that can be represented as a rectangle table consisting of n rows and m columns. Any cell of the map either belongs to one of three states, or is an area where it is allowed to build a road, or is an area where the construction of the road is not allowed. A cell is called passable, if it belongs to one of the states, or the road was built in this cell. From any passable cells you can move up, down, right and left, if the cell that corresponds to the movement exists and is passable.

Your task is to construct a road inside a minimum number of cells, so that it would be possible to get from any cell of any state to any cell of any other state using only passable cells.

It is guaranteed that initially it is possible to reach any cell of any state from any cell of this state, moving only along its cells. It is also guaranteed that for any state there is at least one cell that belongs to it.

Input

The first line of the input contains the dimensions of the map n and m (1 ≤ n, m ≤ 1000) — the number of rows and columns respectively.

Each of the next n lines contain m characters, describing the rows of the map. Digits from 1 to 3 represent the accessory to the corresponding state. The character '.' corresponds to the cell where it is allowed to build a road and the character '#' means no construction is allowed in this cell.

Output

Print a single integer — the minimum number of cells you need to build a road inside in order to connect all the cells of all states. If such a goal is unachievable, print -1.

Examples

Input

4 5
11..2
#..22
#.323
.#333

Output

2

Input

1 5
1#2#3


Output

-1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
range = xrange
input = raw_input

inp = sys.stdin.read().split(); ii = 0

n = int(inp[ii]); ii += 1
m = int(inp[ii]); ii += 1

M = [0]*(m + 2)
for _ in range(n):
    M.append(0)
    for c in inp[ii]:
        if c == '#':
            M.append(0)
        elif c == '.':
            M.append(1)
        else:
            M.append(ord(c) - 47)
    ii += 1
    M.append(0)
M += [0]*(m + 2)

n += 2
m += 2

A = []
B = []
C = []
for i in range(n*m):
    if M[i] == 2:
        A.append(i)
    elif M[i] == 3:
        B.append(i)
    elif M[i] == 4:
        C.append(i)

inf = 10**8

Adist = [inf]*(n*m)
Bdist = [inf]*(n*m)
Cdist = [inf]*(n*m)

def dister(bfs, dist):
    for node in bfs:
        dist[node] = 0
    for node in bfs:
        d = dist[node] + 1
        for nei in node - 1, node + 1, node - m, node + m:
            if M[nei] and dist[nei] == inf:
                dist[nei] = d
                bfs.append(nei)
dister(list(A), Adist)
dister(list(B), Bdist)
dister(list(C), Cdist)

besta = inf
for i in range(n*m):
    besta = min(besta, Adist[i] + Bdist[i] + Cdist[i] - 2)

def opter(dist1, dist2):
    besta = inf
    for i in range(n*m):
        besta = min(besta, dist1[i] + dist2[i] - 1)
    return besta

besta1 = opter(Adist, Bdist)
besta2 = opter(Adist, Cdist)
besta3 = opter(Bdist, Cdist)

alt1 = besta
alt2 = besta1 + besta2
alt3 = besta1 + besta3
alt4 = besta2 + besta3

alt = min(alt1, alt2, alt3, alt4)
print alt if alt < 10**7 else -1
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import json
import random
from bootcamp import Basebootcamp
from collections import deque

class Cthreestatesbootcamp(Basebootcamp):
    def __init__(self, n=5, m=5, min_country_cells=3, max_country_cells=5, max_roads=20):
        self.n = n
        self.m = m
        self.min_country_cells = min_country_cells
        self.max_country_cells = max_country_cells
        self.max_roads = max_roads

    def case_generator(self):
        grid = [['.'] * self.m for _ in range(self.n)]
        countries = ['1', '2', '3']
        placed = {'1': False, '2': False, '3': False}

        for i in range(self.n):
            for j in range(self.m):
                if random.random() < 0.3:
                    grid[i][j] = '#'
                elif not placed['1']:
                    grid[i][j] = '1'
                    placed['1'] = True
                elif not placed['2']:
                    grid[i][j] = '2'
                    placed['2'] = True
                elif not placed['3']:
                    grid[i][j] = '3'
                    placed['3'] = True
                else:
                    if random.random() < 0.5:
                        grid[i][j] = random.choice(['1', '2', '3'])
                    else:
                        grid[i][j] = random.choice(['.', '#'])

        for country in countries:
            if not any(cell == country for row in grid for cell in row):
                i, j = random.randint(0, self.n-1), random.randint(0, self.m-1)
                grid[i][j] = country

        question = {'grid': grid}
        return json.loads(json.dumps(question))

    @staticmethod
    def prompt_func(question_case):
        grid = question_case['grid']
        prompt = (
            "You are an expert in road construction and connectivity analysis. "
            "Please solve the following problem:\n\n"
            "Given a map of size {}x{}, where each cell can be:\n"
            "- '1', '2', or '3' representing regions of three countries.\n"
            "- '.' indicating a cell where a road can be built.\n"
            "- '#' indicating a cell where construction is not allowed.\n\n"
            "Your task is to determine the minimum number of cells where roads must be built so that all three countries are fully connected. "
            "If it's impossible to connect all countries, return -1.\n\n"
            "The map is as follows:\n"
            "{}\n\n"
            "Please provide your answer within [answer] tags. The answer should be an integer representing the minimum number of roads needed or -1 if impossible."
        ).format(len(grid), len(grid[0]), '\n'.join(''.join(row) for row in grid))
        return prompt

    @staticmethod
    def extract_output(output):
        start_tag = "[answer]"
        end_tag = "[/answer]"
        start = output.rfind(start_tag)
        if start == -1:
            return None
        end = output.find(end_tag, start + len(start_tag))
        if end == -1:
            return None
        answer = output[start + len(start_tag):end].strip()
        return answer

    @classmethod
    def _verify_correction(cls, solution, identity):
        grid = identity['grid']
        n = len(grid)
        m = len(grid[0])
        M = []
        for i in range(n):
            M += [0] + [cell for cell in grid[i]] + [0]
        M = [0] * (m + 2) + M + [0] * (m + 2)
        n += 2
        m += 2

        A, B, C = [], [], []
        for i in range(n * m):
            x, y = divmod(i, m)
            if 0 < x < n-1 and 0 < y < m-1:
                cell = M[i]
                if cell == '2':
                    A.append(i)
                elif cell == '3':
                    B.append(i)
                elif cell == '1':
                    C.append(i)

        inf = 10**8
        Adist = [inf] * (n * m)
        Bdist = [inf] * (n * m)
        Cdist = [inf] * (n * m)

        def dister(bfs, dist):
            queue = deque(bfs)
            for node in queue:
                dist[node] = 0
            while queue:
                node = queue.popleft()
                d = dist[node] + 1
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni = node + dx * m + dy
                    nx, ny = divmod(ni, m)
                    if 0 <= nx < n and 0 <= ny < m and M[ni] != '#' and dist[ni] == inf:
                        dist[ni] = d
                        queue.append(ni)

        if not A or not B or not C:
            return solution == -1

        dister(A, Adist)
        dister(B, Bdist)
        dister(C, Cdist)

        besta = inf
        for i in range(n * m):
            if Adist[i] != inf and Bdist[i] != inf and Cdist[i] != inf:
                besta = min(besta, Adist[i] + Bdist[i] + Cdist[i] - 2)

        def opter(dist1, dist2):
            best = inf
            for i in range(n * m):
                if dist1[i] != inf and dist2[i] != inf:
                    best = min(best, dist1[i] + dist2[i] - 1)
            return best

        besta1 = opter(Adist, Bdist)
        besta2 = opter(Adist, Cdist)
        besta3 = opter(Bdist, Cdist)

        alt1 = besta
        alt2 = besta1 + besta2
        alt3 = besta1 + besta3
        alt4 = besta2 + besta3

        alt = min(alt1, alt2, alt3, alt4)

        expected = alt if alt < 10**7 else -1
        return int(solution) == expected

# 使用示例
if __name__ == "__main__":
    bootcamp = Cthreestatesbootcamp()
    case = bootcamp.case_generator()
    prompt = Cthreestatesbootcamp.prompt_func(case)
    print(prompt)
    # response = get_response(prompt, "LLM")  # 假设有一个获取LLM响应的函数
    # extracted_output = Cthreestatesbootcamp.extract_output(prompt + response)
    # score = Cthreestatesbootcamp.verify_score(extracted_output, case)
    # print(f"Score: {score}")
