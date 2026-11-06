"""# 

### 谜题描述
A group of n cities is connected by a network of roads. There is an undirected road between every pair of cities, so there are <image> roads in total. It takes exactly y seconds to traverse any single road.

A spanning tree is a set of roads containing exactly n - 1 roads such that it's possible to travel between any two cities using only these roads.

Some spanning tree of the initial network was chosen. For every road in this tree the time one needs to traverse this road was changed from y to x seconds. Note that it's not guaranteed that x is smaller than y.

You would like to travel through all the cities using the shortest path possible. Given n, x, y and a description of the spanning tree that was chosen, find the cost of the shortest path that starts in any city, ends in any city and visits all cities exactly once.

Input

The first line of the input contains three integers n, x and y (2 ≤ n ≤ 200 000, 1 ≤ x, y ≤ 109).

Each of the next n - 1 lines contains a description of a road in the spanning tree. The i-th of these lines contains two integers ui and vi (1 ≤ ui, vi ≤ n) — indices of the cities connected by the i-th road. It is guaranteed that these roads form a spanning tree.

Output

Print a single integer — the minimum number of seconds one needs to spend in order to visit all the cities exactly once.

Examples

Input

5 2 3
1 2
1 3
3 4
5 3


Output

9


Input

5 3 2
1 2
1 3
3 4
5 3


Output

8

Note

In the first sample, roads of the spanning tree have cost 2, while other roads have cost 3. One example of an optimal path is <image>.

In the second sample, we have the same spanning tree, but roads in the spanning tree cost 3, while other roads cost 2. One example of an optimal path is <image>.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def rlist(t):
    return map(t, raw_input().split())


def read_int_list():
    return rlist(int)


def write_list(lst, divider=\" \"):
    print divider.join(map(str, lst))


def read_graph(vertices, edges):
    graph = [[] for _ in xrange(vertices)]
    for _ in xrange(edges):
        v, u = read_int_list()
        graph[v-1].append(u-1)
        graph[u-1].append(v-1)
    return graph


def dfs(graph, start=0):
    track = [[start, -1]]
    stack = [start]
    visited_neighbours = [0 for _ in xrange(len(graph))]  # -1 for closed
    while stack:
        curr = stack[-1]
        if visited_neighbours[curr] == len(graph[curr]):
            visited_neighbours[curr] = -1
            stack.pop()
        else:
            neighbour = graph[curr][visited_neighbours[curr]]
            visited_neighbours[curr] += 1
            if not visited_neighbours[neighbour]:  # current neighbour unvisited
                stack.append(neighbour)
                track.append([neighbour, curr])
    return track


n, x, y = read_int_list()
graph = read_graph(n, n-1)
maxans = [[0, 0] for i in xrange(n)]
if x >= y:
    flag = False
    for i in graph:
        if len(i) == (n-1):
            flag = True
            break
    if flag:
        print x + y * (n-2)
    else:
        print y * (n-1)
else:
    track = dfs(graph)
    track.reverse()
    for curr, prev in track:
        summ = 0
        goodcnt = 0
        for v in graph[curr]:
            if v != prev:
                summ += maxans[v][0]
                if maxans[v][1]:
                    goodcnt += 1
        maxans[curr][0] = summ + min(2, goodcnt)
        maxans[curr][1] = 0 if goodcnt >= 2 else 1
    ans = maxans[0][0]
    print x * ans + y * (n-1-ans)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Dhamiltonianspanningtreebootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=10, **params):
        self.min_n = min_n
        self.max_n = max_n
        self.params = params

    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        edges = []
        # Generate a random spanning tree using parent assignment method
        for i in range(2, n + 1):
            parent = random.randint(1, i - 1)
            edges.append([i, parent])
        x = random.randint(1, 10)
        y = random.randint(1, 10)
        # Build adjacency list (0-based)
        graph = [[] for _ in range(n)]
        for u, v in edges:
            u0 = u - 1
            v0 = v - 1
            graph[u0].append(v0)
            graph[v0].append(u0)
        # Calculate correct answer based on reference solution logic
        if x >= y:
            has_center = any(len(adj) == n - 1 for adj in graph)
            correct = x + y * (n - 2) if has_center else y * (n - 1)
        else:
            track = self.dfs(graph)
            track.reverse()
            maxans = [[0, 0] for _ in range(n)]
            for curr, prev in track:
                summ = 0
                goodcnt = 0
                for neighbor in graph[curr]:
                    if neighbor != prev:
                        summ += maxans[neighbor][0]
                        if maxans[neighbor][1]:
                            goodcnt += 1
                maxans[curr][0] = summ + min(2, goodcnt)
                maxans[curr][1] = 0 if goodcnt >= 2 else 1
            ans = maxans[0][0]
            correct = x * ans + y * (n - 1 - ans)
        return {
            'n': n,
            'x': x,
            'y': y,
            'edges': edges,
            'correct_answer': correct
        }

    @staticmethod
    def dfs(graph, start=0):
        track = [[start, -1]]
        stack = [start]
        visited = [0] * len(graph)  # 0: unprocessed, -1: processed
        while stack:
            curr = stack[-1]
            if visited[curr] >= len(graph[curr]):
                visited[curr] = -1
                stack.pop()
            else:
                neighbor_idx = visited[curr]
                neighbor = graph[curr][neighbor_idx]
                visited[curr] += 1
                if visited[neighbor] == 0:
                    stack.append(neighbor)
                    track.append([neighbor, curr])
        return track

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        x = question_case['x']
        y = question_case['y']
        edges = question_case['edges']
        edge_list = '\n'.join([f"{u} {v}" for u, v in edges])
        return f"""你是一个专业的算法竞赛选手，需要解决以下问题：

**题目背景**

{n}个城市通过道路网络相连。每对城市之间都有一条双向道路，总共有{n*(n-1)//2}条路。初始所有道路通行时间为{y}秒。之后选定了其中一棵生成树（即连接所有城市的{n-1}条无环路），树中的道路通行时间被改为{x}秒。

**任务**

找到一条访问所有城市恰好一次的最短路径，并计算其总时间。

**输入格式**

第一行：三个整数n, x, y（2 ≤ n ≤ 2e5）
随后{n-1}行：每行两个整数表示生成树中的边

**输入实例**

{n} {x} {y}
{edge_list}

**输出格式**

一个整数表示最短时间，确保答案在[answer]标签内，如：[answer]123[/answer]。

请给出答案："""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except:
            nums = re.findall(r'-?\d+', last_match)
            return int(nums[-1]) if nums else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
