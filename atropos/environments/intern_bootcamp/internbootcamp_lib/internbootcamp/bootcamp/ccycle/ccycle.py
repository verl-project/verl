"""# 

### 谜题描述
A tournament is a directed graph without self-loops in which every pair of vertexes is connected by exactly one directed edge. That is, for any two vertexes u and v (u ≠ v) exists either an edge going from u to v, or an edge from v to u.

You are given a tournament consisting of n vertexes. Your task is to find there a cycle of length three.

Input

The first line contains an integer n (1 ≤ n ≤ 5000). Next n lines contain the adjacency matrix A of the graph (without spaces). Ai, j = 1 if the graph has an edge going from vertex i to vertex j, otherwise Ai, j = 0. Ai, j stands for the j-th character in the i-th line.

It is guaranteed that the given graph is a tournament, that is, Ai, i = 0, Ai, j ≠ Aj, i (1 ≤ i, j ≤ n, i ≠ j).

Output

Print three distinct vertexes of the graph a1, a2, a3 (1 ≤ ai ≤ n), such that Aa1, a2 = Aa2, a3 = Aa3, a1 = 1, or \"-1\", if a cycle whose length equals three does not exist. 

If there are several solutions, print any of them.

Examples

Input

5
00100
10000
01001
11101
11000


Output

1 3 2 

Input

5
01111
00000
01000
01100
01110


Output

-1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin, stdout
input = stdin.readline
import gc
gc.disable()

def f():
    p, q, n = [0], [0], int(input())
    input()
    for i in range(1, n):
        t = input()[: i]
        if '0' in t:
            if '1' in t:
                for l, j in enumerate(p): 
                    if t[j] == '1': 
                        for r, j in enumerate(q):
                            if t[j] == '0':                             
                                if l + r == i: break
                                return str(p[l] + 1) + ' ' + str(q[r] + 1) + ' ' + str(i + 1)
                        break
                p.insert(l, i)
                q.insert(i - l, i)
            else: 
                p.append(i)
                q = [i] + q
        else: 
            p = [i] + p
            q.append(i)
    return -1
print(f())
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random

from bootcamp import Basebootcamp

class Ccyclebootcamp(Basebootcamp):
    def __init__(self, n=5, has_cycle=True):
        self.n = n
        self.has_cycle = has_cycle

    def case_generator(self):
        n = self.n
        has_cycle = self.has_cycle

        # 处理n < 3的情况，无法形成环
        if n < 3 and has_cycle:
            print("Warning: For n < 3, a cycle of length 3 is impossible. Setting has_cycle to False.")
            has_cycle = False

        adj_matrix = []
        if has_cycle:
            # 初始化一个n x n的矩阵，初始为0
            matrix = [[0 for _ in range(n)] for _ in range(n)]
            # 随机选择三个顶点形成环
            if n >= 3:
                # 随机选择三个不同的顶点
                vertices = random.sample(range(n), 3)
                a, b, c = vertices
                # 构造环：a→b, b→c, c→a
                matrix[a][b] = 1
                matrix[b][c] = 1
                matrix[c][a] = 1
                # 处理其他顶点与环顶点之间的关系
                for i in range(n):
                    if i not in vertices:
                        # 选择i与环顶点的连接方式，这里假设i胜过环中的两个顶点，输给另一个
                        # 这只是一个示例，具体可根据需要调整
                        for j in vertices:
                            if j == a:
                                matrix[i][j] = 1  # i胜过a
                            elif j == b:
                                matrix[i][j] = 1  # i胜过b
                            else:
                                matrix[j][i] = 1  # c胜过i
                        # 处理i与其他非环顶点
                        for j in range(i + 1, n):
                            if j not in vertices:
                                matrix[i][j] = 1  # i胜过j
                # 处理非环顶点之间的关系
                for i in range(n):
                    for j in range(i + 1, n):
                        if i not in vertices and j not in vertices:
                            matrix[i][j] = 1  # i胜过j
            else:
                # 当n <3时，无法形成环，设置为无环
                has_cycle = False
                matrix = [[0 for _ in range(n)] for _ in range(n)]
                for i in range(n):
                    for j in range(i + 1, n):
                        matrix[i][j] = 1
        else:
            # 传递锦标赛，i < j则i→j
            matrix = [[0 for _ in range(n)] for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    matrix[i][j] = 1

        # 将矩阵转换为字符串列表
        adj_str = [''.join(map(str, row)) for row in matrix]
        case = {
            'n': n,
            'adj_matrix': adj_str,
            'has_cycle': has_cycle,
            'cycle_vertices': vertices if has_cycle and n >=3 else None
        }
        return case

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        adj_matrix = question_case['adj_matrix']
        prompt = "你正在分析一个包含{}个顶点的锦标赛有向图。锦标赛的性质是，对于任意两个不同的顶点u和v，恰好存在一条有向边，要么u→v，要么v→u。你的任务是找出三个顶点a1, a2, a3，使得a1→a2，a2→a3，a3→a1。如果不存在这样的环，输出-1。请将你的答案放在[answer]和[/answer]标签之间。".format(n)
        prompt += "\n\n顶点的邻接矩阵如下：\n"
        for i in range(n):
            prompt += "顶点 {} 的邻接字符串：{}\n".format(i + 1, adj_matrix[i])
        prompt += "\n例如，输出可能是：\n[answer]1 3 2[/answer]\n或者\n[answer]-1[/answer]"
        return prompt

    @staticmethod
    def extract_output(output):
        pattern = r'\[answer\](.*?)\[/answer\]'
        matches = re.findall(pattern, output, re.DOTALL)
        if not matches:
            return None
        answer = matches[-1].strip()
        if answer == '-1':
            return -1
        else:
            parts = answer.split()
            if len(parts) != 3:
                return None
            try:
                a1, a2, a3 = map(int, parts)
                return (a1, a2, a3)
            except:
                return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution == -1:
            return not identity['has_cycle']
        else:
            a1, a2, a3 = solution
            a1 -= 1  # 转换为0-based索引
            a2 -= 1
            a3 -= 1
            adj_matrix = identity['adj_matrix']
            # 检查a1→a2
            if a1 < 0 or a1 >= len(adj_matrix) or a2 < 0 or a2 >= len(adj_matrix[a1]) or adj_matrix[a1][a2] != '1':
                return False
            # 检查a2→a3
            if a2 < 0 or a2 >= len(adj_matrix) or a3 < 0 or a3 >= len(adj_matrix[a2]) or adj_matrix[a2][a3] != '1':
                return False
            # 检查a3→a1
            if a3 < 0 or a3 >= len(adj_matrix) or a1 < 0 or a1 >= len(adj_matrix[a3]) or adj_matrix[a3][a1] != '1':
                return False
            return True
