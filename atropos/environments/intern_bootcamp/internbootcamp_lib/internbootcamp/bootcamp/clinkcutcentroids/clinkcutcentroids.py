"""# 

### 谜题描述
Fishing Prince loves trees, and he especially loves trees with only one centroid. The tree is a connected graph without cycles.

A vertex is a centroid of a tree only when you cut this vertex (remove it and remove all edges from this vertex), the size of the largest connected component of the remaining graph is the smallest possible.

For example, the centroid of the following tree is 2, because when you cut it, the size of the largest connected component of the remaining graph is 2 and it can't be smaller.

<image>

However, in some trees, there might be more than one centroid, for example:

<image>

Both vertex 1 and vertex 2 are centroids because the size of the largest connected component is 3 after cutting each of them.

Now Fishing Prince has a tree. He should cut one edge of the tree (it means to remove the edge). After that, he should add one edge. The resulting graph after these two operations should be a tree. He can add the edge that he cut.

He wants the centroid of the resulting tree to be unique. Help him and find any possible way to make the operations. It can be proved, that at least one such way always exists.

Input

The input consists of multiple test cases. The first line contains an integer t (1≤ t≤ 10^4) — the number of test cases. The description of the test cases follows.

The first line of each test case contains an integer n (3≤ n≤ 10^5) — the number of vertices.

Each of the next n-1 lines contains two integers x, y (1≤ x,y≤ n). It means, that there exists an edge connecting vertices x and y.

It's guaranteed that the given graph is a tree.

It's guaranteed that the sum of n for all test cases does not exceed 10^5.

Output

For each test case, print two lines.

In the first line print two integers x_1, y_1 (1 ≤ x_1, y_1 ≤ n), which means you cut the edge between vertices x_1 and y_1. There should exist edge connecting vertices x_1 and y_1.

In the second line print two integers x_2, y_2 (1 ≤ x_2, y_2 ≤ n), which means you add the edge between vertices x_2 and y_2.

The graph after these two operations should be a tree.

If there are multiple solutions you can print any.

Example

Input


2
5
1 2
1 3
2 4
2 5
6
1 2
1 3
1 4
2 5
2 6


Output


1 2
1 2
1 3
2 3

Note

Note that you can add the same edge that you cut.

In the first test case, after cutting and adding the same edge, the vertex 2 is still the only centroid.

In the second test case, the vertex 2 becomes the only centroid after cutting the edge between vertices 1 and 3 and adding the edge between vertices 2 and 3.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import math

def calc(n, graph):
    visited = [False] * (n+1)
    subtree = [0] * (n+1)

    # dfs over tree finding subtree sizes
    stack = [1]
    while stack:
        node = stack[-1]
        if visited[node]:
            stack.pop()
            subtree[node] = 1
            for c in graph[node]:
                subtree[node] += subtree[c]
            continue
        visited[node] = True
        for c in graph[node]:
            if not visited[c]:
                stack.append(c)

    visited = [False] * (n+1)
    centroids = []
    # find centroid
    stack = [1]
    centroid = None
    while stack:
        node = stack.pop()
        visited[node] = True
        appended = False
        for c in graph[node]:
            if not visited[c] and subtree[c] > n // 2:
                stack.append(c)
                appended = True
        if appended:
            continue
        centroid = node
        break

    centroids.append(centroid)
    # found a centroid. is it unique?
    for c in graph[centroid]:
        if not visited[c] and subtree[c] >= math.ceil(float(n) / 2):
            centroids.append(c)
    
    if len(centroids) == 1:
        # remove and append any edge
        s = centroids[0]
        e = graph[s][0]
        return ((e,s), (e,s))

    # two centroids, have to fix
    # move single node to other centroid?
    heavy_node = centroids[1]
    stack = [(centroid,heavy_node)]
    light_edge = None
    while stack:
        par, node = stack.pop()
        if subtree[node] == 1:
            light_edge = (par,node)
            break
        for c in graph[node]:
            if c != par:
                stack.append((node,c))

    return (light_edge,(centroid,light_edge[1]))

def main():
    sol = ''
    num_lines = int(input())
    for i in range(num_lines):
        graph = {}
        n = int(input())
        for e in range(n-1):
            line = raw_input().split(' ')
            s, e = int(line[0]), int(line[1])
            if s not in graph:
                graph[s] = []
            if e not in graph:
                graph[e] = []
            graph[s].append(e)
            graph[e].append(s)
        t = calc(n, graph)
        sol += str(t[0][0]) + ' ' + str(t[0][1]) + '\n' + str(t[1][0]) + ' ' + str(t[1][1])
        if i < num_lines-1:
            sol += '\n'
    print(sol)

# g = {}
# for i in range(1,100000):
#     g[i] = [i+1]
#     g[i+1] = [i]
# t = calc(100000,g)
# print(str(t[0][0]) + ' ' + str(t[0][1]) + '\n' + str(t[1][0]) + ' ' + str(t[1][1]) + '\n')
main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import defaultdict
from bootcamp import Basebootcamp

class Clinkcutcentroidsbootcamp(Basebootcamp):
    def __init__(self, min_n=3, max_n=20):
        self.min_n = min_n
        self.max_n = max_n
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        if n == 1:
            return {'n': 1, 'edges': []}
        if n == 2:
            return {'n': 2, 'edges': [(1, 2)]}

        # 使用改进的Prüfer序列生成器
        prufer = [random.randint(1, n) for _ in range(n-2)]
        degree = [1] * (n+1)
        for node in prufer:
            degree[node] += 1

        edges = []
        ptr = 1
        while degree[ptr] != 1:
            ptr += 1
        leaf = ptr

        for node in prufer:
            edges.append((leaf, node))
            degree[leaf] -= 1
            degree[node] -= 1
            if degree[node] == 1 and node < ptr:
                leaf = node
            else:
                ptr += 1
                while ptr <= n and degree[ptr] != 1:
                    ptr += 1
                leaf = ptr

        edges.append((leaf, next(i for i in range(1, n+1) if degree[i] == 1 and i != leaf)))
        return {'n': n, 'edges': [(min(u,v), max(u,v)) for u,v in edges]}
    
    @staticmethod
    def prompt_func(question_case):
        edges = question_case['edges']
        n = question_case['n']
        edges_str = "\n".join(f"{u} {v}" for u, v in edges)
        return f"""Given a tree with {n} vertices connected by these edges:
{edges_str}

Find an edge to cut and an edge to add so that the resulting tree has exactly one centroid. Format your answer as:

[answer]
cut_u cut_v
add_u add_v
[/answer]"""

    @staticmethod
    def extract_output(output):
        try:
            answer_block = output.split('[answer]')[-1].split('[/answer]')[0].strip()
            lines = [line.strip() for line in answer_block.split('\n') if line.strip()]
            cut = tuple(map(int, lines[0].split()))
            add = tuple(map(int, lines[1].split()))
            return (cut, add)
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # Edge case validation
        if not solution or len(solution) != 2:
            return False
        cut, add = solution
        n = identity['n']
        edges = {frozenset((u, v)) for u, v in identity['edges']}
        
        # Validate cut edge exists
        if frozenset(cut) not in edges:
            return False
        
        # Build new edge set
        new_edges = [e for e in edges if e != frozenset(cut)]
        new_edges.append(frozenset(add))
        if len(new_edges) != n-1:
            return False
        
        # Check connectivity
        adj = defaultdict(list)
        for e in new_edges:
            u, v = e
            adj[u].append(v)
            adj[v].append(u)
        
        visited = set()
        stack = [1]  # Trees are connected by definition
        while stack:
            u = stack.pop()
            if u not in visited:
                visited.add(u)
                for v in adj[u]:
                    if v not in visited:
                        stack.append(v)
        if len(visited) != n:
            return False
        
        # Centroid verification
        centroids = cls.find_centroids(n, adj)
        return len(centroids) == 1

    @staticmethod
    def find_centroids(n, adj):
        subtree = [0]*(n+1)
        
        def dfs(u, parent):
            subtree[u] = 1
            for v in adj[u]:
                if v != parent:
                    dfs(v, u)
                    subtree[u] += subtree[v]
        
        dfs(1, -1)  # Root at node 1
        
        centroids = []
        min_max = float('inf')
        for u in range(1, n+1):
            max_size = max(
                (n - subtree[u], 
                 max((subtree[v] for v in adj[u] if v != parent), default=0))
                for parent in [-1]  # Simplified check
            )[0]
            if max_size < min_max:
                min_max = max_size
                centroids = [u]
            elif max_size == min_max:
                centroids.append(u)
        return list(set(centroids))
