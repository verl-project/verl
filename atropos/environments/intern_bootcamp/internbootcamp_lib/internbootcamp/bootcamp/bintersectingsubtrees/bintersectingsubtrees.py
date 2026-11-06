"""# 

### 谜题描述
You are playing a strange game with Li Chen. You have a tree with n nodes drawn on a piece of paper. All nodes are unlabeled and distinguishable. Each of you independently labeled the vertices from 1 to n. Neither of you know the other's labelling of the tree.

You and Li Chen each chose a subtree (i.e., a connected subgraph) in that tree. Your subtree consists of the vertices labeled x_1, x_2, …, x_{k_1} in your labeling, Li Chen's subtree consists of the vertices labeled y_1, y_2, …, y_{k_2} in his labeling. The values of x_1, x_2, …, x_{k_1} and y_1, y_2, …, y_{k_2} are known to both of you.

<image> The picture shows two labelings of a possible tree: yours on the left and Li Chen's on the right. The selected trees are highlighted. There are two common nodes.

You want to determine whether your subtrees have at least one common vertex. Luckily, your friend Andrew knows both labelings of the tree. You can ask Andrew at most 5 questions, each of which is in one of the following two forms: 

  * A x: Andrew will look at vertex x in your labeling and tell you the number of this vertex in Li Chen's labeling. 
  * B y: Andrew will look at vertex y in Li Chen's labeling and tell you the number of this vertex in your labeling. 



Determine whether the two subtrees have at least one common vertex after asking some questions. If there is at least one common vertex, determine one of your labels for any of the common vertices.

Interaction

Each test consists of several test cases.

The first line of input contains a single integer t (1 ≤ t ≤ 100) — the number of test cases.

For each testcase, your program should interact in the following format.

The first line contains a single integer n (1 ≤ n ≤ 1 000) — the number of nodes in the tree.

Each of the next n-1 lines contains two integers a_i and b_i (1≤ a_i, b_i≤ n) — the edges of the tree, indicating an edge between node a_i and b_i according to your labeling of the nodes.

The next line contains a single integer k_1 (1 ≤ k_1 ≤ n) — the number of nodes in your subtree.

The next line contains k_1 distinct integers x_1,x_2,…,x_{k_1} (1 ≤ x_i ≤ n) — the indices of the nodes in your subtree, according to your labeling. It is guaranteed that these vertices form a subtree.

The next line contains a single integer k_2 (1 ≤ k_2 ≤ n) — the number of nodes in Li Chen's subtree.

The next line contains k_2 distinct integers y_1, y_2, …, y_{k_2} (1 ≤ y_i ≤ n) — the indices (according to Li Chen's labeling) of the nodes in Li Chen's subtree. It is guaranteed that these vertices form a subtree according to Li Chen's labelling of the tree's nodes.

Test cases will be provided one by one, so you must complete interacting with the previous test (i.e. by printing out a common node or -1 if there is not such node) to start receiving the next one.

You can ask the Andrew two different types of questions. 

  * You can print \"A x\" (1 ≤ x ≤ n). Andrew will look at vertex x in your labeling and respond to you with the number of this vertex in Li Chen's labeling. 
  * You can print \"B y\" (1 ≤ y ≤ n). Andrew will look at vertex y in Li Chen's labeling and respond to you with the number of this vertex in your labeling. 



You may only ask at most 5 questions per tree.

When you are ready to answer, print \"C s\", where s is your label of a vertex that is common to both subtrees, or -1, if no such vertex exists. Printing the answer does not count as a question. Remember to flush your answer to start receiving the next test case. 

After printing a question do not forget to print end of line and flush the output. Otherwise, you will get Idleness limit exceeded. To do this, use:

  * fflush(stdout) or cout.flush() in C++; 
  * System.out.flush() in Java; 
  * flush(output) in Pascal; 
  * stdout.flush() in Python; 
  * see documentation for other languages. 



If the judge responds with -1, it means that you asked more queries than allowed, or asked an invalid query. Your program should immediately terminate (for example, by calling exit(0)). You will receive Wrong Answer; it means that you asked more queries than allowed, or asked an invalid query. If you ignore this, you can get other verdicts since your program will continue to read from a closed stream.

Hack Format

To hack, use the following format. Note that you can only hack with one test case.

The first line should contain a single integer t (t=1).

The second line should contain a single integer n (1 ≤ n ≤ 1 000).

The third line should contain n integers p_1, p_2, …, p_n (1≤ p_i≤ n) — a permutation of 1 to n. This encodes the labels that Li Chen chose for his tree. In particular, Li Chen chose label p_i for the node you labeled i.

Each of the next n-1 lines should contain two integers a_i and b_i (1≤ a_i, b_i≤ n). These edges should form a tree.

The next line should contain a single integer k_1 (1 ≤ k_1 ≤ n).

The next line should contain k_1 distinct integers x_1,x_2,…,x_{k_1} (1 ≤ x_i ≤ n). These vertices should form a subtree.

The next line should contain a single integer k_2 (1 ≤ k_2 ≤ n).

The next line should contain k_2 distinct integers y_1, y_2, …, y_{k_2} (1 ≤ y_i ≤ n). These vertices should form a subtree in Li Chen's tree according to the permutation above.

Examples

Input

1
3
1 2
2 3
1
1
1
2
2
1


Output

A 1
B 2
C 1


Input

2
6
1 2
1 3
1 4
4 5
4 6
4
1 3 4 5
3
3 5 2
3
6
1 2
1 3
1 4
4 5
4 6
3
1 2 3
3
4 1 6
5


Output

B 2
C 1
A 1
C -1

Note

For the first sample, Li Chen's hidden permutation is [2, 3, 1], and for the second, his hidden permutation is [5, 3, 2, 4, 1, 6] for both cases.

In the first sample, there is a tree with three nodes in a line. On the top, is how you labeled the tree and the subtree you chose, and the bottom is how Li Chen labeled the tree and the subtree he chose: 

<image>

In the first question, you ask Andrew to look at node 1 in your labelling and tell you the label of it in Li Chen's labelling. Andrew responds with 2. At this point, you know that both of your subtrees contain the same node (i.e. node 1 according to your labeling), so you can output \"C 1\" and finish. However, you can also ask Andrew to look at node 2 in Li Chen's labelling and tell you the label of it in your labelling. Andrew responds with 1 (this step was given with the only reason — to show you how to ask questions).

For the second sample, there are two test cases. The first looks is the one from the statement: 

<image>

We first ask \"B 2\", and Andrew will tell us 3. In this case, we know 3 is a common vertex, and moreover, any subtree with size 3 that contains node 3 must contain node 1 as well, so we can output either \"C 1\" or \"C 3\" as our answer.

In the second case in the second sample, the situation looks as follows: 

<image>

In this case, you know that the only subtree of size 3 that doesn't contain node 1 is subtree 4,5,6. You ask Andrew for the label of node 1 in Li Chen's labelling and Andrew says 5. In this case, you know that Li Chen's subtree doesn't contain node 1, so his subtree must be consist of the nodes 4,5,6 (in your labelling), thus the two subtrees have no common nodes.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys

def bfs(edges,root,arr):
    q = [root]
    visited = [0]*(1+n)
    visited[root] = 1
    des = [0]*(1+n)
    for i in arr:
        des[i] = 1

    i = 0
    while i < len(q):
        v = q[i]
        if des[v] == 1:
            return v
        for node in edges[v]:
            if not visited[node]:
                visited[node] = 1
                q.append(node)


        i += 1


t = int(input())
for _ in range(t):
    n = int(input())
    edges = [[]for i in range(1+n)]
    for i in range(n-1):
        a,b = map(int,raw_input().split(\" \"))
        edges[a].append(b)
        edges[b].append(a)
    k1 = int(input())
    arr1 = map(int,raw_input().split(\" \"))
    k2 = int(input())
    arr2 = map(int,raw_input().split(\" \"))
    print \"B\",arr2[0]
    sys.stdout.flush()
    v = int(input())
    node = bfs(edges,v,arr1)
    print \"A\",node
    sys.stdout.flush()
    v = int(input())
    if v in arr2:
        print \"C\",node
    else:
        print \"C -1\"
    sys.stdout.flush()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import json
import random
from collections import deque
from bootcamp import Basebootcamp

class Bintersectingsubtreesbootcamp(Basebootcamp):
    def __init__(self, max_n=1000, default_n=3, max_questions=5):
        self.max_n = max_n
        self.default_n = default_n
        self.max_questions = max_questions

    def case_generator(self):
        """增强型案例生成，保证双方子树的有效性"""
        n = random.randint(1, self.max_n)
        
        # 生成合法树结构
        edges = self._generate_spanning_tree(n)
        
        # 生成双向标号映射
        permutation = list(range(1, n+1))
        random.shuffle(permutation)
        reverse_perm = [0]*(n+1)
        for i in range(n):
            reverse_perm[permutation[i]] = i+1
        
        # 生成用户子树（保证连通）
        k1 = random.randint(1, n)
        x_list = self._find_connected_subgraph(edges, k1)
        
        # 生成李晨子树（基于映射后的树结构）
        lc_edges = [(permutation[u-1], permutation[v-1]) for u, v in edges]
        k2 = random.randint(1, n)
        y_list = self._find_connected_subgraph(lc_edges, k2)
        
        return {
            'n': n,
            'edges': edges,
            'permutation': permutation,
            'k1': k1,
            'x_list': sorted(x_list),
            'k2': k2,
            'y_list': sorted(y_list)
        }

    @staticmethod
    def prompt_func(case) -> str:
        edges_str = '\n'.join(f"{u} {v}" for u, v in case['edges'])
        return f"""树结构（你的标号）：
节点数：{case['n']}
边列表：
{edges_str}

你的子树节点：{case['x_list']}
李晨的子树节点（他的标号）：{case['y_list']}

通过至多5次A/B查询确定共同节点。答案置于[answer]标签内。例：[answer]3[/answer] 或 [answer]-1[/answer]"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.IGNORECASE)
        if not matches: return None
        try: return int(matches[-1].strip())
        except: return None

    @classmethod
    def _verify_correction(cls, solution, case):
        x_set = set(case['x_list'])
        y_set = set(case['y_list'])
        perm = case['permutation']
        
        # 验证solution格式
        if solution == -1:
            return all(perm[x-1] not in y_set for x in x_set)
        else:
            return solution in x_set and perm[solution-1] in y_set

    # 增强的辅助方法
    @staticmethod
    def _generate_spanning_tree(n):
        """生成保证连通的树结构"""
        if n == 1: return []
        nodes = list(range(1, n+1))
        random.shuffle(nodes)
        root = nodes[0]
        edges = []
        connected = {root}
        for node in nodes[1:]:
            neighbor = random.choice(list(connected))
            edges.append((neighbor, node))
            connected.add(node)
        return edges

    def _find_connected_subgraph(self, edges, k):
        """DFS生成连通子图"""
        adj = [[] for _ in range(self.max_n+2)]  # 兼容大节点编号
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        start = random.choice([u for u, v in edges] + [v for u, v in edges]) if edges else 1
        visited = set()
        stack = [start]
        while stack and len(visited) < k:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                neighbors = adj[node]
                random.shuffle(neighbors)
                stack.extend(neighbors)
        return sorted(list(visited)[:k])
