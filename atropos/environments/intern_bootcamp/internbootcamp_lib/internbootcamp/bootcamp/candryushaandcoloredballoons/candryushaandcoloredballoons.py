"""# 

### 谜题描述
Andryusha goes through a park each day. The squares and paths between them look boring to Andryusha, so he decided to decorate them.

The park consists of n squares connected with (n - 1) bidirectional paths in such a way that any square is reachable from any other using these paths. Andryusha decided to hang a colored balloon at each of the squares. The baloons' colors are described by positive integers, starting from 1. In order to make the park varicolored, Andryusha wants to choose the colors in a special way. More precisely, he wants to use such colors that if a, b and c are distinct squares that a and b have a direct path between them, and b and c have a direct path between them, then balloon colors on these three squares are distinct.

Andryusha wants to use as little different colors as possible. Help him to choose the colors!

Input

The first line contains single integer n (3 ≤ n ≤ 2·105) — the number of squares in the park.

Each of the next (n - 1) lines contains two integers x and y (1 ≤ x, y ≤ n) — the indices of two squares directly connected by a path.

It is guaranteed that any square is reachable from any other using the paths.

Output

In the first line print single integer k — the minimum number of colors Andryusha has to use.

In the second line print n integers, the i-th of them should be equal to the balloon color on the i-th square. Each of these numbers should be within range from 1 to k.

Examples

Input

3
2 3
1 3


Output

3
1 3 2 

Input

5
2 3
5 3
4 3
1 3


Output

5
1 3 2 5 4 

Input

5
2 1
3 2
4 3
5 4


Output

3
1 2 3 1 2 

Note

In the first sample the park consists of three squares: 1 → 3 → 2. Thus, the balloon colors have to be distinct.

<image> Illustration for the first sample.

In the second example there are following triples of consequently connected squares: 

  * 1 → 3 → 2
  * 1 → 3 → 4
  * 1 → 3 → 5
  * 2 → 3 → 4
  * 2 → 3 → 5
  * 4 → 3 → 5

We can see that each pair of squares is encountered in some triple, so all colors have to be distinct. <image> Illustration for the second sample.

In the third example there are following triples: 

  * 1 → 2 → 3
  * 2 → 3 → 4
  * 3 → 4 → 5

We can see that one or two colors is not enough, but there is an answer that uses three colors only. <image> Illustration for the third sample.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
v=[[] for i in range(200005)]
a=[0]*200005
q=[]
m=0
n=input()
for i in range(n-1):
	x,y=map(int,raw_input().split())
	v[x].append(y)
	v[y].append(x)
q.append((1,0))
a[1]=1
for i in range(n):
	x,p=q[i]
	c=0
	for y in v[x]:
		if y==p:
			continue
		c+=1
		while c==a[x] or c==a[p]:
			c+=1
		a[y]=c
		q.append((y,x))
	m=max(m,c)
print m
for i in range(n):
	print a[i+1],
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from collections import deque
from bootcamp import Basebootcamp

class Candryushaandcoloredballoonsbootcamp(Basebootcamp):
    def __init__(self, tree_type='random', min_nodes=3, max_nodes=10):
        self.tree_type = tree_type
        self.min_nodes = max(3, min_nodes)
        self.max_nodes = max(self.min_nodes, max_nodes)
    
    def case_generator(self):
        n = random.randint(self.min_nodes, self.max_nodes)
        edges = []
        if self.tree_type == 'chain':
            edges = [(i, i+1) for i in range(1, n)]
        elif self.tree_type == 'star':
            if n < 1:
                n = 3
            center = 1
            edges = [(center, i) for i in range(2, n+1)]
        elif self.tree_type == 'random':
            edges = self.generate_random_tree(n)
        else:
            raise ValueError("Invalid tree type")
        return {"n": n, "edges": edges}
    
    @staticmethod
    def generate_random_tree(n):
        if n == 1:
            return []
        if n == 2:
            return [(1, 2)]
        prufer = [random.randint(1, n) for _ in range(n-2)]
        degree = [0] * (n + 1)
        for node in prufer:
            degree[node] += 1
        leaves = []
        for i in range(1, n+1):
            if degree[i] == 0:
                leaves.append(i)
        edges = []
        for node in prufer:
            if not leaves:
                break
            leaf = leaves.pop()
            edges.append((min(node, leaf), max(node, leaf)))
            degree[node] -= 1
            if degree[node] == 0:
                leaves.append(node)
            leaves = [i for i in range(1, n+1) if degree[i] == 0]
        leaves = [i for i in range(1, n+1) if degree[i] == 0]
        if len(leaves) >= 2:
            edges.append((min(leaves[0], leaves[1]), max(leaves[0], leaves[1])))
        return edges
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        edges = question_case['edges']
        input_lines = [f"{n}"] + [f"{x} {y}" for x, y in edges]
        input_str = '\n'.join(input_lines)
        prompt = (
            "Candryushaandcoloredballoons需要为公园的广场分配气球颜色。公园由n个广场和(n-1)条路径组成树状结构。要求若三个连续相连的广场颜色必须互不相同。请找出最小颜色数k，并给出每个广场的颜色。\n\n"
            "输入格式：\n"
            "第一行为n，接下来(n-1)行每行两个整数表示路径连接的广场。\n\n"
            "当前问题输入：\n"
            f"{input_str}\n\n"
            "请按照以下格式输出答案：\n"
            "[answer]\n"
            "k\n"
            "c1 c2 ... cn\n"
            "[/answer]"
        )
        return prompt
    
    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        last_answer = answer_blocks[-1].strip()
        lines = last_answer.split('\n')
        if len(lines) < 2:
            return None
        try:
            k = int(lines[0].strip())
            colors = list(map(int, lines[1].strip().split()))
            return {'k': k, 'colors': colors}
        except (ValueError, IndexError):
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or 'k' not in solution or 'colors' not in solution:
            return False
        user_k = solution['k']
        user_colors = solution['colors']
        n = identity['n']
        edges = identity['edges']
        if len(user_colors) != n:
            return False
        adj = [[] for _ in range(n+1)]
        for x, y in edges:
            adj[x].append(y)
            adj[y].append(x)
        max_degree = max(len(nodes) for nodes in adj[1:n+1])
        correct_k = max_degree + 1
        if user_k != correct_k:
            return False
        if any(c < 1 or c > user_k for c in user_colors):
            return False
        color_of = {i: user_colors[i-1] for i in range(1, n+1)}
        for b in range(1, n+1):
            neighbors = adj[b]
            for i in range(len(neighbors)):
                a = neighbors[i]
                for j in range(i+1, len(neighbors)):
                    c = neighbors[j]
                    a_color = color_of[a]
                    b_color = color_of[b]
                    c_color = color_of[c]
                    if a_color == b_color or a_color == c_color or b_color == c_color:
                        return False
        return True
