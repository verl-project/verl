"""# 

### 谜题描述
This is an interactive problem.

Little Dormi was faced with an awkward problem at the carnival: he has to guess the edges of an unweighted tree of n nodes! The nodes of the tree are numbered from 1 to n.

The game master only allows him to ask one type of question:

  * Little Dormi picks a node r (1 ≤ r ≤ n), and the game master will reply with an array d_1, d_2, …, d_n, where d_i is the length of the shortest path from node r to i, for all 1 ≤ i ≤ n.



Additionally, to make the game unfair challenge Little Dormi the game master will allow at most ⌈n/2⌉ questions, where ⌈ x ⌉ denotes the smallest integer greater than or equal to x.

Faced with the stomach-churning possibility of not being able to guess the tree, Little Dormi needs your help to devise a winning strategy!

Note that the game master creates the tree before the game starts, and does not change it during the game.

Input

The first line of input contains the integer n (2 ≤ n ≤ 2 000), the number of nodes in the tree.

You will then begin interaction.

Output

When your program has found the tree, first output a line consisting of a single \"!\" followed by n-1 lines each with two space separated integers a and b, denoting an edge connecting nodes a and b (1 ≤ a, b ≤ n). Once you are done, terminate your program normally immediately after flushing the output stream.

You may output the edges in any order and an edge (a,b) is considered the same as an edge (b,a). Answering is not considered as a query.

Interaction

After taking input, you may make at most ⌈n/2⌉ queries. Each query is made in the format \"? r\", where r is an integer 1 ≤ r ≤ n that denotes the node you want to pick for that query.

You will then receive n space separated integers d_1, d_2, …, d_n, where d_i is the length of the shortest path from node r to i, followed by a newline.

After printing a query do not forget to output end of line and flush the output. Otherwise, you will get Idleness limit exceeded. To do this, use:

  * fflush(stdout) or cout.flush() in C++; 
  * System.out.flush() in Java; 
  * flush(output) in Pascal; 
  * stdout.flush() in Python; 
  * see documentation for other languages. 



If at any point you make an invalid query or try to make more than ⌈ n/2 ⌉ queries, the interaction will terminate immediately and you will receive a Wrong Answer verdict.

Hacks

To hack a solution, use the following format.

The first line contains the integer n (2 ≤ n ≤ 2 000).

The next n−1 lines contain two integers u and v (1 ≤ u,v ≤ n) denoting an edge between u and v (u ≠ v). These n-1 edges must form a tree.

Examples

Input


4

0 1 2 2

1 0 1 1

Output


? 1

? 2

!
4 2
1 2
2 3


Input


5

2 2 1 1 0


Output


? 5

!
4 5
3 5
2 4
1 3

Note

Here is the tree from the first example.

<image>

Notice that the edges can be output in any order.

Additionally, here are the answers for querying every single node in example 1:

  * 1: [0,1,2,2] 
  * 2: [1,0,1,1] 
  * 3: [2,1,0,2] 
  * 4: [2,1,2,0]



Below is the tree from the second example interaction.

<image>

Lastly, here are the answers for querying every single node in example 2:

  * 1: [0,4,1,3,2] 
  * 2: [4,0,3,1,2] 
  * 3: [1,3,0,2,1] 
  * 4: [3,1,2,0,1] 
  * 5: [2,2,1,1,0]

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
import random
import collections

test = False
edges = [(1, 2) ,(1, 3) ,(1, 4) ,(2, 5) ,(2, 6) ,(3, 7) ,(3, 8) ,(4, 9) ,(4, 10) ,(5, 11) ,(5, 12)]
adj = collections.defaultdict(list)
for u,v in edges:
	adj[u].append(v)
	adj[v].append(u)

def ask(l, n):

	if test:
		q = collections.deque([(l,0)])
		vis = set([l])
		h = [+float('inf') for _ in range(0, n+1)]
		while(q):
			u,dis = q.popleft()
			h[u] = dis
			for v in adj[u]:
				if v not in vis:
					vis.add(v)
					q.append((v,dis+1))

		return h[1:]

	query = '? {}'.format(l)
	print query
	sys.stdout.flush()
	return map(int,raw_input().split())


def work(u, bis, res):
	for v,val in enumerate(bis):
		if val == 1:
			uu,vv = u, v+1
			if uu > vv:
				uu,vv = vv,uu
			res.add((uu,vv))



n = int(raw_input()) if not(test) else 12
bis = ask(1, n)
res = set([])

cc = collections.defaultdict(collections.deque)
for i,di in enumerate(bis):
	if i == 0:
		work(1, bis, res)
		continue
	cc[di % 2].append(i+1)

for u in (cc[1] if len(cc[1]) <= len(cc[0]) else cc[0]):
	work(u, ask(u,n), res)

print '!'
for u,v in res:
	print u,v
#print res  == set(edges)
sys.stdout.flush()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Dlosttreebootcamp(Basebootcamp):
    def __init__(self, **params):
        """
        初始化训练场参数，默认随机生成节点数范围2-20
        """
        self.params = params

    def case_generator(self):
        """
        生成随机树结构实例
        """
        n = self.params.get('n', random.randint(2, 20))
        
        def generate_random_tree(size):
            if size == 1:
                return []
            nodes = list(range(1, size+1))
            random.shuffle(nodes)
            edges = []
            for i in range(1, size):
                parent = random.choice(nodes[:i])
                edges.append([nodes[i], parent])
            return edges
        
        edges = generate_random_tree(n)
        return {'n': n, 'edges': edges}

    @staticmethod
    def prompt_func(question_case) -> str:
        """
        生成包含谜题背景、规则和格式要求的完整问题描述
        """
        n = question_case['n']
        query_limit = (n + 1) // 2  # ⌈n/2⌉
        
        prompt = f"""你是嘉年华游戏的参与者，需要猜出包含{n}个节点的树结构。节点编号为1到{n}，树边需要完全还原。

## 游戏规则
1. 你可以进行最多{query_limit}次距离查询
2. 每次查询格式："? r" (1 ≤ r ≤ {n})
3. 每次返回n个整数表示各节点到r的最短距离

## 胜利条件
通过有限次数的查询确定所有树边。正确输出格式：
1. 第一行为"!"
2. 后续n-1行每行两个用空格分隔的整数表示边

## 当前题目
需要还原的树共有{n}个节点。请设计查询策略并输出正确答案。

请将最终答案按以下格式包裹在[answer]标签内：
[answer]
!
a1 b1
a2 b2
...[/answer]"""

        return prompt

    @staticmethod
    def extract_output(output):
        """
        从模型输出中提取最后一个[answer]块内的边数据
        """
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
            
        last_block = answer_blocks[-1].strip()
        edges = []
        for line in last_block.split('\n'):
            line = line.strip()
            if line == '!' or not line:
                continue
            match = re.fullmatch(r'\s*(\d+)\s+(\d+)\s*', line)
            if match:
                try:
                    a, b = int(match.group(1)), int(match.group(2))
                    edges.append((a, b))
                except:
                    continue
        return edges if edges else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """
        验证答案边集合与原始树结构是否一致
        """
        if not solution:
            return False
        
        n = identity['n']
        if len(solution) != n - 1:
            return False

        try:
            # 将提交的答案转换为标准化边集合
            submitted = {tuple(sorted(e)) for e in solution}
            # 原始树的标准边集合
            expected = {tuple(sorted(edge)) for edge in identity['edges']}
        except:
            return False

        return submitted == expected and len(submitted) == n - 1
