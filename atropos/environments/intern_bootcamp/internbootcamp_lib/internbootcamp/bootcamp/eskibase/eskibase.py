"""# 

### 谜题描述
A ski base is planned to be built in Walrusland. Recently, however, the project is still in the constructing phase. A large land lot was chosen for the construction. It contains n ski junctions, numbered from 1 to n. Initially the junctions aren't connected in any way.

In the constructing process m bidirectional ski roads will be built. The roads are built one after another: first the road number 1 will be built, then the road number 2, and so on. The i-th road connects the junctions with numbers ai and bi.

Track is the route with the following properties: 

  * The route is closed, that is, it begins and ends in one and the same junction.
  * The route contains at least one road. 
  * The route doesn't go on one road more than once, however it can visit any junction any number of times. 



Let's consider the ski base as a non-empty set of roads that can be divided into one or more tracks so that exactly one track went along each road of the chosen set. Besides, each track can consist only of roads from the chosen set. Ski base doesn't have to be connected.

Two ski bases are considered different if they consist of different road sets.

After building each new road the Walrusland government wants to know the number of variants of choosing a ski base based on some subset of the already built roads. The government asks you to help them solve the given problem.

Input

The first line contains two integers n and m (2 ≤ n ≤ 105, 1 ≤ m ≤ 105). They represent the number of junctions and the number of roads correspondingly. Then on m lines follows the description of the roads in the order in which they were built. Each road is described by a pair of integers ai and bi (1 ≤ ai, bi ≤ n, ai ≠ bi) — the numbers of the connected junctions. There could be more than one road between a pair of junctions.

Output

Print m lines: the i-th line should represent the number of ways to build a ski base after the end of construction of the road number i. The numbers should be printed modulo 1000000009 (109 + 9).

Examples

Input

3 4
1 3
2 3
1 2
1 2


Output

0
0
1
3

Note

Let us have 3 junctions and 4 roads between the junctions have already been built (as after building all the roads in the sample): 1 and 3, 2 and 3, 2 roads between junctions 1 and 2. The land lot for the construction will look like this:

<image>

The land lot for the construction will look in the following way:

<image>

We can choose a subset of roads in three ways:

<image>

In the first and the second ways you can choose one path, for example, 1 - 2 - 3 - 1. In the first case you can choose one path 1 - 2 - 1.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import random

def FindSet(dsu, x):
	if dsu[x] != x:
		dsu[x] = FindSet(dsu, dsu[x])
	return dsu[x]

def Unite(dsu, x, y):
	x = FindSet(dsu, x)
	y = FindSet(dsu, y)
	if random.random() > 0.5:
		dsu[x] = y
	else:
		dsu[y] = x

mod = 10 ** 9 + 9
n, m = map(int, raw_input().split())
dsu = range(n + 1)
res = 1
for i in xrange(m):
	u, v = map(int, raw_input().split())
	if FindSet(dsu, u) != FindSet(dsu, v):
		print res - 1
	else:
		res = (res * 2) % mod
		print res - 1
	Unite(dsu, u, v)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

mod = 10**9 + 9

class Eskibasebootcamp(Basebootcamp):
    def __init__(self, n=3, m=4):
        self.n = n
        self.m = m
    
    def case_generator(self):
        # 分离边生成与合并操作的随机性
        edge_rng = random.Random()
        edge_seed = random.getrandbits(32)
        edge_rng.seed(edge_seed)
        
        # 生成边列表
        edges = []
        for _ in range(self.m):
            while True:
                ai = edge_rng.randint(1, self.n)
                bi = edge_rng.randint(1, self.n)
                if ai != bi:
                    edges.append((ai, bi))
                    break
        
        # 创建独立的合并随机生成器
        unite_rng = random.Random()
        unite_seed = random.getrandbits(32)
        unite_rng.seed(unite_seed)
        
        # 计算预期输出
        dsu = list(range(self.n + 1))
        res = 1
        expected_outputs = []
        
        for u, v in edges:
            u_root = self.find_set(dsu, u)
            v_root = self.find_set(dsu, v)
            
            if u_root != v_root:
                expected_outputs.append((res - 1) % mod)
                # 使用独立随机源决定合并方向
                if unite_rng.random() > 0.5:
                    dsu[u_root] = v_root
                else:
                    dsu[v_root] = u_root
            else:
                res = (res * 2) % mod
                expected_outputs.append((res - 1) % mod)
        
        return {
            "n": self.n,
            "m": self.m,
            "edges": edges,
            "expected_outputs": expected_outputs
        }

    @staticmethod
    def find_set(dsu, x):
        if dsu[x] != x:
            dsu[x] = Eskibasebootcamp.find_set(dsu, dsu[x])
        return dsu[x]

    @staticmethod
    def prompt_func(question_case) -> str:
        input_desc = f"{question_case['n']} {question_case['m']}\n" + \
                     '\n'.join(f"{u} {v}" for u, v in question_case['edges'])
        
        return f"""Walrusland滑雪基地建设问题

任务背景：
我们正在规划建设一个滑雪基地网络，包含{question_case['n']}个枢纽（编号1-{question_case['n']}），按顺序修建{question_case['m']}条双向雪道。每次完成道路建设后，需要计算当前所有已建道路中可以组成滑雪基地的方案数（模1000000009）。

滑雪基地定义：
1. 由非空道路集合构成
2. 可以划分为若干闭合路线（track），每个track满足：
   - 闭合路径（起点终点相同）
   - 每条道路最多使用一次
   - 可以使用任意多次枢纽
3. 不同道路集合视为不同的基地

输入格式：
{input_desc}

输出要求：
共输出{question_case['m']}行，每行对应修建完第i条道路后的方案数

请将答案按顺序放在[answer]和[/answer]之间，每行一个数值。例如：
[answer]
0
0
1
3
[/answer]

当前问题输入：
{input_desc}

请计算并输出正确结果："""

    @staticmethod
    def extract_output(output):
        import re
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        
        # 提取最后一个有效答案块并验证完整性
        last_block = answer_blocks[-1].strip()
        solutions = []
        valid_lines = 0
        
        for line in last_block.split('\n'):
            line = line.strip()
            if line:
                try:
                    num = int(line) % mod
                    solutions.append(num)
                    valid_lines += 1
                except ValueError:
                    continue
        
        # 严格验证行数匹配
        if valid_lines != len(solutions) or valid_lines == 0:
            return None
        return solutions

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 严格验证答案长度和内容
        return solution == identity['expected_outputs'] if solution else False
