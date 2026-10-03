"""# 

### 谜题描述
I have an undirected graph consisting of n nodes, numbered 1 through n. Each node has at most two incident edges. For each pair of nodes, there is at most an edge connecting them. No edge connects a node to itself.

I would like to create a new graph in such a way that: 

  * The new graph consists of the same number of nodes and edges as the old graph. 
  * The properties in the first paragraph still hold. 
  * For each two nodes u and v, if there is an edge connecting them in the old graph, there is no edge connecting them in the new graph. 



Help me construct the new graph, or tell me if it is impossible.

Input

The first line consists of two space-separated integers: n and m (1 ≤ m ≤ n ≤ 105), denoting the number of nodes and edges, respectively. Then m lines follow. Each of the m lines consists of two space-separated integers u and v (1 ≤ u, v ≤ n; u ≠ v), denoting an edge between nodes u and v.

Output

If it is not possible to construct a new graph with the mentioned properties, output a single line consisting of -1. Otherwise, output exactly m lines. Each line should contain a description of edge in the same way as used in the input format.

Examples

Input

8 7
1 2
2 3
4 5
5 6
6 8
8 7
7 4


Output

1 4
4 6
1 6
2 7
7 5
8 5
2 8


Input

3 2
1 2
2 3


Output

-1


Input

5 4
1 2
2 3
3 4
4 1


Output

1 3
3 5
5 2
2 4

Note

The old graph of the first example:

<image>

A possible new graph for the first example:

<image>

In the second example, we cannot create any new graph.

The old graph of the third example:

<image>

A possible new graph for the third example:

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import random
s = raw_input().split()

n = int(s[0])
m = int(s[1])

graf = set([])

for i in xrange(m):
    s = raw_input().split()
    u = int(s[0])
    v = int(s[1])
    graf.add((u,v))
    
    #print graf
if m*4>n*(n-1):
    print -1
else:
    edges = 0
    tries = 0
    listv = range(1,n+1)
    while edges < m:
        tries += 1
        if (tries > 10000):
            break
        edges = 0

        random.shuffle(listv)
        for i in xrange(m):
            u = listv[i-1]
            v = listv[i]
            edges = i+1
            if ((u,v) in graf) or ((v,u) in graf):
                edges = 0
                break

    if (tries > 10000):
        print -1
    else:
        for i in xrange(m):
            print listv[i-1], listv[i]
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import defaultdict
from bootcamp import Basebootcamp

class Egraphreconstructionbootcamp(Basebootcamp):
    def __init__(self, max_nodes=10, prob_impossible=0.3):
        super().__init__()
        self.max_nodes = max_nodes
        self.prob_impossible = prob_impossible  # 控制不可解案例生成概率

    def case_generator(self):
        """智能案例生成：平衡可解/不可解案例"""
        # 特殊已知不可解案例
        if random.random() < 0.2:
            return {'n': 3, 'm': 2, 'edges': [(1,2), (2,3)], 'impossible': True}
        
        # 按概率尝试生成不可解案例
        if random.random() < self.prob_impossible:
            case = self._generate_impossible_case()
            if case:
                return case
        
        # 生成可解案例（确保存在解）
        for _ in range(100):
            n = random.randint(3, self.max_nodes)
            max_m = min(n, n*(n-1)//2)
            if max_m == 0:
                continue
            m = random.randint(1, max_m)
            case = self._generate_solvable_case(n, m)
            if case:
                return case
        
        # 回退到简单案例
        return {'n': 4, 'm': 2, 'edges': [(1,2), (3,4)], 'impossible': False}

    def _generate_impossible_case(self):
        """生成真正不可解的案例（数学约束+构造验证）"""
        for _ in range(100):
            # 1. 满足数学边界条件
            n = random.randint(5, self.max_nodes)  # n至少5以保证可能无法构造
            min_m = (n*(n-1)//4) + 1
            max_m = min(2*n, n*(n-1)//2)  # 边数上限调整为实际可能值
            if min_m > max_m:
                continue
            m = random.randint(min_m, max_m)
            
            # 2. 生成合法原始图
            edges = []
            degrees = defaultdict(int)
            candidates = [(i,j) for i in range(1,n+1) for j in range(i+1,n+1)]
            random.shuffle(candidates)
            
            for u, v in candidates:
                if len(edges) >= m:
                    break
                if degrees[u] < 2 and degrees[v] < 2:
                    edges.append((u, v))
                    degrees[u] += 1
                    degrees[v] += 1
            
            # 3. 验证实际无法构造解
            if len(edges) == m and self._is_truly_impossible(n, m, edges):
                return {'n': n, 'm': m, 'edges': edges, 'impossible': True}
        return None

    def _is_truly_impossible(self, n, m, original_edges):
        """双重验证：数学条件+构造验证"""
        # 数学条件验证
        if 4*m > n*(n-1):
            return True
        
        # 实际构造尝试（优化版）
        original_set = {tuple(sorted(e)) for e in original_edges}
        for _ in range(1000):
            # 使用参考算法的方法
            nodes = list(range(1, n+1))
            random.shuffle(nodes)
            valid = True
            for i in range(m):
                u = nodes[i-1]
                v = nodes[i]
                if tuple(sorted((u, v))) in original_set:
                    valid = False
                    break
            if valid:
                return False  # 存在解
        return True  # 未找到解

    def _generate_solvable_case(self, n, m):
        """生成保证可解的案例"""
        # 生成解方案
        nodes = list(range(1, n+1))
        random.shuffle(nodes)
        solution_edges = {tuple(sorted((nodes[i-1], nodes[i]))) for i in range(m)}
        
        # 生成原始边（排除解方案）
        edges = []
        degrees = defaultdict(int)
        candidates = [(i,j) for i in range(1,n+1) for j in range(i+1,n+1)
                     if tuple(sorted((i,j))) not in solution_edges]
        random.shuffle(candidates)
        
        for u, v in candidates:
            if len(edges) >= m:
                break
            if degrees[u] < 2 and degrees[v] < 2:
                edges.append((u, v))
                degrees[u] += 1
                degrees[v] += 1
        
        if len(edges) == m:
            return {'n': n, 'm': m, 'edges': edges, 'impossible': False}
        return None

    @staticmethod
    def prompt_func(question_case):
        edges = '\n'.join(f"{u} {v}" for u, v in question_case['edges'])
        return f"""You are a graph theory expert. Solve the following problem:

Original Graph:
- Nodes: {question_case['n']}
- Edges ({question_case['m']}):
{edges}

Task:
1. Create a new graph with same node/edge counts
2. Each node must have ≤2 edges
3. No overlapping edges with original
4. Output format:
   - If impossible: only '-1'
   - Else: {question_case['m']} edges in ascending order

Put your final answer within [ANSWER] tags like:
[ANSWER]
1 2
3 4
...[/ANSWER]"""

    @staticmethod
    def extract_output(output):
        import re
        # 优先匹配标签内容
        tag_match = re.search(r'\[ANSWER\](.*?)\[/ANSWER\]', output, re.DOTALL | re.IGNORECASE)
        if tag_match:
            content = tag_match.group(1).strip()
            if re.match(r'^\s*-1\s*$', content):
                return -1
            edges = []
            for line in content.splitlines():
                line = line.strip()
                if re.fullmatch(r'\d+\s+\d+', line):
                    u, v = map(int, line.split())
                    edges.append((min(u,v), max(u,v)))
            return edges if edges else None
        
        # 次优匹配：最后的-1或边序列
        lines = [line.strip() for line in output.splitlines()]
        for line in reversed(lines):
            if line == '-1':
                return -1
        edges = []
        for line in reversed(lines):
            if re.fullmatch(r'\d+\s+\d+', line):
                u, v = map(int, line.split())
                edges.append((min(u,v), max(u,v)))
            elif edges:
                break
        return list(reversed(edges)) if edges else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        original_edges = {tuple(sorted(e)) for e in identity['edges']}
        n, m = identity['n'], identity['m']
        
        if solution == -1:
            # 双重验证机制
            return cls._is_truly_impossible_wrapper(n, m, original_edges)
        
        # 验证正解
        if len(solution) != m:
            return False
        
        seen = set()
        degrees = defaultdict(int)
        for u, v in solution:
            # 数值有效性
            if not (1 <= u <= n) or not (1 <= v <= n) or u == v:
                return False
            edge = (min(u,v), max(u,v))
            
            # 冲突检查
            if edge in seen or edge in original_edges:
                return False
            seen.add(edge)
            
            # 度数检查
            degrees[u] += 1
            degrees[v] += 1
            if degrees[u] > 2 or degrees[v] > 2:
                return False
        
        return True

    @classmethod
    def _is_truly_impossible_wrapper(cls, n, m, original_edges):
        """统一的不可解验证逻辑"""
        # 快速数学检查
        if 4*m > n*(n-1):
            return True
        
        # 构造试错（优化性能）
        original_set = original_edges
        for _ in range(1000):
            nodes = list(range(1, n+1))
            random.shuffle(nodes)
            conflict = False
            for i in range(m):
                u = nodes[i-1]
                v = nodes[i]
                if tuple(sorted((u, v))) in original_set:
                    conflict = True
                    break
            if not conflict:
                return False  # 存在解
        return True
