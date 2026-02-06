"""# 

### 谜题描述
A tree of size n is an undirected connected graph consisting of n vertices without cycles.

Consider some tree with n vertices. We call a tree invariant relative to permutation p = p1p2... pn, if for any two vertices of the tree u and v the condition holds: \"vertices u and v are connected by an edge if and only if vertices pu and pv are connected by an edge\".

You are given permutation p of size n. Find some tree size n, invariant relative to the given permutation.

Input

The first line contains number n (1 ≤ n ≤ 105) — the size of the permutation (also equal to the size of the sought tree).

The second line contains permutation pi (1 ≤ pi ≤ n).

Output

If the sought tree does not exist, print \"NO\" (without the quotes).

Otherwise, print \"YES\", and then print n - 1 lines, each of which contains two integers — the numbers of vertices connected by an edge of the tree you found. The vertices are numbered from 1, the order of the edges and the order of the vertices within the edges does not matter.

If there are multiple solutions, output any of them.

Examples

Input

4
4 3 2 1


Output

YES
4 1
4 2
1 3


Input

3
3 1 2


Output

NO

Note

In the first sample test a permutation transforms edge (4, 1) into edge (1, 4), edge (4, 2) into edge (1, 3) and edge (1, 3) into edge (4, 2). These edges all appear in the resulting tree.

It can be shown that in the second sample test no tree satisfies the given condition.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n=input()
a=map(lambda x:int(x)-1,raw_input().split())
p=q=-1
for i in xrange(n):
  if a[a[i]]==i:
    p,q=i,a[i]
  if a[i]==i:
    print\"YES\"
    for j in xrange(n):
      if i!=j:
        print i+1,j+1
    exit()
if p<0 or q<0:
  print\"NO\"
  exit()
r=[(p,q)]
v=[0]*n
v[p]=v[q]=1
for i in xrange(n):
  if not v[i]:
    r.append((p,i))
    v[i]=1
    t=0
    x=a[i]
    while x!=i:
      r.append((p if t else q,x))
      v[x]=1
      t=1-t
      x=a[x]
    if not t:
      print\"NO\"
      exit()
print\"YES\"
for x,y in r:
  print x+1,y+1
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

def solve_permutation_tree(n, p):
    # 处理n=1的情况
    if n == 1:
        return (True, [])
    
    for i in range(n):
        if p[i] == i:
            edges = []
            for j in range(n):
                if i != j:
                    edges.append((i, j))
            return (True, edges)
    
    p_pair = -1
    q_pair = -1
    for i in range(n):
        if p[p[i]] == i and i != p[i]:
            p_pair = i
            q_pair = p[i]
            break
    
    if p_pair == -1 or q_pair == -1 or p_pair == q_pair:
        return (False, [])
    
    visited = [False] * n
    visited[p_pair] = visited[q_pair] = True
    edges = [(p_pair, q_pair)]
    
    for i in range(n):
        if not visited[i]:
            current_edges = []
            current_edges.append((p_pair, i))
            visited[i] = True
            x = p[i]
            t = 0
            while x != i:
                next_node = p[x]
                if t % 2 == 0:
                    current_node = q_pair
                else:
                    current_node = p_pair
                current_edges.append((current_node, x))
                visited[x] = True
                t += 1
                x = next_x = p[x]
                if next_x == x:  # 防止无限循环
                    break
            if t % 2 == 0:
                return (False, [])
            edges.extend(current_edges)
    
    return (True, edges)

def is_valid_solution(n, p, edges):
    if n == 1:
        return len(edges) == 0
    
    if len(edges) != n - 1:
        return False
    
    parent = list(range(n + 1))
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    
    for u, v in edges:
        if u < 1 or u > n or v < 1 or v > n:
            return False
        root_u = find(u)
        root_v = find(v)
        if root_u == root_v:
            return False
        parent[root_v] = root_u
    
    for i in range(1, n + 1):
        if find(i) != find(1):
            return False
    
    edge_set = set(frozenset((u, v)) for u, v in edges)
    
    for u, v in edges:
        pu = p[u - 1]
        pv = p[v - 1]
        permuted_edge = frozenset((pu, pv))
        if permuted_edge not in edge_set:
            return False
    
    return True

class Dinvarianceoftreebootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = params.copy()
        self.params.setdefault('min_n', 1)
        self.params.setdefault('max_n', 10)
        self.params.setdefault('n', None)
    
    def case_generator(self):
        params = self.params
        n = params['n']
        if n is None:
            min_n = params['min_n']
            max_n = params['max_n']
            n = random.randint(min_n, max_n)
        
        p = list(range(1, n + 1))
        random.shuffle(p)
        p_0based = [x - 1 for x in p]
        possible, edges = solve_permutation_tree(n, p_0based)
        
        case = {
            'n': n,
            'p': p.copy()
        }
        if possible:
            if n == 1:
                case['solution'] = []
            else:
                case['solution'] = [(u + 1, v + 1) for u, v in edges]
        return case
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        p = question_case['p']
        prompt = f"""Given a permutation p of size {n}, determine if there exists a tree of {n} vertices that remains invariant under p. The permutation p is: {p}

Output "YES" followed by the edges of the tree if it exists, otherwise output "NO". Enclose your answer within [answer] tags."""
        return prompt
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        answer_block = matches[-1].strip()
        lines = [line.strip() for line in answer_block.split('\n') if line.strip()]
        if not lines:
            return None
        
        first_line = lines[0].upper()
        if first_line == 'YES':
            edges = []
            for line in lines[1:]:
                parts = line.split()
                if len(parts) != 2:
                    return None
                try:
                    u = int(parts[0])
                    v = int(parts[1])
                except ValueError:
                    return None
                edges.append((u, v))
            if len(edges) != (len(lines)-1) or (len(edges) > 0 and len(edges) != (question_case['n'] -1)):
                return None
            return {'answer': 'YES', 'edges': edges}
        elif first_line == 'NO':
            return {'answer': 'NO'}
        else:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False
        answer = solution.get('answer')
        n = identity['n']
        p_list = identity['p']
        if answer == 'NO':
            return 'solution' not in identity
        elif answer == 'YES':
            if 'solution' not in identity:
                return False
            submitted_edges = solution.get('edges', [])
            return is_valid_solution(n, p_list, submitted_edges)
        return False
