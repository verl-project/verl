"""# 

### 谜题描述
Harry, Ron and Hermione have figured out that Helga Hufflepuff's cup is a horcrux. Through her encounter with Bellatrix Lestrange, Hermione came to know that the cup is present in Bellatrix's family vault in Gringott's Wizarding Bank. 

The Wizarding bank is in the form of a tree with total n vaults where each vault has some type, denoted by a number between 1 to m. A tree is an undirected connected graph with no cycles.

The vaults with the highest security are of type k, and all vaults of type k have the highest security.

There can be at most x vaults of highest security. 

Also, if a vault is of the highest security, its adjacent vaults are guaranteed to not be of the highest security and their type is guaranteed to be less than k.

Harry wants to consider every possibility so that he can easily find the best path to reach Bellatrix's vault. So, you have to tell him, given the tree structure of Gringotts, the number of possible ways of giving each vault a type such that the above conditions hold.

Input

The first line of input contains two space separated integers, n and m — the number of vaults and the number of different vault types possible. (1 ≤ n ≤ 105, 1 ≤ m ≤ 109).

Each of the next n - 1 lines contain two space separated integers ui and vi (1 ≤ ui, vi ≤ n) representing the i-th edge, which shows there is a path between the two vaults ui and vi. It is guaranteed that the given graph is a tree.

The last line of input contains two integers k and x (1 ≤ k ≤ m, 1 ≤ x ≤ 10), the type of the highest security vault and the maximum possible number of vaults of highest security.

Output

Output a single integer, the number of ways of giving each vault a type following the conditions modulo 109 + 7.

Examples

Input

4 2
1 2
2 3
1 4
1 2


Output

1


Input

3 3
1 2
1 3
2 1


Output

13


Input

3 1
1 2
1 3
1 1


Output

0

Note

In test case 1, we cannot have any vault of the highest security as its type is 1 implying that its adjacent vaults would have to have a vault type less than 1, which is not allowed. Thus, there is only one possible combination, in which all the vaults have type 2.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import os
import sys
from atexit import register
from io import BytesIO
sys.stdin = BytesIO(os.read(0, os.fstat(0).st_size))
sys.stdout = BytesIO()
register(lambda: os.write(1, sys.stdout.getvalue()))
input = lambda: sys.stdin.readline().rstrip('\r\n')
raw_input = lambda: sys.stdin.readline().rstrip('\r\n')

n,m = map(int,raw_input().split(\" \"))

mod = int(1e9+7)

edges = [[]for i in range(n+1)]
for i in range(n-1):
    u,v = map(int,raw_input().split(\" \"))
    edges[v].append(u)
    edges[u].append(v)

cnt,x = map(int,raw_input().split(\" \"))
dp = [ [0]*33 for i in range(n+1)]

q = [1]
visited= [0]*(1+n)
visited[1] = 1
i = 0
rank = [0]*(n+1)
rank[1] = 1
while i < n:
    v = q[i]
    for node in edges[v]:
        if not visited[node]:
            q.append(node)
            visited[node] = 1
            rank[node] = rank[v]+1
     
    i += 1 
tmp = [0]*33
for v in q[::-1]:
    f = dp[v]
    f[0]  = cnt-1
    f[12] = 1
    f[22] = m-cnt

    for son in edges[v]:
        if rank[son] < rank[v]:
            continue
        for i in range(33):
            tmp[i] = f[i]
        g = dp[son]
        for j in range(x+1):
            for i in range(x-j+1):
                        if j == 0:
                            k = 0
                            f[k*11+i+j] = ((g[j]+g[11+j]+g[22+j])*tmp[k*11+i] )%mod
                            k = 1
                            f[k*11+i+j] = (g[j]*tmp[k*11+i] )%mod
                            k = 2
                            f[k*11+i+j] = ((g[j]+dp[son][k*11+j])*tmp[k*11+i] )%mod
                        else:
                            k = 0
                            f[k*11+i+j] = (f[k*11+i+j]+(g[j]+g[11+j]+g[22+j])*tmp[k*11+i] )%mod
                            k = 1
                            f[k*11+i+j] = (f[k*11+i+j]+g[j]*tmp[k*11+i] )%mod
                            k = 2
                            f[k*11+i+j] = (f[k*11+i+j]+(g[j]+g[k*11+j])*tmp[k*11+i] )%mod
                    #print v,k*11+i+j,dp[v][k*11+i+j]
            
          
    
print (sum(dp[v][:x+1])+sum(dp[v][11:11+x+1])+sum(dp[v][22:22+x+1]))%mod
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import subprocess
import re
from bootcamp import Basebootcamp

MOD = 10**9 + 7

class Chelgahufflepuffscupbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = {
            'n_max': 5,
            'm_min': 1,
            'm_max': 10**9,
            'k_min': 1,
            'x_max': 10,
        }
        self.params.update(params)
    
    def case_generator(self):
        n = random.randint(1, self.params['n_max'])
        edges = []
        if n > 1:
            parent = {}
            nodes = list(range(1, n+1))
            for i in range(2, n+1):
                parent_node = random.choice(nodes[:i-1])
                if random.choice([True, False]):
                    edge = (parent_node, i)
                else:
                    edge = (i, parent_node)
                edges.append(edge)
        
        m = random.randint(self.params['m_min'], self.params['m_max'])
        k = random.randint(self.params['k_min'], m)
        x = random.randint(1, min(self.params['x_max'], 10))
        
        input_data = self._construct_input(n, m, edges, k, x)
        answer = self._run_reference_solution(input_data)
        
        return {
            'n': n,
            'm': m,
            'edges': edges,
            'k': k,
            'x': x,
            'answer': answer
        }
    
    def _construct_input(self, n, m, edges, k, x):
        input_lines = [f"{n} {m}"]
        input_lines.extend(f"{u} {v}" for u, v in edges)
        input_lines.append(f"{k} {x}")
        return '\n'.join(input_lines) + '\n'
    
    def _run_reference_solution(self, input_data):
        try:
            completed_process = subprocess.run(
                ['python', 'reference.py'],
                input=input_data.encode(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
                check=True
            )
            output = completed_process.stdout.decode().strip()
            return output if output else '0'
        except:
            return '0'
    
    @staticmethod
    def prompt_func(question_case):
        input_lines = [f"{question_case['n']} {question_case['m']}"]
        input_lines.extend(f"{u} {v}" for u, v in question_case['edges'])
        input_lines.append(f"{question_case['k']} {question_case['x']}")
        input_str = '\n'.join(input_lines)
        
        prompt = f"""Harry needs to calculate valid vault type assignments under these rules:

- Tree with {question_case['n']} vaults, types 1-{question_case['m']}
- Max {question_case['x']} vaults of type {question_case['k']} (highest security)
- Adjacent vaults to type {question_case['k']} must have lower types

Input:
{input_str}

Compute the answer modulo {MOD}. Put your final answer within [answer] tags like [answer]42[/answer]."""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return int(solution) == int(identity['answer'])
        except:
            return False
