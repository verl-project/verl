"""# 

### 谜题描述
Your city has n junctions. There are m one-way roads between the junctions. As a mayor of the city, you have to ensure the security of all the junctions.

To ensure the security, you have to build some police checkposts. Checkposts can only be built in a junction. A checkpost at junction i can protect junction j if either i = j or the police patrol car can go to j from i and then come back to i.

Building checkposts costs some money. As some areas of the city are more expensive than others, building checkpost at some junctions might cost more money than other junctions.

You have to determine the minimum possible money needed to ensure the security of all the junctions. Also you have to find the number of ways to ensure the security in minimum price and in addition in minimum number of checkposts. Two ways are different if any of the junctions contains a checkpost in one of them and do not contain in the other.

Input

In the first line, you will be given an integer n, number of junctions (1 ≤ n ≤ 105). In the next line, n space-separated integers will be given. The ith integer is the cost of building checkpost at the ith junction (costs will be non-negative and will not exceed 109).

The next line will contain an integer m (0 ≤ m ≤ 3·105). And each of the next m lines contains two integers ui and vi (1 ≤ ui, vi ≤ n; u ≠ v). A pair ui, vi means, that there is a one-way road which goes from ui to vi. There will not be more than one road between two nodes in the same direction.

Output

Print two integers separated by spaces. The first one is the minimum possible money needed to ensure the security of all the junctions. And the second one is the number of ways you can ensure the security modulo 1000000007 (109 + 7).

Examples

Input

3
1 2 3
3
1 2
2 3
3 2


Output

3 1


Input

5
2 8 0 6 0
6
1 4
1 3
2 4
3 4
4 5
5 1


Output

8 2


Input

10
1 3 2 2 1 3 1 4 10 10
12
1 2
2 3
3 1
3 4
4 5
5 6
5 7
6 4
7 3
8 9
9 10
10 9


Output

15 6


Input

2
7 91
2
1 2
2 1


Output

7 1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys

#sys.stdin = open('inputfile.txt')

def solve():
    \"\"\"def FindRings():
        S = []
        v = set(range(1,n+1))
        visited = set()
        
        while len(v) > 0:
            vertex = v.pop()
            Queue = [vertex]
            while len(Queue) > 0:
                vertex = Queue.pop()
                visited.add(vertex)
                k = 0
                for i in Graph.get(vertex, []):
                    if i not in visited:
                        k += 1
                        Queue.append(i)
                if k > 0:
                    Queue.insert(-k, vertex)
                else:
                    S.append(vertex)
                    if vertex in v:
                        v.remove(vertex)
        Answer = []
        allvisited = set()
        while len(S) > 0:
            vertex = S.pop()
            if vertex in allvisited:
                continue
            visited = set()
            Queue = [vertex]
            while len(Queue) > 0:
                vertex = Queue.pop(0)
                for i in RGraph.get(vertex, []):
                    if i not in visited and i not in allvisited:
                        Queue.append(i)
                if vertex not in allvisited:
                    visited.add(vertex)
                    allvisited.add(vertex)
                
            Answer.append(visited)
        
        return Answer\"\"\"
                    
    def FindRings():
        preorder={}
        lowlink={}
        scc_found={}
        scc_queue = []
        scc_list=[]
        i=0
        for source in xrange(1, n+1):
            if source not in scc_found:
                queue=[source]
                while queue:
                    v=queue[-1]
                    if v not in preorder:
                        i=i+1
                        preorder[v]=i
                    done=1
                    v_nbrs=Graph.get(v, [])
                    for w in v_nbrs:
                        if w not in preorder:
                            queue.append(w)
                            done=0
                            break
                    if done==1:
                        lowlink[v]=preorder[v]
                        for w in v_nbrs:
                            if w not in scc_found:
                                if preorder[w]>preorder[v]:
                                    lowlink[v]=min([lowlink[v],lowlink[w]])
                                else:
                                    lowlink[v]=min([lowlink[v],preorder[w]])
                        queue.pop()
                        if lowlink[v]==preorder[v]:
                            scc_found[v]=True
                            scc=[v]
                            while scc_queue and preorder[scc_queue[-1]]>preorder[v]:
                                k=scc_queue.pop()
                                scc_found[k]=True
                                scc.append(k)
                            scc_list.append(scc)
                        else:
                            scc_queue.append(v)
        return scc_list

    Lines = sys.stdin.readlines()
    line = 0
    
    n = int(Lines[line])
    line += 1
    Prices = [int(i) for i in Lines[line].split()]
    line += 1
    m = int(Lines[line])
    line += 1
    Graph = {}
    RGraph = {}
    for _ in range(m):
        u, v = [int(i) for i in Lines[line].split()]
        line += 1
        if u not in Graph:
            Graph[u] = []
        if v not in RGraph:
            RGraph[v] = []
        Graph[u].append(v)
        RGraph[v].append(u)
    
    #print time.time() - t    
    Rings = FindRings()
    #print time.time() - t    
    
    total = 0
    permutation = 1
    for r in Rings:
        best = 1e10
        counter = 0
        for i in r:
            if Prices[i-1] < best:
                best = Prices[i-1]
                counter = 0
            if Prices[i-1] == best:
                counter += 1
        total += best
        permutation = (permutation * counter) % 1000000007
    
    print total, permutation % 1000000007


solve()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from itertools import combinations
import re
from bootcamp import Basebootcamp

class Ccheckpostsbootcamp(Basebootcamp):
    def __init__(self, n_min=2, n_max=10, cost_min=0, cost_max=100, max_scc_count=3):
        self.n_min = n_min
        self.n_max = n_max
        self.cost_min = cost_min
        self.cost_max = cost_max
        self.max_scc_count = max_scc_count

    def split_n_into_k(self, n, k):
        if k == 1:
            return [n]
        dividers = sorted(random.sample(range(1, n), k-1))
        prev = 0
        parts = []
        for d in dividers:
            parts.append(d - prev)
            prev = d
        parts.append(n - prev)
        return parts

    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        max_possible_k = min(n, self.max_scc_count)
        k = random.randint(1, max_possible_k)
        s_list = self.split_n_into_k(n, k)
        nodes = list(range(1, n+1))
        scc_nodes = []
        start = 0
        for s in s_list:
            end = start + s
            scc_nodes.append(nodes[start:end])
            start = end
        
        internal_edges = []
        for scc in scc_nodes:
            s = len(scc)
            if s >= 2:
                for i in range(s):
                    u = scc[i]
                    v = scc[(i+1) % s]
                    internal_edges.append((u, v))
        
        cross_edges = []
        existing_edges = set(internal_edges)
        for i in range(len(scc_nodes)):
            for j in range(i+1, len(scc_nodes)):
                if random.random() < 0.3:
                    u = random.choice(scc_nodes[i])
                    v = random.choice(scc_nodes[j])
                    if (u, v) not in existing_edges and u != v:
                        cross_edges.append((u, v))
                        existing_edges.add((u, v))
        
        edges = internal_edges + cross_edges
        m = len(edges)
        
        costs = []
        for scc in scc_nodes:
            min_cost = random.randint(self.cost_min, self.cost_max)
            num_min = random.randint(1, len(scc))
            selected = random.sample(scc, num_min)
            for node in scc:
                if node in selected:
                    costs.append(min_cost)
                else:
                    costs.append(min_cost + random.randint(1, 10))
        
        return {
            'n': n,
            'costs': costs,
            'm': m,
            'edges': edges,
            'scc_list': scc_nodes
        }

    @staticmethod
    def prompt_func(question_case):
        input_lines = [
            str(question_case['n']),
            ' '.join(map(str, question_case['costs'])),
            str(question_case['m'])
        ] + [f"{u} {v}" for u, v in question_case['edges']]
        input_text = '\n'.join(input_lines)
        prompt = f"""As the mayor, you need to secure all junctions with minimum cost. Each checkpost can protect reachable junctions. Find the minimal total cost and number of ways (mod 1e9+7).

Input:
{input_text}

Output two integers: minimal cost and number of ways. Place your answer within [answer] and [/answer], e.g., [answer]15 6[/answer]."""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        parts = last_match.split()
        if len(parts) != 2:
            return None
        try:
            int(parts[0]), int(parts[1])
            return f"{parts[0]} {parts[1]}"
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or len(solution.split()) != 2:
            return False
        try:
            total, ways = map(int, solution.split())
        except ValueError:
            return False
        
        calc_total = 0
        calc_ways = 1
        MOD = 10**9 +7
        for scc in identity['scc_list']:
            costs = [identity['costs'][n-1] for n in scc]
            min_cost = min(costs)
            cnt = costs.count(min_cost)
            calc_total += min_cost
            calc_ways = (calc_ways * cnt) % MOD
        
        return total == calc_total and calc_ways == (ways % MOD)
