"""# 

### 谜题描述
Valera had an undirected connected graph without self-loops and multiple edges consisting of n vertices. The graph had an interesting property: there were at most k edges adjacent to each of its vertices. For convenience, we will assume that the graph vertices were indexed by integers from 1 to n.

One day Valera counted the shortest distances from one of the graph vertices to all other ones and wrote them out in array d. Thus, element d[i] of the array shows the shortest distance from the vertex Valera chose to vertex number i.

Then something irreparable terrible happened. Valera lost the initial graph. However, he still has the array d. Help him restore the lost graph.

Input

The first line contains two space-separated integers n and k (1 ≤ k < n ≤ 105). Number n shows the number of vertices in the original graph. Number k shows that at most k edges were adjacent to each vertex in the original graph.

The second line contains space-separated integers d[1], d[2], ..., d[n] (0 ≤ d[i] < n). Number d[i] shows the shortest distance from the vertex Valera chose to the vertex number i.

Output

If Valera made a mistake in his notes and the required graph doesn't exist, print in the first line number -1. Otherwise, in the first line print integer m (0 ≤ m ≤ 106) — the number of edges in the found graph.

In each of the next m lines print two space-separated integers ai and bi (1 ≤ ai, bi ≤ n; ai ≠ bi), denoting the edge that connects vertices with numbers ai and bi. The graph shouldn't contain self-loops and multiple edges. If there are multiple possible answers, print any of them.

Examples

Input

3 2
0 1 1


Output

3
1 2
1 3
3 2


Input

4 2
2 0 1 3


Output

3
1 3
1 4
2 3


Input

3 1
0 0 0


Output

-1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, k = map(int, raw_input().split())
v = [int(i) for i in raw_input().split()]

s = [0] * n
for x in v:
    s[x] += 1

idx = [[] for i in range(n)]
for i, val in enumerate(v):
    idx[val].append(i + 1)

for i in range(n - 1):
    if s[0] != 1 or s[i] * (k - (i != 0)) < s[i + 1]:
        print -1
        break
else:
    print n - 1
    ans = []
    for i in range(1, n):
        z = 0
        c = 0
        for x in idx[i]:
            ans.append(str(idx[i - 1][z]) + ' ' + str(x))
            c += 1
            if c == k - (i != 1):
                c = 0
                z += 1
    print('\n'.join(ans))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import defaultdict
from math import ceil

class Crestoregraphbootcamp(Basebootcamp):
    def __init__(self, max_n=20, **params):
        self.max_n = max_n
        super().__init__(**params)
    
    def case_generator(self):
        # Generate valid or invalid cases
        generate_valid = random.random() < 0.7  # 70% valid, 30% invalid
        n = random.randint(2, self.max_n)
        valid = True

        # Generate initial valid layers (s)
        s = [1]
        remaining = n - 1
        while remaining > 0:
            next_s = random.randint(1, remaining)
            s.append(next_s)
            remaining -= next_s

        # Calculate minimal required k
        max_k = 0
        for i in range(len(s)-1):
            current_s = s[i]
            next_s = s[i+1]
            required_k = ceil(next_s / current_s) + (i != 0)
            max_k = max(max_k, required_k)
        k = max_k

        # Generate d array
        d = []
        for i, count in enumerate(s):
            d.extend([i] * count)
        random.shuffle(d)

        # Introduce invalid conditions if needed
        if not generate_valid:
            invalid_type = random.choice([0, 1, 2, 3])
            if invalid_type == 0:
                # Break s[0] to invalid count
                s[0] = 0
                d = []
                for i, cnt in enumerate(s):
                    d.extend([i] * cnt)
                d[random.randint(0, len(d)-1)] = 0  # Ensure at least one 0
            elif invalid_type == 1 and max_k > 1:
                # Reduce k to make insufficient
                k = random.randint(1, max_k - 1)
            elif invalid_type == 2 and len(s) > 1:
                # Disrupt d array hierarchy
                max_d = len(s) - 1
                if max_d + 1 < n:
                    idx = random.randint(0, len(d)-1)
                    d[idx] = max_d + 1
            elif invalid_type == 3 and len(s) > 2:
                # Make a layer exceed parent capacity
                i = random.randint(1, len(s)-2)
                s[i+1] = s[i] * (k - 1) + 1
                d = []
                for layer, cnt in enumerate(s):
                    d.extend([layer] * cnt)
                random.shuffle(d)
        
        return {'n': n, 'k': k, 'd': d}
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k = question_case['k']
        d = question_case['d']
        input_lines = [f"{n} {k}", ' '.join(map(str, d))]
        input_str = '\n'.join(input_lines)
        prompt = f"""Valera had an undirected connected graph without self-loops or multiple edges, where each vertex has at most {k} edges. He recorded the shortest distances from one vertex to all others in array d. Your task is to determine if the graph can be restored. If not, output -1. Otherwise, output the number of edges followed by the edges. Format your answer within [answer] and [/answer].
Input:
{input_str}"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        answer = matches[-1].strip()
        lines = [line.strip() for line in answer.split('\n') if line.strip()]
        if not lines:
            return None
        if lines[0] == '-1':
            return '-1'
        try:
            m = int(lines[0])
            edges = []
            for line in lines[1:m+1]:
                parts = line.split()
                if len(parts) != 2:
                    continue
                a, b = map(int, parts)
                if a == b:
                    continue
                a, b = sorted((a, b))
                edges.append((a, b))
            edges = list(set(edges))
            edges.sort()
            return {'m': len(edges), 'edges': edges}
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        def reference_solve(n, k, d_list):
            v = d_list
            s = [0] * n
            for x in v:
                if x >= n or x < 0:
                    return (-1, [])
                s[x] += 1

            if s[0] != 1:
                return (-1, [])
            
            max_d = max(v)
            for i in range(max_d + 1):
                if s[i] == 0:
                    return (-1, [])
            
            for i in range(max_d):
                allowed = s[i] * (k - (i != 0))
                if s[i+1] > allowed:
                    return (-1, [])
            
            idx = [[] for _ in range(n)]
            for node, dist in enumerate(v, 1):
                idx[dist].append(node)
            
            edges = []
            try:
                for dist in range(1, max_d + 1):
                    parents = idx[dist-1]
                    children = idx[dist]
                    if not parents or not children:
                        return (-1, [])
                    
                    slots_per_parent = k - (1 if dist-1 !=0 else 0)
                    required_parents = ceil(len(children) / slots_per_parent)
                    if len(parents) < required_parents:
                        return (-1, [])
                    
                    for i, child in enumerate(children):
                        parent_idx = i // slots_per_parent
                        if parent_idx >= len(parents):
                            return (-1, [])
                        edges.append((parents[parent_idx], child))
            except:
                return (-1, [])
            
            unique_edges = set()
            degrees = defaultdict(int)
            for a, b in edges:
                if a == b:
                    continue
                a, b = sorted((a, b))
                unique_edges.add((a, b))
                degrees[a] += 1
                degrees[b] += 1
                if degrees[a] > k or degrees[b] > k:
                    return (-1, [])
            
            return (len(unique_edges), sorted(unique_edges))

        n = identity['n']
        k = identity['k']
        d_list = identity['d']
        ref_m, ref_edges = reference_solve(n, k, d_list)
        
        if solution == '-1':
            return ref_m == -1
        
        if isinstance(solution, dict):
            user_m = solution.get('m', 0)
            user_edges = set(tuple(e) for e in solution.get('edges', []))
            if ref_m == -1:
                return False
            
            # Check edge count and content
            if user_m != ref_m or user_edges != set(ref_edges):
                return False
            
            # Check degree constraints
            degrees = defaultdict(int)
            for a, b in user_edges:
                degrees[a] += 1
                degrees[b] += 1
                if degrees[a] > k or degrees[b] > k:
                    return False
            
            # Check connectivity and distances
            try:
                adj = defaultdict(list)
                for a, b in user_edges:
                    adj[a].append(b)
                    adj[b].append(a)
                
                source = d_list.index(0) + 1
                visited = {source: 0}
                queue = [source]
                for node in queue:
                    for neighbor in adj[node]:
                        if neighbor not in visited:
                            visited[neighbor] = visited[node] + 1
                            queue.append(neighbor)
                
                for i, d_val in enumerate(d_list):
                    node = i + 1
                    if visited.get(node, -1) != d_val:
                        return False
            except:
                return False
            
            return True
        
        return False
