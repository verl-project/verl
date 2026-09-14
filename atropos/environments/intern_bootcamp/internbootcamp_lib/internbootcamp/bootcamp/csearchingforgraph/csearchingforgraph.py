"""# 

### 谜题描述
Let's call an undirected graph of n vertices p-interesting, if the following conditions fulfill: 

  * the graph contains exactly 2n + p edges; 
  * the graph doesn't contain self-loops and multiple edges; 
  * for any integer k (1 ≤ k ≤ n), any subgraph consisting of k vertices contains at most 2k + p edges. 



A subgraph of a graph is some set of the graph vertices and some set of the graph edges. At that, the set of edges must meet the condition: both ends of each edge from the set must belong to the chosen set of vertices. 

Your task is to find a p-interesting graph consisting of n vertices.

Input

The first line contains a single integer t (1 ≤ t ≤ 5) — the number of tests in the input. Next t lines each contains two space-separated integers: n, p (5 ≤ n ≤ 24; p ≥ 0; <image>) — the number of vertices in the graph and the interest value for the appropriate test. 

It is guaranteed that the required graph exists.

Output

For each of the t tests print 2n + p lines containing the description of the edges of a p-interesting graph: the i-th line must contain two space-separated integers ai, bi (1 ≤ ai, bi ≤ n; ai ≠ bi) — two vertices, connected by an edge in the resulting graph. Consider the graph vertices numbered with integers from 1 to n. 

Print the answers to the tests in the order the tests occur in the input. If there are multiple solutions, you can print any of them.

Examples

Input

1
6 0


Output

1 2
1 3
1 4
1 5
1 6
2 3
2 4
2 5
2 6
3 4
3 5
3 6

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
t = int(raw_input())

def compute(n, p):
    fault = False
    want = (2 * n) + p
    for i in range(n):
        if fault: break
        for j in range(i + 1, n):
            if want == 0:
                fault = True
                break
            else:
                print i + 1, j + 1
                want -= 1


for x in range(t):
    inp = raw_input().split()
    n, p = int(inp[0]), int(inp[1])
    compute(n, p)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from typing import Dict, Any, List, Optional, Set, Tuple

from bootcamp import Basebootcamp

class Csearchingforgraphbootcamp(Basebootcamp):
    def __init__(self, min_n: int = 5, max_n: int = 24, max_t: int = 5):
        self.min_n = min_n
        self.max_n = max_n
        self.max_t = max_t

    def case_generator(self) -> Dict[str, Any]:
        t = random.randint(1, self.max_t)
        tests = []
        for _ in range(t):
            n = random.randint(self.min_n, self.max_n)
            max_p = (n * (n - 1) // 2) - 2 * n
            if max_p < 0:
                max_p = 0  # Ensure p is non-negative
            p = random.randint(0, max_p)
            tests.append({'n': n, 'p': p})
        return {'tests': tests}

    @staticmethod
    def prompt_func(question_case: Dict[str, Any]) -> str:
        tests = question_case['tests']
        input_lines = [str(len(tests))]
        for test in tests:
            input_lines.append(f"{test['n']} {test['p']}")
        input_str = '\n'.join(input_lines)

        return f"""You are a programming expert. Solve the p-interesting graph construction problem.

**Problem Rules**:
1. The graph must contain exactly 2n + p edges (n = number of vertices)
2. No self-loops or duplicate edges
3. Any k-vertex subgraph (1 ≤ k ≤ n) must contain ≤ 2k + p edges

**Input Format**:
First line: t (number of test cases)
Next t lines: n and p values

**Output Format**:
For each test case, output 2n + p edges in any order

**Sample Input**:
1
6 0

**Sample Output**:
1 2
1 3
1 4
1 5
1 6
2 3
2 4
2 5
2 6
3 4
3 5
3 6

**Current Input**:
{input_str}

Place your answer between [answer] tags:
[answer]
{{your_answer}}
[/answer]"""

    @staticmethod
    def extract_output(output: str) -> Optional[List[str]]:
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        last_answer = answer_blocks[-1].strip()
        lines = [line.strip() for line in last_answer.split('\n') if line.strip()]
        return lines if lines else None

    @classmethod
    def _verify_correction(cls, solution: List[str], identity: Dict[str, Any]) -> bool:
        try:
            # Step 1: Parse user's solution
            all_edges = []
            for line in solution:
                a, b = sorted(map(int, line.strip().split()))
                if a == b or a < 1:
                    return False
                all_edges.append((a, b))
            
            # Check for duplicates
            if len(all_edges) != len(set(all_edges)):
                return False

            edge_ptr = 0
            # Process each test case
            for test_case in identity['tests']:
                n = test_case['n']
                p = test_case['p']
                required = 2 * n + p
                available = len(all_edges) - edge_ptr
                
                # Check if enough edges
                if available < required:
                    return False
                
                # Extract edges for this test case
                test_edges = set(all_edges[edge_ptr:edge_ptr+required])
                edge_ptr += required
                
                # Validate edges are within vertex range
                for a, b in test_edges:
                    if b > n or a > n:
                        return False
                
                # Generate canonical solution
                canonical_edges = set()
                want = required
                for i in range(n):
                    if want <= 0:
                        break
                    for j in range(i+1, n):
                        if want <= 0:
                            break
                        canonical_edges.add((i+1, j+1))
                        want -= 1
                
                # Compare edge sets
                if test_edges != canonical_edges:
                    # Check if alternative valid structure exists
                    total_edges = len(test_edges)
                    if total_edges != required:
                        return False
                    
                    # Validate subgraph conditions (basic check for small n)
                    if n <= 10:
                        adj = {v: set() for v in range(1, n+1)}
                        for a, b in test_edges:
                            adj[a].add(b)
                            adj[b].add(a)
                        
                        from itertools import combinations
                        valid = True
                        for k in range(1, n+1):
                            max_allowed = 2 * k + p
                            # Check all k-combinations of vertices
                            for vertices in combinations(range(1, n+1), k):
                                sub_edges = 0
                                for i in range(len(vertices)):
                                    for j in range(i+1, len(vertices)):
                                        if vertices[j] in adj[vertices[i]]:
                                            sub_edges += 1
                                if sub_edges > max_allowed:
                                    valid = False
                                    break
                            if not valid:
                                return False
            return edge_ptr == len(all_edges)
        except Exception:
            return False
