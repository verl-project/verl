"""# 

### 谜题描述
Arseny likes to organize parties and invite people to it. However, not only friends come to his parties, but friends of his friends, friends of friends of his friends and so on. That's why some of Arseny's guests can be unknown to him. He decided to fix this issue using the following procedure.

At each step he selects one of his guests A, who pairwise introduces all of his friends to each other. After this action any two friends of A become friends. This process is run until all pairs of guests are friends.

Arseny doesn't want to spend much time doing it, so he wants to finish this process using the minimum number of steps. Help Arseny to do it.

Input

The first line contains two integers n and m (1 ≤ n ≤ 22; <image>) — the number of guests at the party (including Arseny) and the number of pairs of people which are friends.

Each of the next m lines contains two integers u and v (1 ≤ u, v ≤ n; u ≠ v), which means that people with numbers u and v are friends initially. It's guaranteed that each pair of friends is described not more than once and the graph of friendship is connected.

Output

In the first line print the minimum number of steps required to make all pairs of guests friends.

In the second line print the ids of guests, who are selected at each step.

If there are multiple solutions, you can output any of them.

Examples

Input

5 6
1 2
1 3
2 3
2 5
3 4
4 5


Output

2
2 3 

Input

4 4
1 2
1 3
1 4
3 4


Output

1
1 

Note

In the first test case there is no guest who is friend of all other guests, so at least two steps are required to perform the task. After second guest pairwise introduces all his friends, only pairs of guests (4, 1) and (4, 2) are not friends. Guest 3 or 5 can introduce them.

In the second test case guest number 1 is a friend of all guests, so he can pairwise introduce all guests in one step.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#include <bits/stdc++.h>
using namespace std;
const int MAXN = 22;
int N, M, aim_state, min_steps, min_pset, edges[MAXN], vis[1 << MAXN];
void DFS(int state, int pset, int steps) {
  if (steps >= min_steps) return;
  if (vis[pset]) return;
  vis[pset] = true;
  if (state == aim_state) {
    min_steps = steps;
    min_pset = pset;
    return;
  }
  for (int i = 0; i < N; ++i) {
    if ((1 << i) & (state ^ pset)) {
      if ((state | edges[i]) > state)
        DFS(state | edges[i], pset | (1 << i), steps + 1);
    }
  }
}
inline void init() {
  scanf(\"%d%d\", &N, &M);
  for (int i = 0, u, v; i < M; ++i) {
    scanf(\"%d%d\", &u, &v);
    --u;
    --v;
    edges[u] |= 1 << v;
    edges[v] |= 1 << u;
  }
  for (int i = 0; i < N; ++i) edges[i] |= 1 << i;
}
inline int solve() {
  if (M == N * (N - 1) / 2) return 0;
  aim_state = (1 << N) - 1;
  min_steps = N + 1;
  min_pset = aim_state;
  for (int i = 0; i < N; ++i) DFS(edges[i], 1 << i, 1);
  return min_steps;
}
int main() {
  init();
  int steps = solve();
  if (steps) {
    printf(\"%d\n\", steps);
    bool first = true;
    for (int i = 0; i < N; ++i)
      if (min_pset & (1 << i)) {
        if (first)
          first = false;
        else
          printf(\" \");
        printf(\"%d\", i + 1);
      }
    printf(\"\n\");
  } else
    printf(\"0\n\");
  return 0;
}
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import deque
import re
from bootcamp import Basebootcamp

class Epartybootcamp(Basebootcamp):
    def __init__(self, n=5):
        self.params = {'n': n}
    
    def case_generator(self):
        n = self.params['n']
        edges = self._generate_connected_graph(n)
        return {
            'n': n,
            'm': len(edges),
            'edges': edges
        }
    
    @staticmethod
    def _generate_connected_graph(n):
        edges = set()
        nodes = list(range(1, n+1))
        random.shuffle(nodes)
        
        # Generate a spanning tree
        for i in range(1, n):
            j = random.randint(0, i-1)
            u, v = sorted((nodes[j], nodes[i]))
            edges.add((u, v))
        
        # Add additional edges
        all_edges = [(u, v) for u in range(1, n+1) for v in range(u+1, n+1)]
        remaining = [e for e in all_edges if e not in edges]
        max_possible = n * (n-1) // 2
        m = random.randint(n-1, max_possible)
        edges.update(random.sample(remaining, k=m - (n-1)))
        
        return sorted(edges)
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        edges = question_case['edges']
        edges_str = '\n'.join(f"{u} {v}" for u, v in edges)
        
        return f"""You are at a party organized by Arseny. The goal is to help him introduce all guests to each other with the minimum number of steps. The process is: in each step, select a guest who will pairwise introduce all their current friends. After this step, all pairs of their friends become friends. This continues until all pairs are friends.

Input:
{n} {m}
{edges_str}

Your task is to determine the minimal number of steps and the sequence of guest IDs to select. 

Format your answer as:
[answer]
<number_of_steps>
<space_separated_guest_ids>
[/answer]

Example:
[answer]
1
1
[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        
        answer = matches[-1].strip().split()
        if len(answer) < 2:
            return None
        
        try:
            steps = int(answer[0])
            sequence = list(map(int, answer[1:1+steps]))
            if len(sequence) != steps:
                return None
            return {'steps': steps, 'sequence': sequence}
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            n = identity['n']
            edges = identity['edges']
            min_steps = cls._solve_min_steps(n, edges)
            
            user_steps = solution.get('steps', -1)
            sequence = solution.get('sequence', [])
            
            if user_steps != min_steps or len(sequence) != min_steps:
                return False
            
            # Simulate the process
            edge_set = set(tuple(sorted(e)) for e in edges)
            for a in sequence:
                friends = set()
                for u, v in edge_set:
                    if u == a:
                        friends.add(v)
                    elif v == a:
                        friends.add(u)
                
                new_edges = [(u, v) for u in friends for v in friends if u < v]
                edge_set.update(new_edges)
            
            return len(edge_set) == n * (n-1) // 2
        except:
            return False

    @staticmethod
    def _solve_min_steps(n, edges):
        edges_list = [(u, v) for u in range(1, n+1) for v in range(u+1, n+1)]
        edge_to_bit = {e: i for i, e in enumerate(edges_list)}
        
        initial_mask = 0
        for u, v in edges:
            initial_mask |= 1 << edge_to_bit[(u, v) if u < v else (v, u)]
        
        target = (1 << len(edges_list)) - 1
        if initial_mask == target:
            return 0
        
        visited = {initial_mask: 0}
        queue = deque([(initial_mask, 0)])
        
        while queue:
            mask, steps = queue.popleft()
            
            for a in range(1, n+1):
                friends = set()
                for u in range(1, n+1):
                    if u == a:
                        continue
                    e = tuple(sorted((a, u)))
                    if mask & (1 << edge_to_bit[e]):
                        friends.add(u)
                
                friends.add(a)
                new_mask = mask
                for i in friends:
                    for j in friends:
                        if i < j:
                            new_mask |= 1 << edge_to_bit[(i, j)]
                
                if new_mask == target:
                    return steps + 1
                if new_mask not in visited or steps + 1 < visited[new_mask]:
                    visited[new_mask] = steps + 1
                    queue.append((new_mask, steps + 1))
        
        return n  # Fallback
