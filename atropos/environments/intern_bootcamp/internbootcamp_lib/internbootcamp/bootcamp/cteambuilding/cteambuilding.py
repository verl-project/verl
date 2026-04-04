"""# 

### 谜题描述
The new academic year has started, and Berland's university has n first-year students. They are divided into k academic groups, however, some of the groups might be empty. Among the students, there are m pairs of acquaintances, and each acquaintance pair might be both in a common group or be in two different groups.

Alice is the curator of the first years, she wants to host an entertaining game to make everyone know each other. To do that, she will select two different academic groups and then divide the students of those groups into two teams. The game requires that there are no acquaintance pairs inside each of the teams.

Alice wonders how many pairs of groups she can select, such that it'll be possible to play a game after that. All students of the two selected groups must take part in the game.

Please note, that the teams Alice will form for the game don't need to coincide with groups the students learn in. Moreover, teams may have different sizes (or even be empty).

Input

The first line contains three integers n, m and k (1 ≤ n ≤ 500 000; 0 ≤ m ≤ 500 000; 2 ≤ k ≤ 500 000) — the number of students, the number of pairs of acquaintances and the number of groups respectively.

The second line contains n integers c_1, c_2, ..., c_n (1 ≤ c_i ≤ k), where c_i equals to the group number of the i-th student.

Next m lines follow. The i-th of them contains two integers a_i and b_i (1 ≤ a_i, b_i ≤ n), denoting that students a_i and b_i are acquaintances. It's guaranteed, that a_i ≠ b_i, and that no (unordered) pair is mentioned more than once.

Output

Print a single integer — the number of ways to choose two different groups such that it's possible to select two teams to play the game.

Examples

Input


6 8 3
1 1 2 2 3 3
1 3
1 5
1 6
2 5
2 6
3 4
3 5
5 6


Output


2


Input


4 3 3
1 1 2 2
1 2
2 3
3 4


Output


3


Input


4 4 2
1 1 1 2
1 2
2 3
3 1
1 4


Output


0


Input


5 5 2
1 2 1 2 1
1 2
2 3
3 4
4 5
5 1


Output


0

Note

The acquaintances graph for the first example is shown in the picture below (next to each student there is their group number written).

<image>

In that test we can select the following groups:

  * Select the first and the second groups. For instance, one team can be formed from students 1 and 4, while other team can be formed from students 2 and 3. 
  * Select the second and the third group. For instance, one team can be formed 3 and 6, while other team can be formed from students 4 and 5. 
  * We can't select the first and the third group, because there is no way to form the teams for the game. 



In the second example, we can select any group pair. Please note, that even though the third group has no students, we still can select it (with some other group) for the game.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import os
import sys
from atexit import register
from io import BytesIO

sys.stdout = BytesIO()
register(lambda: os.write(1, sys.stdout.getvalue()))

input = BytesIO(os.read(0, os.fstat(0).st_size)).readline
def main():
    class UnionFind():
        def __init__(self, n):
            self.n = n
            self.root = [-1] * (n + 1)
            self.rnk = [0] * (n + 1)

        def find_root(self, x):
            while self.root[x] >= 0:
                x = self.root[x]
            return x

        def unite(self, x, y):
            x = self.find_root(x)
            y = self.find_root(y)
            if x == y:
                return
            elif self.rnk[x] > self.rnk[y]:
                self.root[x] += self.root[y]
                self.root[y] = x
            else:
                self.root[y] += self.root[x]
                self.root[x] = y
                if self.rnk[x] == self.rnk[y]:
                    self.rnk[y] += 1

        def unite_root(self, x, y):
            # assert self.root[x] < 0 and self.root[y] < 0
            if self.rnk[x] > self.rnk[y]:
                self.root[x] += self.root[y]
                self.root[y] = x
            else:
                self.root[y] += self.root[x]
                self.root[x] = y
                if self.rnk[x] == self.rnk[y]:
                    self.rnk[y] += 1

        def isSameGroup(self, x, y):
            return self.find_root(x) == self.find_root(y)

        def size(self, x):
            return -self.root[self.find_root(x)]

    N, M, K = map(int, input().split())
    C = [0] + list(map(int, input().split()))
    V_color = [[] for _ in range(K+1)]
    for v in range(1, N+1):
        V_color[C[v]].append(v)
    E = {}
    UF = UnionFind(2 * N)
    for _ in range(M):
        a, b = map(int, input().split())
        ca, cb = C[a], C[b]
        if ca == cb:
            UF.unite(a, b + N)
            UF.unite(a + N, b)
        else:
            if ca > cb:
                ca, cb = cb, ca
            c_ab = ca * (K+1) + cb
            if c_ab in E:
                E[c_ab].append((a, b))
            else:
                E[c_ab] = [(a, b)]

    ok_color_num = 0
    ok_color = [0] * (K+1)
    root_c = [set() for _ in range(K+1)]
    for c in range(1, K+1):
        ok = 1
        for v in V_color[c]:
            r1 = UF.find_root(v)
            r2 = UF.find_root(v+N)
            if r1 == r2:
                ok = 0
                break
            else:
                root_c[c].add(min(r1, r2))
        if ok:
            ok_color_num += 1
            ok_color[c] = 1
        else:
            for v in V_color[c]:
                UF.root[v] = -1

    R = [-1] * (N*2+1)
    for v in range(1, 2*N+1):
        R[v] = UF.find_root(v)

    ans = (ok_color_num - 1) * ok_color_num // 2
    root_ori = UF.root[:]
    for c_ab in E:
        ca, cb = divmod(c_ab, K+1)
        if ok_color[ca] * ok_color[cb] == 0:
            continue

        ok = 1
        for a, b in E[c_ab]:
            ra = R[a]
            rb = R[b]
            if ra <= N:
                ra2 = ra + N
            else:
                ra2 = ra - N
            if rb <= N:
                rb2 = rb + N
            else:
                rb2 = rb - N
            r_ra = UF.find_root(ra)
            r_rb2 = UF.find_root(rb2)
            if r_ra != r_rb2:
                UF.unite_root(r_ra, r_rb2)
                r_ra2 = UF.find_root(ra2)
                r_rb = UF.find_root(rb)
                if r_ra2 == r_rb:
                    ok = 0
                    break
                else:
                    UF.unite_root(r_ra2, r_rb)
        \"\"\"
        if len(root_c[ca]) > len(root_c[cb]):
            ca, cb = cb, ca
        for v in root_c[ca]:
            if UF.isSameGroup(v, v+N):
                ok = 0
                break
        \"\"\"
        if ok == 0:
            ans -= 1
        for v in root_c[ca]:
            UF.root[v] = root_ori[v]
            UF.root[v+N] = root_ori[v]
        for v in root_c[cb]:
            UF.root[v] = root_ori[v]
            UF.root[v+N] = root_ori[v]
    print(ans)


if __name__ == '__main__':
    main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from itertools import combinations
from bootcamp import Basebootcamp

class UnionFind:
    def __init__(self, n):
        self.n = n
        self.root = [-1] * (n + 1)
        self.rnk = [0] * (n + 1)
    
    def find_root(self, x):
        while self.root[x] >= 0:
            x = self.root[x]
        return x
    
    def unite(self, x, y):
        x = self.find_root(x)
        y = self.find_root(y)
        if x == y:
            return
        elif self.rnk[x] > self.rnk[y]:
            self.root[x] += self.root[y]
            self.root[y] = x
        else:
            self.root[y] += self.root[x]
            self.root[x] = y
            if self.rnk[x] == self.rnk[y]:
                self.rnk[y] += 1
    
    def unite_root(self, x, y):
        if self.rnk[x] > self.rnk[y]:
            self.root[x] += self.root[y]
            self.root[y] = x
        else:
            self.root[y] += self.root[x]
            self.root[x] = y
            if self.rnk[x] == self.rnnk[y]:
                self.rnk[y] += 1
    
    def isSameGroup(self, x, y):
        return self.find_root(x) == self.find_root(y)
    
    def size(self, x):
        return -self.root[self.find_root(x)]

def compute_answer(n, m, k, group_assignments, edges):
    C = [0] + group_assignments
    V_color = [[] for _ in range(k+1)]
    for v in range(1, n+1):
        V_color[C[v]].append(v)
    
    E = {}
    UF = UnionFind(2 * n)
    for a, b in edges:
        ca, cb = C[a], C[b]
        if ca == cb:
            UF.unite(a, b + n)
            UF.unite(a + n, b)
        else:
            if ca > cb:
                ca, cb = cb, ca
            c_ab = ca * (k + 1) + cb
            if c_ab in E:
                E[c_ab].append((a, b))
            else:
                E[c_ab] = [(a, b)]
    
    ok_color = [0] * (k + 1)
    root_c = [set() for _ in range(k + 1)]
    ok_color_num = 0
    for c in range(1, k + 1):
        ok = 1
        for v in V_color[c]:
            r1 = UF.find_root(v)
            r2 = UF.find_root(v + n)
            if r1 == r2:
                ok = 0
                break
            root_c[c].add(min(r1, r2))
        if ok:
            ok_color[c] = 1
            ok_color_num += 1
        else:
            for v in V_color[c]:
                UF.root[v] = -1
    
    R = [-1] * (2 * n + 1)
    for v in range(1, 2 * n + 1):
        R[v] = UF.find_root(v)
    
    ans = (ok_color_num - 1) * ok_color_num // 2
    root_ori = UF.root.copy()
    
    for c_ab in E:
        ca, cb = divmod(c_ab, k + 1)
        if ok_color[ca] * ok_color[cb] == 0:
            continue
        
        ok = 1
        for a, b in E[c_ab]:
            ra = R[a]
            rb = R[b]
            if ra <= n:
                ra2 = ra + n
            else:
                ra2 = ra - n
            if rb <= n:
                rb2 = rb + n
            else:
                rb2 = rb - n
            r_ra = UF.find_root(ra)
            r_rb2 = UF.find_root(rb2)
            if r_ra != r_rb2:
                UF.unite_root(r_ra, r_rb2)
                r_ra2_new = UF.find_root(ra2)
                r_rb_new = UF.find_root(rb)
                if r_ra2_new == r_rb_new:
                    ok = 0
                    break
                UF.unite_root(r_ra2_new, r_rb_new)
        if not ok:
            ans -= 1
        
        for v in root_c[ca]:
            UF.root[v] = root_ori[v]
            UF.root[v + n] = root_ori[v + n]
        for v in root_c[cb]:
            UF.root[v] = root_ori[v]
            UF.root[v + n] = root_ori[v + n]
    
    return ans

class Cteambuildingbootcamp(Basebootcamp):
    def __init__(self, max_students=10, max_groups=5, max_edges=20):
        self.max_students = max_students
        self.max_groups = max_groups
        self.max_edges = max_edges
    
    def case_generator(self):
        while True:
            try:
                n = random.randint(1, self.max_students)
                k = random.randint(2, self.max_groups)
                max_possible_edges = n * (n - 1) // 2
                m = random.randint(0, min(self.max_edges, max_possible_edges))
                
                # Assign groups, allowing empty groups
                c = [random.randint(1, k) for _ in range(n)]
                
                # Generate all possible edges and sample m unique ones
                students = list(range(1, n+1))
                all_possible_edges = list(combinations(students, 2))
                random.shuffle(all_possible_edges)
                edges = all_possible_edges[:m]
                
                # Compute correct answer
                correct_answer = compute_answer(n, m, k, c, edges)
                
                return {
                    'n': n,
                    'm': m,
                    'k': k,
                    'c': c,
                    'edges': edges,
                    'correct_answer': correct_answer
                }
            except:
                continue
    
    @staticmethod
    def prompt_func(question_case):
        input_desc = f"n = {question_case['n']}, m = {question_case['m']}, k = {question_case['k']}\n"
        input_desc += f"Group assignments: {question_case['c']}\n"
        if question_case['edges']:
            input_desc += "Acquaintance pairs:\n" + "\n".join(f"{a} {b}" for a, b in question_case['edges'])
        else:
            input_desc += "There are no acquaintance pairs."
        
        prompt = f"""The new academic year has started, and Berland University has {question_case['n']} first-year students divided into {question_case['k']} academic groups. There are {question_case['m']} pairs of acquaintances. Alice, the curator, needs to select two different groups such that all students from these groups can be divided into two teams with no acquaintances within each team.

**Task**
Compute how many valid pairs of groups can be selected. 

**Input**
{input_desc}

**Rules**
1. Selected groups must be different.
2. All students from the selected groups must participate.
3. Teams must have no acquaintances within them.
4. Teams can be of any size, including empty.

**Output**
Provide the number of valid group pairs inside [answer] tags. Example: [answer]3[/answer]"""

        return prompt
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity.get('correct_answer', None)
