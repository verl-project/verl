"""# 

### 谜题描述
Autumn came late to the kingdom of Far Far Away. The harvest was exuberant and it is now time to get ready for the winter. As most people celebrate the Harvest festival, Simon the Caretaker tries to solve a very non-trivial task of how to find place for the agricultural equipment in the warehouse.

He's got problems with some particularly large piece of equipment, which is, of course, turboplows. The problem is that when a turboplow is stored, it takes up not some simply rectangular space. It takes up a T-shaped space like on one of the four pictures below (here character \"#\" stands for the space occupied by the turboplow and character \".\" stands for the free space):
    
    
    ###      ..#      .#.      #..  
    .#.      ###      .#.      ###  
    .#.      ..#      ###      #..  
    

Simon faced a quite natural challenge: placing in the given n × m cells warehouse the maximum number of turboplows. As one stores the turboplows, he can rotate them in any manner (so that they take up the space like on one of the four pictures above). However, two turboplows cannot \"overlap\", that is, they cannot share the same cell in the warehouse.

Simon feels that he alone cannot find the optimal way of positioning the plugs in the warehouse that would maximize their quantity. Can you help him?

Input

The only line contains two space-separated integers n and m — the sizes of the warehouse (1 ≤ n, m ≤ 9).

Output

In the first line print the maximum number of turboplows that can be positioned in the warehouse. In each of the next n lines print m characters. Use \".\" (dot) to mark empty space and use successive capital Latin letters (\"A\" for the first turboplow, \"B\" for the second one and so on until you reach the number of turboplows in your scheme) to mark place for the corresponding turboplows considering that they are positioned in the optimal manner in the warehouse. The order in which you number places for the turboplows does not matter. If there are several optimal solutions for a warehouse of the given size, print any of them.

Examples

Input

3 3


Output

1
AAA
.A.
.A.


Input

5 6


Output

4
A..C..
AAAC..
ABCCCD
.B.DDD
BBB..D


Input

2 2


Output

0
..
..

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from itertools import izip
n,m = map(int,raw_input().split())
rf = False
if n>m: n,m,rf=m,n,True
w=[1,-1][m==9]
Z=((7,2,2),(2,2,7),(1,7,1),(4,7,4))
Zx = [[(i<<x,j<<x,k<<x)for i,j,k in Z] for x in xrange(n)]
q=[(0,)*m]
d,pr={q[0]:0},{q[0]:None}
def put(p,x,y,i,j,k):
    res = False
    pp = list(p)
    for vi,vj,vk in Zx[x]:
        if i&vi or j&vj or k&vk: continue
        pp[y]=i|vi
        pp[y+1]=j|vj
        pp[y+2]=k|vk
        pc = tuple(pp)
        res = True
        if pc in d: continue
        d[pc]=d[p]+1
        pr[pc]=p
        q.append(pc)
    return res
for p in q:
    jm,im = m,n
    for j in xrange(1,m-1):
        if j>jm:break
        p1,p2,p3 = p[j-1:j+2]
        for i in xrange(1,n-1):
            if i>im: break            
            if p2&(3<<i): continue
            if p1&(1<<i) and p2&(1<<(i-1)): continue
            if put(p,i-1,j-1,p1,p2,p3) and im==n:
                im,jm=i+w,j-1

z,i,l=-1,0,'A'
for k,v in d.iteritems():
    if v>z:
        z,i=v,k
r = [['.']*m for _ in xrange(n)]
while pr[i]:
    p = pr[i]
    for y in xrange(m):
        for x in xrange(n):
            if i[y]&(1<<x) and not p[y]&(1<<x):
                r[x][y]=l
    i,l = p,chr(ord(l)+1)
print z
if rf: r=zip(*r)
for l in r:
    print ''.join(l)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from collections import defaultdict
from bootcamp import Basebootcamp

def solve_turboplow(n, m):
    rf = False
    if n > m:
        n, m, rf = m, n, True
    w = -1 if m == 9 else 1
    Z = ((7, 2, 2), (2, 2, 7), (1, 7, 1), (4, 7, 4))
    Zx = []
    for x in range(n):
        current = []
        for i, j, k in Z:
            current.append((i << x, j << x, k << x))
        Zx.append(current)
    q = [tuple([0] * m)]
    d = {q[0]: 0}
    pr = {q[0]: None}

    def put(p, x, y, i, j, k):
        res = False
        pp = list(p)
        for vi, vj, vk in Zx[x]:
            if (i & vi) or (j & vj) or (k & vk):
                continue
            pp[y] = i | vi
            if y + 1 >= m:
                continue
            pp[y+1] = j | vj
            if y + 2 >= m:
                continue
            pp[y+2] = k | vk
            pc = tuple(pp)
            if pc in d:
                continue
            d[pc] = d[p] + 1
            pr[pc] = p
            q.append(pc)
            res = True
        return res

    for p in q:
        jm = m
        im = n
        for j in range(1, m - 1):
            if j > jm:
                break
            if j + 1 >= m:
                continue
            p1, p2, p3 = p[j-1], p[j], p[j+1]
            for i in range(1, n - 1):
                if i > im:
                    break
                if p2 & (3 << i):
                    continue
                if (p1 & (1 << i)) and (p2 & (1 << (i-1))):
                    continue
                if put(p, i-1, j-1, p1, p2, p3) and im == n:
                    im = i + w
                    jm = j - 1

    max_k = -1
    best_key = None
    for key, value in d.items():
        if value > max_k:
            max_k = value
            best_key = key

    if best_key is None:
        return 0, ['.' * m for _ in range(n)]

    r = [['.'] * m for _ in range(n)]
    current = best_key
    l = 'A'
    while pr.get(current) is not None:
        prev = pr[current]
        for y in range(m):
            for x in range(n):
                if (current[y] & (1 << x)) and not (prev[y] & (1 << x)):
                    r[x][y] = l
        current = prev
        l = chr(ord(l) + 1)

    if rf:
        transposed = []
        for col in range(m):
            transposed_row = []
            for row in range(n):
                transposed_row.append(r[row][col])
            transposed.append(''.join(transposed_row))
        r = transposed
    else:
        r = [''.join(row) for row in r]

    return max_k, r

def is_valid_t_shape(coords):
    if len(coords) != 5:
        return False
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    translated = set((r - min_r, c - min_c) for r, c in coords)
    patterns = [
        {(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)},
        {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)},
        {(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)},
        {(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)},
    ]
    return translated in patterns

class Chelpcaretakerbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=9, min_m=1, max_m=9):
        self.min_n = min_n
        self.max_n = max_n
        self.min_m = min_m
        self.max_m = max_m

    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        m = random.randint(self.min_m, self.max_m)
        return {
            'n': n,
            'm': m
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        prompt = f"""Autumn has come to the kingdom of Far Far Away, and Simon the Caretaker needs to store turboplows in a {n}x{m} warehouse. Each turboplow occupies a T-shaped area that can be rotated in any of four directions. The goal is to place the maximum number of turboplows without overlapping.

Your task is to determine the maximum number of turboplows that can fit and provide a valid layout. The first line of output should be the maximum number. The next {n} lines should each contain {m} characters representing the warehouse layout, using '.' for empty cells and successive letters (A, B, etc.) for each turboplow.

Format your answer exactly as follows between [answer] and [/answer]:

[answer]
{{max_number}}
{{row_1}}
{{row_2}}
...
{{row_{n}}}
[/answer]

Example for a 3x3 warehouse:
[answer]
1
AAA
.A.
.A.
[/answer]"""
        return prompt

    @staticmethod
    def extract_output(output):
        pattern = r'\[answer\](.*?)\[\/answer\]'
        matches = re.findall(pattern, output, re.DOTALL)
        if not matches:
            return None
        answer_content = matches[-1].strip()
        lines = [line.strip() for line in answer_content.split('\n') if line.strip()]
        if len(lines) < 1:
            return None
        try:
            k = int(lines[0])
        except ValueError:
            return None
        layout = lines[1:]
        return {'k': k, 'layout': layout}

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        n = identity['n']
        m = identity['m']
        user_k = solution['k']
        user_layout = solution['layout']

        if len(user_layout) != n:
            return False
        for row in user_layout:
            if len(row) != m:
                return False

        correct_k, _ = solve_turboplow(n, m)
        if user_k != correct_k:
            return False

        cells = defaultdict(list)
        for i in range(n):
            for j in range(m):
                char = user_layout[i][j]
                if char != '.':
                    cells[char].append((i, j))

        all_coords = []
        for char, coords in cells.items():
            if len(coords) != 5:
                return False
            if not is_valid_t_shape(coords):
                return False
            all_coords.extend(coords)

        if len(all_coords) != len(set(all_coords)):
            return False

        return True
