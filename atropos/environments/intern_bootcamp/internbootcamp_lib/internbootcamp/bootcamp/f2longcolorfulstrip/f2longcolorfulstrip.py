"""# 

### 谜题描述
This is the second subtask of problem F. The only differences between this and the first subtask are the constraints on the value of m and the time limit. It is sufficient to solve this subtask in order to hack it, but you need to solve both subtasks in order to hack the first one.

There are n+1 distinct colours in the universe, numbered 0 through n. There is a strip of paper m centimetres long initially painted with colour 0. 

Alice took a brush and painted the strip using the following process. For each i from 1 to n, in this order, she picks two integers 0 ≤ a_i < b_i ≤ m, such that the segment [a_i, b_i] is currently painted with a single colour, and repaints it with colour i. 

Alice chose the segments in such a way that each centimetre is now painted in some colour other than 0. Formally, the segment [i-1, i] is painted with colour c_i (c_i ≠ 0). Every colour other than 0 is visible on the strip.

Count the number of different pairs of sequences \\{a_i\}_{i=1}^n, \\{b_i\}_{i=1}^n that result in this configuration. 

Since this number may be large, output it modulo 998244353.

Input

The first line contains a two integers n, m (1 ≤ n ≤ 500, n ≤ m ≤ 10^6) — the number of colours excluding the colour 0 and the length of the paper, respectively.

The second line contains m space separated integers c_1, c_2, …, c_m (1 ≤ c_i ≤ n) — the colour visible on the segment [i-1, i] after the process ends. It is guaranteed that for all j between 1 and n there is an index k such that c_k = j.

Output

Output a single integer — the number of ways Alice can perform the painting, modulo 998244353.

Examples

Input


3 3
1 2 3


Output


5


Input


2 3
1 2 1


Output


1


Input


2 3
2 1 2


Output


0


Input


7 7
4 5 1 6 2 3 7


Output


165


Input


8 17
1 3 2 2 7 8 2 5 5 4 4 4 1 1 6 1 1


Output


20

Note

In the first example, there are 5 ways, all depicted in the figure below. Here, 0 is white, 1 is red, 2 is green and 3 is blue.

<image>

Below is an example of a painting process that is not valid, as in the second step the segment 1 3 is not single colour, and thus may not be repainted with colour 2.

<image>

In the second example, Alice must first paint segment 0 3 with colour 1 and then segment 1 2 with colour 2. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
range = xrange
input = raw_input

MOD = 998244353 

m,n = [int(x) for x in input().split()]
C = [int(x) - 1 for x in input().split()]

C2 = [C[0]]
for c in C:
    if C2[-1] != c: 
        C2.append(c)
C = C2
n = len(C)

if n > 2*m:
    print 0
    sys.exit()

pos = [[] for _ in range(n)]
for i in range(n):
    pos[C[i]].append(i)


DP = [[1]*(n + 1) for _ in range(n + 1)]
for le in range(1, n + 1):
    for i in range(n - le + 1):
        j = i + le
        k1 = min(range(i,j), key = C.__getitem__)
        c = C[k1]

        k1 = pos[c][0]
        k2 = pos[c][-1]

        ans1 = 0
        for split in range(i,k1 + 1):
            ans1 += DP[i][split] * DP[split][k1] % MOD   
        
        ans2 = 0
        for split in range(k2+1, j + 1):
            ans2 += DP[k2 + 1][split] * DP[split][j] % MOD

        posc = pos[c]
        for ind in range(len(posc) - 1):
            ans1 = (ans1 * DP[posc[ind] + 1][posc[ind + 1]] % MOD)

        DP[i][j] = int((ans1 % MOD) * (ans2 % MOD) % MOD)

print DP[0][n]
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

MOD = 998244353

def compute_answer(n_input, m_input, c_list):
    # Correctly map problem's n (number of colors) and m (strip length) to reference code's variables
    m_code = n_input  # Reference code's m represents problem's n (number of colors)
    n_code = m_input  # Reference code's n represents problem's m (strip length)

    C = [x - 1 for x in c_list]
    
    # Compress consecutive duplicates
    if not C:
        return 0
    C2 = [C[0]]
    for c in C[1:]:
        if C2[-1] != c:
            C2.append(c)
    new_n = len(C2)
    
    # Check if compressed length exceeds 2*m_code (problem's n)
    if new_n > 2 * m_code:
        return 0
    
    pos = [[] for _ in range(m_code)]
    for i in range(new_n):
        c = C2[i]
        if c >= m_code or c < 0:
            return 0
        pos[c].append(i)
    
    # Verify all colors are present
    for color in range(m_code):
        if not pos[color]:
            return 0
    
    DP = [[1] * (new_n + 1) for _ in range(new_n + 1)]
    
    for le in range(1, new_n + 1):
        for i in range(new_n - le + 1):
            j = i + le
            min_color = min(C2[i:j])
            min_indices = [p for p in range(i, j) if C2[p] == min_color]
            if not min_indices:
                DP[i][j] = 0
                continue
            
            first = min(min_indices)
            last = max(min_indices)
            
            # Calculate left part
            left = 0
            for k in range(i, first + 1):
                left = (left + DP[i][k] * DP[k][first]) % MOD
            
            # Calculate right part
            right = 0
            for k in range(last + 1, j + 1):
                right = (right + DP[last + 1][k] * DP[k][j]) % MOD
            
            # Calculate middle parts between occurrences of min_color
            middle = 1
            color_positions = pos[min_color]
            for idx in range(len(color_positions) - 1):
                prev = color_positions[idx]
                next_p = color_positions[idx + 1]
                if prev < i or next_p >= j:
                    continue
                middle = (middle * DP[prev + 1][next_p]) % MOD
            
            DP[i][j] = (left * right % MOD) * middle % MOD
    
    return DP[0][new_n]

class F2longcolorfulstripbootcamp(Basebootcamp):
    def __init__(self, max_n=5, max_m=10):
        self.max_n = max_n
        self.max_m = max_m
    
    def case_generator(self):
        # Ensure valid problem constraints: n >=1, m >=n
        n = random.randint(1, self.max_n)
        m = random.randint(n, self.max_m)
        
        # Generate initial c list containing all colors 1..n
        c = list(range(1, n+1))
        # Add remaining elements randomly
        if m > n:
            c += [random.randint(1, n) for _ in range(m - n)]
        # Shuffle to create random configuration
        random.shuffle(c)
        
        return {
            'n': n,
            'm': m,
            'c': c
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        c = question_case['c']
        c_str = ' '.join(map(str, c))
        problem = f"""You are a programming competition participant. Solve the following problem and enclose your answer within [answer] and [/answer] tags.

Problem:
Calculate the number of valid ways Alice could have painted the strip. The initial strip is color 0, and each step repaints a segment to a new color. The result must match the given configuration.

Input:
The first line contains two integers n and m: {n} {m}.
The second line contains {m} integers: {c_str}.

Output:
The number of valid ways modulo 998244353. Provide your answer inside [answer] tags."""
        return problem
    
    @staticmethod
    def extract_output(output):
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
        n = identity['n']
        m = identity['m']
        c = identity['c']
        correct = compute_answer(n, m, c)
        return (solution % MOD) == (correct % MOD)
