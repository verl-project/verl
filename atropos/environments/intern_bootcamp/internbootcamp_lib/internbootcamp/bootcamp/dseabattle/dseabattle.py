"""# 

### 谜题描述
Galya is playing one-dimensional Sea Battle on a 1 × n grid. In this game a ships are placed on the grid. Each of the ships consists of b consecutive cells. No cell can be part of two ships, however, the ships can touch each other.

Galya doesn't know the ships location. She can shoot to some cells and after each shot she is told if that cell was a part of some ship (this case is called \"hit\") or not (this case is called \"miss\").

Galya has already made k shots, all of them were misses.

Your task is to calculate the minimum number of cells such that if Galya shoot at all of them, she would hit at least one ship.

It is guaranteed that there is at least one valid ships placement.

Input

The first line contains four positive integers n, a, b, k (1 ≤ n ≤ 2·105, 1 ≤ a, b ≤ n, 0 ≤ k ≤ n - 1) — the length of the grid, the number of ships on the grid, the length of each ship and the number of shots Galya has already made.

The second line contains a string of length n, consisting of zeros and ones. If the i-th character is one, Galya has already made a shot to this cell. Otherwise, she hasn't. It is guaranteed that there are exactly k ones in this string. 

Output

In the first line print the minimum number of cells such that if Galya shoot at all of them, she would hit at least one ship.

In the second line print the cells Galya should shoot at.

Each cell should be printed exactly once. You can print the cells in arbitrary order. The cells are numbered from 1 to n, starting from the left.

If there are multiple answers, you can print any of them.

Examples

Input

5 1 2 1
00100


Output

2
4 2


Input

13 3 2 3
1000000010001


Output

2
7 11

Note

There is one ship in the first sample. It can be either to the left or to the right from the shot Galya has already made (the \"1\" character). So, it is necessary to make two shots: one at the left part, and one at the right part.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/usr/bin/env python

n,a,b,k = map(int,raw_input().split())
s = raw_input()

# Find all segments of zeroes
prev = '1'
segments = []
start = None
length = 0
idx = 0
s = s+'1'
while (idx < len(s)):
  if (prev=='1') and (s[idx]=='0'):
    start = idx
    length = 0
  if (s[idx]=='0'):
    length+=1
  if (prev=='0') and (s[idx]=='1'):
    segments.append((start, length))
  prev=s[idx]
  idx+=1

# Find all positions where a ship could be
positions = []
for (start, length) in segments:
  p=start+b-1
  while (p < (start+length)):
    positions.append(p+1)
    p+=b


# We can throw away a-1 positions
positions = positions[a-1:]
print len(positions)
print \" \".join(map(str,positions))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Dseabattlebootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n = params.get('n', 8)
        self.a = params.get('a', 2)
        self.b = params.get('b', 3)
        self.k = params.get('k', 2)

    def case_generator(self):
        while True:
            a = random.randint(1, 3)
            b = random.randint(1, 3)
            min_n = a * b
            max_n = min_n + 10
            n = random.randint(min_n, max_n)
            possible_k_max = n - a * b
            if possible_k_max < 0:
                continue
            k = random.randint(0, possible_k_max)
            ships = self.generate_ships(n, a, b)
            if ships is None:
                continue
            occupied = set()
            for s in ships:
                for i in range(b):
                    occupied.add(s + i)
            available = [i for i in range(n) if i not in occupied]
            if len(available) < k:
                continue
            selected_shots = random.sample(available, k)
            s_list = ['0'] * n
            for idx in selected_shots:
                s_list[idx] = '1'
            s = ''.join(s_list)
            if s.count('1') != k:
                continue
            return {
                'n': n,
                'a': a,
                'b': b,
                'k': k,
                's': s
            }

    @staticmethod
    def generate_ships(n, a, b):
        occupied = set()
        ships = []
        for _ in range(a):
            possible_starts = []
            for s in range(n - b + 1):
                conflict = False
                for i in range(b):
                    if (s + i) in occupied:
                        conflict = True
                        break
                if not conflict:
                    possible_starts.append(s)
            if not possible_starts:
                return None
            s = random.choice(possible_starts)
            ships.append(s)
            for i in range(b):
                occupied.add(s + i)
        return ships

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = question_case['a']
        b = question_case['b']
        k = question_case['k']
        s = question_case['s']
        prompt = f"""You are playing a one-dimensional Sea Battle game on a 1×{n} grid. There are {a} ships placed on the grid, each consisting of {b} consecutive cells. Ships cannot overlap but can be adjacent. Galya has made {k} shots, all of which missed. Your task is to determine the minimal number of cells to shoot such that at least one ship is hit. The grid is represented by a string of '0's and '1's, where '1's indicate previously shot cells (all misses).

Input format:
The first line contains four integers: n, a, b, k.
The second line contains the string of length n.

For this problem, the input is:
{n} {a} {b} {k}
{s}

Output format:
The first line must contain the minimal number of cells to shoot. The second line must list the cell numbers in any order. Each cell must be numbered from 1 to n.

Please provide your answer within [answer] and [/answer] tags. For example:

[answer]
2
4 2
[/answer]"""
        return prompt

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        answer_block = matches[-1].strip()
        lines = answer_block.split('\n')
        if len(lines) < 2:
            return None
        try:
            count = int(lines[0].strip())
            cells = list(map(int, lines[1].strip().split()))
            if len(cells) != count:
                return None
            return f"{count}\n{' '.join(map(str, cells))}"
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        a = identity['a']
        b = identity['b']
        s_input = identity['s']
        s = s_input + '1'
        segments = []
        prev = '1'
        start = None
        length = 0
        idx = 0
        while idx < len(s):
            if prev == '1' and s[idx] == '0':
                start = idx
                length = 0
            if s[idx] == '0':
                length += 1
            if prev == '0' and s[idx] == '1':
                segments.append((start, length))
            prev = s[idx]
            idx += 1
        positions = []
        for seg_start, seg_length in segments:
            p = seg_start + b - 1
            while p < seg_start + seg_length:
                positions.append(p + 1)
                p += b
        positions.sort()
        correct_positions = positions[a-1:]
        try:
            lines = solution.strip().split('\n')
            if len(lines) < 2:
                return False
            user_count = int(lines[0].strip())
            user_cells = list(map(int, lines[1].strip().split()))
        except:
            return False
        if user_count != len(correct_positions):
            return False
        user_set = set(user_cells)
        correct_set = set(correct_positions)
        return user_set == correct_set and len(user_cells) == user_count
