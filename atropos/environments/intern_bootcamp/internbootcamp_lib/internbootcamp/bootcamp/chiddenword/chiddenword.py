"""# 

### 谜题描述
Let’s define a grid to be a set of tiles with 2 rows and 13 columns. Each tile has an English letter written in it. The letters don't have to be unique: there might be two or more tiles with the same letter written on them. Here is an example of a grid:
    
    
    ABCDEFGHIJKLM  
    NOPQRSTUVWXYZ

We say that two tiles are adjacent if they share a side or a corner. In the example grid above, the tile with the letter 'A' is adjacent only to the tiles with letters 'B', 'N', and 'O'. A tile is not adjacent to itself.

A sequence of tiles is called a path if each tile in the sequence is adjacent to the tile which follows it (except for the last tile in the sequence, which of course has no successor). In this example, \"ABC\" is a path, and so is \"KXWIHIJK\". \"MAB\" is not a path because 'M' is not adjacent to 'A'. A single tile can be used more than once by a path (though the tile cannot occupy two consecutive places in the path because no tile is adjacent to itself).

You’re given a string s which consists of 27 upper-case English letters. Each English letter occurs at least once in s. Find a grid that contains a path whose tiles, viewed in the order that the path visits them, form the string s. If there’s no solution, print \"Impossible\" (without the quotes).

Input

The only line of the input contains the string s, consisting of 27 upper-case English letters. Each English letter occurs at least once in s.

Output

Output two lines, each consisting of 13 upper-case English characters, representing the rows of the grid. If there are multiple solutions, print any of them. If there is no solution print \"Impossible\".

Examples

Input

ABCDEFGHIJKLMNOPQRSGTUVWXYZ


Output

YXWVUTGHIJKLM
ZABCDEFSRQPON


Input

BUVTYZFQSNRIWOXXGJLKACPEMDH


Output

Impossible

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
s = raw_input()

st = 0
en = 0

ans = [['.' for _ in xrange(13)] for _ in xrange(2)]

for i in xrange(ord('A'),ord('Z')+1):
    c = chr(i)
    st = s.find(c)
    en = s.find(c,st+1)
    if en != -1:
        break

if st+1 == en:
    print \"Impossible\"
else:
    l = (en-st)
    l += (l%2)
    ss = 13-(l/2)
    p = [ss,0]
    dr = 1
    for i in xrange(st,en):
        ans[p[1]][p[0]] = s[i]
        if (p[0]+dr == 13):
            p[1] += 1
            dr *= -1
        else:
            p[0] += dr
    p = [ss-1,0]
    dr = -1
    a = s[:st]
    b = s[en+1:]
    bf = a[::-1]+ b[::-1]
    for i in xrange(len(bf)):
        if p[0]<0:
            p[0] = 0
            p[1] = 1
            dr = 1
        ans[p[1]][p[0]] = bf[i]
        p[0] += dr
    print \"\".join(ans[0])
    print \"\".join(ans[1])
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import string
import random
from bootcamp import Basebootcamp

class Chiddenwordbootcamp(Basebootcamp):
    def __init__(self, force_possible=None, **params):
        super().__init__(**params)
        self.force_possible = force_possible
    
    def case_generator(self):
        while True:
            letters = list(string.ascii_uppercase)
            duplicate_char = random.choice(letters)
            s_list = letters.copy()
            s_list.append(duplicate_char)
            random.shuffle(s_list)
            s = ''.join(s_list)
            
            st = s.find(duplicate_char)
            en = s.find(duplicate_char, st + 1)
            if en == -1:
                continue
            
            if self.force_possible is None:
                break
            elif self.force_possible:
                if en - st > 1:
                    break
            else:
                if en - st == 1:
                    break
        
        solution = self.generate_solution(s)
        expected_output = solution if solution != "Impossible" else solution
        return {
            's': s,
            'expected_output': expected_output
        }
    
    @staticmethod
    def generate_solution(s_input):
        s = s_input
        st = 0
        en = 0
        ans = [['.' for _ in range(13)] for _ in range(2)]
        found = False
        for i in range(ord('A'), ord('Z') + 1):
            c = chr(i)
            st = s.find(c)
            if st == -1:
                continue
            en = s.find(c, st + 1)
            if en != -1:
                found = True
                break
        if not found:
            return "Impossible"
        
        if st + 1 == en:
            return "Impossible"
        else:
            l = (en - st)
            l += l % 2
            ss = 13 - (l // 2)
            p = [ss, 0]
            dr = 1
            for i in range(st, en):
                ans[p[1]][p[0]] = s[i]
                if p[0] + dr == 13:
                    p[1] += 1
                    dr *= -1
                else:
                    p[0] += dr
            p = [ss - 1, 0]
            dr = -1
            a = s[:st]
            b = s[en + 1:]
            bf = a[::-1] + b[::-1]
            for i in range(len(bf)):
                if p[0] < 0:
                    p[0] = 0
                    p[1] = 1
                    dr = 1
                ans[p[1]][p[0]] = bf[i]
                p[0] += dr
            row0 = ''.join(ans[0])
            row1 = ''.join(ans[1])
            return [row0, row1]
    
    @staticmethod
    def prompt_func(question_case):
        s = question_case['s']
        prompt = f"""You are given a string s of 27 uppercase English letters, where each letter appears at least once. Your task is to construct a 2-row by 13-column grid where each tile contains a letter. The grid must contain a path that spells the string s when traversed consecutively. Adjacent tiles are those sharing a side or a corner. The path must visit adjacent tiles (including diagonally adjacent) in sequence. If no such grid exists, output 'Impossible'. Otherwise, output the two rows of the grid. Ensure your answer is formatted with the two rows enclosed within [answer] tags. 

Input string s: {s}

Put your answer within [answer] and [/answer] tags. For example, if the solution is:
YXWVUTGHIJKLM
ZABCDEFSRQPON
Your answer should be formatted as:
[answer]YXWVUTGHIJKLM[/answer]
[answer]ZABCDEFSRQPON[/answer]

If impossible, write:
[answer]Impossible[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        pattern = re.compile(r'\[answer\](.*?)\[\/answer\]', re.DOTALL | re.IGNORECASE)
        matches = pattern.findall(output)
        if not matches:
            return None
        last_match = matches[-1].strip()
        if 'impossible' in last_match.lower():
            return 'Impossible'
        rows = [line.strip() for line in last_match.split('\n') if line.strip()]
        if len(rows) == 2 and len(rows[0]) == 13 and len(rows[1]) == 13 and rows[0].isupper() and rows[1].isupper():
            return rows
        return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity['expected_output']
        if expected == "Impossible":
            return solution == "Impossible"
        else:
            return solution == expected
