"""# 

### 谜题描述
You are given an n × m rectangular table consisting of lower case English letters. In one operation you can completely remove one column from the table. The remaining parts are combined forming a new table. For example, after removing the second column from the table
    
    
      
    abcd  
    edfg  
    hijk  
    

we obtain the table:
    
    
      
    acd  
    efg  
    hjk  
    

A table is called good if its rows are ordered from top to bottom lexicographically, i.e. each row is lexicographically no larger than the following one. Determine the minimum number of operations of removing a column needed to make a given table good.

Input

The first line contains two integers — n and m (1 ≤ n, m ≤ 100).

Next n lines contain m small English letters each — the characters of the table.

Output

Print a single number — the minimum number of columns that you need to remove in order to make the table good.

Examples

Input

1 10
codeforces


Output

0


Input

4 4
case
care
test
code


Output

2


Input

5 4
code
forc
esco
defo
rces


Output

4

Note

In the first sample the table is already good.

In the second sample you may remove the first and third column.

In the third sample you have to remove all the columns (note that the table where all rows are empty is considered good by definition).

Let strings s and t have equal length. Then, s is lexicographically larger than t if they are not equal and the character following the largest common prefix of s and t (the prefix may be empty) in s is alphabetically larger than the corresponding character of t.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n, m = map(int, raw_input().split())
r = []
c = {}
remove = 0
for _ in range(n):
    r.append(list(raw_input()))

row = 0
while row != n:
    if row+1 < n and r[row] > r[row+1]:
        s1 = r[row]
        s2 = r[row+1]
        for col in range(m):
            if col not in c and s1[col] > s2[col]:
                c[col] = True
                remove+=1
                for i, s in enumerate(r):
                    s[col] = ''
                    r[i] = s
                row = 0
                break
    else:
        row+=1
print remove
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cremovingcolumnsbootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=10, m_min=1, m_max=10):
        self.n_min = n_min
        self.n_max = n_max
        self.m_min = m_min
        self.m_max = m_max

    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        m = random.randint(self.m_min, self.m_max)
        rows = [''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(m)) for _ in range(n)]
        correct_answer = self.compute_min_removals(n, m, rows)
        return {
            'n': n,
            'm': m,
            'rows': rows,
            'correct_answer': correct_answer
        }

    @staticmethod
    def compute_min_removals(n, m, rows):
        if n <= 1:
            return 0
        deleted = set()
        fixed = [False] * (n-1)
        for col in range(m):
            delete_col = False
            for i in range(n-1):
                if fixed[i]:
                    continue
                if rows[i][col] > rows[i+1][col]:
                    delete_col = True
                    break
            if delete_col:
                deleted.add(col)
            else:
                for i in range(n-1):
                    if not fixed[i] and rows[i][col] < rows[i+1][col]:
                        fixed[i] = True
        return len(deleted)

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        rows = question_case['rows']
        input_lines = [f"{n} {m}"] + rows
        input_block = '\n'.join(input_lines)
        problem_text = f"""You are given a {n}×{m} rectangular table of lowercase English letters. Your task is to determine the minimum number of columns you need to remove so that the remaining table's rows are ordered lexicographically from top to bottom.

Lexicographical order is defined as follows: two strings are compared based on the first position where they differ. The string with the smaller character at that position is considered lexicographically smaller. If one string is a prefix of the other, the shorter string is considered smaller.

For example, after removing the second column from the table:

abcd
edfg
hijk

the resulting table is:

acd
efg
hjk

Your goal is to find the minimum number of columns that must be removed to make the table good. A table is considered good if each row is lexicographically no larger than the next row.

Input format:
- The first line contains two integers n (number of rows) and m (number of columns).
- The next n lines each contain exactly m lowercase letters representing the table.

Now, solve the following specific case:

Input:
{input_block}

Please provide your answer as a single integer enclosed within [answer] and [/answer] tags. For example, if your answer is 3, write it as [answer]3[/answer]."""
        return problem_text

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return int(solution.strip()) == identity['correct_answer']
        except (ValueError, KeyError):
            return False
