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
import sys

def isLexicographical(s):
	return s == sorted(s)
	
def concatenate(s,t):
	return s+t

n,m = map(int, sys.stdin.readline().split())
l = [list(sys.stdin.readline())[0:-1] for fruta in range(n)]
columnas = [[l[i][j] for i in range(n) ] for j in range(m)]

sol = ['' for fruta in range(n)]
c = 0
i = 0
while i < m:
	test = map(concatenate,sol,columnas[i])
	if not(isLexicographical(test)):
		c += 1
	else:
		sol = test
	i += 1

print c
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def compute_min_removals(n, m, table_rows):
    sol = ['' for _ in range(n)]
    count = 0
    for col_idx in range(m):
        current_col = [row[col_idx] for row in table_rows]
        temp = [sol[i] + current_col[i] for i in range(n)]
        if temp == sorted(temp):
            sol = temp
        else:
            count += 1
    return count

class Aremovingcolumnsbootcamp(Basebootcamp):
    def __init__(self, n_range=(1, 5), m_range=(1, 5)):
        self.n_min, self.n_max = n_range
        self.m_min, self.m_max = m_range

    def case_generator(self):
        n = random.randint(self.n_min, self.n_max)
        m = random.randint(self.m_min, self.m_max)
        table = [''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=m)) for _ in range(n)]
        correct_answer = compute_min_removals(n, m, table)
        return {
            'n': n,
            'm': m,
            'table': table,
            'correct': correct_answer  # Key必须与_verify_correction一致
        }

    @staticmethod
    def prompt_func(question_case):
        input_lines = [f"{question_case['n']} {question_case['m']}"] + question_case['table']
        
        # 修复的字符串拼接结构
        problem_description = (
            "You are given a rectangular table of lowercase English letters. Your task is to determine the "
            "minimum number of columns you need to remove so that the rows become lexicographically ordered "
            "from top to bottom.\n\n"
            
            "Lexicographical Order Rules:\n"
            "1. A string 's' is lexicographically smaller than 't' if:\n"
            "   - At the first position where they differ, 's' has a character smaller than 't'\n"
            "   - All preceding characters are identical\n"
            "   - Example: 'apple' < 'apply' (diff at 4th character)\n\n"
            
            "Special Cases:\n"
            "- An empty table (all columns removed) is considered valid\n"
            "- Single-row tables are always valid (requires 0 removals)\n\n"
            
            "Input Format:\n"
            "- Line 1: Two integers n (rows) and m (columns)\n"
            "- Next n lines: Each contains exactly m lowercase letters\n\n"
            
            "Output Format:\n"  # 修复此处换行符错误
            "- A single integer indicating the minimal number of columns to remove\n\n"
            
            "Example Input 1:\n"
            "1 10\ncodeforces\n\n"
            "Example Output 1:\n[answer]0[/answer]\n\n"
            
            "Example Input 2:\n"
            "4 4\ncase\ncare\ntest\ncode\n\n"
            "Example Output 2:\n[answer]2[/answer]\n\n"
            
            "Now solve this problem:\n"
            f"{chr(10).join(input_lines)}\n"
            "Put your final answer within [answer] and [/answer] tags."
        )
        return problem_description

    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        try:
            return int(answer_blocks[-1].strip())
        except (ValueError, TypeError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct']  # 确保与case_generator的键一致
