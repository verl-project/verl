"""# 

### 谜题描述
During the lesson small girl Alyona works with one famous spreadsheet computer program and learns how to edit tables.

Now she has a table filled with integers. The table consists of n rows and m columns. By ai, j we will denote the integer located at the i-th row and the j-th column. We say that the table is sorted in non-decreasing order in the column j if ai, j ≤ ai + 1, j for all i from 1 to n - 1.

Teacher gave Alyona k tasks. For each of the tasks two integers l and r are given and Alyona has to answer the following question: if one keeps the rows from l to r inclusive and deletes all others, will the table be sorted in non-decreasing order in at least one column? Formally, does there exist such j that ai, j ≤ ai + 1, j for all i from l to r - 1 inclusive.

Alyona is too small to deal with this task and asks you to help!

Input

The first line of the input contains two positive integers n and m (1 ≤ n·m ≤ 100 000) — the number of rows and the number of columns in the table respectively. Note that your are given a constraint that bound the product of these two integers, i.e. the number of elements in the table.

Each of the following n lines contains m integers. The j-th integers in the i of these lines stands for ai, j (1 ≤ ai, j ≤ 109).

The next line of the input contains an integer k (1 ≤ k ≤ 100 000) — the number of task that teacher gave to Alyona.

The i-th of the next k lines contains two integers li and ri (1 ≤ li ≤ ri ≤ n).

Output

Print \"Yes\" to the i-th line of the output if the table consisting of rows from li to ri inclusive is sorted in non-decreasing order in at least one column. Otherwise, print \"No\".

Example

Input

5 4
1 2 3 5
3 1 3 2
4 5 2 3
5 5 3 2
4 4 3 4
6
1 1
2 5
4 5
3 5
1 3
1 5


Output

Yes
No
Yes
Yes
Yes
No

Note

In the sample, the whole table is not sorted in any column. However, rows 1–3 are sorted in column 1, while rows 4–5 are sorted in column 3.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
getline = lambda: [int(x) for x in raw_input().split()]

n,m = getline()
a = [getline() for _ in range(n)]
k = int(raw_input())
q = [getline() for _ in range(k)]

v = range(n)
for i in range(m):
	l = 0
	climbs = []
	for j in range(1,n):
		if a[j][i] >= a[j-1][i]:
			continue
		else:
			climbs.append((l,j))
			l=j
	climbs.append((l,n))
	
	for l,r in climbs:
		for w in range(l, r):
			v[w] = max(v[w], r-1)

for l,r in q:
	l,r = l-1, r-1
	if r <= v[l]:
		print \"Yes\"
	else:
		print \"No\"
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

def generate_column(n):
    if n == 0:
        return []
    column = []
    prev = random.randint(1, 10**9)
    column.append(prev)
    for _ in range(n-1):
        if random.random() < 0.2:
            current = random.randint(1, prev-1)
        else:
            current = random.randint(prev, 10**9)
        column.append(current)
        prev = current
    return column

def preprocess_v(table):
    n = len(table)
    if n == 0:
        return []
    m = len(table[0]) if n > 0 else 0
    v = list(range(n))  # 初始为0-based索引的row值
    for col in range(m):
        column = [table[i][col] for i in range(n)]
        climbs = []
        current_start = 0
        for i in range(1, n):
            if column[i] < column[i-1]:
                climbs.append((current_start, i))
                current_start = i
        climbs.append((current_start, n))
        for (l, r) in climbs:
            for w in range(l, r):
                if w < n:
                    v[w] = max(v[w], r - 1)
    return v

class Calyonaandspreadsheetbootcamp(Basebootcamp):
    def __init__(self, max_nm=10**5, max_n=10000, max_m=10000):
        self.max_nm = max_nm
        self.max_n = max_n
        self.max_m = max_m

    def case_generator(self):
        while True:
            n = random.randint(1, min(self.max_n, self.max_nm))
            m = random.randint(1, min(self.max_m, self.max_nm // max(n, 1)))
            if n * m <= self.max_nm:
                break
        
        # Generate columns
        columns = []
        for _ in range(m):
            col = generate_column(n)
            columns.append(col)
        
        # Transpose to get rows
        table = []
        for i in range(n):
            row = [columns[j][i] for j in range(m)]
            table.append(row)
        
        v = preprocess_v(table)
        
        # Generate query
        l = random.randint(1, n)
        r = random.randint(l, n)
        
        return {
            'n': n,
            'm': m,
            'table': table,
            'l': l,
            'r': r,
            'v': v
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        m = question_case['m']
        table = question_case['table']
        l = question_case['l']
        r = question_case['r']
        
        prompt = f"""You are helping Alyona analyze a spreadsheet table. The table contains {n} rows and {m} columns of integers. Your task is to determine if there exists at least one column that remains non-decreasing when only considering rows {l} to {r} inclusive.

Table Data:
"""
        for idx, row in enumerate(table):
            prompt += f"Row {idx+1}: {' '.join(map(str, row))}\n"
        
        prompt += f"""
Rules:
1. A column is non-decreasing if every element from row {l} to row {r-1} is ≤ the next element in the same column.
2. Check all columns - if any column satisfies this condition, the answer is Yes.

Format your answer as either [answer]Yes[/answer] or [answer]No[/answer], using exactly this format."""

        return prompt
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.IGNORECASE | re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        if last_match.lower() in ['yes', 'no']:
            return last_match.capitalize()
        return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        l = identity['l']
        r = identity['r']
        v = identity['v']
        l_index = l - 1
        r_index = r - 1
        expected = 'Yes' if r_index <= v[l_index] else 'No'
        return solution.lower() == expected.lower()
