"""# 

### 谜题描述
You are given n integers. You need to choose a subset and put the chosen numbers in a beautiful rectangle (rectangular matrix). Each chosen number should occupy one of its rectangle cells, each cell must be filled with exactly one chosen number. Some of the n numbers may not be chosen.

A rectangle (rectangular matrix) is called beautiful if in each row and in each column all values are different.

What is the largest (by the total number of cells) beautiful rectangle you can construct? Print the rectangle itself.

Input

The first line contains n (1 ≤ n ≤ 4⋅10^5). The second line contains n integers (1 ≤ a_i ≤ 10^9).

Output

In the first line print x (1 ≤ x ≤ n) — the total number of cells of the required maximum beautiful rectangle. In the second line print p and q (p ⋅ q=x): its sizes. In the next p lines print the required rectangle itself. If there are several answers, print any.

Examples

Input


12
3 1 4 1 5 9 2 6 5 3 5 8


Output


12
3 4
1 2 3 5
3 1 5 4
5 6 8 9


Input


5
1 1 1 1 1


Output


1
1 1
1

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import math
n = input()
arr = map(int,raw_input().split(\" \"))
 
def add(dic,key,v):
    if not dic.has_key(key):
        dic[key]  = v
    else:
        dic[key] += v
dic = {}
for i in range(n):
    add(dic,arr[i],1)
 
keys = sorted(dic.keys(),key = lambda x:dic[x])
cnts = [0]*(n+1)
psum = [0]*(n+1)
 
i = len(keys)-1
j = n
while i >=0:
    while j >dic[keys[i]]:
        cnts[j-1] = cnts[j]
        j-= 1
    cnts[j] = cnts[j]+1
    i -= 1
while j >= 1:
    cnts[j-1] = cnts[j]
    j -= 1
 
i,j = 0,0
while i<len(keys):
    while j<dic[keys[i]]:
        psum[j+1] = psum[j]
        j += 1
    psum[j] = psum[j]+dic[keys[i]]
    i += 1
while j < n:
    psum[j+1] = psum[j]
    j += 1
maxv = 0
maxp = None
for row in  range(1,int(math.sqrt(n))+1):
    s = cnts[row]*row + psum[row-1]
    col = s/row
    if col < row:
        continue
    if col*row > maxv:
        maxv = col*row
        maxp = col,row
arr = []
col,row = maxp
print maxv
print row,col
for k in keys:
    if dic[k] >row:
        arr+= [k]*row
    else:
        arr += [k]*dic[k]
mat = [[0]*col for i in range(row)]
i,j = 0,0
k = len(arr) -1
for j in range(col):
    for i in range(row):
        mat[i][(i+j)%col] = arr[k]
        k -= 1
for i in range(row):
    print \" \".join(map(str,mat[i]))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
from collections import defaultdict
import random
import re

class Fbeautifulrectanglebootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = params.get('params', {})
        self.max_case_size = self.params.get('max_case_size', 15)

    def case_generator(self):
        mode = random.choice(['unique', 'all_same', 'mixed'])
        input_arr = []
        
        if mode == 'unique':
            n = random.randint(1, self.max_case_size)
            input_arr = random.sample(range(1, 2*self.max_case_size), n)
        elif mode == 'all_same':
            n = random.randint(1, self.max_case_size)
            input_arr = [random.randint(1, 10)] * n
        else:
            input_arr = []
            for _ in range(random.randint(2,3)):
                val = random.randint(1, 10)
                input_arr += [val] * random.randint(2,5)
            unique_vals = random.sample(range(11, 20), random.randint(2,5))
            input_arr += unique_vals
            random.shuffle(input_arr)
            input_arr = input_arr[:self.max_case_size]
        
        # 确保至少有一个元素
        if not input_arr:
            input_arr = [random.randint(1,10)]
        
        x, p, q, matrix = self.solve_max_beautiful_rectangle(input_arr)
        return {
            'input_arr': input_arr,
            'x': x,
            'p': p,
            'q': q,
            'matrix': matrix
        }

    @staticmethod
    def solve_max_beautiful_rectangle(arr):
        from collections import defaultdict
        dic = defaultdict(int)
        for num in arr:
            dic[num] += 1
        
        n = len(arr)
        if n == 0:
            return (0, 0, 0, [])
        
        keys = sorted(dic.keys(), key=lambda x: dic[x])
        
        # 正确初始化cnts数组
        cnts = [0]*(n+1)
        j = n
        for k in reversed(keys):
            freq = dic[k]
            while j > freq:
                if j-1 >=0:
                    cnts[j-1] = cnts[j]
                j -=1
            if j >=0:
                cnts[j] +=1
        while j >0:
            cnts[j-1] = cnts[j]
            j -=1
        
        # 正确初始化psum数组
        psum = [0]*(n+1)
        j = 0
        for k in keys:
            freq = dic[k]
            while j < freq:
                if j+1 <=n:
                    psum[j+1] = psum[j]
                j +=1
            if j <=n:
                psum[j] += freq
        while j <n:
            psum[j+1] = psum[j]
            j +=1
        
        max_area = 0
        best_dims = (1, 1)
        max_row = int(math.sqrt(n)) if n>0 else 0
        for row in range(1, max_row+1):
            if row >n:
                break
            s = cnts[row] * row + psum[row]
            cols = s // row if row !=0 else 0
            if cols >= row and row * cols > max_area:
                max_area = row * cols
                best_dims = (row, cols)
        
        if max_area ==0:
            # 处理全相同元素的情况
            if not dic:
                return (0,0,0,[])
            max_elem = max(dic.keys(), key=lambda x:dic[x])
            return (1,1,1, [[max_elem]])
        
        row, cols = best_dims
        selected = []
        for k in reversed(keys):
            freq = dic[k]
            add_count = min(freq, row)
            selected += [k]*add_count
        
        matrix = [[0]*cols for _ in range(row)]
        index = len(selected)-1
        for c in range(cols):
            for r in range(row):
                matrix[r][(r+c)%cols] = selected[index]
                index -=1
                if index <0:
                    break
        
        return (max_area, row, cols, matrix)

    @staticmethod
    def prompt_func(question_case):
        arr_str = ' '.join(map(str, question_case['input_arr']))
        return (
            f"给定数组，构造最大美丽矩形。规则：\n"
            "1. 矩阵每行、每列元素必须唯一\n"
            "2. 输出格式：\n第一行x（总单元格数）\n第二行p q（行列）\n随后p行每行q个整数\n"
            f"输入数组：n={len(question_case['input_arr'])}\n{arr_str}\n"
            "将答案放在[answer]标签内。示例：\n[answer]\n6\n2 3\n1 2 3\n4 5 6\n[/answer]"
        )

    @staticmethod
    def extract_output(output):
        pattern = r'\[answer\](.*?)\[\/answer\]'
        matches = re.findall(pattern, output, flags=re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        lines = [l.strip() for l in last_match.split('\n') if l.strip()]
        
        try:
            if len(lines) <2:
                return None
            x = int(lines[0])
            p, q = map(int, lines[1].split())
            if p*q !=x or x <=0:
                return None
            matrix_lines = lines[2:2+p]
            if len(matrix_lines)!=p:
                return None
            matrix = []
            for line in matrix_lines:
                row = list(map(int, line.strip().split()))
                if len(row)!=q:
                    return None
                matrix.append(row)
            return {'x':x, 'p':p, 'q':q, 'matrix':matrix}
        except ValueError:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            # 基础尺寸验证
            if solution['x'] != identity['x']:
                return False
            if solution['p'] * solution['q'] != solution['x']:
                return False
            if len(solution['matrix']) != solution['p']:
                return False
            if any(len(row)!=solution['q'] for row in solution['matrix']):
                return False
            
            # 元素频率验证
            input_freq = defaultdict(int)
            for num in identity['input_arr']:
                input_freq[num] +=1
            output_freq = defaultdict(int)
            for row in solution['matrix']:
                for num in row:
                    output_freq[num] +=1
                    if output_freq[num] > input_freq[num]:
                        return False
            
            # 行列唯一性验证
            for row in solution['matrix']:
                if len(row) != len(set(row)):
                    return False
            for c in range(solution['q']):
                column = [solution['matrix'][r][c] for r in range(solution['p'])]
                if len(column)!=len(set(column)):
                    return False
            return True
        except Exception as e:
            return False
