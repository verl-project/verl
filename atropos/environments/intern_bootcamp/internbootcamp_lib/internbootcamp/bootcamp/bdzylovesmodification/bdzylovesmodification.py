"""# 

### 谜题描述
As we know, DZY loves playing games. One day DZY decided to play with a n × m matrix. To be more precise, he decided to modify the matrix with exactly k operations.

Each modification is one of the following:

  1. Pick some row of the matrix and decrease each element of the row by p. This operation brings to DZY the value of pleasure equal to the sum of elements of the row before the decreasing. 
  2. Pick some column of the matrix and decrease each element of the column by p. This operation brings to DZY the value of pleasure equal to the sum of elements of the column before the decreasing. 



DZY wants to know: what is the largest total value of pleasure he could get after performing exactly k modifications? Please, help him to calculate this value.

Input

The first line contains four space-separated integers n, m, k and p (1 ≤ n, m ≤ 103; 1 ≤ k ≤ 106; 1 ≤ p ≤ 100).

Then n lines follow. Each of them contains m integers representing aij (1 ≤ aij ≤ 103) — the elements of the current row of the matrix.

Output

Output a single integer — the maximum possible total pleasure value DZY could get.

Examples

Input

2 2 2 2
1 3
2 4


Output

11


Input

2 2 5 2
1 3
2 4


Output

11

Note

For the first sample test, we can modify: column 2, row 2. After that the matrix becomes:
    
    
      
    1 1  
    0 0  
      
    

For the second sample test, we can modify: column 2, row 2, row 1, column 1, column 2. After that the matrix becomes:
    
    
      
    -3 -3  
    -2 -2  
      
    

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from heapq import heappush,heappop,heapify
n,m,k,p=map(int,raw_input().split())
row=[]
col=[]
heapify(row)
heapify(col)
arr=[]
for i in range(n):
    s=map(int,raw_input().split())
    heappush(row,-sum(s))
    arr.append(s)
for i in range(m):
    s=0
    for j in range(n):
        s-=arr[j][i]
    heappush(col,s)
pr={}
pc={}
rr=0
rc=0
pr[0]=0
pc[0]=0
for h in range(k):
    a=-(heappop(row))
    b=-(heappop(col))
    temp=a
    a-=m*p
    heappush(row,-a)
    rr+=temp
    pr[h+1]=rr

    temp=b
    b-=n*p
    heappush(col,-b)
    rc+=temp
    pc[h+1]=rc
#print pr
ans=-10**20
for i in range(0,k+1):
    ans=max(ans,pc[i]+pr[k-i]-i*(k-i)*p)
print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import heapq
from heapq import heappush, heappop
import random
from bootcamp import Basebootcamp

def calculate_max_pleasure(n, m, k, p, matrix):
    # 计算初始行和列的总和
    row_sums = [sum(row) for row in matrix]
    col_sums = [sum(matrix[i][j] for i in range(n)) for j in range(m)]

    # 初始化最大堆（使用负数实现最小堆模拟最大堆）
    row_heap = [-s for s in row_sums]
    heapq.heapify(row_heap)
    col_heap = [-s for s in col_sums]
    heapq.heapify(col_heap)

    # 预计算所有可能的行操作收益
    pr = {0: 0}
    current_sum = 0
    for h in range(1, k+1):
        if not row_heap:
            break
        current = -heappop(row_heap)
        current_sum += current
        pr[h] = current_sum
        heappush(row_heap, -(current - m*p))  # 更新行总和

    # 预计算所有可能的列操作收益
    pc = {0: 0}
    current_sum = 0
    for h in range(1, k+1):
        if not col_heap:
            break
        current = -heappop(col_heap)
        current_sum += current
        pc[h] = current_sum
        heappush(col_heap, -(current - n*p))  # 更新列总和

    # 穷举所有可能的行、列操作组合
    max_total = -float('inf')
    for i in pr:
        j = k - i
        if j >= 0 and j in pc:
            total = pr[i] + pc[j] - i*j*p
            max_total = max(max_total, total)
    
    return max_total if max_total != -float('inf') else 0

class Bdzylovesmodificationbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = {
            'n_range': (2, 3),
            'm_range': (2, 3),
            'k_range': (1, 5),
            'p_range': (1, 5),
            'a_min': 1,
            'a_max': 10
        }
        self.params.update(params)
    
    def case_generator(self):
        # 生成随机参数
        n = random.randint(*self.params['n_range'])
        m = random.randint(*self.params['m_range'])
        k = random.randint(*self.params['k_range'])
        p = random.randint(*self.params['p_range'])
        a_min = self.params['a_min']
        a_max = self.params['a_max']
        
        # 生成随机矩阵
        matrix = [[random.randint(a_min, a_max) for _ in range(m)] for _ in range(n)]
        
        # 计算正确答案
        correct_answer = calculate_max_pleasure(n, m, k, p, matrix)
        
        return {
            'n': n,
            'm': m,
            'k': k,
            'p': p,
            'matrix': matrix,
            'correct_answer': correct_answer
        }
    
    @staticmethod
    def prompt_func(question_case):
        input_lines = [f"{question_case['n']} {question_case['m']} {question_case['k']} {question_case['p']}"]
        input_lines.extend(' '.join(map(str, row)) for row in question_case['matrix'])
        input_str = '\n'.join(input_lines)
        
        problem = f"""你正在解决一个矩阵修改问题，目标是通过恰好k次操作获得最大快乐值。仔细阅读规则后，计算以下输入的结果：

规则：
1. 每次操作可选择一行或一列，并将其每个元素减少p
2. 每次操作的快乐值是该行/列操作前的元素总和
3. 必须执行恰好k次操作，最大化总快乐值

输入格式：
第一行包含n, m, k, p
接下来n行每行m个整数表示矩阵

输入数据：
{input_str}

请输出最大总快乐值，并将答案放在[answer]和[/answer]之间，例如：[answer]42[/answer]。答案应为整数。"""
        return problem
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
