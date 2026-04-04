"""# 

### 谜题描述
During the break, we decided to relax and play dominoes. Our box with Domino was empty, so we decided to borrow the teacher's dominoes.

The teacher responded instantly at our request. He put nm dominoes on the table as an n × 2m rectangle so that each of the n rows contained m dominoes arranged horizontally. Each half of each domino contained number (0 or 1).

We were taken aback, and the teacher smiled and said: \"Consider some arrangement of dominoes in an n × 2m matrix. Let's count for each column of the matrix the sum of numbers in this column. Then among all such sums find the maximum one. Can you rearrange the dominoes in the matrix in such a way that the maximum sum will be minimum possible? Note that it is prohibited to change the orientation of the dominoes, they all need to stay horizontal, nevertheless dominoes are allowed to rotate by 180 degrees. As a reward I will give you all my dominoes\".

We got even more taken aback. And while we are wondering what was going on, help us make an optimal matrix of dominoes.

Input

The first line contains integers n, m (1 ≤ n, m ≤ 103).

In the next lines there is a description of the teachers' matrix. Each of next n lines contains m dominoes. The description of one domino is two integers (0 or 1), written without a space — the digits on the left and right half of the domino.

Output

Print the resulting matrix of dominoes in the format: n lines, each of them contains m space-separated dominoes.

If there are multiple optimal solutions, print any of them.

Examples

Input

2 3
01 11 00
00 01 11


Output

11 11 10
00 00 01


Input

4 1
11
10
01
00


Output

11
10
01
00

Note

Consider the answer for the first sample. There, the maximum sum among all columns equals 1 (the number of columns is 6, and not 3). Obviously, this maximum can't be less than 1, then such matrix is optimal.

Note that the dominoes can be rotated by 180 degrees.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin, stdout
def main():
    m, n = map(int, stdin.readline().split())
    k = [0] * 3
    for b in stdin.read().split():
        if b == '11':
            k[0] += 1
        elif b == '00':
            k[2] += 1
        else:
            k[1] += 1
    rest = [k[0] % n, k[1] % (2 * n), k[2] % n]
    if rest[0] + rest[1] + rest[2]:
        if rest[0] + rest[1] + rest[2] <= n:
            stdout.write(' '.join(['11'] * rest[0] + ['10'] * rest[1] + ['00'] * rest[2]))
            stdout.write('\n')
        else:
            l0 = ['10'] * (rest[1] - rest[1] / 2)
            l1 = ['01'] * (rest[1] / 2)
            for l in (l0, l1):
                while rest[0] and len(l) < n:
                    l.append('11')
                    rest[0] -= 1
            for l in (l0, l1):
                while rest[2] and len(l) < n:
                    l.append('00')
                    rest[2] -= 1
            l2 = ['11'] * rest[0] + ['00'] * rest[2]
            stdout.write('\n'.join(' '.join(l) for l in (l0, l1, l2) if l))
            stdout.write('\n')
    b = [('11', k[0] / n), ('10', k[1] / (2 * n)), ('01', k[1] / (2 * n)), ('00', k[2] / n)]
    stdout.write('\n'.join('\n'.join(' '.join(s for i in xrange(n)) for j in xrange(rep)) for s, rep in b if rep))
main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import math
from collections import defaultdict

class Cdominoesbootcamp(Basebootcamp):
    def __init__(self, n=2, m=3):
        self.n = n
        self.m = m

    def case_generator(self):
        # 生成合法多米诺配置
        domino_set = self.generate_valid_set()
        optimal_matrix = self.build_optimal_matrix(domino_set)
        input_matrix = self.scramble_matrix(optimal_matrix)
        
        # 计算理论最优值
        total_S = sum(int(a)+int(b) for row in input_matrix for a,b in row)
        optimal_max = math.ceil(total_S / (2 * self.m))
        
        return {
            'input_matrix': input_matrix,
            'optimal_matrix': optimal_matrix,
            'n': self.n,
            'm': self.m,
            'optimal_max': optimal_max,
        }

    def generate_valid_set(self):
        """生成包含可优化空间的有效集合"""
        total = self.n * self.m
        while True:
            types = ['00', '01', '10', '11']
            probs = [0.25, 0.25, 0.25, 0.25]
            dominoes = random.choices(types, weights=probs, k=total)
            if sum(1 for d in dominoes if d in ['01','10']) > 0:
                return dominoes

    def build_optimal_matrix(self, domino_set):
        """按照官方解题算法构建最优矩阵"""
        # 统计类型
        k = defaultdict(int)
        for d in domino_set:
            if d in ['00','11']:
                k[d] += 1
            else:
                k['mix'] += 1

        # 初始化二维矩阵
        matrix = [[] for _ in range(self.n)]
        
        # 类型划分（参考官方解法）
        a = k['11'] // self.n
        b = (k['mix'] // 2) // self.n
        c = k['00'] // self.n
        
        # 基础分配
        for row in matrix:
            row += ['11']*a
            row += ['01']*b
            row += ['10']*b
            row += ['00']*c
        
        # 余数处理
        rem_11 = k['11'] % self.n
        rem_mix = k['mix'] % (2*self.n)
        rem_00 = k['00'] % self.n
        
        # Phase 1: 分配余数11
        for i in range(rem_11):
            matrix[i].append('11')
            
        # Phase 2: 分配余数mix
        for i in range(rem_mix):
            matrix[i%self.n].append('01' if i%2 else '10')
            
        # Phase 3: 分配余数00
        for i in range(rem_00):
            matrix[i].append('00')

        # 填充并校验每行长度
        for row in matrix:
            random.shuffle(row)
            while len(row) < self.m:
                # 异常处理：补充虚拟domino（理论上不应触发）
                row.append('00')
            del row[self.m:]  # 精确截断

        return matrix

    def scramble_matrix(self, matrix):
        """生成随机输入矩阵"""
        scrambled = []
        for row in matrix:
            new_row = []
            for d in row:
                if d in ['01','10']:
                    new_row.append(random.choice([d, d[::-1]]))
                else:
                    new_row.append(d)
            random.shuffle(new_row)
            scrambled.append(new_row)
        return scrambled

    @staticmethod
    def prompt_func(question_case):
        matrix = question_case['input_matrix']
        return f"""Rearrange the {question_case['n']}x{2*question_case['m']} domino matrix to minimize maximum column sum. Original matrix:
""" + '\n'.join(' '.join(row) for row in matrix) + """

Rules:
1. Keep dominoes horizontal but can rotate
2. Reorder dominoes within each row
3. Format answer with {question_case['n']} lines of {question_case['m']} dominoes

Put your answer between [answer] and [/answer] tags."""

    @staticmethod
    def extract_output(output):
        import re
        answers = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answers: return None
        matrix = []
        for line in answers[-1].strip().split('\n'):
            if line.strip():
                matrix.append(line.strip().split())
        return matrix if len(matrix) == 0 or len(matrix[0]) == matrix[0].__len__() else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 尺寸验证
        if len(solution) != identity['n']: return False
        if any(len(row)!=identity['m'] for row in solution): return False
        
        # 多米诺来源验证
        orig = defaultdict(int)
        for row in identity['input_matrix']:
            for d in row:
                key = d if d in ['00','11'] else '01'
                orig[key] += 1
                
        sol = defaultdict(int)
        for row in solution:
            for d in row:
                key = d if d in ['00','11'] else '01'
                sol[key] += 1
        if orig != sol: return False
        
        # 列和验证
        columns = [0]*(2*identity['m'])
        for row in solution:
            for i, domino in enumerate(row):
                columns[2*i] += int(domino[0])
                columns[2*i+1] += int(domino[1])
        return max(columns) == identity['optimal_max']
