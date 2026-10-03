"""# 

### 谜题描述
There are n people in this world, conveniently numbered 1 through n. They are using burles to buy goods and services. Occasionally, a person might not have enough currency to buy what he wants or needs, so he borrows money from someone else, with the idea that he will repay the loan later with interest. Let d(a,b) denote the debt of a towards b, or 0 if there is no such debt.

Sometimes, this becomes very complex, as the person lending money can run into financial troubles before his debtor is able to repay his debt, and finds himself in the need of borrowing money. 

When this process runs for a long enough time, it might happen that there are so many debts that they can be consolidated. There are two ways this can be done:

  1. Let d(a,b) > 0 and d(c,d) > 0 such that a ≠ c or b ≠ d. We can decrease the d(a,b) and d(c,d) by z and increase d(c,b) and d(a,d) by z, where 0 < z ≤ min(d(a,b),d(c,d)). 
  2. Let d(a,a) > 0. We can set d(a,a) to 0. 



The total debt is defined as the sum of all debts:

$$$\Sigma_d = ∑_{a,b} d(a,b)$$$

Your goal is to use the above rules in any order any number of times, to make the total debt as small as possible. Note that you don't have to minimise the number of non-zero debts, only the total debt.

Input

The first line contains two space separated integers n (1 ≤ n ≤ 10^5) and m (0 ≤ m ≤ 3⋅ 10^5), representing the number of people and the number of debts, respectively.

m lines follow, each of which contains three space separated integers u_i, v_i (1 ≤ u_i, v_i ≤ n, u_i ≠ v_i), d_i (1 ≤ d_i ≤ 10^9), meaning that the person u_i borrowed d_i burles from person v_i.

Output

On the first line print an integer m' (0 ≤ m' ≤ 3⋅ 10^5), representing the number of debts after the consolidation. It can be shown that an answer always exists with this additional constraint.

After that print m' lines, i-th of which contains three space separated integers u_i, v_i, d_i, meaning that the person u_i owes the person v_i exactly d_i burles. The output must satisfy 1 ≤ u_i, v_i ≤ n, u_i ≠ v_i and 0 < d_i ≤ 10^{18}.

For each pair i ≠ j, it should hold that u_i ≠ u_j or v_i ≠ v_j. In other words, each pair of people can be included at most once in the output.

Examples

Input

3 2
1 2 10
2 3 5


Output

2
1 2 5
1 3 5


Input

3 3
1 2 10
2 3 15
3 1 10


Output

1
2 3 5


Input

4 2
1 2 12
3 4 8


Output

2
1 2 12
3 4 8


Input

3 4
2 3 1
2 3 2
2 3 4
2 3 8


Output

1
2 3 15

Note

In the first example the optimal sequence of operations can be the following:

  1. Perform an operation of the first type with a = 1, b = 2, c = 2, d = 3 and z = 5. The resulting debts are: d(1, 2) = 5, d(2, 2) = 5, d(1, 3) = 5, all other debts are 0; 
  2. Perform an operation of the second type with a = 2. The resulting debts are: d(1, 2) = 5, d(1, 3) = 5, all other debts are 0. 



In the second example the optimal sequence of operations can be the following:

  1. Perform an operation of the first type with a = 1, b = 2, c = 3, d = 1 and z = 10. The resulting debts are: d(3, 2) = 10, d(2, 3) = 15, d(1, 1) = 10, all other debts are 0; 
  2. Perform an operation of the first type with a = 2, b = 3, c = 3, d = 2 and z = 10. The resulting debts are: d(2, 2) = 10, d(3, 3) = 10, d(2, 3) = 5, d(1, 1) = 10, all other debts are 0; 
  3. Perform an operation of the second type with a = 2. The resulting debts are: d(3, 3) = 10, d(2, 3) = 5, d(1, 1) = 10, all other debts are 0; 
  4. Perform an operation of the second type with a = 3. The resulting debts are: d(2, 3) = 5, d(1, 1) = 10, all other debts are 0; 
  5. Perform an operation of the second type with a = 1. The resulting debts are: d(2, 3) = 5, all other debts are 0. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from __future__ import division, print_function
def main():
    n, m = map(int,input().split())
    debts = [0]*(n+1)
    for i in range(m):
        u, v, d = map(int,input().split())
        debts[u]+=d
        debts[v]-=d
    negatives = []
    positives = []
    for i in range(1,n+1):
        if debts[i] > 0:
            positives.append([i,debts[i]])
        elif debts[i] < 0:
            negatives.append([i,debts[i]])
    result = []
    j = 0
    for item in positives:
        while item[1]>0:
            if abs(negatives[j][1])>=item[1]:
                result.append((item[0],negatives[j][0],item[1]))
                negatives[j][1]+=item[1]
                item[1]=0
                
                if negatives[j][1]==0:
                    j+=1
            else :
                result.append((item[0],negatives[j][0],abs(negatives[j][1])))
                item[1]+=negatives[j][1]
                negatives[j][1]=0
                
                j+=1
    print(len(result))
    for item in result:
        print(item[0],item[1],item[2])
    

######## Python 2 and 3 footer by Pajenegod and c1729

# Note because cf runs old PyPy3 version which doesn't have the sped up
# unicode strings, PyPy3 strings will many times be slower than pypy2.
# There is a way to get around this by using binary strings in PyPy3
# but its syntax is different which makes it kind of a mess to use.

# So on cf, use PyPy2 for best string performance.

py2 = round(0.5)
if py2:
    from future_builtins import ascii, filter, hex, map, oct, zip
    range = xrange

import os, sys
from io import IOBase, BytesIO

BUFSIZE = 8192
class FastIO(BytesIO):
    newlines = 0

    def __init__(self, file):
        self._file = file
        self._fd = file.fileno()
        self.writable = \"x\" in file.mode or \"w\" in file.mode
        self.write = super(FastIO, self).write if self.writable else None

    def _fill(self):
        s = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
        self.seek((self.tell(), self.seek(0,2), super(FastIO, self).write(s))[0])
        return s

    def read(self):
        while self._fill(): pass
        return super(FastIO,self).read()

    def readline(self):
        while self.newlines == 0:
            s = self._fill(); self.newlines = s.count(b\"\n\") + (not s)
        self.newlines -= 1
        return super(FastIO, self).readline()

    def flush(self):
        if self.writable:
            os.write(self._fd, self.getvalue())
            self.truncate(0), self.seek(0)

class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        if py2:
            self.write = self.buffer.write
            self.read = self.buffer.read
            self.readline = self.buffer.readline
        else:
            self.write = lambda s:self.buffer.write(s.encode('ascii'))
            self.read = lambda:self.buffer.read().decode('ascii')
            self.readline = lambda:self.buffer.readline().decode('ascii')


sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
input = lambda: sys.stdin.readline().rstrip('\r\n')

# Cout implemented in Python
import sys
class ostream:
    def __lshift__(self,a):
        sys.stdout.write(str(a))
        return self
cout = ostream()
endl = '\n'

# Read all remaining integers in stdin, type is given by optional argument, this is fast
def readnumbers(zero = 0):
    conv = ord if py2 else lambda x:x
    A = []; numb = zero; sign = 1; i = 0; s = sys.stdin.buffer.read()
    try:
        while True:
            if s[i] >= b'0' [0]:
                numb = 10 * numb + conv(s[i]) - 48
            elif s[i] == b'-' [0]: sign = -1
            elif s[i] != b'\r' [0]:
                A.append(sign*numb)
                numb = zero; sign = 1
            i += 1
    except:pass
    if s and s[-1] >= b'0' [0]:
        A.append(sign*numb)
    return A

if __name__== \"__main__\":
  main()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import json
import random
import re
from bootcamp import Basebootcamp

class Ddecreasingdebtsbootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=5, min_m=0, max_m=10, d_min=1, d_max=100):
        self.min_n = min_n
        self.max_n = max_n
        self.min_m = min_m
        self.max_m = max_m
        self.d_min = d_min
        self.d_max = d_max
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        max_possible_m = min(self.max_m, 3 * n * (n-1))  # 允许每个(u,v)出现多次
        m = random.randint(self.min_m, max_possible_m)
        
        debts = []
        for _ in range(m):
            u = random.randint(1, n)
            v = random.randint(1, n)
            while u == v:
                v = random.randint(1, n)
            d = random.randint(self.d_min, self.d_max)
            debts.append([u, v, d])
        
        # 计算原始净债务数组
        original_net = [0] * (n + 1)
        for u, v, d in debts:
            original_net[u] += d
            original_net[v] -= d
        
        # 生成正确的solution
        positives = []
        negatives = []
        for i in range(1, n +1):
            if original_net[i] > 0:
                positives.append( (i, original_net[i]) )
            elif original_net[i] < 0:
                negatives.append( (i, original_net[i]) )
        
        result = []
        ptr_neg = 0
        for u_pos, remaining_pos in positives:
            while remaining_pos > 0 and ptr_neg < len(negatives):
                v_neg, remaining_neg = negatives[ptr_neg]
                transfer = min(remaining_pos, -remaining_neg)
                result.append( (u_pos, v_neg, transfer) )
                remaining_pos -= transfer
                # 更新负数债务的剩余量
                new_neg = remaining_neg + transfer
                if new_neg == 0:
                    ptr_neg += 1
                else:
                    negatives[ptr_neg] = (v_neg, new_neg)
        
        # 直接转换为最终结果（参考代码保证无重复）
        final_result = [[u, v, d] for u, v, d in result]
        
        identity = {
            'input': {
                'n': n,
                'm': m,
                'debts': debts
            },
            'original_net': original_net,
            'expected_total': sum(val for val in original_net if val > 0)
        }
        return identity
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['input']['n']
        m = question_case['input']['m']
        debts = question_case['input']['debts']
        debts_desc = "\n".join([f"{u} {v} {d}" for u, v, d in debts])
        prompt = f"""你是债务合并专家，需要根据以下规则将债务合并以使总债务最小化：

规则说明：
1. 债务转移：若存在两笔债务d(a,b) >0 和 d(c,d) >0（其中a≠c或b≠d），可以选择减少这两笔债务z（z为两者中的较小值），同时增加d(c,b)和d(a,d)各z。
2. 消除自环债务：若某人有自环债务d(a,a) >0，可将其置零。

输入数据：
- 第1行：两个整数n（人数）和m（初始债务数量）
- 接下来m行：每行三个整数u, v, d，表示u欠v共d burles

任务：
应用上述规则，输出处理后的债务，使得总债务最小。输出格式为：
- 第1行：剩余债务数量m'
- 随后m'行：每行三个整数u, v, d，表示u欠v的最终债务d

当前问题：
n = {n}, m = {m}
债务列表：
{debts_desc}

请按照上述格式要求，将最终答案严格按以下格式放置在[answer]和[/answer]之间：
[answer]
m'
u1 v1 d1
u2 v2 d2
...
[/answer]

确保：
1. 每个(u,v)对唯一且u≠v，d>0
2. 所有数值为普通整数（无科学计数法）"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        content = matches[-1].strip()
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return None
        try:
            m_prime = int(lines[0])
        except:
            return None
        if len(lines) != m_prime + 1:
            return None
        debts = []
        seen = set()
        for line in lines[1:]:
            parts = line.split()
            if len(parts) !=3:
                return None
            try:
                u = int(parts[0])
                v = int(parts[1])
                d = int(parts[2])
                # 检查是否科学计数法（如1e5）
                if 'e' in line.lower():
                    return None
            except:
                return None
            if u == v or d <=0 or (u, v) in seen:
                return None
            seen.add((u, v))
            debts.append((u, v, d))
        return debts
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            expected_total = identity['expected_total']
            return expected_total == 0
        input_info = identity['input']
        n = input_info['n']
        original_net = identity['original_net']
        expected_total = identity['expected_total']
        
        # 验证solution的格式
        seen = set()
        current_total =0
        solution_net = [0]*(n+1)
        for u, v, d in solution:
            if u == v or d <=0 or (u, v) in seen:
                return False
            seen.add((u, v))
            solution_net[u] += d
            solution_net[v] -= d
            current_total +=d
        
        # 验证净债务是否一致
        for i in range(1, n+1):
            if solution_net[i] != original_net[i]:
                return False
        
        # 验证总债务是否正确
        return current_total == expected_total
