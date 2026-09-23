"""# 

### 谜题描述
Now Serval is a junior high school student in Japari Middle School, and he is still thrilled on math as before. 

As a talented boy in mathematics, he likes to play with numbers. This time, he wants to play with numbers on a rooted tree.

A tree is a connected graph without cycles. A rooted tree has a special vertex called the root. A parent of a node v is the last different from v vertex on the path from the root to the vertex v. Children of vertex v are all nodes for which v is the parent. A vertex is a leaf if it has no children.

The rooted tree Serval owns has n nodes, node 1 is the root. Serval will write some numbers into all nodes of the tree. However, there are some restrictions. Each of the nodes except leaves has an operation max or min written in it, indicating that the number in this node should be equal to the maximum or minimum of all the numbers in its sons, respectively. 

Assume that there are k leaves in the tree. Serval wants to put integers 1, 2, …, k to the k leaves (each number should be used exactly once). He loves large numbers, so he wants to maximize the number in the root. As his best friend, can you help him?

Input

The first line contains an integer n (2 ≤ n ≤ 3⋅ 10^5), the size of the tree.

The second line contains n integers, the i-th of them represents the operation in the node i. 0 represents min and 1 represents max. If the node is a leaf, there is still a number of 0 or 1, but you can ignore it.

The third line contains n-1 integers f_2, f_3, …, f_n (1 ≤ f_i ≤ i-1), where f_i represents the parent of the node i.

Output

Output one integer — the maximum possible number in the root of the tree.

Examples

Input


6
1 0 1 1 0 1
1 2 2 2 2


Output


1


Input


5
1 0 1 0 1
1 1 1 1


Output


4


Input


8
1 0 0 1 0 1 1 0
1 1 2 2 3 3 3


Output


4


Input


9
1 1 0 0 1 0 1 0 1
1 1 2 2 3 3 4 4


Output


5

Note

Pictures below explain the examples. The numbers written in the middle of the nodes are their indices, and the numbers written on the top are the numbers written in the nodes.

In the first example, no matter how you arrange the numbers, the answer is 1.

<image>

In the second example, no matter how you arrange the numbers, the answer is 4.

<image>

In the third example, one of the best solution to achieve 4 is to arrange 4 and 5 to nodes 4 and 5.

<image>

In the fourth example, the best solution is to arrange 5 to node 5.

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
py2 = round(0.5)

if py2:
    from future_builtins import ascii, filter, hex, map, oct, zip
    range = xrange
    from cStringIO import StringIO as BytesIO
else:
    from io import BytesIO
    from builtins import str as __str__
    str = lambda x=b'': x if type(x) is bytes else __str__(x).encode()

import os, sys
from io import IOBase

# FastIO for PyPy2 and PyPy3 (works with interactive) by Pajenegod
class FastI(object):
    def __init__(self, fd=0, buffersize=2**14):
        self.stream = stream = BytesIO(); self.bufendl = 0
        def read2buffer():
            s = os.read(fd, buffersize + os.fstat(fd).st_size); pos = stream.tell()
            stream.seek(0,2); stream.write(s); stream.seek(pos); return s
        self.read2buffer = read2buffer
    
    # Read entire input
    def read(self):
        while self.read2buffer(): pass
        return self.stream.read() if self.stream.tell() else self.stream.getvalue()
    
    def readline(self):
        while self.bufendl == 0: s = self.read2buffer(); self.bufendl += s.count(b\"\n\") + (not s)
        self.bufendl -= 1; return self.stream.readline()
    
    def input(self): return self.readline().rstrip(b'\r\n')
    
    # Read all remaining integers, type is given by optional argument, this is fast
    def readnumbers(self, zero=0):
        conv = ord if py2 else lambda x:x
        A = []; numb = zero; sign = 1; i = 0; s = self.read()
        try:
            while True:
                if s[i] >= b'0' [0]:
                    numb = 10 * numb + conv(s[i]) - 48
                elif s[i] == b'-' [0]: sign = -1
                elif s[i] != b'\r' [0]:
                    A.append(sign*numb)
                    numb = 0; sign = 1
                i += 1
        except:pass
        if s and s[-1] >= b'0' [0]:
            A.append(sign*numb)
        return A

class FastO(IOBase):
    def __init__(self, fd=1):
        stream = BytesIO()
        self.flush = lambda: os.write(1, stream.getvalue()) and not stream.truncate(0) and stream.seek(0)
        self.write = stream.write if py2 else lambda s: stream.write(s.encode())

sys.stdin, sys.stdout = FastI(), FastO()
input = sys.stdin.input

import sys
class ostream:
    def __lshift__(self,a):
        sys.stdout.write(str(a))
        return self
    def tie(self, val):pass
    def flush(self):sys.stdout.flush()
cout = ostream()
endl = b'\n'

class istream:
    inp = None
    def __rlshift__(a,b):
        if type(b)==tuple or type(b)==list:
            return type(b)(type(c)(a.get()) for c in b)
        return type(b)(a.get())
    def get(a):
        while not a.inp:
            a.inp = sys.stdin.readline().split(); a.inp.reverse()
        return a.inp.pop()
cin = istream()

n = 0
n <<= cin

A = [0]*n
A <<= cin

P = [0]*(n-1)
P <<= cin

P = [0] + [p-1 for p in P]

counter = [0]*n
for i in range(1,n):
    counter[P[i]] += 1

leaves = [i for i in range(n) if not counter[i]]
k = len(leaves)

score = [0]*n
teller = [[] for _ in range(n)]

Q = leaves[:]
for node in Q:
    if A[node]==0:
        # min
        if teller[node]:
            score[node] = sum(teller[node])
        else:
            score[node] = 1
    else:
        if teller[node]:
            score[node] = min(teller[node])
        else:
            score[node] = 1
    
    counter[P[node]] -= 1
    teller[P[node]].append(score[node])
    if counter[P[node]] == 0:
        Q.append(P[node])

cout << k+1-score[0] << endl;
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from collections import deque
from bootcamp import Basebootcamp

class Dservalandrootedtreebootcamp(Basebootcamp):
    def __init__(self, min_nodes=2, max_nodes=20):
        self.params = {'min_nodes': min_nodes, 'max_nodes': max_nodes}
    
    def case_generator(self):
        n = random.randint(self.params['min_nodes'], self.params['max_nodes'])
        
        # 生成合法树结构 (0-based)
        parent = [0]*n
        for i in range(1, n):
            parent[i] = random.randint(0, i-1)
        
        # 生成operations
        children = [[] for _ in range(n)]
        for i in range(1, n):
            children[parent[i]].append(i)
        operations = [
            random.choice([0,1]) if children[i] else random.choice([0,1])
            for i in range(n)
        ]
        
        # 转换输入格式 (1-based父节点)
        input_parents = [str(parent[i]+1) for i in range(1, n)]
        
        # 计算正确答案
        correct = self.calculate_answer(n, operations, parent)
        
        return {
            'n': n,
            'operations': operations,
            'parents': input_parents,
            'correct_answer': correct
        }

    def calculate_answer(self, n, ops, parent):
        # 构建children (0-based)
        children = [[] for _ in range(n)]
        for i in range(1, n):
            children[parent[i]].append(i)
        
        # 确定叶子节点
        leaves = [i for i in range(n) if not children[i]]
        k = len(leaves)
        
        # 反向传播计算
        dp = [1]*n
        reverse_children = [[] for _ in range(n)]
        for i in range(n):
            for child in children[i]:
                reverse_children[child].append(i)
        
        indegree = [len(children[i]) for i in range(n)]
        q = deque(leaves)
        processing_order = []
        
        while q:
            u = q.popleft()
            processing_order.append(u)
            for p in reverse_children[u]:
                indegree[p] -= 1
                if indegree[p] == 0:
                    q.append(p)
        
        for u in reversed(processing_order):
            if not children[u]:
                continue
                
            if ops[u] == 1:  # max
                dp[u] = min(dp[v] for v in children[u])
            else:  # min
                dp[u] = sum(dp[v] for v in children[u])
        
        return k + 1 - dp[0]
    
    @staticmethod
    def prompt_func(question_case) -> str:
        return (
            f"Solve the tree optimization puzzle with:\n"
            f"n={question_case['n']}\n"
            f"Operations: {question_case['operations']}\n"
            f"Parent list: {' '.join(question_case['parents'])}\n"
            "Put your answer between [answer] and [/answer]."
        )
    
    @staticmethod 
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None
    
    @staticmethod
    def _verify_correction(solution, identity):
        return solution == identity['correct_answer']
