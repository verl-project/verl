"""# 

### 谜题描述
There are n lamps on a line, numbered from 1 to n. Each one has an initial state off (0) or on (1).

You're given k subsets A_1, …, A_k of \{1, 2, ..., n\}, such that the intersection of any three subsets is empty. In other words, for all 1 ≤ i_1 < i_2 < i_3 ≤ k, A_{i_1} ∩ A_{i_2} ∩ A_{i_3} = ∅.

In one operation, you can choose one of these k subsets and switch the state of all lamps in it. It is guaranteed that, with the given subsets, it's possible to make all lamps be simultaneously on using this type of operation.

Let m_i be the minimum number of operations you have to do in order to make the i first lamps be simultaneously on. Note that there is no condition upon the state of other lamps (between i+1 and n), they can be either off or on.

You have to compute m_i for all 1 ≤ i ≤ n.

Input

The first line contains two integers n and k (1 ≤ n, k ≤ 3 ⋅ 10^5).

The second line contains a binary string of length n, representing the initial state of each lamp (the lamp i is off if s_i = 0, on if s_i = 1).

The description of each one of the k subsets follows, in the following format:

The first line of the description contains a single integer c (1 ≤ c ≤ n) — the number of elements in the subset.

The second line of the description contains c distinct integers x_1, …, x_c (1 ≤ x_i ≤ n) — the elements of the subset.

It is guaranteed that: 

  * The intersection of any three subsets is empty; 
  * It's possible to make all lamps be simultaneously on using some operations. 

Output

You must output n lines. The i-th line should contain a single integer m_i — the minimum number of operations required to make the lamps 1 to i be simultaneously on.

Examples

Input


7 3
0011100
3
1 4 6
3
3 4 7
2
2 3


Output


1
2
3
3
3
3
3


Input


8 6
00110011
3
1 3 8
5
1 2 5 6 7
2
6 8
2
3 5
2
4 7
1
2


Output


1
1
1
1
1
1
4
4


Input


5 3
00011
3
1 2 3
1
4
3
3 4 5


Output


1
1
1
1
1


Input


19 5
1001001001100000110
2
2 3
2
5 6
2
8 9
5
12 13 14 15 16
1
19


Output


0
1
1
1
2
2
2
3
3
3
3
4
4
4
4
4
4
4
5

Note

In the first example: 

  * For i = 1, we can just apply one operation on A_1, the final states will be 1010110; 
  * For i = 2, we can apply operations on A_1 and A_3, the final states will be 1100110; 
  * For i ≥ 3, we can apply operations on A_1, A_2 and A_3, the final states will be 1111111. 



In the second example: 

  * For i ≤ 6, we can just apply one operation on A_2, the final states will be 11111101; 
  * For i ≥ 7, we can apply operations on A_1, A_3, A_4, A_6, the final states will be 11111111. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
range = xrange
input = raw_input
 
n, k = [int(x) for x in input().split()]
A = [+(c == '0') for c in input()]
inp = [int(x) for x in sys.stdin.read().split()]; ii = 0

B = []
for _ in range(k):
    m = inp[ii]; ii += 1
    B.append([a - 1 for a in inp[ii: ii + m]]); ii += m

buckets = [[] for _ in range(n)]
for j in range(k):
    for a in B[j]:
        buckets[a].append(j)

#####################
parent = list(range(2 * k))
size = [1] * (2 * k)
state = [0] * (2 * k)
true = [1] * k + [0] * k
cost = 0


def find(a):
    acopy = a
    while a != parent[a]:
        a = parent[a]
    while acopy != a:
        parent[acopy], acopy = a, parent[acopy]
    return a

def calcer(a):
    a = find(a)
    b = find((a + k) % (2 * k))
    if state[a] == 0:
        return min(true[a], true[b])
    elif state[a] == 1:
        return true[b]
    else:
        return true[a]

def union(a, b):
    a, b = find(a), find(b)
    if a != b:
        if size[a] < size[b]:
            a, b = b, a

        parent[b] = a
        size[a] += size[b]
        state[a] |= state[b]
        true[a] += true[b]

def on(a):
    global cost
    a = find(a)
    if state[a] == 0:
        b = find((a + k) % (2 * k))
        cost -= calcer(a)
        state[a] = 1
        state[b] = 2 
        cost += calcer(a)

def same(a, b):
    global cost
    a = find(a)
    b = find(b)
    if a != b:
        cost -= calcer(a)
        cost -= calcer(b)
        union(a, b)
        union((a + k) % (2 * k), (b + k) % (2 * k))
        cost += calcer(a)

######

out = []
for i in range(n):
    bucket = buckets[i]
    if not bucket:
        pass
    elif len(bucket) == 1:
        j = bucket[0]
        if not A[i]:
            on(j)
        else:
            on(j + k)
    else:
        j1, j2 = bucket
        if not A[i]:
            same(j1, j2)
        else:
            same(j1, j2 + k)
    out.append(cost)
print '\n'.join(str(x) for x in out)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from io import StringIO
import sys
from bootcamp import Basebootcamp

class Cprefixenlightenmentbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.n_min = params.get('n_min', 5)
        self.n_max = params.get('n_max', 10)
        self.k_min = params.get('k_min', 1)
        self.k_max = params.get('k_max', 5)
        self.subset_size_min = params.get('subset_size_min', 1)
        self.subset_size_max = params.get('subset_size_max', 3)
        self.max_retries = params.get('max_retries', 1000)
    
    def case_generator(self):
        for _ in range(self.max_retries):
            n = random.randint(self.n_min, self.n_max)
            k = random.randint(self.k_min, self.k_max)
            count = [0] * n
            subsets = []
            valid = True
            for _ in range(k):
                c = random.randint(self.subset_size_min, min(self.subset_size_max, n))
                available = [x for x in range(n) if count[x] < 2]
                if len(available) < c:
                    valid = False
                    break
                selected = random.sample(available, c)
                subset = [x + 1 for x in selected]
                subsets.append(subset)
                for x in selected:
                    count[x] += 1
            if not valid:
                continue
            x = [random.choice([0, 1]) for _ in range(k)]
            initial_state = []
            for i in range(n):
                sum_x = sum(x[j] for j in range(k) if (i + 1) in subsets[j])
                final_state = sum_x % 2
                initial_state.append(str(final_state ^ 1))
            initial_state_str = ''.join(initial_state)
            case = {
                'n': n,
                'k': k,
                'initial_state': initial_state_str,
                'subsets': subsets
            }
            return case
        raise ValueError("Failed to generate valid case after retries")
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k = question_case['k']
        initial_state = question_case['initial_state']
        subsets = question_case['subsets']
        subset_str = '\n'.join([f"{len(subset)}\n{' '.join(map(str, subset))}" for subset in subsets])
        prompt = f"""You are solving a lamp switching puzzle. The task is to find the minimum operations required to turn on the first i lamps for all i from 1 to {n}.

Problem Input:
{n} {k}
{initial_state}
{subset_str}

Rules:
- Each switch toggles all lamps in its subset.
- Any three subsets have empty intersection.
- You need to output {n} lines where the i-th line is the minimum operations to turn on the first i lamps.

Output your answer within [answer] and [/answer], each m_i on a separate line. Example:
[answer]
0
1
...
[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        solution = []
        for line in matches[-1].strip().split('\n'):
            line = line.strip()
            if line and line.isdigit():
                solution.append(int(line))
            elif line:  # 包含非数字内容视为无效
                return None
        return solution  # 长度验证在_verify_correction中进行
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution or len(solution) != identity['n']:
            return False
        try:
            computed = cls.compute_solution(
                identity['n'], identity['k'],
                identity['initial_state'], identity['subsets']
            )
            return solution == computed
        except Exception as e:
            print(f"Verification error: {e}")
            return False
    
    @staticmethod
    def compute_solution(n, k, initial_state, subsets):
        input_lines = [f"{n} {k}", initial_state]
        for subset in subsets:
            input_lines.append(str(len(subset)))
            input_lines.append(' '.join(map(str, subset)))
        input_str = '\n'.join(input_lines)
        
        # 重新实现参考代码逻辑（Python3适配）
        from io import StringIO
        input_buf = StringIO(input_str)
        
        n_ref, k_ref = map(int, input_buf.readline().split())
        s_ref = input_buf.readline().strip()
        A_ref = [1 if c == '0' else 0 for c in s_ref]
        
        B_ref = []
        for _ in range(k_ref):
            c = int(input_buf.readline())
            elements = list(map(int, input_buf.readline().split()))
            B_ref.append([x - 1 for x in elements])  # 转换为0-based
        
        buckets = [[] for _ in range(n_ref)]
        for j in range(k_ref):
            for a in B_ref[j]:
                buckets[a].append(j)
        
        parent = list(range(2 * k_ref))
        size = [1] * (2 * k_ref)
        state = [0] * (2 * k_ref)
        true = [1] * k_ref + [0] * k_ref
        cost = 0

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]  # 路径压缩
                a = parent[a]
            return a

        def calcer(a):
            a_root = find(a)
            b_root = find((a_root + k_ref) % (2 * k_ref))
            if state[a_root] == 0:
                return min(true[a_root], true[b_root])
            elif state[a_root] == 1:
                return true[b_root]
            else:
                return true[a_root]

        def union(a, b):
            nonlocal cost
            a_root = find(a)
            b_root = find(b)
            if a_root != b_root:
                # 按秩合并
                if size[a_root] < size[b_root]:
                    a_root, b_root = b_root, a_root
                parent[b_root] = a_root
                size[a_root] += size[b_root]
                state[a_root] |= state[b_root]
                true[a_root] += true[b_root]

        def on(a):
            nonlocal cost
            a_root = find(a)
            if state[a_root] == 0:
                b_root = find((a_root + k_ref) % (2 * k_ref))
                cost -= calcer(a_root)
                state[a_root] = 1
                state[b_root] = 2
                cost += calcer(a_root)

        out = []
        for i in range(n_ref):
            bucket = buckets[i]
            if not bucket:
                pass
            elif len(bucket) == 1:
                j = bucket[0]
                if A_ref[i]:
                    on(j)
                else:
                    on(j + k_ref)
            else:
                j1, j2 = bucket
                if A_ref[i]:
                    union(j1, j2)
                else:
                    union(j1, j2 + k_ref)
            out.append(cost)
        
        return out
