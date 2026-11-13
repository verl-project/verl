"""# 

### 谜题描述
You've got an array, consisting of n integers a1, a2, ..., an. Also, you've got m queries, the i-th query is described by two integers li, ri. Numbers li, ri define a subsegment of the original array, that is, the sequence of numbers ali, ali + 1, ali + 2, ..., ari. For each query you should check whether the corresponding segment is a ladder. 

A ladder is a sequence of integers b1, b2, ..., bk, such that it first doesn't decrease, then doesn't increase. In other words, there is such integer x (1 ≤ x ≤ k), that the following inequation fulfills: b1 ≤ b2 ≤ ... ≤ bx ≥ bx + 1 ≥ bx + 2... ≥ bk. Note that the non-decreasing and the non-increasing sequences are also considered ladders.

Input

The first line contains two integers n and m (1 ≤ n, m ≤ 105) — the number of array elements and the number of queries. The second line contains the sequence of integers a1, a2, ..., an (1 ≤ ai ≤ 109), where number ai stands for the i-th array element.

The following m lines contain the description of the queries. The i-th line contains the description of the i-th query, consisting of two integers li, ri (1 ≤ li ≤ ri ≤ n) — the boundaries of the subsegment of the initial array.

The numbers in the lines are separated by single spaces.

Output

Print m lines, in the i-th line print word \"Yes\" (without the quotes), if the subsegment that corresponds to the i-th query is the ladder, or word \"No\" (without the quotes) otherwise. 

Examples

Input

8 6
1 2 1 3 3 5 2 1
1 3
2 3
2 4
8 8
1 4
5 8


Output

Yes
Yes
No
Yes
No
Yes

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
R=lambda:map(int,raw_input().split())
n,m=R()
a=R()
q=[i for i in range(n)]
w=[i for i in range(n)]
for i in reversed(range(n-1)):
    if (a[i]<=a[i+1]):q[i]=q[i+1]
    if (a[i]>=a[i+1]):w[i]=w[i+1]
for i in xrange(m):
    l,r=R()
    if (w[q[l-1]]>=r-1):
        print \"Yes\"
    else:print \"No\"
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Cladderbootcamp(Basebootcamp):
    def __init__(self, n=8, m=6, max_val=10**9, seed=None):
        super().__init__()
        self.n = n
        self.m = m
        self.max_val = max_val
        self.seed = seed
        if seed is not None:
            random.seed(seed)
    
    def case_generator(self):
        if self.seed is not None:
            random.seed(self.seed)
        
        # 生成随机数组时允许非单调序列（含波峰波谷）
        a = [random.randint(1, self.max_val) for _ in range(self.n)]
        n = self.n

        # 预处理数组
        q = list(range(n))  # 每个位置能向右扩展到的最远非递减位置
        w = list(range(n))  # 从q[i]开始能向右扩展到的最远非递增位置
        
        for i in reversed(range(n-1)):  # 从倒数第二位向前遍历
            if a[i] <= a[i+1]:
                q[i] = q[i+1]
            if a[i] >= a[i+1]:
                w[i] = w[i+1]

        # 生成多样化查询（包含全段、单元素、随机范围）
        queries = []
        answers = []
        for _ in range(self.m):
            # 生成合法的1-based区间
            li = random.randint(1, n)
            ri = random.randint(li, n)
            queries.append([li, ri])
            
            # 转换为0-based索引
            l_0, r_0 = li-1, ri-1
            peak_pos = q[l_0]
            final_pos = w[peak_pos]
            answers.append("Yes" if final_pos >= r_0 else "No")

        return {
            'n': self.n,
            'm': self.m,
            'a': a,
            'queries': queries,
            'answers': answers
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        a_str = ' '.join(map(str, question_case['a']))
        queries = [f"{q[0]} {q[1]}" for q in question_case['queries']]
        return f"""You need to determine if subarrays form ladder sequences. 

**Rules:**
1. A ladder first non-decreases then non-increases (can be entirely non-decreasing or non-increasing)
2. Array indices are 1-based

**Input:**
- Array: {a_str}
- Queries:
{chr(10).join(queries)}

**Output Format:**
Put each answer (Yes/No) on separate lines within [answer] tags like:
[answer]
Yes
No
...
[/answer]"""

    @staticmethod
    def extract_output(output):
        # 匹配最后一个完整的答案块
        matches = re.findall(
            r'\[answer\](.*?)\[\/answer\]', 
            output, 
            flags=re.DOTALL|re.IGNORECASE
        )
        if not matches:
            return None
        
        answers = []
        for line in matches[-1].strip().splitlines():
            line = line.strip()
            if re.fullmatch(r'yes', line, re.I):
                answers.append("Yes")
            elif re.fullmatch(r'no', line, re.I):
                answers.append("No")
        
        return answers if len(answers) > 0 else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # solution需要与预设答案完全匹配
        return solution == identity.get('answers', [])
