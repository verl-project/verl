"""# 

### 谜题描述
Slime has a sequence of positive integers a_1, a_2, …, a_n.

In one operation Orac can choose an arbitrary subsegment [l … r] of this sequence and replace all values a_l, a_{l + 1}, …, a_r to the value of median of \\{a_l, a_{l + 1}, …, a_r\}.

In this problem, for the integer multiset s, the median of s is equal to the ⌊ (|s|+1)/(2)⌋-th smallest number in it. For example, the median of \{1,4,4,6,5\} is 4, and the median of \{1,7,5,8\} is 5.

Slime wants Orac to make a_1 = a_2 = … = a_n = k using these operations.

Orac thinks that it is impossible, and he does not want to waste his time, so he decided to ask you if it is possible to satisfy the Slime's requirement, he may ask you these questions several times.

Input

The first line of the input is a single integer t: the number of queries.

The first line of each query contains two integers n\ (1≤ n≤ 100 000) and k\ (1≤ k≤ 10^9), the second line contains n positive integers a_1,a_2,...,a_n\ (1≤ a_i≤ 10^9)

The total sum of n is at most 100 000.

Output

The output should contain t lines. The i-th line should be equal to 'yes' if it is possible to make all integers k in some number of operations or 'no', otherwise. You can print each letter in lowercase or uppercase.

Example

Input


5
5 3
1 5 2 6 1
1 6
6
3 2
1 2 3
4 3
3 1 2 3
10 3
1 2 3 4 5 6 7 8 9 10


Output


no
yes
yes
no
yes

Note

In the first query, Orac can't turn all elements into 3.

In the second query, a_1=6 is already satisfied.

In the third query, Orac can select the complete array and turn all elements into 2.

In the fourth query, Orac can't turn all elements into 3.

In the fifth query, Orac can select [1,6] at first and then select [2,10].

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin
input = stdin.readline

output = []

T = int(input())

for test in range(T):
    n,k = map(int, input().split())

    A = list(map(int, input().split()))
    flag = False

    if n == 1:
        output.append(\"yes\" if A[0] == k else \"no\")
        continue

    if k not in A:
        output.append(\"no\")
        continue

    for t in range(2):
        if t == 1:
            A = A[::-1]

        prefix = [1 if A[0] >= k else -1]
        for a in A[1:]:
            val = 1 if a >= k else -1
            prefix.append(prefix[-1] + val)

        maxRight = [0]*n
        maxRight[-1] = -10**10

        for i in range(n-2, -1, -1):
            maxRight[i] = max(maxRight[i+1], prefix[i+1])

        for i in range(n):
            if A[i] >= k and maxRight[i] >= prefix[i]:
                flag = True

    output.append(\"yes\" if flag else \"no\")

print(\"\n\".join(output))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Boracandmediansbootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_k=10**9, max_value=10**9):
        self.max_n = max_n
        self.max_k = max_k
        self.max_value = max_value  # 允许生成与max_k相等的数值
    
    def case_generator(self):
        n = random.randint(1, self.max_n)
        include_k = random.choice([True, False])
        
        k = random.randint(1, self.max_k)
        array = []
        
        # 强制包含k的逻辑
        if include_k:
            array = [random.randint(1, self.max_value) for _ in range(n)]
            mandatory_pos = random.randint(0, n-1)
            array[mandatory_pos] = k
        else:
            while len(array) < n:
                val = random.randint(1, self.max_value)
                if val != k:
                    array.append(val)
            # 当无法构造不包含k的情况时自动转换类型
            if k in array:
                include_k = True

        expected = self.compute_expected(n, k, array)
        return {
            'n': n,
            'k': k,
            'array': array,
            'expected': expected
        }
    
    def compute_expected(self, n, k, array):
        """完整实现参考代码逻辑"""
        if n == 1:
            return 'yes' if array[0] == k else 'no'
        if k not in array:
            return 'no'
        flag = False
        A = array.copy()
        for t in range(2):
            if t == 1:
                A = A[::-1]
            prefix = [1 if A[0] >= k else -1]
            for a in A[1:]:
                val = 1 if a >= k else -1
                prefix.append(prefix[-1] + val)
            maxRight = [0] * len(A)
            if len(A) > 0:
                maxRight[-1] = -10**10
                for i in range(len(A)-2, -1, -1):
                    maxRight[i] = max(maxRight[i+1], prefix[i+1])
                for i in range(len(A)):
                    if A[i] >= k and maxRight[i] >= prefix[i]:
                        flag = True
        return 'yes' if flag else 'no'
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        k = question_case['k']
        array = question_case['array']
        return f"""Slime has a sequence of {n} positive integers: {array}
Orac can perform operations to replace any subsegment with its median. The median is defined as the ⌊(size+1)/2⌋-th smallest element. 
Determine if it's possible to make all elements equal to {k} through any number of operations. 

Output format:
[answer]{{yes|no}}[/answer]"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](yes|no)\[/answer\]', output, re.IGNORECASE)
        return matches[-1].lower() if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return str(solution).lower() == identity['expected'].lower()
