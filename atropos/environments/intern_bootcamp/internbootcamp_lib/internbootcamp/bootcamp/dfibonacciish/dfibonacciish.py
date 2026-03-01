"""# 

### 谜题描述
Yash has recently learnt about the Fibonacci sequence and is very excited about it. He calls a sequence Fibonacci-ish if 

  1. the sequence consists of at least two elements 
  2. f0 and f1 are arbitrary 
  3. fn + 2 = fn + 1 + fn for all n ≥ 0. 



You are given some sequence of integers a1, a2, ..., an. Your task is rearrange elements of this sequence in such a way that its longest possible prefix is Fibonacci-ish sequence.

Input

The first line of the input contains a single integer n (2 ≤ n ≤ 1000) — the length of the sequence ai.

The second line contains n integers a1, a2, ..., an (|ai| ≤ 109).

Output

Print the length of the longest possible Fibonacci-ish prefix of the given sequence after rearrangement.

Examples

Input

3
1 2 -1


Output

3


Input

5
28 35 7 14 21


Output

4

Note

In the first sample, if we rearrange elements of the sequence as  - 1, 2, 1, the whole sequence ai would be Fibonacci-ish.

In the second sample, the optimal way to rearrange elements is <image>, <image>, <image>, <image>, 28.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n=int(raw_input())
arr=list(map(int,raw_input().split()))
dic={}
for i in arr:
    dic[i]=dic.get(i,0)+1
ans=1
for i in range(n):
    for j in range(n):
        if i!=j:
            if arr[i]==arr[j]==0:
                ans=max(ans,dic[0])
            else:
                a,b=arr[i],arr[j]
                dic[a]-=1
                ln=1
                while(dic.get(b,0)):
                    dic[b]-=1
                    a,b=b,a+b
                    ln+=1
                ans=max(ans,ln)
                while(ln>0):
                    dic[a]+=1
                    a,b=b-a,a
                    ln-=1
print(ans)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Dfibonacciishbootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=1000):
        """
        初始化参数校验和范围限制
        """
        self.min_n = max(2, min_n)
        self.max_n = min(1000, max(max_n, self.min_n))
        if self.min_n > self.max_n:
            raise ValueError("Invalid range: min_n cannot be larger than max_n")

    def case_generator(self):
        """
        改进后的案例生成逻辑，确保生成有效案例
        """
        # 全零案例处理
        if random.random() < 0.2:  # 20%概率生成全零案例
            n = random.randint(self.min_n, self.max_n)
            return {
                'n': n,
                'arr': [0]*n,
                'expected': n
            }
        
        # 常规案例生成（确保所有元素可用于构建序列）
        for _ in range(100):  # 最大尝试次数
            # 生成非零初始值
            a, b = 0, 0
            while a == 0 and b == 0:
                a = random.randint(-10, 10)
                b = random.randint(-10, 10)
            
            # 构建合法序列
            seq = [a, b]
            while len(seq) < self.max_n:
                next_val = seq[-1] + seq[-2]
                if abs(next_val) > 1e9:
                    break
                seq.append(next_val)
            
            # 确保序列长度符合要求
            if self.min_n <= len(seq) <= self.max_n:
                shuffled = seq.copy()
                random.shuffle(shuffled)
                return {
                    'n': len(seq),
                    'arr': shuffled,
                    'expected': len(seq)
                }
        
        # 生成失败时返回最小有效案例
        return {
            'n': 2,
            'arr': [1, 1],
            'expected': 2
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        """增强格式规范说明"""
        return f"""You are given a sequence of {question_case['n']} integers: {' '.join(map(str, question_case['arr']))}
        
Your task is to rearrange them to form the longest possible Fibonacci-ish sequence. The sequence must:
1. Contain at least 2 elements
2. Follow the rule: each element after the first two is the sum of the preceding two

Output ONLY the maximum length as:
[answer]length[/answer]

Example1:
Input: 3 1 2 -1 → [answer]3[/answer]

Your turn:"""

    @staticmethod
    def extract_output(output):
        """严格正则匹配"""
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """精确验证逻辑"""
        return solution == identity['expected']
