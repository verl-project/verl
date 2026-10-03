"""# 

### 谜题描述
A bracket sequence is a string containing only characters \"(\" and \")\". A regular bracket sequence is a bracket sequence that can be transformed into a correct arithmetic expression by inserting characters \"1\" and \"+\" between the original characters of the sequence. For example, bracket sequences \"()()\" and \"(())\" are regular (the resulting expressions are: \"(1)+(1)\" and \"((1+1)+1)\"), and \")(\", \"(\" and \")\" are not.

Subsequence is a sequence that can be derived from another sequence by deleting some elements without changing the order of the remaining elements.

You are given a regular bracket sequence s and an integer number k. Your task is to find a regular bracket sequence of length exactly k such that it is also a subsequence of s.

It is guaranteed that such sequence always exists.

Input

The first line contains two integers n and k (2 ≤ k ≤ n ≤ 2 ⋅ 10^5, both n and k are even) — the length of s and the length of the sequence you are asked to find.

The second line is a string s — regular bracket sequence of length n.

Output

Print a single string — a regular bracket sequence of length exactly k such that it is also a subsequence of s.

It is guaranteed that such sequence always exists.

Examples

Input

6 4
()(())


Output

()()


Input

8 8
(()(()))


Output

(()(()))

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
[n,k] = map(int,raw_input().split())
bracket = raw_input()
result = []
i = 0
for i in bracket:
    if i == \")\" and n>k:
        result.pop()
        n-=2
    else:
        result.append(i)
print \"\".join(result)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Cbracketsubsequencebootcamp(Basebootcamp):
    def __init__(self, max_insertions=3):
        self.max_insertions = max_insertions
    
    def case_generator(self):
        # 生成目标k长度序列
        k = 2 * random.randint(1, 6)  # 允许更大范围k值
        t = self.generate_regular_sequence(k)
        
        # 插入多层括号结构
        insertions = random.randint(0, self.max_insertions)
        s = t
        for _ in range(insertions):
            # 随机插入合法括号对
            pos = random.randint(0, len(s))
            s = s[:pos] + '()' + s[pos:]
        
        assert self.is_regular(s), f"Invalid regular sequence: {s}"
        return {'n': len(s), 'k': k, 's': s}
    
    @staticmethod
    def generate_regular_sequence(length):
        """生成更复杂的嵌套结构"""
        if length == 0:
            return ""
        if length == 2:
            return "()"
        choice = random.random()
        if choice < 0.5:
            return "(" + Cbracketsubsequencebootcamp.generate_regular_sequence(length-2) + ")"
        else:
            split = random.randint(1, length//2 - 1) * 2
            return (Cbracketsubsequencebootcamp.generate_regular_sequence(split) +
                    Cbracketsubsequencebootcamp.generate_regular_sequence(length-split))
    
    @staticmethod
    def prompt_func(question_case):
        # 保持原prompt结构，增加示例多样性
        return f"""给定长度为{question_case['n']}的正则括号序列：
{question_case['s']}

请找出长度为{question_case['k']}的正则子序列。将答案置于[answer][/answer]标签内。"""

    @staticmethod
    def extract_output(output):
        # 强化格式校验
        matches = re.findall(r'\[answer\]\s*([()]+)\s*\[/answer\]', output)
        return matches[-1] if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 添加更严格的校验
        return (
            len(solution) == identity['k'] and
            solution.count('(') == solution.count(')') and
            cls.is_regular(solution) and
            cls.is_subsequence(solution, identity['s'])
        )
    
    @staticmethod
    def is_regular(s):
        balance = 0
        for c in s:
            balance += 1 if c == '(' else -1
            if balance < 0:
                return False
        return balance == 0
    
    @staticmethod
    def is_subsequence(sub, s):
        it = iter(s)
        return all(c in it for c in sub)
