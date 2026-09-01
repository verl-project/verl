"""# 

### 谜题描述
Serval soon said goodbye to Japari kindergarten, and began his life in Japari Primary School.

In his favorite math class, the teacher taught him the following interesting definitions.

A parenthesis sequence is a string, containing only characters \"(\" and \")\".

A correct parenthesis sequence is a parenthesis sequence that can be transformed into a correct arithmetic expression by inserting characters \"1\" and \"+\" between the original characters of the sequence. For example, parenthesis sequences \"()()\", \"(())\" are correct (the resulting expressions are: \"(1+1)+(1+1)\", \"((1+1)+1)\"), while \")(\" and \")\" are not. Note that the empty string is a correct parenthesis sequence by definition.

We define that |s| as the length of string s. A strict prefix s[1... l] (1≤ l< |s|) of a string s = s_1s_2... s_{|s|} is string s_1s_2... s_l. Note that the empty string and the whole string are not strict prefixes of any string by the definition.

Having learned these definitions, he comes up with a new problem. He writes down a string s containing only characters \"(\", \")\" and \"?\". And what he is going to do, is to replace each of the \"?\" in s independently by one of \"(\" and \")\" to make all strict prefixes of the new sequence not a correct parenthesis sequence, while the new sequence should be a correct parenthesis sequence.

After all, he is just a primary school student so this problem is too hard for him to solve. As his best friend, can you help him to replace the question marks? If there are many solutions, any of them is acceptable.

Input

The first line contains a single integer |s| (1≤ |s|≤ 3 ⋅ 10^5), the length of the string.

The second line contains a string s, containing only \"(\", \")\" and \"?\".

Output

A single line contains a string representing the answer.

If there are many solutions, any of them is acceptable.

If there is no answer, print a single line containing \":(\" (without the quotes).

Examples

Input


6
(?????


Output


(()())

Input


10
(???(???(?


Output


:(

Note

It can be proved that there is no solution for the second sample, so print \":(\".

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys,math
from fractions import gcd
from bisect import bisect_left
from collections import defaultdict
from io import BytesIO
sys.stdin = BytesIO(sys.stdin.read())
input = lambda: sys.stdin.readline().rstrip('\r\n')
n = int(input())
s = input()
#n,m,h = [int(x) for x in input().split()]

no = s.count('(')
nc = s.count(')')
nq = s.count('?')
if n % 2:
    print(':(')
    exit()
h = n // 2
if no > h or nc > h or s[0] == ')':
    print(':(')
    exit()
cur = 0
cc = 0
res = list(s)
for i in range(n):
    if s[i] == '(':
        cur += 1
    elif s[i] == ')':
        cur -= 1
        if cur < 1 and i < n-1:
            print(':(')
            exit()
    elif s[i] == '?':
        if no + cc < h:
            cur += 1
            res[i] = '('
            cc += 1
        else:
            cur -= 1
            res[i] = ')'
            if cur < 1 and i < n-1:
                print(':(')
                exit()
print(''.join(res))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cservalandparenthesissequencebootcamp(Basebootcamp):
    def __init__(self, min_length=2, max_length=20, replace_prob=0.3):
        min_length = max(2, min_length)
        max_length = max(min_length, max_length)
        self.min_length = (min_length + 1) // 2 * 2  # 强制偶数
        self.max_length = (max_length // 2) * 2
        self.replace_prob = replace_prob

    def case_generator(self):
        # 随机生成长度（保证偶数）
        n = random.randint(self.min_length // 2, self.max_length // 2) * 2
        # 生成随机包含 (、)、? 的字符串
        s = []
        # 首字符允许生成 ) 来制造无法解决的情况
        first_char = random.choice(['(', ')', '?'])
        s.append(first_char)
        # 剩余字符随机生成
        for _ in range(n-1):
            s.append(random.choice(['(', ')', '?']))
        # 随机替换部分非?字符为?
        for i in range(n):
            if s[i] != '?' and random.random() < self.replace_prob:
                s[i] = '?'
        return {'n': n, 's': ''.join(s)}

    @staticmethod
    def solve_parenthesis(s):
        n = len(s)
        if n % 2 != 0:
            return ':('
        h = n // 2
        no = s.count('(')
        nc = s.count(')')
        nq = n - no - nc
        if no > h or nc > h or s[0] == ')':
            return ':('
        res = list(s)
        open_needed = h - no
        close_needed = h - nc
        if open_needed < 0 or close_needed < 0:
            return ':('
        # 遍历填充?
        cur_balance = 0
        for i in range(n):
            if res[i] == '(':
                cur_balance += 1
            elif res[i] == ')':
                cur_balance -= 1
                if cur_balance < 1 and i < n-1:
                    return ':('
            elif res[i] == '?':
                # 优先填 ( 的条件
                if open_needed > 0:
                    res[i] = '('
                    cur_balance += 1
                    open_needed -= 1
                else:
                    res[i] = ')'
                    cur_balance -= 1
                    close_needed -= 1
                # 检查中间非法情况
                if cur_balance < 0 or (cur_balance < 1 and i < n-1):
                    return ':('
        # 最终平衡检查
        return ''.join(res) if cur_balance == 0 else ':('

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        s = question_case['s']
        prompt = (
            "Replace '?' in the string to form a valid parenthesis sequence where all strict prefixes are invalid.\n"
            f"Input length: {n}\n"
            f"Input string: {s}\n\n"
            "Output the solution string or ':('. Enclose your answer within [answer]...[/answer]."
        )
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if matches:
            last_match = matches[-1].strip()
            return last_match if last_match else None
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        s = identity['s']
        n = identity['n']
        # 处理无解情况
        if solution == ':(':
            return cls.solve_parenthesis(s) == ':('
        # 格式检查
        if len(solution) != n:
            return False
        # 原始字符串固定位置检查
        for i in range(n):
            if s[i] != '?' and solution[i] != s[i]:
                return False
        # 整体有效性检查
        if not cls.is_valid_parenthesis(solution):
            return False
        # 严格前缀检查
        for l in range(1, n):
            if cls.is_valid_parenthesis(solution[:l]):
                return False
        return True

    @staticmethod
    def is_valid_parenthesis(s):
        balance = 0
        for c in s:
            balance += 1 if c == '(' else -1
            if balance < 0:
                return False
        return balance == 0
