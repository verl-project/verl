"""# 

### 谜题描述
Sereja has a bracket sequence s1, s2, ..., sn, or, in other words, a string s of length n, consisting of characters \"(\" and \")\".

Sereja needs to answer m queries, each of them is described by two integers li, ri (1 ≤ li ≤ ri ≤ n). The answer to the i-th query is the length of the maximum correct bracket subsequence of sequence sli, sli + 1, ..., sri. Help Sereja answer all queries.

You can find the definitions for a subsequence and a correct bracket sequence in the notes.

Input

The first line contains a sequence of characters s1, s2, ..., sn (1 ≤ n ≤ 106) without any spaces. Each character is either a \"(\" or a \")\". The second line contains integer m (1 ≤ m ≤ 105) — the number of queries. Each of the next m lines contains a pair of integers. The i-th line contains integers li, ri (1 ≤ li ≤ ri ≤ n) — the description of the i-th query.

Output

Print the answer to each question on a single line. Print the answers in the order they go in the input.

Examples

Input

())(())(())(
7
1 1
2 3
1 2
1 12
8 12
5 11
2 10


Output

0
0
2
10
4
6
6

Note

A subsequence of length |x| of string s = s1s2... s|s| (where |s| is the length of string s) is string x = sk1sk2... sk|x| (1 ≤ k1 < k2 < ... < k|x| ≤ |s|).

A correct bracket sequence is a bracket sequence that can be transformed into a correct aryphmetic expression by inserting characters \"1\" and \"+\" between the characters of the string. For example, bracket sequences \"()()\", \"(())\" are correct (the resulting expressions \"(1)+(1)\", \"((1+1)+1)\"), and \")(\" and \"(\" are not.

For the third query required sequence will be «()».

For the fourth query required sequence will be «()(())(())».

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  sem título.py
#  
#  Copyright 2020 Alencar <Alencar@ALENCAR-PC>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import os
from sys import stdin
 
import sys
 
 
def solution():
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'wb')
    a = [0]
    for x in stdin.next().rstrip('\r\n'):
        a.append(a[-1] + (1 if x == '(' else -1))
    stdin.next()
    tree = [a]
    while len(a) >= 2:
        a = [min(a[i], a[i + 1]) for i in xrange(0, len(a) - 1, 2)]
        tree.append(a)
    a = tree[0]
    for line in stdin:
        left, right = map(int, line.split())
        left -= 1
 
        ans = 1000 * 1000 * 1000
        l = left
        r = right + 1
        for t in tree:
            if l & 1:
                ans = min(ans, t[l])
                l += 1
            if r & 1:
                r -= 1
                ans = min(ans, t[r])
            l /= 2
            r /= 2
            if l == r:
                break
        ans = right - left - (a[left] - ans) - (a[right] - ans)
        sys.stdout.write('%d\n' % ans)
 
 
solution()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re
import math

class Cserejaandbracketsbootcamp(Basebootcamp):
    def __init__(self, n=20, m=5):
        """
        参数:
            n: 括号序列长度 (默认20，最小值5)
            m: 查询数量 (默认5，最小值1)
        """
        self.n = max(n, 5)  # 保证最小长度
        self.m = max(m, 1)  # 至少1个查询
    
    def case_generator(self):
        # 生成更合理的括号序列（包含平衡和不平衡区域）
        s = []
        stack = []
        positions = list(range(self.n))
        random.shuffle(positions)
        pairs = min(self.n//2, 10)  # 生成至少部分合法对
        
        # 生成基础合法对
        for _ in range(pairs):
            if len(positions) >= 2:
                o = positions.pop()
                c = positions.pop()
                s.extend(['']*(max(o,c)+1 - len(s)))
                s[o] = '('
                s[c] = ')'
        
        # 填充剩余位置
        for i in range(self.n):
            if not s[i]:
                s[i] = random.choice(['(', ')'])
        
        # 平衡调整
        s = ''.join(s)
        balance = 0
        final_s = []
        for c in s:
            if c == '(':
                balance += 1
                final_s.append(c)
            else:
                if balance > 0:
                    balance -= 1
                    final_s.append(c)
                else:
                    final_s.append('(')  # 强制平衡
                    balance += 1
        s = ''.join(final_s)
        
        # 生成多样化的查询区间（包含有效区间和随机区间）
        valid_regions = self.find_valid_regions(s)
        queries = []
        
        # 生成有效区域查询
        for _ in range(min(self.m//2, len(valid_regions))):
            l, r = random.choice(valid_regions)
            li = random.randint(l+1, r+1)  # 1-based
            ri = random.randint(li, r+1)
            queries.append((li, ri))
        
        # 补充边界测试用例
        queries.extend([
            (1, 1),  # 单字符测试
            (1, len(s)),  # 全范围测试
            (max(1, len(s)-3), len(s))  # 尾部测试
        ][:min(3, self.m-len(queries))])
        
        # 补充随机区间
        while len(queries) < self.m:
            li = random.randint(1, self.n)
            ri = random.randint(li, self.n)
            queries.append((li, ri))
        
        random.shuffle(queries)
        return {
            's': s,
            'queries': queries,
            'answers': self.compute_answers(s, queries)
        }
    
    def find_valid_regions(self, s):
        # 寻找有效括号子序列区域
        stack = []
        valid = []
        max_len = 0
        start = 0
        for i, c in enumerate(s):
            if c == '(':
                stack.append(i)
            else:
                if stack:
                    stack.pop()
                    if not stack:
                        valid.append((start, i))
                    else:
                        valid.append((stack[-1]+1, i))
                else:
                    start = i + 1
        return valid if valid else [(0, len(s)-1)]
    
    @staticmethod
    def compute_answers(s, queries):
        n = len(s)
        a = [0]*(n+1)
        for i in range(1, n+1):
            a[i] = a[i-1] + (1 if s[i-1] == '(' else -1)
        
        # 构建Sparse Table
        log_table = [0]*(n+2)
        for i in range(2, n+2):
            log_table[i] = log_table[i//2] + 1
        
        k_max = log_table[n] + 1 if n > 0 else 0
        st = [[0]*(n+1) for _ in range(k_max)]
        st[0] = a.copy()
        
        for k in range(1, k_max):
            for i in range(n+1 - (1 << k) + 1):
                st[k][i] = min(st[k-1][i], st[k-1][i + (1 << (k-1))])
        
        answers = []
        for li, ri in queries:
            l = li - 1
            r = ri
            length = r - l + 1
            k = log_table[length]
            mid = r - (1 << k) + 1
            
            min_val = min(st[k][l], st[k][mid])
            ans = (ri - li + 1) - (a[l] - min_val) - (a[r] - min_val)
            answers.append(max(ans // 1, 0))  # 确保结果为整数
        
        return answers
    
    @staticmethod
    def prompt_func(question_case) -> str:
        s = question_case['s']
        queries = question_case['queries']
        return f"""给定由括号组成的字符串s和m个查询，每个查询指定区间[l, r]，要求计算该区间内最长合法括号子序列的长度。合法括号序列定义为可以正确闭合的括号组合。

输入格式：
s（长度n）
m
l1 r1
...
lm rm

输出格式：
m行，每行对应查询结果

特别注意：
1. 区间是闭区间[li, ri]
2. 输出结果必须为非负整数
3. 答案必须严格按输入顺序输出

当前问题：
s = {s}
m = {len(queries)}
查询区间：
""" + '\n'.join(f"{l} {r}" for l, r in queries) + """

请将最终答案按顺序放置在[answer]标签内，例如：
[answer]
0
4
6
[/answer]"""

    @staticmethod
    def extract_output(output):
        # 增强多格式匹配能力
        pattern = r'(?:<answer>|\[answer\]|答案：?)([\s\d\n]+)(?:<\/answer>|\[\/answer\]|)'
        matches = re.findall(pattern, output, re.IGNORECASE)
        if not matches:
            return None
        
        numbers = []
        for match in matches:
            nums = re.findall(r'\b\d+\b', match)
            numbers.extend(map(int, nums))
        
        # 取最后一个完整答案块
        if numbers and len(numbers) >= len(matches[-1].split()):
            return numbers[-len(matches[-1].split()):]
        return numbers if numbers else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity.get('answers', [])
        return isinstance(solution, list) and solution == expected
