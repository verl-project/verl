"""# 

### 谜题描述
Ringo found a string s of length n in his [yellow submarine](https://www.youtube.com/watch?v=m2uTFF_3MaA). The string contains only lowercase letters from the English alphabet. As Ringo and his friends love palindromes, he would like to turn the string s into a palindrome by applying two types of operations to the string. 

The first operation allows him to choose i (2 ≤ i ≤ n-1) and to append the substring s_2s_3 … s_i (i - 1 characters) reversed to the front of s.

The second operation allows him to choose i (2 ≤ i ≤ n-1) and to append the substring s_i s_{i + 1}… s_{n - 1} (n - i characters) reversed to the end of s.

Note that characters in the string in this problem are indexed from 1.

For example suppose s=abcdef. If he performs the first operation with i=3 then he appends cb to the front of s and the result will be cbabcdef. Performing the second operation on the resulted string with i=5 will yield cbabcdefedc.

Your task is to help Ringo make the entire string a palindrome by applying any of the two operations (in total) at most 30 times. The length of the resulting palindrome must not exceed 10^6

It is guaranteed that under these constraints there always is a solution. Also note you do not have to minimize neither the number of operations applied, nor the length of the resulting string, but they have to fit into the constraints.

Input

The only line contains the string S (3 ≤ |s| ≤ 10^5) of lowercase letters from the English alphabet.

Output

The first line should contain k (0≤ k ≤ 30) — the number of operations performed.

Each of the following k lines should describe an operation in form L i or R i. L represents the first operation, R represents the second operation, i represents the index chosen.

The length of the resulting palindrome must not exceed 10^6.

Examples

Input


abac


Output


2
R 2
R 5


Input


acccc


Output


2
L 4
L 2


Input


hannah


Output


0

Note

For the first example the following operations are performed:

abac → abacab → abacaba

The second sample performs the following operations: acccc → cccacccc → ccccacccc

The third example is already a palindrome so no operations are required.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def check(c):
    for i in xrange(len(c)):
        if c[len(c) - i - 1] != c[i]:
            print i
            return False
    return True
def process(st):
    print 5
    st = st + st[1:-1][::-1]
    print st
    a, b, H, b = st[0], st[-1], st[2:-1], st[-1]
    print a, b, H, b
    print b+a+b+H+b
    print b+a+b+H+b+H+b+a
    print b+a+b+H+b+H+b+a+b
    c = b+a+b+H+b+H+b+a+b 
    print check(c)
    
def ans(st):
    print 4
    st = st + st[1:-1][::-1]
    a, b, H, b = st[0], st[-1], st[2:-1], st[-1]
    #print a, b, H, b
    print 'R', 2
    #print b+a+b+H+b
    print 'L', 2
    #print b+a+b+H+b+H+b+a
    print 'R', 2
    #print b+a+b+H+b+H+b+a+b
    print 'R', len(b+a+b+H+b+H+b+a) - 1
    #c = b+a+b+H+b+H+b+a+b 
    #print c
    #print check(c)
def main():
    t = input()
    for _ in xrange(t):
        n = input()
        process(n)

if __name__ == '__main__':
    #main()
    s = raw_input()
    #process(s)
    ans(s)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import string
from bootcamp import Basebootcamp

class Cpalindromifierbootcamp(Basebootcamp):
    def __init__(self, min_length=3, max_length=100, palindrome_prob=0.2):
        """
        参数说明:
            min_length: 生成字符串最小长度 (≥3)
            max_length: 生成字符串最大长度 (≤1e5)
            palindrome_prob: 直接生成回文字符串的概率
        """
        self.min_length = max(3, min_length)
        self.max_length = min(100000, max_length)  # 遵守题目上限
        self.palindrome_prob = palindrome_prob

    def case_generator(self):
        """生成满足题目长度限制的字符串"""
        length = random.randint(self.min_length, self.max_length)
        
        if random.random() < self.palindrome_prob:
            s = self._generate_palindrome(length)
        else:
            s = ''.join(random.choices(string.ascii_lowercase, k=length))
            # 确保生成的字符串不是回文
            while s == s[::-1]:
                s = ''.join(random.choices(string.ascii_lowercase, k=length))
        
        return {'s': s}

    @staticmethod
    def prompt_func(question_case) -> str:
        s = question_case['s']
        return f"""Ringo found a string s consisting of lowercase letters. He wants to turn it into a palindrome using at most 30 operations. The allowed operations are:

1. L i: Prepend the reversed substring from the 2nd to i-th character (2 ≤ i ≤ current length -1)
2. R i: Append the reversed substring from the i-th to second-last character (2 ≤ i ≤ current length -1)

The final string must be a palindrome with length ≤ 1,000,000.

Current string: {s}

Provide your solution in the format:

k
op1 i1
op2 i2
...

Enclose your answer within [answer] and [/answer] tags. For example:

[answer]
2
R 2
R 5
[/answer]"""

    @staticmethod
    def extract_output(output):
        # 支持嵌套标签的情况，查找最后一个匹配的标签对
        start_tag = '[answer]'
        end_tag = '[/answer]'
        
        last_start = output.rfind(start_tag)
        if last_start == -1:
            return None
        
        last_end = output.find(end_tag, last_start)
        if last_end == -1:
            return None
        
        return output[last_start+len(start_tag):last_end].strip()

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            lines = solution.strip().split('\n')
            if not lines:
                return False
            
            # 解析操作数量
            try:
                k = int(lines[0].strip())
            except ValueError:
                return False
            
            # 验证操作数量范围
            if k < 0 or k > 30:
                return False
            
            # 验证操作行数匹配
            if len(lines[1:]) != k:
                return False
            
            operations = []
            current = identity['s']
            total_len = len(current)
            
            for line in lines[1:k+1]:
                parts = line.strip().split()
                if len(parts) != 2:
                    return False
                
                op, i_str = parts
                if op not in ('L', 'R'):
                    return False
                
                # 验证i的有效性
                try:
                    i = int(i_str)
                except ValueError:
                    return False
                
                m = len(current)
                if i < 2 or i > m - 1:
                    return False  # 违反操作规则
                
                # 执行操作（验证阶段不实际执行，只检查有效性）
                if op == 'L':
                    new_part_len = i - 1  # 新增字符数
                else:
                    new_part_len = (m - 1) - (i - 1) + 1  # 原题中i到n-1的子串长度
                
                total_len += new_part_len
                if total_len > 1e6:
                    return False
                
                operations.append((op, i))
            
            # 实际操作验证
            current = identity['s']
            for op, i in operations:
                m = len(current)
                
                if op == 'L':
                    # 取s[1:i] (1-based到i-1) 对应Python的[1-1:i-1] => [0:i-1]
                    part = current[0:i-1]  # s_2到s_i的原始部分
                    prepend = part[::-1]
                    current = prepend + current
                else:
                    # 取s[i-1:m-1] (原题i到n-1的Python表示)
                    part = current[i-1:m-1]
                    append_part = part[::-1]
                    current = current + append_part
            
            return current == current[::-1]
        
        except Exception as e:
            return False

    @staticmethod
    def _generate_palindrome(length):
        """生成随机回文字符串的优化版本"""
        half = (length + 1) // 2
        chars = [random.choice(string.ascii_lowercase) for _ in range(half)]
        return ''.join(chars + chars[:-1 if length % 2 else None][::-1])
