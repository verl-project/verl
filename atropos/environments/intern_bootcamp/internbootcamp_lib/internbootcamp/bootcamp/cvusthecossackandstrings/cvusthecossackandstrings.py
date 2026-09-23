"""# 

### 谜题描述
Vus the Cossack has two binary strings, that is, strings that consist only of \"0\" and \"1\". We call these strings a and b. It is known that |b| ≤ |a|, that is, the length of b is at most the length of a.

The Cossack considers every substring of length |b| in string a. Let's call this substring c. He matches the corresponding characters in b and c, after which he counts the number of positions where the two strings are different. We call this function f(b, c).

For example, let b = 00110, and c = 11000. In these strings, the first, second, third and fourth positions are different.

Vus the Cossack counts the number of such substrings c such that f(b, c) is even.

For example, let a = 01100010 and b = 00110. a has four substrings of the length |b|: 01100, 11000, 10001, 00010. 

  * f(00110, 01100) = 2;
  * f(00110, 11000) = 4;
  * f(00110, 10001) = 4;
  * f(00110, 00010) = 1.



Since in three substrings, f(b, c) is even, the answer is 3.

Vus can not find the answer for big strings. That is why he is asking you to help him.

Input

The first line contains a binary string a (1 ≤ |a| ≤ 10^6) — the first string.

The second line contains a binary string b (1 ≤ |b| ≤ |a|) — the second string.

Output

Print one number — the answer.

Examples

Input


01100010
00110


Output


3


Input


1010111110
0110


Output


4

Note

The first example is explained in the legend.

In the second example, there are five substrings that satisfy us: 1010, 0101, 1111, 1111.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#!/usr/bin/env python

a, b = raw_input(), raw_input()

c1 = 0
c2 = 0
res = 0
for i in range(len(b)):
    if a[i] == '1':
        c1 += 1
    if b[i] == '1':
        c2 += 1

d2 = c2 % 2
res += (c1 % 2) == d2

for i in range(len(b), len(a)):
    if a[i] == '1':
        c1 += 1
    if a[i-len(b)] == '1':
        c1 -= 1
    res += (c1 % 2) == d2

print res
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Cvusthecossackandstringsbootcamp(Basebootcamp):
    def __init__(self, a_min_len=5, a_max_len=20, **kwargs):
        super().__init__(**kwargs)
        self.a_min_len = max(a_min_len, 1)
        self.a_max_len = max(a_max_len, self.a_min_len + 1)
        
    def case_generator(self):
        # 保证 a长度 ≥ b长度
        a_len = random.randint(self.a_min_len, self.a_max_len)
        b_len = random.randint(1, a_len)  # 确保1 ≤ |b| ≤ |a|
        
        # 生成合法二进制字符串对
        while True:
            a = ''.join(random.choices('01', k=a_len))
            b = ''.join(random.choices('01', k=b_len))
            
            # 计算正确答案
            expected = self._calculate_ground_truth(a, b)
            if expected > 0:  # 确保至少存在有效解
                return {
                    'a': a,
                    'b': b,
                    '_ground_truth': expected
                }

    def _calculate_ground_truth(self, a, b):
        """基于参考代码的高效实现"""
        m = len(b)
        n = len(a)
        if m > n:
            return 0
        
        c2 = sum(1 for c in b if c == '1') % 2
        c1 = sum(1 for c in a[:m] if c == '1')
        res = (c1 % 2) == c2
        
        for i in range(m, n):
            c1 += (a[i] == '1') - (a[i - m] == '1')
            res += (c1 % 2) == c2
        return res

    @staticmethod
    def prompt_func(question_case):
        a = question_case['a']
        b = question_case['b']
        return f"""给定两个二进制字符串a和b，其中|b| ≤ |a|。计算a中所有长度为|b|的子串c，使得b与c的对应位差异数为偶数的子串数量。

输入：
a = {a} (长度={len(a)})
b = {b} (长度={len(b)})

请输出确切的整数答案，格式示例：[answer]5[/answer]"""

    @staticmethod
    def extract_output(output):
        try:
            matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
            return int(matches[-1].strip()) if matches else None
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['_ground_truth']

# 验证示例
if __name__ == "__main__":
    # 测试用例1：基础案例
    test_case = {
        'a': '01100010',
        'b': '00110',
        '_ground_truth': 3
    }
    assert Cvusthecossackandstringsbootcamp._verify_correction(3, test_case)
    
    # 测试用例2：边界检查
    edge_case = {
        'a': '1',
        'b': '1',
        '_ground_truth': 1
    }
    assert Cvusthecossackandstringsbootcamp._verify_correction(1, edge_case)
    
    # 测试案例生成
    bootcamp = Cvusthecossackandstringsbootcamp(a_min_len=3, a_max_len=10)
    for _ in range(3):
        case = bootcamp.case_generator()
        assert len(case['b']) <= len(case['a'])
        print(f"生成案例：\na: {case['a']}\nb: {case['b']}\n答案：{case['_ground_truth']}\n")
