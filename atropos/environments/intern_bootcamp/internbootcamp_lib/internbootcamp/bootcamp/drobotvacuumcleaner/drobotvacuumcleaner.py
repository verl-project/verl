"""# 

### 谜题描述
Pushok the dog has been chasing Imp for a few hours already.

<image>

Fortunately, Imp knows that Pushok is afraid of a robot vacuum cleaner. 

While moving, the robot generates a string t consisting of letters 's' and 'h', that produces a lot of noise. We define noise of string t as the number of occurrences of string \"sh\" as a subsequence in it, in other words, the number of such pairs (i, j), that i < j and <image> and <image>. 

The robot is off at the moment. Imp knows that it has a sequence of strings ti in its memory, and he can arbitrary change their order. When the robot is started, it generates the string t as a concatenation of these strings in the given order. The noise of the resulting string equals the noise of this concatenation.

Help Imp to find the maximum noise he can achieve by changing the order of the strings.

Input

The first line contains a single integer n (1 ≤ n ≤ 105) — the number of strings in robot's memory.

Next n lines contain the strings t1, t2, ..., tn, one per line. It is guaranteed that the strings are non-empty, contain only English letters 's' and 'h' and their total length does not exceed 105.

Output

Print a single integer — the maxumum possible noise Imp can achieve by changing the order of the strings.

Examples

Input

4
ssh
hs
s
hhhs


Output

18


Input

2
h
s


Output

1

Note

The optimal concatenation in the first sample is ssshhshhhs.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
v=(raw_input() for _ in range(input()))
w=0
z=0
for x in ''.join(a[2] for a in sorted([(s.count('s'),len(s),s) for s in v],lambda a,b:1 if a[0]*b[1]<a[1]*b[0] else -1)):
    if x=='h':
        z+=w
    else:
        w+=1
print z
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
from functools import cmp_to_key
from bootcamp import Basebootcamp
import random

class Drobotvacuumcleanerbootcamp(Basebootcamp):
    def __init__(self, max_n=10, max_str_len=5, s_prob=0.5):
        self.max_n = max_n
        self.max_str_len = max_str_len
        self.s_prob = s_prob

    def case_generator(self):
        n = random.randint(1, self.max_n)
        strings = []
        for _ in range(n):
            length = random.randint(1, self.max_str_len)
            s = ''.join(['s' if random.random() < self.s_prob else 'h' for _ in range(length)])
            strings.append(s)

        def compute_internal_sh(s):
            s_count = 0
            sh = 0
            for c in s:
                if c == 's':
                    s_count += 1
                else:
                    sh += s_count
            return sh

        def compare(a, b):
            a_s, a_len = a[0], a[1]
            b_s, b_len = b[0], b[1]
            left = a_s * b_len
            right = b_s * a_len
            if left > right:
                return -1
            elif left < right:
                return 1
            else:
                return 0

        tuples = [(s.count('s'), len(s), s) for s in strings]
        sorted_tuples = sorted(tuples, key=cmp_to_key(compare))

        internal_sh = sum(compute_internal_sh(s) for s in strings)

        cross_sh = 0
        total_s = 0
        for s_count, _, s in sorted_tuples:
            h_count = len(s) - s_count
            cross_sh += total_s * h_count
            total_s += s_count  # 累加原始字符串的s数量

        return {
            'n': n,
            'strings': strings,
            'correct_noise': internal_sh + cross_sh
        }

    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        strings = question_case['strings']
        str_list = '\n'.join(strings)
        return f"""Imp需要找出通过重新排序字符串可获得的最大噪声值。噪声定义为字符串中'sh'子序列的数量。

输入格式：
- 第一行：整数n（字符串数量）
- 后n行：仅包含's'和'h'的字符串

问题实例：
n = {n}
字符串列表：
{str_list}

请输出最大噪声值（整数），使用[answer]答案[/answer]标签包裹。示例：[answer]18[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return int(matches[-1].strip()) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_noise']
