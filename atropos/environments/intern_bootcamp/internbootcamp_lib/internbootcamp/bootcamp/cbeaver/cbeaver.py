"""# 

### 谜题描述
After Fox Ciel got off a bus, she found that the bus she was on was a wrong bus and she lost her way in a strange town. However, she fortunately met her friend Beaver Taro and asked which way to go to her castle. Taro's response to her was a string s, and she tried to remember the string s correctly.

However, Ciel feels n strings b1, b2, ... , bn are really boring, and unfortunately she dislikes to remember a string that contains a boring substring. To make the thing worse, what she can remember is only the contiguous substring of s.

Determine the longest contiguous substring of s that does not contain any boring string, so that she can remember the longest part of Taro's response.

Input

In the first line there is a string s. The length of s will be between 1 and 105, inclusive.

In the second line there is a single integer n (1 ≤ n ≤ 10). Next n lines, there is a string bi (1 ≤ i ≤ n). Each length of bi will be between 1 and 10, inclusive.

Each character of the given strings will be either a English alphabet (both lowercase and uppercase) or a underscore ('_') or a digit. Assume that these strings are case-sensitive.

Output

Output in the first line two space-separated integers len and pos: the length of the longest contiguous substring of s that does not contain any bi, and the first position of the substring (0-indexed). The position pos must be between 0 and |s| - len inclusive, where |s| is the length of string s.

If there are several solutions, output any.

Examples

Input

Go_straight_along_this_street
5
str
long
tree
biginteger
ellipse


Output

12 4


Input

IhaveNoIdea
9
I
h
a
v
e
N
o
I
d


Output

0 0


Input

unagioisii
2
ioi
unagi


Output

5 5

Note

In the first sample, the solution is traight_alon.

In the second sample, the solution is an empty string, so the output can be «0 0», «0 1», «0 2», and so on.

In the third sample, the solution is either nagio or oisii.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#   Grader: codeforces
#   79C_beaver
#   Varot Premtoon 8 Nov 2554

s = raw_input()
n = int(raw_input())
a = []
for i in range(0,n): a.append(raw_input())
mx = ind = cl = 0
for i in range(len(s)-1, -1, -1):
    cl += 1
    for w in a:
        if len(w) <= cl and s[i : i + len(w)] == w:
            cl = len(w)-1
    if cl > mx:
        mx, ind = cl, i        

print mx, ind
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import string
import re
from bootcamp import Basebootcamp

allowed_chars = string.ascii_letters + string.digits + '_'

def generate_random_string(length, chars=allowed_chars):
    return ''.join(random.choice(chars) for _ in range(length))

def compute_solution(s, a):
    mx = 0
    ind = 0
    cl = 0  # Current length
    for i in reversed(range(len(s))):  # 优化循环写法
        cl += 1
        for w in a:
            w_len = len(w)
            if w_len > cl:
                continue
            if s[i:i + w_len] == w:
                cl = min(cl, w_len - 1)
        if cl > mx or (cl == mx and i < ind):
            mx = cl
            ind = i
    return mx, ind

class Cbeaverbootcamp(Basebootcamp):
    def __init__(self, s_min_length=20, s_max_length=100, target_min_len=5, 
                 target_max_len=20, boring_count=5, boring_substr_min_len=1, 
                 boring_substr_max_len=5, special_case_prob=0.2):
        self.s_min_length = s_min_length
        self.s_max_length = s_max_length
        self.target_min_len = target_min_len
        self.target_max_len = target_max_len
        self.boring_count = min(boring_count, 10)  # 遵守题目约束n≤10
        self.boring_substr_min_len = max(boring_substr_min_len, 1)
        self.boring_substr_max_len = min(boring_substr_max_len, 10)  # 遵守题目约束长度≤10
        self.special_case_prob = special_case_prob

    def _generate_valid_substring(self, min_len, max_len, forbidden):
        """生成不包含任何forbidden子串的有效字符串"""
        for _ in range(100):
            candidate = generate_random_string(random.randint(min_len, max_len))
            if all(b not in candidate for b in forbidden):
                return candidate
        return generate_random_string(max_len)  # 回退机制

    def case_generator(self):
        # 处理全无效的特殊案例
        if random.random() < self.special_case_prob:
            s_len = random.randint(5, 15)
            s = generate_random_string(s_len)
            unique_substrings = {c for c in s}  # 单字符子串
            unique_substrings.update([s[i:i+2] for i in range(len(s)-1)])  # 双字符子串
            
            n = min(len(unique_substrings), self.boring_count)
            selected = random.sample(sorted(unique_substrings), n)
            return {
                's': s,
                'n': n,
                'boring': selected
            }

        # 正常案例生成
        target_len = random.randint(self.target_min_len, self.target_max_len)
        # 生成核心有效区域
        while True:
            target = generate_random_string(target_len)
            # 确保target自身不包含后续生成的boring子串
            boring_list = [
                self._generate_valid_substring(
                    self.boring_substr_min_len,
                    self.boring_substr_max_len,
                    [target]
                ) for _ in range(random.randint(1, self.boring_count))
            ]
            if all(b not in target for b in boring_list):
                break

        # 构建上下文环境
        def build_context(max_parts=2):
            context = []
            for _ in range(random.randint(0, max_parts)):
                valid_part = self._generate_valid_substring(3, 8, boring_list)
                context.append(valid_part)
                context.append(random.choice(boring_list))
            return ''.join(context)

        prefix = build_context(max_parts=2)
        suffix = build_context(max_parts=1)

        return {
            's': prefix + target + suffix,
            'n': len(boring_list),
            'boring': boring_list
        }

    @staticmethod
    def prompt_func(question_case):
        input_lines = [question_case['s'], str(question_case['n'])] + question_case['boring']
        input_example = '\n'.join(input_lines)
        return f"""字符串处理问题：寻找最长有效子串

给定主字符串s和若干禁止子串，找出s中最长的连续子串，满足：
1. 不包含任何禁止子串
2. 若有多个相同长度的解，输出起始位置最小的
3. 允许空字符串解（输出0 0）

输入格式：
{sorted(input_lines)}

输出要求：
两个整数：长度 起始位置（0-based）

示例输入：
Go_straight_along_this_street
5
str
long
tree
biginteger
ellipse

示例输出：
12 4

请将答案用[answer]标签包裹，如：[answer]12 4[/answer]"""

    @staticmethod
    def extract_output(output):
        pattern = r'\[answer\]\s*(\d+)\s+(\d+)\s*\[/answer\]'
        matches = re.findall(pattern, output, re.IGNORECASE)
        if not matches:
            return None
        last_match = matches[-1]
        try:
            return (int(last_match[0]), int(last_match[1]))
        except (ValueError, IndexError):
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 类型检查
        if not isinstance(solution, tuple) or len(solution) != 2:
            return False
        len_val, pos_val = solution
        
        s = identity['s']
        boring = identity['boring']
        s_len = len(s)
        
        # 有效性验证
        if len_val == 0:
            return 0 <= pos_val <= s_len  # 允许任意合法位置
        
        # 边界检查
        if pos_val < 0 or len_val < 0 or pos_val + len_val > s_len:
            return False
        
        # 子串内容检查
        substr = s[pos_val:pos_val+len_val]
        if any(b in substr for b in boring):
            return False
        
        # 最优性验证
        max_len, earliest_pos = compute_solution(s, boring)
        return len_val == max_len and pos_val >= earliest_pos
