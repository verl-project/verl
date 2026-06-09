"""# 

### 谜题描述
A Martian boy is named s — he has got this name quite recently from his parents for his coming of age birthday. Now he enjoys looking for his name everywhere. If he sees that he can obtain his name from some string by removing zero or more letters (at that, the remaining letters remain in the same order), he gets happy. For example, if s=«aba», then strings «baobab», «aabbaa», «helloabahello» make him very happy and strings «aab», «baaa» and «helloabhello» do not.

However rather than being happy once, he loves twice as much being happy twice! So, when he got string t as a present, he wanted to cut it in two parts (the left part and the right part) so that each part made him happy.

Help s determine the number of distinct ways to cut the given string t into two parts in the required manner.

Input

The first line contains string s, consisting of lowercase English letters. The length of string s is from 1 to 1000 letters.

The second line contains string t, that also consists of lowercase English letters. The length of string t is from 1 to 106 letters.

Output

Print the sought number of ways to cut string t in two so that each part made s happy. 

Examples

Input

aba
baobababbah


Output

2


Input

mars
sunvenusearthmarsjupitersaturnuranusneptune


Output

0

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
# tested by Hightail - https://github.com/dj3500/hightail
from __future__ import print_function


read, read_array = raw_input, lambda: raw_input().split()
read_int, read_int_array = lambda: int(read()), lambda: [int(p) for p in read().split()]
read_float, read_float_array = lambda: float(read()), lambda: [float(p) for p in read().split()]
write, write_array = lambda *args, **kw: print(*args, **kw), lambda arr, **kw: print(*arr, **kw)


s = read()
t = read()

i = -1
for c in s:
    i = t.find(c, i + 1)
    if i == -1:
        write(0)
        exit(0)
l = i

i = len(t)
for c in reversed(s):
    i = t.rfind(c, 0, i)
r = i

if l >= r:
    write(0)
else:
    write(r - l)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
import string
from bootcamp import Basebootcamp

class Cnamequestbootcamp(Basebootcamp):
    def __init__(self, s_min_len=1, s_max_len=5, t_min_len=10, t_max_len=100):
        self.s_min_len = s_min_len
        self.s_max_len = s_max_len
        self.t_min_len = max(t_min_len, 2)  # 确保至少可以分割
        self.t_max_len = t_max_len

    def _generate_valid_t(self, s):
        """生成有效t字符串（保证s是t的子序列且存在分割点）"""
        # 生成左右部分各包含s的构造
        left = []
        ptr = 0
        for c in s:
            # 在字符前添加随机前缀
            left.append(''.join(random.choices(string.ascii_lowercase, k=random.randint(0, 3))))
            left.append(c)
            ptr += 1
        left.append(''.join(random.choices(string.ascii_lowercase, k=random.randint(0, 3))))
        
        right = []
        ptr = 0
        for c in s:
            # 在字符后添加随机后缀
            right.append(c)
            right.append(''.join(random.choices(string.ascii_lowercase, k=random.randint(0, 3))))
            ptr += 1
        
        return (''.join(left) + ''.join(right)).replace('\x00', '')  # 防止空字符

    def _generate_invalid_t(self, s):
        """生成无效t字符串（保证至少有一半不满足条件）"""
        # 首先生成有效左半部分
        left = []
        ptr = 0
        for c in s:
            left.append(''.join(random.choices(string.ascii_lowercase, k=random.randint(0, 2))))
            left.append(c)
        left = ''.join(left)
        
        # 生成无效右半部分（不包含s）
        right = ''.join(random.choices(string.ascii_lowercase, 
                      k=random.randint(len(s)+1, len(s)*2)))
        while self._is_subsequence(s, right):
            right = ''.join(random.choices(string.ascii_lowercase, 
                          k=random.randint(len(s)+1, len(s)*2)))
        
        return left + right

    def _is_subsequence(self, s, t):
        """正确实现子序列判断"""
        it = iter(t)
        return all(c in it for c in s)

    def case_generator(self):
        # 随机生成s
        s_len = random.randint(self.s_min_len, self.s_max_len)
        s = ''.join(random.choices(string.ascii_lowercase, k=s_len))
        
        # 控制有效性比例
        if random.random() < 0.5:
            t = self._generate_valid_t(s)
            # 随机插入噪声字符
            insert_pos = random.randint(0, len(t))
            noise = ''.join(random.choices(string.ascii_lowercase, 
                         k=random.randint(1,3)))
            t = t[:insert_pos] + noise + t[insert_pos:]
        else:
            t = self._generate_invalid_t(s)
        
        # 长度调整
        t = self._adjust_length(t)
        return {'s': s, 't': t}

    def _adjust_length(self, t):
        """确保t长度在合理范围内"""
        t = t[:self.t_max_len]
        while len(t) < self.t_min_len:
            t += random.choice(string.ascii_lowercase)
        return t

    @staticmethod
    def prompt_func(question_case) -> str:
        s = question_case['s']
        t = question_case['t']
        return f"""火星男孩需要将字符串t分割为左右两部分，每部分都包含s的子序列。字符串s是"{s}"，字符串t是"{t}"。请计算有效分割方式的数量，并将最终答案放在[answer]标签内。示例：[answer]2[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return matches[-1] if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        s, t = identity['s'], identity['t']
        
        try:
            # 参考算法的完整实现
            i = -1
            for c in s:
                i = t.find(c, i + 1)
                if i == -1:
                    return int(solution) == 0
            
            l = i
            i = len(t)
            for c in reversed(s):
                i = t.rfind(c, 0, i)
                if i == -1:
                    return int(solution) == 0
            
            r = i
            correct = max(0, r - l) if l < r else 0
            return correct == int(solution)
        except:
            return False
