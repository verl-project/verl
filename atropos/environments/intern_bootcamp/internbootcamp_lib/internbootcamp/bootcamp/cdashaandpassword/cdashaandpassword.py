"""# 

### 谜题描述
After overcoming the stairs Dasha came to classes. She needed to write a password to begin her classes. The password is a string of length n which satisfies the following requirements:

  * There is at least one digit in the string, 
  * There is at least one lowercase (small) letter of the Latin alphabet in the string, 
  * There is at least one of three listed symbols in the string: '#', '*', '&'. 

<image>

Considering that these are programming classes it is not easy to write the password.

For each character of the password we have a fixed string of length m, on each of these n strings there is a pointer on some character. The i-th character displayed on the screen is the pointed character in the i-th string. Initially, all pointers are on characters with indexes 1 in the corresponding strings (all positions are numbered starting from one).

During one operation Dasha can move a pointer in one string one character to the left or to the right. Strings are cyclic, it means that when we move the pointer which is on the character with index 1 to the left, it moves to the character with the index m, and when we move it to the right from the position m it moves to the position 1.

You need to determine the minimum number of operations necessary to make the string displayed on the screen a valid password. 

Input

The first line contains two integers n, m (3 ≤ n ≤ 50, 1 ≤ m ≤ 50) — the length of the password and the length of strings which are assigned to password symbols. 

Each of the next n lines contains the string which is assigned to the i-th symbol of the password string. Its length is m, it consists of digits, lowercase English letters, and characters '#', '*' or '&'.

You have such input data that you can always get a valid password.

Output

Print one integer — the minimum number of operations which is necessary to make the string, which is displayed on the screen, a valid password. 

Examples

Input

3 4
1**2
a3*0
c4**


Output

1


Input

5 5
#*&amp;#*
*a1c&amp;
&amp;q2w*
#a3c#
*&amp;#*&amp;


Output

3

Note

In the first test it is necessary to move the pointer of the third string to one left to get the optimal answer. 

<image>

In the second test one of possible algorithms will be: 

  * to move the pointer of the second symbol once to the right. 
  * to move the pointer of the third symbol twice to the right. 

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import math


def inp():
    return int(raw_input().strip())


def inp_arr():
    return map(int, raw_input().strip().split())


[n, m] = inp_arr()

ss = []

for _ in xrange(n):
    ss.append(raw_input().strip())

ns = []
cs = []
ps = []

for s in ss:
    num_dist = 100000000
    char_dist = 10000000
    special_dist = 10000000
    for i in xrange(m):
        c = s[i]
        if ord(c) in range(97, 123):
            char_dist = min(i, m - i, char_dist)
        if c.isdigit():
            num_dist = min(i, m - i, num_dist)
        if c in {'#', '*', '&'}:
            special_dist = min(i, m - i, special_dist)
    ns.append(num_dist)
    cs.append(char_dist)
    ps.append(special_dist)

ans = 10000000
#print ns, cs, ps

for i in xrange(n):
    for j in xrange(n):
        if j == i:
            continue
        for k in xrange(n):
            if k == j or k == i:
                continue
            if ns[i] + cs[j] + ps[k] < ans:
                ans = ns[i] + cs[j] + ps[k]

print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import math
import random
import re
from bootcamp import Basebootcamp

class Cdashaandpasswordbootcamp(Basebootcamp):
    def __init__(self, **params):
        """
        初始化密码谜题训练场参数，确保参数范围有效性
        """
        super().__init__(**params)
        # 参数有效性约束（n≥3且m≥1）
        self.n = min(max(params.get('n', 3), 3), 50)
        self.m = max(params.get('m', 4), 1)
    
    def case_generator(self):
        """
        生成保证有效性的谜题实例（确保必然存在三种字符类型）
        """
        case = {
            "n": self.n,
            "m": self.m,
            "strings": []
        }
        
        # 生成必须包含数字的字符串（固定至少一个数字）
        s_digit = [random.choice('0123456789')]
        for _ in range(self.m-1):
            s_digit.append(random.choice('0123456789abcdefghijklmnopqrstuvwxyz#*&'))
        random.shuffle(s_digit)
        case["strings"].append(''.join(s_digit))
        
        # 生成必须包含小写字母的字符串（固定至少一个字母）
        s_alpha = [random.choice('abcdefghijklmnopqrstuvwxyz')]
        for _ in range(self.m-1):
            s_alpha.append(random.choice('0123456789abcdefghijklmnopqrstuvwxyz#*&'))
        random.shuffle(s_alpha)
        case["strings"].append(''.join(s_alpha))
        
        # 生成必须包含特殊符号的字符串（固定至少一个符号）
        s_special = [random.choice('#*&')]
        for _ in range(self.m-1):
            s_special.append(random.choice('0123456789abcdefghijklmnopqrstuvwxyz#*&'))
        random.shuffle(s_special)
        case["strings"].append(''.join(s_special))
        
        # 生成剩余字符串（随机类型）
        for _ in range(3, self.n):
            s = [
                random.choice('0123456789abcdefghijklmnopqrstuvwxyz#*&')
                for _ in range(self.m)
            ]
            case["strings"].append(''.join(s))
        
        return case
    
    @staticmethod
    def prompt_func(question_case) -> str:
        """
        转换为符合题目输入格式的严格问题描述
        """
        return (
            "Dasha needs to set a password for programming class. The password must:\n"
            "1. Contain at least one digit (0-9)\n"
            "2. Contain at least one lowercase letter (a-z)\n"
            "3. Contain at least one of '#', '*' or '&'\n\n"
            "Each password character has a cyclic string. All pointers start at position 1.\n"
            "Find the minimal moves to form a valid password.\n\n"
            "Input format:\n"
            f"{question_case['n']} {question_case['m']}\n" +
            "\n".join(question_case['strings']) +
            "\n\nOutput the integer answer within [answer] tags like [answer]3[/answer]"
        )
    
    @staticmethod
    def extract_output(output):
        """
        严格匹配最后一个[answer]标签内的整数
        """
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except (ValueError, TypeError):
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        """
        精确计算最小操作步数的验证逻辑
        """
        n, m = identity["n"], identity["m"]
        strings = identity["strings"]
        
        # 预计算每个字符串的最短移动距离
        num_dists = [math.inf] * n
        alpha_dists = [math.inf] * n
        special_dists = [math.inf] * n
        
        for i, s in enumerate(strings):
            for pos in range(m):
                # 计算从初始位置(0 index)到pos的移动步数
                move_cost = min(pos, m - pos)
                char = s[pos]
                if char.isdigit():
                    num_dists[i] = min(num_dists[i], move_cost)
                elif char.islower():
                    alpha_dists[i] = min(alpha_dists[i], move_cost)
                elif char in {'#', '*', '&'}:
                    special_dists[i] = min(special_dists[i], move_cost)
        
        # 遍历所有三元组组合
        min_operations = math.inf
        for i in range(n):
            for j in range(n):
                if j == i:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    total = num_dists[i] + alpha_dists[j] + special_dists[k]
                    min_operations = min(min_operations, total)
        
        return solution == min_operations
