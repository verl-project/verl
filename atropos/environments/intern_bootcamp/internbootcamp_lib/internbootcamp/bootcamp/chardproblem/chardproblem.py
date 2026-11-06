"""# 

### 谜题描述
Vasiliy is fond of solving different tasks. Today he found one he wasn't able to solve himself, so he asks you to help.

Vasiliy is given n strings consisting of lowercase English letters. He wants them to be sorted in lexicographical order (as in the dictionary), but he is not allowed to swap any of them. The only operation he is allowed to do is to reverse any of them (first character becomes last, second becomes one before last and so on).

To reverse the i-th string Vasiliy has to spent ci units of energy. He is interested in the minimum amount of energy he has to spent in order to have strings sorted in lexicographical order.

String A is lexicographically smaller than string B if it is shorter than B (|A| < |B|) and is its prefix, or if none of them is a prefix of the other and at the first position where they differ character in A is smaller than the character in B.

For the purpose of this problem, two equal strings nearby do not break the condition of sequence being sorted lexicographically.

Input

The first line of the input contains a single integer n (2 ≤ n ≤ 100 000) — the number of strings.

The second line contains n integers ci (0 ≤ ci ≤ 109), the i-th of them is equal to the amount of energy Vasiliy has to spent in order to reverse the i-th string. 

Then follow n lines, each containing a string consisting of lowercase English letters. The total length of these strings doesn't exceed 100 000.

Output

If it is impossible to reverse some of the strings such that they will be located in lexicographical order, print  - 1. Otherwise, print the minimum total amount of energy Vasiliy has to spent.

Examples

Input

2
1 2
ba
ac


Output

1


Input

3
1 3 1
aa
ba
ac


Output

1


Input

2
5 5
bbb
aaa


Output

-1


Input

2
3 3
aaa
aa


Output

-1

Note

In the second sample one has to reverse string 2 or string 3. To amount of energy required to reverse the string 3 is smaller.

In the third sample, both strings do not change after reverse and they go in the wrong order, so the answer is  - 1.

In the fourth sample, both strings consists of characters 'a' only, but in the sorted order string \"aa\" should go before string \"aaa\", thus the answer is  - 1.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from sys import stdin

n = input()
c = map(int, raw_input().strip().split())

dp = [[-1 for i in xrange(2)] for j in xrange(n)]
dp[0] = [0, c[0]]

flag = 0

inp = stdin.readlines()

ctr = 1
while ctr < len(inp):
    s = inp[ctr - 1].strip()
    t = inp[ctr].strip()
    u = s[::-1]
    v = t[::-1]

    f1 = (t >= s)
    f2 = (t >= u)
    f3 = (v >= s)
    f4 = (v >= u)
    
    if dp[ctr - 1] == [-1, -1]:
        flag = 1
        print -1
        break
    elif dp[ctr - 1][0] == -1:
        if f2: dp[ctr][0] = dp[ctr - 1][1]
        if f4: dp[ctr][1] = c[ctr] + dp[ctr - 1][1]
        elif not f2 and not f4:
            dp[ctr] = [-1, -1]
            print -1
            flag = 1
            break
    elif dp[ctr - 1][1] == -1:
        if f1: dp[ctr][0] = dp[ctr - 1][0]
        if f3: dp[ctr][1] = c[ctr] + dp[ctr - 1][0]
        elif not f1 and not f3:
            dp[ctr] = [-1, -1]
            print -1
            flag = 1
            break
    else:
        # for t
        if not f1 and not f2: dp[ctr][0] = -1
        elif f1 and not f2: dp[ctr][0] = dp[ctr - 1][0]
        elif f2 and not f1: dp[ctr][0] = dp[ctr - 1][1]
        elif f1 and f2: dp[ctr][0] = min(dp[ctr - 1])

        # for v
        if not f3 and not f4: dp[ctr][1] = -1
        elif f3 and not f4: dp[ctr][1] = c[ctr] + dp[ctr - 1][0]
        elif f4 and not f3: dp[ctr][1] = c[ctr] + dp[ctr - 1][1]
        elif f3 and f4: dp[ctr][1] = c[ctr] + min(dp[ctr - 1])
        
    ctr += 1

if not flag:
    x = dp[n - 1][0]
    y = dp[n - 1][1]

    if x == -1 and y == -1: print -1
    elif x == -1: print y
    elif y == -1: print x
    else: print min(x, y)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import string
from bootcamp import Basebootcamp
import re

class Chardproblembootcamp(Basebootcamp):
    def __init__(self, max_n=5, max_string_length=10, min_n=2, **kwargs):
        super().__init__(**kwargs)
        self.min_n = min_n
        self.max_n = max_n
        self.max_string_length = max_string_length

    def case_generator(self):
        """生成包含边界情况的测试案例"""
        n = random.randint(self.min_n, self.max_n)
        c = [random.randint(0, 10) for _ in range(n)]
        strings = self._generate_strings_with_edge_cases(n)
        
        case = {
            'n': n,
            'c': c,
            'strings': strings,
            'expected_output': self._solve_case(n, c, strings)
        }
        return case

    def _generate_strings_with_edge_cases(self, n):
        """生成包含前缀、相同字符串等边界情况的序列"""
        strings = []
        if random.random() < 0.3:
            base = self._random_string()
            strings.append(base)
            for _ in range(n-1):
                strings.append(base + self._random_string(1))
        elif random.random() < 0.3: 
            s = self._random_string()
            strings = [s] * n
        else:
            total_length = 0
            for _ in range(n):
                max_len = min(self.max_string_length, 100000 - total_length)
                if max_len <=0:
                    s = ''
                else:
                    length = random.randint(1, max_len)
                    s = ''.join(random.choices(string.ascii_lowercase, k=length))
                    total_length += length
                strings.append(s)
        return strings

    def _random_string(self, length=None):
        """生成随机长度的字符串"""
        if length is None:
            length = random.randint(1, self.max_string_length)
        return ''.join(random.choices(string.ascii_lowercase, k=length))

    def _solve_case(self, n, c, strings):
        """动态规划求解正确结果 (完整实现)"""
        dp = [[-1] * 2 for _ in range(n)]
        dp[0][0] = 0
        dp[0][1] = c[0]
        possible = True

        for i in range(1, n):
            prev = strings[i-1]
            current = strings[i]
            prev_rev = prev[::-1]
            current_rev = current[::-1]

            dp_i0 = -1
            dp_i1 = -1

            # 处理不反转当前字符串的情况
            if dp[i-1][0] != -1 and current >= prev:
                dp_i0 = dp[i-1][0]
            if dp[i-1][1] != -1 and current >= prev_rev:
                if dp_i0 == -1 or dp[i-1][1] < dp_i0:
                    dp_i0 = dp[i-1][1]

            # 处理反转当前字符串的情况
            cost = c[i]
            if dp[i-1][0] != -1 and current_rev >= prev:
                dp_i1 = dp[i-1][0] + cost
            if dp[i-1][1] != -1 and current_rev >= prev_rev:
                candidate = dp[i-1][1] + cost
                if dp_i1 == -1 or candidate < dp_i1:
                    dp_i1 = candidate

            dp[i][0] = dp_i0
            dp[i][1] = dp_i1

            if dp[i][0] == -1 and dp[i][1] == -1:
                possible = False
                break

        if not possible:
            return -1

        final0 = dp[-1][0]
        final1 = dp[-1][1]
        if final0 == -1 and final1 == -1:
            return -1
        return min(filter(lambda x: x != -1, [final0, final1])) if final0 != -1 and final1 != -1 else max(final0, final1)

    @staticmethod
    def prompt_func(question_case) -> str:
        input_example = f"{question_case['n']}\n"
        input_example += ' '.join(map(str, question_case['c'])) + '\n'
        input_example += '\n'.join(question_case['strings'])
        return f"""请解决下列字符串排序能量消耗问题：
{input_example}
答案格式：[answer]答案[/answer]，如[answer]-1[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            return int(matches[-1].strip())
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['expected_output']
