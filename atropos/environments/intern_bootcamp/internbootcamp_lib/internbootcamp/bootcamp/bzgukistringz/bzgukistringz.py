"""# 

### 谜题描述
Professor GukiZ doesn't accept string as they are. He likes to swap some letters in string to obtain a new one.

GukiZ has strings a, b, and c. He wants to obtain string k by swapping some letters in a, so that k should contain as many non-overlapping substrings equal either to b or c as possible. Substring of string x is a string formed by consecutive segment of characters from x. Two substrings of string x overlap if there is position i in string x occupied by both of them.

GukiZ was disappointed because none of his students managed to solve the problem. Can you help them and find one of possible strings k?

Input

The first line contains string a, the second line contains string b, and the third line contains string c (1 ≤ |a|, |b|, |c| ≤ 105, where |s| denotes the length of string s).

All three strings consist only of lowercase English letters. 

It is possible that b and c coincide.

Output

Find one of possible strings k, as described in the problem statement. If there are multiple possible answers, print any of them.

Examples

Input

aaa
a
b


Output

aaa

Input

pozdravstaklenidodiri
niste
dobri


Output

nisteaadddiiklooprrvz

Input

abbbaaccca
ab
aca


Output

ababacabcc

Note

In the third sample, this optimal solutions has three non-overlaping substrings equal to either b or c on positions 1 – 2 (ab), 3 – 4 (ab), 5 – 7 (aca). In this sample, there exist many other optimal solutions, one of them would be acaababbcc.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def solve():
    a = raw_input()
    b = raw_input()
    c = raw_input()
    acount = [0] * 26
    for char in a: acount[ord(char) - ord('a')]+=1
    bcount = [0] * 26
    for char in b: bcount[ord(char) - ord('a')]+=1
    ccount = [0] * 26
    for char in c: ccount[ord(char) - ord('a')]+=1
    bres = 0
    cres = 0
    for bs in xrange(len(a) + 1):
        acountclone = list(acount)
        works = True
        for i in range(26):
            acountclone[i] -= bcount[i] * bs
            if acountclone[i] < 0: works = False
        cs = 1000000000
        for i in range(26): 
            if ccount[i] != 0: cs = min(cs, acountclone[i] / ccount[i])
        if works and bs + cs > bres + cres:
            bres = bs
            cres = cs
    for i in range(26):
        acount[i] -= bres * bcount[i] + cres * ccount[i]
    ans = b * bres + c * cres
    for i in range(26):
        while acount[i] > 0:
            ans += chr(i + ord('a'))
            acount[i] -= 1
    print(ans)
        
solve()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from collections import Counter
import random
import string
import re
from bootcamp import Basebootcamp

def compute_max_counts(a, b, c):
    a_counts = [0] * 26
    for char in a:
        a_counts[ord(char) - ord('a')] += 1

    b_counts = [0] * 26
    for char in b:
        b_counts[ord(char) - ord('a')] += 1

    c_counts = [0] * 26
    for char in c:
        c_counts[ord(char) - ord('a')] += 1

    best_bs = 0
    best_cs = 0
    max_total = 0

    # 模拟原题代码，枚举bs到a的长度+1
    max_bs = len(a)
    for bs in range(0, max_bs + 1):
        possible = True
        a_clone = a_counts.copy()
        for i in range(26):
            required = bs * b_counts[i]
            if a_clone[i] < required:
                possible = False
                break
            a_clone[i] -= required
        if not possible:
            continue

        # 计算c的最大次数
        cs = float('inf')
        for i in range(26):
            if c_counts[i] == 0:
                continue
            available = a_clone[i]
            if available < c_counts[i]:
                cs = 0
                break
            cs = min(cs, available // c_counts[i])
        if cs == float('inf'):
            cs = 0

        total = bs + cs
        if total > max_total or (total == max_total and cs > best_cs):
            max_total = total
            best_bs = bs
            best_cs = cs

    return best_bs, best_cs

def count_max_substrings(k_str, b, c):
    subs = []
    len_b, len_c = len(b), len(c)
    if len_b > 0:
        subs.append((len_b, b))
    if len_c > 0 and b != c:
        subs.append((len_c, c))

    n = len(k_str)
    dp = [0] * (n + 1)

    for i in range(n):
        dp[i + 1] = max(dp[i + 1], dp[i])
        for length, sub in subs:
            if i + length > n:
                continue
            if k_str[i:i + length] == sub:
                dp[i + length] = max(dp[i + length], dp[i] + 1)
    return dp[n]

class Bzgukistringzbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = params
        self.max_b_length = params.get('max_b_length', 5)
        self.max_c_length = params.get('max_c_length', 5)
        self.max_a_length = params.get('max_a_length', 20)
        self.min_length = params.get('min_length', 1)
        self.max_attempts = params.get('max_attempts', 1000)

    def case_generator(self):
        for _ in range(self.max_attempts):
            b = ''.join(random.choices(string.ascii_lowercase, k=random.randint(self.min_length, self.max_b_length)))
            c = ''.join(random.choices(string.ascii_lowercase, k=random.randint(self.min_length, self.max_c_length)))

            a_length = random.randint(max(len(b), len(c), self.min_length), self.max_a_length)
            
            # 确保a至少包含足够的字符生成一个b或一个c
            a_chars = []
            if random.choice([True, False]) and len(b) > 0:
                a_chars.extend(list(b))
            elif len(c) > 0:
                a_chars.extend(list(c))
            
            remaining = a_length - len(a_chars)
            if remaining > 0:
                a_chars += random.choices(string.ascii_lowercase, k=remaining)
            a = ''.join(random.sample(a_chars, len(a_chars)))  # 打乱顺序避免简单排列

            best_bs, best_cs = compute_max_counts(a, b, c)
            if best_bs + best_cs > 0:
                k_str = b * best_bs + c * best_cs
                remaining_chars = []
                a_counter = Counter(a)
                for char in k_str:
                    a_counter[char] -= 1
                for char, count in a_counter.items():
                    remaining_chars.extend([char] * count)
                random.shuffle(remaining_chars)
                k_str += ''.join(remaining_chars)
                
                # 验证生成字符串的字符计数
                if Counter(k_str) != Counter(a):
                    continue
                
                return {
                    'a': a,
                    'b': b,
                    'c': c,
                    'best_bs': best_bs,
                    'best_cs': best_cs,
                    'max_total': best_bs + best_cs
                }
        
        raise RuntimeError("Failed to generate valid case after multiple attempts")

    @staticmethod
    def prompt_func(question_case):
        a = question_case['a']
        b = question_case['b']
        c = question_case['c']
        return f'''GukiZ教授有一个字符串重组谜题需要解决。给定三个字符串a、b、c，你需要将a中的字符重新排列，生成新字符串k，使得k中包含尽可能多的非重叠子串b或c。子串不能重叠，且必须完全匹配。

输入：
a = "{a}"
b = "{b}"
c = "{c}"

请输出重组后的字符串k，确保其是a的一个排列，并将最终答案包裹在[answer]和[/answer]标签中。例如：
[answer]
examplekstring
[/answer]'''

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 检查solution是否是a的正确排列
        if Counter(solution) != Counter(identity['a']):
            return False
        
        # 计算最大非重叠子串数目
        calculated = count_max_substrings(solution, identity['b'], identity['c'])
        return calculated == identity['max_total']
