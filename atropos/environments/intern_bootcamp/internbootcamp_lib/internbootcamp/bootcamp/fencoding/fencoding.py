"""# 

### 谜题描述
Polycarp invented a new way to encode strings. Let's assume that we have string T, consisting of lowercase English letters. Let's choose several pairs of letters of the English alphabet in such a way that each letter occurs in at most one pair. Then let's replace each letter in T with its pair letter if there is a pair letter for it. For example, if you chose pairs (l, r), (p, q) and (a, o), then word \"parallelogram\" according to the given encoding principle transforms to word \"qolorreraglom\".

Polycarpus already has two strings, S and T. He suspects that string T was obtained after applying the given encoding method from some substring of string S. Find all positions mi in S (1 ≤ mi ≤ |S| - |T| + 1), such that T can be obtained fro substring SmiSmi + 1... Smi + |T| - 1 by applying the described encoding operation by using some set of pairs of English alphabet letters

Input

The first line of the input contains two integers, |S| and |T| (1 ≤ |T| ≤ |S| ≤ 2·105) — the lengths of string S and string T, respectively.

The second and third line of the input contain strings S and T, respectively. Both strings consist only of lowercase English letters.

Output

Print number k — the number of suitable positions in string S.

In the next line print k integers m1, m2, ..., mk — the numbers of the suitable positions in the increasing order.

Examples

Input

11 5
abacabadaba
acaba


Output

3
1 3 7


Input

21 13
paraparallelogramgram
qolorreraglom


Output

1
5

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
raw_input()
ss = raw_input()
tt = raw_input()

def df(s):
    c = {}
    res = []
    for i, x in enumerate(s):
        res.append(i - c[x] if x in c else 0)
        c[x] = i
    return res

s = df(ss)
t = df(tt)
p = []
l = []
for i, x in enumerate(t):
    if not x:
        p.append(i)
        l.append(ord(tt[i]) - 97)
nt = len(l)

def prefix_func(s):
    pi = [0] * len(s)
    for i in xrange(1, len(s)):
        j = pi[i - 1]
        while j > 0 and not (s[i] == s[j] or (not s[j] and s[i] > j)):
            j = pi[j - 1]
        pi[i] = j + 1 if s[i] == s[j] or (not s[j] and s[i] > j) else j

    return pi

pi = prefix_func(t + [-1] + s)
n = len(t)
res = []
ss = [ord(x) - 97 for x in ss]

def check(pos):
    d = [-1] * 26
    for i in xrange(nt):
        j = p[i]
        x = ss[j + pos]
        y = l[i]
        if d[x] >= 0 and d[x] != y:
            return False
        if d[y] >= 0 and d[y] != x:
            return False
        d[x] = y
        d[y] = x
    return True

for i, x in enumerate(pi):
    j = i - 2 * n
    if x != n or not check(j):
        continue
    res.append(j + 1)
print len(res)
print ' '.join(map(str, res))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import string
import random
import re
from bootcamp import Basebootcamp

class Fencodingbootcamp(Basebootcamp):
    def __init__(self, min_s_length=5, max_s_length=20, max_t_ratio=0.5, pair_prob=0.3):
        self.min_s_length = min_s_length
        self.max_s_length = max_s_length
        self.max_t_ratio = max_t_ratio
        self.pair_prob = pair_prob
    
    def generate_pairs(self):
        letters = list(string.ascii_lowercase)
        random.shuffle(letters)
        pairs = {}
        i = 0
        while i < len(letters) - 1:
            if random.random() < self.pair_prob:
                a, b = letters[i], letters[i+1]
                pairs[a], pairs[b] = b, a
                i += 2
            else:
                i += 1
        return pairs
    
    def case_generator(self):
        n = random.randint(self.min_s_length, self.max_s_length)
        m_max = max(1, min(n, int(n * self.max_t_ratio)))
        m = random.randint(1, m_max)
        S = ''.join(random.choices(string.ascii_lowercase, k=n))
        pos = random.randint(0, n - m)
        S_sub = S[pos:pos+m]
        pairs = self.generate_pairs()
        T = ''.join([pairs.get(c, c) for c in S_sub])
        return {'S': S, 'T': T}
    
    @staticmethod
    def prompt_func(question_case):
        S = question_case['S']
        T = question_case['T']
        problem = f"""你是一个编程竞赛选手，请解决以下问题：

Polycarp发明了一种字符串编码方式。规则为：选择若干对字母，每个字母只能在一个对中出现。然后将字符串中的每个字母替换为它的配对字母（如果存在配对）。例如，选择对(l,r)和(p,q)后，"parallelogram"会被编码为"qolorreraglom"。

给定两个字符串S和T，请找出S中所有起始位置（1-based），使得从该位置开始的长|T|的子串经上述编码后得到T。

输入：
S长度为{len(S)}，T长度为{len(T)}。
S = "{S}"
T = "{T}"

输出：
第一行输出k（匹配位置数），第二行按升序输出k个位置。

答案请放入[answer]标签，例如：
[answer]
3
1 3 7
[/answer]"""
        return problem.strip()
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        content = matches[-1].strip()
        lines = content.split('\n')
        if len(lines) < 2:
            return None
        try:
            k = int(lines[0].strip())
            positions = list(map(int, lines[1].strip().split()))
            if len(positions) != k or positions != sorted(positions):
                return None
        except:
            return None
        return positions
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if solution is None:
            return False
        S = identity['S']
        T = identity['T']
        correct = cls.find_correct_positions(S, T)
        return solution == correct

    @staticmethod
    def find_correct_positions(S, T):
        len_S, len_T = len(S), len(T)
        if len_T > len_S or len_T == 0:
            return []
        
        def compute_diff(s):
            last = {}
            diff = []
            for i, c in enumerate(s):
                diff.append(i - last[c] if c in last else 0)
                last[c] = i
            return diff
        
        s_diff = compute_diff(S)
        t_diff = compute_diff(T)
        
        p, l = [], []
        for i, val in enumerate(t_diff):
            if val == 0:
                p.append(i)
                l.append(ord(T[i]) - ord('a'))
        nt = len(p)
        
        combined = t_diff + [-1] + s_diff
        
        def prefix_func(arr):
            pi = [0] * len(arr)
            for i in range(1, len(arr)):
                j = pi[i-1]
                while j > 0 and not (arr[i] == arr[j] or (arr[j] == 0 and arr[i] > j)):
                    j = pi[j-1]
                if arr[i] == arr[j] or (arr[j] == 0 and arr[i] > j):
                    pi[i] = j + 1
                else:
                    pi[i] = 0
            return pi
        
        pi = prefix_func(combined)
        s_nums = [ord(c) - ord('a') for c in S]
        res = []
        
        for i in range(len(pi)):
            if pi[i] != len_T:
                continue
            j = i - 2 * len_T  # 调整索引偏移
            # 检查有效范围
            if j < 0 or j > len_S - len_T:
                continue
            
            # 验证字符映射一致性
            d = [-1] * 26
            valid = True
            for idx in range(nt):
                t_pos = p[idx]
                s_pos = j + t_pos  # 对应的S中的位置
                # 修复这里的语法错误
                if s_pos >= len(s_nums):
                    valid = False
                    break
                s_char = s_nums[s_pos]
                t_char = l[idx]
                
                # 检查双向映射是否一致
                if d[s_char] != -1 and d[s_char] != t_char:
                    valid = False
                    break
                if d[t_char] != -1 and d[t_char] != s_char:
                    valid = False
                    break
                d[s_char] = t_char
                d[t_char] = s_char
            
            if valid:
                res.append(j + 1)  # 转换为1-based索引
        
        return sorted(res)
