"""# 

### 谜题描述
You are given n strings s1, s2, ..., sn consisting of characters 0 and 1. m operations are performed, on each of them you concatenate two existing strings into a new one. On the i-th operation the concatenation saisbi is saved into a new string sn + i (the operations are numbered starting from 1). After each operation you need to find the maximum positive integer k such that all possible strings consisting of 0 and 1 of length k (there are 2k such strings) are substrings of the new string. If there is no such k, print 0.

Input

The first line contains single integer n (1 ≤ n ≤ 100) — the number of strings. The next n lines contain strings s1, s2, ..., sn (1 ≤ |si| ≤ 100), one per line. The total length of strings is not greater than 100.

The next line contains single integer m (1 ≤ m ≤ 100) — the number of operations. m lines follow, each of them contains two integers ai abd bi (1 ≤ ai, bi ≤ n + i - 1) — the number of strings that are concatenated to form sn + i.

Output

Print m lines, each should contain one integer — the answer to the question after the corresponding operation.

Example

Input

5
01
10
101
11111
0
3
1 2
6 5
4 4


Output

1
2
0

Note

On the first operation, a new string \"0110\" is created. For k = 1 the two possible binary strings of length k are \"0\" and \"1\", they are substrings of the new string. For k = 2 and greater there exist strings of length k that do not appear in this string (for k = 2 such string is \"00\"). So the answer is 1.

On the second operation the string \"01100\" is created. Now all strings of length k = 2 are present.

On the third operation the string \"1111111111\" is created. There is no zero, so the answer is 0.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from math import*
n = input()
p = [bin(p)[2:] for p in range(0,512)]
def mset(s):
    ss = set()
    for k in range(0,10):
        for pi in range(0,2 ** k):
            cs = p[pi]
            cs = (k - len(cs)) * \"0\" + cs
            if cs in s:
                ss.add(cs)
    return ss
def q(s):
    for k in range(0,10):
        for pi in range(0,2 ** k):
            cs = p[pi]
            cs = (k - len(cs)) * \"0\" + cs
            if not cs in s:
                return k - 1
s = [[v[:9], v[-9:], mset(v)] for v in [raw_input() for i in range(n)]]
for qa, qb in [[int(v) - 1 for v in raw_input().split()] for i in range(input())]:
    v = [s[qa][0], s[qb][1], mset(s[qa][1] + s[qb][0]) | s[qa][2] | s[qb][2]]
    if len(v[0]) < 9:
        v[0] = (v[0] + s[qb][0])[:9]
    if len(v[1]) < 9:
        v[1] = (s[qa][1] + s[qb][1])[-9:]
    s += [v]
    print(max(q(v[2]),0))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from math import pow

class Dhugestringsbootcamp(Basebootcamp):
    def __init__(self, n_max=5, m_max=3, max_total_length=100):
        self.n_max = min(n_max, 100)
        self.m_max = min(m_max, 100)
        self.max_total_length = min(max_total_length, 100)
    
    def case_generator(self):
        # 生成初始字符串逻辑优化
        initial_strings = []
        total_length = 0
        target_count = random.randint(1, self.n_max)
        
        while len(initial_strings) < target_count and total_length < self.max_total_length:
            remaining = self.max_total_length - total_length
            length = random.randint(1, min(100, remaining))
            s = ''.join(random.choices(['0','1'], k=length))
            initial_strings.append(s)
            total_length += length

        # 生成操作序列逻辑完善
        m = random.randint(1, self.m_max)
        s_states = []
        for s in initial_strings:
            pre = s[:9]
            suf = s[-9:] if len(s)>=9 else s
            s_states.append((
                pre,
                suf,
                self.mset(s)
            ))
        
        operations = []
        answers = []
        for _ in range(m):
            current_count = len(s_states)
            ai = random.randint(1, current_count)
            bi = random.randint(1, current_count)
            operations.append((ai, bi))
            
            # 状态更新逻辑
            a_state = s_states[ai-1]
            b_state = s_states[bi-1]
            
            # 计算新前缀
            new_pre = a_state[0]
            if len(new_pre) < 9:
                new_pre = (new_pre + b_state[0])[:9]
            
            # 计算新后缀
            new_suf = b_state[1]
            if len(new_suf) < 9:
                combined = a_state[1] + b_state[1]
                new_suf = combined[-9:]
            
            # 计算中间组合
            mid_str = a_state[1] + b_state[0]
            mid_set = self.mset(mid_str)
            combined_set = a_state[2].union(b_state[2]).union(mid_set)
            
            # 关键修正：最大k值计算逻辑
            max_k = 0
            for k in range(1, 10):
                required = 2 ** k
                all_exist = True
                for num in range(required):
                    target = bin(num)[2:].zfill(k)
                    if target not in combined_set:
                        all_exist = False
                        break
                if all_exist:
                    max_k = k
            answers.append(max_k if max_k > 0 else 0)
            
            s_states.append((new_pre, new_suf, combined_set))
        
        return {
            'initial': initial_strings,
            'operations': operations,
            'answers': answers
        }

    @staticmethod
    def mset(s):
        substr_set = set()
        for k in range(10):
            for num in range(2**k):
                bin_str = bin(num)[2:].zfill(k)
                if bin_str in s:
                    substr_set.add(bin_str)
        return substr_set

    @staticmethod
    def prompt_func(question_case):
        input_str = f"{len(question_case['initial'])}\n"
        input_str += "\n".join(question_case['initial']) + "\n"
        input_str += f"{len(question_case['operations'])}\n"
        for a, b in question_case['operations']:
            input_str += f"{a} {b}\n"
        
        return f"""You are given {len(question_case['initial'])} binary strings and {len(question_case['operations'])} concatenation operations. After each operation, determine the maximum positive integer k such that all possible binary strings of length k are present as substrings in the new string. If no such k exists, output 0.

Input:
{input_str.strip()}

Output your answers for each operation in order, each on a new line enclosed within [answer] tags. Example:
[answer]
1
2
0
[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(.*?)\s*\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        
        last_match = matches[-1].strip()
        answers = []
        for line in last_match.split('\n'):
            line = line.strip()
            if line.isdigit():
                answers.append(int(line))
        return answers if len(answers) > 0 else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity.get('answers', [])
        return isinstance(solution, list) and solution == expected
