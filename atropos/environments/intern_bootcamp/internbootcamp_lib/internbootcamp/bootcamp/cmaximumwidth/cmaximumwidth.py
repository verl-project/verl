"""# 

### 谜题描述
Your classmate, whom you do not like because he is boring, but whom you respect for his intellect, has two strings: s of length n and t of length m.

A sequence p_1, p_2, …, p_m, where 1 ≤ p_1 < p_2 < … < p_m ≤ n, is called beautiful, if s_{p_i} = t_i for all i from 1 to m. The width of a sequence is defined as max_{1 ≤ i < m} \left(p_{i + 1} - p_i\right).

Please help your classmate to identify the beautiful sequence with the maximum width. Your classmate promised you that for the given strings s and t there is at least one beautiful sequence.

Input

The first input line contains two integers n and m (2 ≤ m ≤ n ≤ 2 ⋅ 10^5) — the lengths of the strings s and t.

The following line contains a single string s of length n, consisting of lowercase letters of the Latin alphabet.

The last line contains a single string t of length m, consisting of lowercase letters of the Latin alphabet.

It is guaranteed that there is at least one beautiful sequence for the given strings.

Output

Output one integer — the maximum width of a beautiful sequence.

Examples

Input


5 3
abbbc
abc


Output


3


Input


5 2
aaaaa
aa


Output


4


Input


5 5
abcdf
abcdf


Output


1


Input


2 2
ab
ab


Output


1

Note

In the first example there are two beautiful sequences of width 3: they are \{1, 2, 5\} and \{1, 4, 5\}.

In the second example the beautiful sequence with the maximum width is \{1, 5\}.

In the third example there is exactly one beautiful sequence — it is \{1, 2, 3, 4, 5\}.

In the fourth example there is exactly one beautiful sequence — it is \{1, 2\}.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from collections import Counter
import math

if __name__==\"__main__\":
    n,m= map(int,raw_input().split(\" \"))
    s1=raw_input().strip()
    s2=raw_input().strip()
    ans1=[0 for k in s2]
    ans2=[0 for k in s2]
    
    b2=0
    for i in range(0,len(s1)):
        if s1[i]==s2[b2]:
            ans1[b2]=i
            b2+=1
            if b2==len(s2):
                break
    
    b2=len(s2)-1
    for i in range(len(s1)-1,-1,-1):
        if s1[i]==s2[b2]:
            ans2[b2]=i
            b2-=1
            if b2<0:
                break
    ans=0
    for i in range(0,len(s2)-1):
        ans=max(ans,ans2[i+1]-ans1[i])
    print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Cmaximumwidthbootcamp(Basebootcamp):
    def __init__(self, min_n=2, max_n=20, min_m=2, max_m=15):
        self.params = {
            'min_n': min_n,
            'max_n': max_n,
            'min_m': min_m,
            'max_m': max_m,
        }
    
    def case_generator(self):
        max_retry = 5
        for _ in range(max_retry):
            try:
                m = random.randint(
                    max(self.params['min_m'], 2),
                    min(self.params['max_m'], self.params['max_n'])
                )
                n = random.randint(
                    max(m, self.params['min_n']),
                    self.params['max_n']
                )

                # 生成策略优化
                gen_strategy = random.choices(
                    ['direct_insert', 'reverse_insert', 'balanced'],
                    weights=[0.3, 0.3, 0.4],
                    k=1
                )[0]

                t = []
                s = []
                
                # 生成逻辑优化
                if gen_strategy == 'direct_insert':
                    t = [chr(97 + random.randint(0, 25)) for _ in range(m)]
                    ptr = 0
                    for i in range(m):
                        gap = random.randint(0, n - m - ptr) if i < m-1 else 0
                        s += [chr(97 + random.randint(0, 25)) for _ in range(gap)]
                        s.append(t[i])
                        ptr += gap + 1
                    s += [chr(97 + random.randint(0, 25)) for _ in range(n - len(s))]
                
                elif gen_strategy == 'reverse_insert':
                    t = [chr(97 + random.randint(0, 25)) for _ in range(m)]
                    remaining_space = n - m
                    gaps = [random.randint(0, remaining_space) for _ in range(m-1)]
                    total_gaps = sum(gaps)
                    
                    if total_gaps > remaining_space:
                        scale = remaining_space / total_gaps
                        gaps = [int(g*scale) for g in gaps]
                    
                    for i in range(m):
                        s.append(t[i])
                        if i < m-1:
                            s += [chr(97 + random.randint(0,25)) for _ in range(gaps[i])]
                    s += [chr(97 + random.randint(0,25)) for _ in range(n - len(s))]
                
                else:  # balanced strategy
                    t = [chr(97 + random.randint(0, 25)) for _ in range(m)]
                    pos = sorted(random.sample(range(n), m))
                    s = [chr(97 + random.randint(0,25)) for _ in range(n)]
                    for i,p in enumerate(pos):
                        s[p] = t[i]

                s = ''.join(s[:n])  # 长度强制对齐
                t = ''.join(t)
                
                # 验证子序列
                def is_subsequence(s, t):
                    it = iter(s)
                    return all(c in it for c in t)
                
                if not is_subsequence(s, t):
                    continue  # 重试

                # 计算正确答案
                ans1 = []
                ptr = 0
                for c in t:
                    while ptr < len(s) and s[ptr] != c:
                        ptr += 1
                    ans1.append(ptr)
                    ptr += 1
                
                ans2 = []
                ptr = len(s) - 1
                for c in reversed(t):
                    while ptr >= 0 and s[ptr] != c:
                        ptr -= 1
                    ans2.append(ptr)
                    ptr -= 1
                ans2.reverse()
                
                max_width = max(ans2[i+1] - ans1[i] for i in range(m-1))

                return {
                    'n': n,
                    'm': m,
                    's': s,
                    't': t,
                    'correct_answer': max_width
                }

            except Exception as e:
                continue
        raise RuntimeError("生成有效案例失败")

    @staticmethod
    def prompt_func(question_case) -> str:
        return f"""请解决以下字符串序列问题：

输入格式：
第一行：n m（2 ≤ m ≤ n）
第二行：s（长度n）
第三行：t（长度m）

问题描述：
寻找s中满足s[p_i] = t[i]的严格递增下标序列p_1 < p_2 < ... < p_m。
定义序列宽度为相邻下标差的最大值，即max(p_{{i+1}} - p_i)。
求所有可能序列中的最大宽度。

输入数据：
{question_case['n']} {question_case['m']}
{question_case['s']}
{question_case['t']}

将答案用[answer]标签包裹，例如：[answer]3[/answer]"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return int(matches[-1].strip()) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
