"""# 

### 谜题描述
Little Elephant loves Furik and Rubik, who he met in a small city Kremenchug.

The Little Elephant has two strings of equal length a and b, consisting only of uppercase English letters. The Little Elephant selects a pair of substrings of equal length — the first one from string a, the second one from string b. The choice is equiprobable among all possible pairs. Let's denote the substring of a as x, and the substring of b — as y. The Little Elephant gives string x to Furik and string y — to Rubik.

Let's assume that f(x, y) is the number of such positions of i (1 ≤ i ≤ |x|), that xi = yi (where |x| is the length of lines x and y, and xi, yi are the i-th characters of strings x and y, correspondingly). Help Furik and Rubik find the expected value of f(x, y).

Input

The first line contains a single integer n (1 ≤ n ≤ 2·105) — the length of strings a and b. The second line contains string a, the third line contains string b. The strings consist of uppercase English letters only. The length of both strings equals n.

Output

On a single line print a real number — the answer to the problem. The answer will be considered correct if its relative or absolute error does not exceed 10 - 6.

Examples

Input

2
AB
BA


Output

0.400000000


Input

3
AAB
CAA


Output

0.642857143

Note

Let's assume that we are given string a = a1a2... a|a|, then let's denote the string's length as |a|, and its i-th character — as ai.

A substring a[l... r] (1 ≤ l ≤ r ≤ |a|) of string a is string alal + 1... ar.

String a is a substring of string b, if there exists such pair of integers l and r (1 ≤ l ≤ r ≤ |b|), that b[l... r] = a.

Let's consider the first test sample. The first sample has 5 possible substring pairs: (\"A\", \"B\"), (\"A\", \"A\"), (\"B\", \"B\"), (\"B\", \"A\"), (\"AB\", \"BA\"). For the second and third pair value f(x, y) equals 1, for the rest it equals 0. The probability of choosing each pair equals <image>, that's why the answer is <image> · 0 +  <image> · 1 +  <image> · 1 +  <image> · 0 +  <image> · 0 =  <image> =  0.4.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
n=input()
a=[map(lambda x:[],range(26)) for _ in (0,1)]
for i in (0,1):
  for j,x in enumerate(raw_input()):
    a[i][ord(x)-ord('A')]+=[j]
s=0
for i in range(26):
  p,q=a[0][i],a[1][i]
  if p and q:
    t,j,w,k,m=0,0,sum(q),len(p),len(q)
    for u,x in enumerate(p):
      while j<m and q[j]<x:t,j=t+q[j],j+1
      s+=(t+j)*(n-x)+(x+1)*(n*(m-j)-(w-t))
print '%.9lf'%(s*6./n/(n+1)/(n*2+1))
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import string
import re
from bootcamp import Basebootcamp

def compute_expected(n, a_str, b_str):
    a = [[[] for _ in range(26)] for __ in range(2)]
    for i, s in enumerate([a_str, b_str]):
        for j, c in enumerate(s):
            idx = ord(c) - ord('A')
            a[i][idx].append(j)
    
    total = 0
    for char_idx in range(26):
        p = a[0][char_idx]
        q = a[1][char_idx]
        if not p or not q:
            continue
        
        q_sum = sum(q)
        p_len = len(p)
        q_len = len(q)
        j = 0
        t = 0
        
        for x in p:
            # 维护双指针找到q中第一个不小于x的位置
            while j < q_len and q[j] < x:
                t += q[j]
                j += 1
            
            # 计算两项贡献（参考原算法逻辑）
            part1 = (t + j * x) * (n - x)
            part2 = (x + 1) * (n * (q_len - j) - (q_sum - t))
            total += part1 + part2
    
    # 计算分母：n*(n+1)*(2n+1)/6
    denominator = n * (n + 1) * (2 * n + 1) / 6
    if denominator == 0:
        return 0.0
    return total * 6.0 / (n * (n + 1) * (2 * n + 1))

class Clittleelephantandfurikandrubikbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=200000):
        self.min_n = min_n
        self.max_n = max_n
    
    def case_generator(self):
        n = random.randint(self.min_n, min(self.max_n, 10))  # 默认测试限制n<=10
        a = ''.join(random.choices(string.ascii_uppercase, k=n))
        b = ''.join(random.choices(string.ascii_uppercase, k=n))
        expected = compute_expected(n, a, b)
        return {
            "n": n,
            "a": a,
            "b": b,
            "expected": expected
        }
    
    @staticmethod
    def prompt_func(question_case):
        n = question_case['n']
        a = question_case['a']
        b = question_case['b']
        prompt = f"""你是数学竞赛选手，请解决以下期望值计算问题：

给定两个长度均为{n}的字符串：
字符串a：{a}
字符串b：{b}

定义所有有效子串对(x,y)为从a和b中分别选取的等长子串。求在所有可能子串对中，x与y在相同位置上字符相等的数量的数学期望。

请先分析问题，给出计算步骤，最后将答案（保留9位小数）放入[answer]标签内。例如：[answer]0.400000000[/answer]"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        # 处理科学计数法（如1e-9等）
        if 'e' in last_match:
            try:
                return "{0:.9f}".format(float(last_match))
            except:
                return None
        return last_match
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            solution_float = float(solution)
            expected = identity['expected']
            absolute_error = abs(solution_float - expected)
            
            # 处理极端情况（如期望为0时）
            if expected == 0:
                return absolute_error < 1e-6
            # 计算相对误差
            relative_error = absolute_error / abs(expected)
            return relative_error < 1e-6 or absolute_error < 1e-6
        except (ValueError, TypeError):
            return False

# 示例验证
if __name__ == "__main__":
    bootcamp = Clittleelephantandfurikandrubikbootcamp(min_n=2, max_n=2)
    case = bootcamp.case_generator()
    case.update({"n": 2, "a": "AB", "b": "BA", "expected": 0.4})
    print("Test case:", case)
    prompt = bootcamp.prompt_func(case)
    print("Generated prompt:\n", prompt)
    test_output = "经过计算，期望值为[answer]0.400000000[/answer]"
    extracted = bootcamp.extract_output(test_output)
    print("Extracted answer:", extracted)
    print("Verification result:", bootcamp._verify_correction(extracted, case))
