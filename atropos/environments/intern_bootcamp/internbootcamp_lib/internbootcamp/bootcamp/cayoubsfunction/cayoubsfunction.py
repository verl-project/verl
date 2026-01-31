"""# 

### 谜题描述
Ayoub thinks that he is a very smart person, so he created a function f(s), where s is a binary string (a string which contains only symbols \"0\" and \"1\"). The function f(s) is equal to the number of substrings in the string s that contains at least one symbol, that is equal to \"1\".

More formally, f(s) is equal to the number of pairs of integers (l, r), such that 1 ≤ l ≤ r ≤ |s| (where |s| is equal to the length of string s), such that at least one of the symbols s_l, s_{l+1}, …, s_r is equal to \"1\". 

For example, if s = \"01010\" then f(s) = 12, because there are 12 such pairs (l, r): (1, 2), (1, 3), (1, 4), (1, 5), (2, 2), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 4), (4, 5).

Ayoub also thinks that he is smarter than Mahmoud so he gave him two integers n and m and asked him this problem. For all binary strings s of length n which contains exactly m symbols equal to \"1\", find the maximum value of f(s).

Mahmoud couldn't solve the problem so he asked you for help. Can you help him? 

Input

The input consists of multiple test cases. The first line contains a single integer t (1 ≤ t ≤ 10^5) — the number of test cases. The description of the test cases follows.

The only line for each test case contains two integers n, m (1 ≤ n ≤ 10^{9}, 0 ≤ m ≤ n) — the length of the string and the number of symbols equal to \"1\" in it.

Output

For every test case print one integer number — the maximum value of f(s) over all strings s of length n, which has exactly m symbols, equal to \"1\".

Example

Input


5
3 1
3 2
3 3
4 0
5 2


Output


4
5
6
0
12

Note

In the first test case, there exists only 3 strings of length 3, which has exactly 1 symbol, equal to \"1\". These strings are: s_1 = \"100\", s_2 = \"010\", s_3 = \"001\". The values of f for them are: f(s_1) = 3, f(s_2) = 4, f(s_3) = 3, so the maximum value is 4 and the answer is 4.

In the second test case, the string s with the maximum value is \"101\".

In the third test case, the string s with the maximum value is \"111\".

In the fourth test case, the only string s of length 4, which has exactly 0 symbols, equal to \"1\" is \"0000\" and the value of f for that string is 0, so the answer is 0.

In the fifth test case, the string s with the maximum value is \"01010\" and it is described as an example in the problem statement.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
 
 
T = input()
 
 
 
 
 
for case in range(T):
 
	n, m = map(int, raw_input().split())
 
	if m >= float(n)/2:
		A = n*(n + 1)/2 - (n - m)
	else:
		# the number of compartments
		c = m + 1
		# splitting the zeros into compartments
		z = n - m
		f_l = z/c
		# calculate how many of f_l and f_u there need 
		n_f_u = z % c
		n_f_l = c - n_f_u
 
		# calculating A
		A = n*(n + 1)/2 - (((f_l + 1)*(f_l + 2)/2)*n_f_u + (f_l*(f_l + 1)/2)*n_f_l)
 
	print A
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Cayoubsfunctionbootcamp(Basebootcamp):
    def __init__(self, n_min=1, n_max=10**9):
        """
        支持极大数值范围的初始化参数
        参数:
            n_min: 可生成的最小n值 (default=1)
            n_max: 可生成的最大n值 (default=1e9)
        """
        self.n_min = n_min
        self.n_max = n_max
    
    def case_generator(self):
        # 30%生成特殊边界情况
        if random.random() < 0.3:
            n = random.randint(self.n_min, self.n_max)
            special_cases = [
                (n, 0),         # 全0字符串
                (n, n),         # 全1字符串
                (n, n//2),      # 临界值情况
                (n, n//2 + 1)   # 临界值+1
            ]
            return dict(zip(['n', 'm'], random.choice(special_cases)))
        
        # 常规随机生成
        n = random.randint(max(1, self.n_min), self.n_max)
        m = random.randint(0, n)
        return {"n": n, "m": m}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        m = question_case['m']
        return f"""### Programming Problem Statement

Given a binary string s of length {n} containing exactly {m} '1's, find the maximum possible value of function f(s). 

**Function Definition:**
f(s) = number of substrings containing at least one '1'

**Substring Definition:**
A substring is any contiguous sequence of characters from index l to r (1 ≤ l ≤ r ≤ n)

**Examples:**
1. Input: n=3, m=1 → Output:4
2. Input: n=5, m=2 → Output:12
3. Input: n=4, m=0 → Output:0

**Your Task:**
Compute the maximum f(s) for n={n}, m={m}. 

**Answer Format Requirements:**
- Return only the integer answer 
- Enclose your final answer with [answer] and [/answer] tags
- Example valid response: [answer]42[/answer]"""

    @staticmethod
    def extract_output(output):
        # 使用非贪婪匹配并支持跨行内容
        matches = re.findall(
            r'\[answer\](.*?)\[/answer\]', 
            output, 
            flags=re.IGNORECASE | re.DOTALL
        )
        if not matches:
            return None
        
        # 提取最后一个答案并清理空白字符
        raw_answer = matches[-1].strip()
        
        # 处理包含逗号分隔的情况
        if ',' in raw_answer:
            raw_answer = raw_answer.replace(',', '')
        
        try:
            return int(raw_answer)
        except (ValueError, TypeError):
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 处理超大规模数值计算
        n = identity['n']
        m = identity['m']
        
        if m == 0:
            return solution == 0
        
        # 初始化总子串数
        total = n * (n + 1) // 2
        
        if m >= n / 2.0:
            expected = total - (n - m)
            return solution == expected
        
        # 分段计算验证
        c = m + 1
        z = n - m
        base, rem = divmod(z, c)
        
        sum_zeros = (
            rem * (base + 1) * (base + 2) // 2 +
            (c - rem) * base * (base + 1) // 2
        )
        expected = total - sum_zeros
        return solution == expected

    @staticmethod
    def calculate_answer(n, m):
        """参考计算方法（用于生成测试答案）"""
        if m == 0:
            return 0
        total = n * (n + 1) // 2
        if m >= n / 2.0:
            return total - (n - m)
        
        c = m + 1
        z = n - m
        base, rem = divmod(z, c)
        return total - (
            rem * (base + 1) * (base + 2) // 2 +
            (c - rem) * base * (base + 1) // 2
        )
