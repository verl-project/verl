"""# 

### 谜题描述
Constanze is the smartest girl in her village but she has bad eyesight.

One day, she was able to invent an incredible machine! When you pronounce letters, the machine will inscribe them onto a piece of paper. For example, if you pronounce 'c', 'o', 'd', and 'e' in that order, then the machine will inscribe \"code\" onto the paper. Thanks to this machine, she can finally write messages without using her glasses.

However, her dumb friend Akko decided to play a prank on her. Akko tinkered with the machine so that if you pronounce 'w', it will inscribe \"uu\" instead of \"w\", and if you pronounce 'm', it will inscribe \"nn\" instead of \"m\"! Since Constanze had bad eyesight, she was not able to realize what Akko did.

The rest of the letters behave the same as before: if you pronounce any letter besides 'w' and 'm', the machine will just inscribe it onto a piece of paper.

The next day, I received a letter in my mailbox. I can't understand it so I think it's either just some gibberish from Akko, or Constanze made it using her machine. But since I know what Akko did, I can just list down all possible strings that Constanze's machine would have turned into the message I got and see if anything makes sense.

But I need to know how much paper I will need, and that's why I'm asking you for help. Tell me the number of strings that Constanze's machine would've turned into the message I got.

But since this number can be quite large, tell me instead its remainder when divided by 10^9+7.

If there are no strings that Constanze's machine would've turned into the message I got, then print 0.

Input

Input consists of a single line containing a string s (1 ≤ |s| ≤ 10^5) — the received message. s contains only lowercase Latin letters.

Output

Print a single integer — the number of strings that Constanze's machine would've turned into the message s, modulo 10^9+7.

Examples

Input


ouuokarinn


Output


4


Input


banana


Output


1


Input


nnn


Output


3


Input


amanda


Output


0

Note

For the first example, the candidate strings are the following: \"ouuokarinn\", \"ouuokarim\", \"owokarim\", and \"owokarinn\".

For the second example, there is only one: \"banana\".

For the third example, the candidate strings are the following: \"nm\", \"mn\" and \"nnn\".

For the last example, there are no candidate strings that the machine can turn into \"amanda\", since the machine won't inscribe 'm'.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
s = raw_input()
flag = True
maxn = int(1e5+5)
fib = [0]*maxn
fib[0] = 1
fib[1] = 1
fib[2] = 2
fib[3] = 3
mod  = int(1e9+7)
for i in range(4,maxn):
	fib[i] = (fib[i-1]+fib[i-2])%mod


for i in range(len(s)):
	if s[i] == \"m\" or s[i] == \"w\":
		flag = False

if not flag:
	print 0
else:
	i = 0
	cnt = 0
	ans = 1
	while i<len(s):
		if s[i]!=\"u\" and s[i]!=\"n\":

			ans = (ans*fib[cnt])%mod
			cnt = 0
		elif s[i] == s[i-1]:
			cnt += 1
		else:

			ans = (ans*fib[cnt])%mod
			cnt = 1
		i += 1

	ans = (ans*fib[cnt])%mod
	print ans
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

MOD = 10**9 + 7
MAX_FIB_LENGTH = 10**5 + 10  # 覆盖题目最大输入长度

class Cconstanzesmachinebootcamp(Basebootcamp):
    fib = [1, 1]  # 类级别预计算斐波那契数列
    for i in range(2, MAX_FIB_LENGTH):
        fib.append((fib[-1] + fib[-2]) % MOD)
    
    def __init__(self, min_length=3, max_length=100, prob_wm=0.3, prob_error=0.2):
        self.min_length = max(min_length, 1)
        self.max_length = min(max_length, 1000)  # 控制生成规模
        self.prob_wm = prob_wm
        self.prob_error = prob_error

    def case_generator(self):
        while True:
            # Phase 1: 生成合法原始字符串
            original = []
            required_length = random.randint(self.min_length, self.max_length)
            
            while len(original) < required_length:
                # 优先注入特殊字符（w/m）
                if random.random() < self.prob_wm:
                    c = random.choice(['w', 'm'])
                    original.append(c)
                    continue
                
                # 生成常规字符（允许u/n但不连续）
                valid_chars = [chr(ord('a') + i) for i in range(26)]
                if original:
                    last_char = original[-1]
                    if last_char == 'u' and 'u' in valid_chars:
                        valid_chars.remove('u')
                    elif last_char == 'n' and 'n' in valid_chars:
                        valid_chars.remove('n')
                
                c = random.choice(valid_chars)
                original.append(c)
            
            # 转换为受控字符串
            s = ''.join(original).replace('w', 'uu').replace('m', 'nn')
            
            # Phase 2: 注入错误字符（如果需要）
            if random.random() < self.prob_error:
                error_pos = random.randint(0, len(s)-1)
                s = s[:error_pos] + random.choice(['m', 'w']) + s[error_pos+1:]
            
            # 合法性检查
            if self.min_length <= len(s) <= self.max_length:
                return {'s': s}

    @staticmethod
    def prompt_func(question_case):
        s = question_case['s']
        return (
            f"Analyze the string '{s}' according to Cconstanzesmachine's machine rules:\n"
            "1. Original characters are transformed as: w→uu, m→nn\n"
            "2. Input containing 'w'/'m' is invalid (output 0)\n"
            "3. Count possible source strings modulo 1e9+7\n\n"
            "Format: [answer]NUMBER[/answer]"
        )

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output, re.DOTALL)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            # 直接计算结果进行验证
            s = identity['s']
            if 'm' in s or 'w' in s:
                return int(solution) == 0
            
            result = 1
            i = 0
            n = len(s)
            while i < n:
                if s[i] not in ('u', 'n'):
                    i += 1
                    continue
                
                j = i
                while j < n and s[j] == s[i]:
                    j += 1
                
                block_len = j - i
                if block_len >= 1:
                    result = (result * cls.fib[block_len]) % MOD
                i = j
            
            expected = result % MOD
            return int(solution) == expected
        except:
            return False
