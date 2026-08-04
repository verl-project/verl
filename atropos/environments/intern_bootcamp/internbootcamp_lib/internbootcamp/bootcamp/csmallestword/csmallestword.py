"""# 

### 谜题描述
IA has so many colorful magnets on her fridge! Exactly one letter is written on each magnet, 'a' or 'b'. She loves to play with them, placing all magnets in a row. However, the girl is quickly bored and usually thinks how to make her entertainment more interesting.

Today, when IA looked at the fridge, she noticed that the word formed by magnets is really messy. \"It would look much better when I'll swap some of them!\" — thought the girl — \"but how to do it?\". After a while, she got an idea. IA will look at all prefixes with lengths from 1 to the length of the word and for each prefix she will either reverse this prefix or leave it as it is. She will consider the prefixes in the fixed order: from the shortest to the largest. She wants to get the lexicographically smallest possible word after she considers all prefixes. Can you help her, telling which prefixes should be chosen for reversing?

A string a is lexicographically smaller than a string b if and only if one of the following holds:

  * a is a prefix of b, but a ≠ b;
  * in the first position where a and b differ, the string a has a letter that appears earlier in the alphabet than the corresponding letter in b.

Input

The first and the only line contains a string s (1 ≤ |s| ≤ 1000), describing the initial string formed by magnets. The string s consists only of characters 'a' and 'b'.

Output

Output exactly |s| integers. If IA should reverse the i-th prefix (that is, the substring from 1 to i), the i-th integer should be equal to 1, and it should be equal to 0 otherwise.

If there are multiple possible sequences leading to the optimal answer, print any of them.

Examples

Input

bbab


Output

0 1 1 0


Input

aaaaa


Output

1 0 0 0 1

Note

In the first example, IA can reverse the second and the third prefix and get a string \"abbb\". She cannot get better result, since it is also lexicographically smallest string obtainable by permuting characters of the initial string.

In the second example, she can reverse any subset of prefixes — all letters are 'a'.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
s = raw_input().strip()
n = len(s)

out = ['0' for i in xrange(n)]

i = 0
while i < n:
    while i < n and s[i] != 'a': i += 1
    while i < n and s[i] == 'a': i += 1

    if i <= n: out[i - 1] = '1'
    else: break

    while i < n and s[i] != 'b': i += 1
    while i < n and s[i] == 'b': i += 1

    if i < n: out[i - 1] = '1'

print ' '.join(out)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class Csmallestwordbootcamp(Basebootcamp):
    def __init__(self, min_length=4, max_length=10, seed=None, alternate_prob=0.3):
        self.min_length = min_length
        self.max_length = max_length
        self.alternate_prob = alternate_prob
        self.rng = random.Random(seed)
    
    @staticmethod
    def generate_reversal_sequence(s):
        """使用贪心算法生成最优反转序列"""
        s_list = list(s)
        n = len(s_list)
        reversal_seq = [0] * n
        current = s_list.copy()
        
        for i in range(1, n+1):
            # 生成反转后的候选字符串
            candidate = current[:i][::-1] + current[i:]
            if candidate < current:
                reversal_seq[i-1] = 1
                current = candidate
        return reversal_seq
    
    @staticmethod
    def apply_reversals(s, reversal_sequence):
        """应用反转序列到原始字符串"""
        current = list(s)
        for idx, flag in enumerate(reversal_sequence, 1):
            if flag:
                current[:idx] = current[:idx][::-1]
        return ''.join(current)
    
    def _generate_string(self):
        """生成多样化的测试字符串"""
        length = self.rng.randint(self.min_length, self.max_length)
        if self.rng.random() < self.alternate_prob:
            base = ['a', 'b'] * (length//2 + 1)
            s = base[:length]
            if self.rng.random() < 0.5:
                s = s[::-1]
        else:
            s = self.rng.choices(['a', 'b'], k=length, weights=[0.5,0.5])
        return ''.join(s)
    
    def case_generator(self):
        while True:
            s = self._generate_string()
            reversal_seq = self.generate_reversal_sequence(s)
            optimal_str = self.apply_reversals(s, reversal_seq)
            
            # 确保该案例有有效操作（允许全相同字符的情况）
            if any(reversal_seq) or all(c == s[0] for c in s):
                return {
                    's': s,
                    'reversal_seq': reversal_seq,
                    'optimal_str': optimal_str
                }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        s = question_case['s']
        n = len(s)
        return f"""IA needs to reverse prefixes to get the lexicographically smallest string. 

Initial string: {s}
Rules:
1. Process prefixes from length 1 to {n} in order
2. For each prefix, output 1 to reverse or 0 to leave
3. The final string must be the smallest possible in dictionary order

Example valid format: [answer]0 1 1 0[/answer]

What's the reversal sequence for this string?"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.I|re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            solution = list(map(int, last_match.split()))
            if not all(v in (0, 1) for v in solution):
                return None
            return solution
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if len(solution) != len(identity['s']):
            return False
        if any(v not in (0, 1) for v in solution):
            return False
        
        # 计算用户答案生成的字符串
        try:
            result = cls.apply_reversals(identity['s'], solution)
        except:
            return False
        
        # 验证是否为最优解
        return result == identity['optimal_str']
