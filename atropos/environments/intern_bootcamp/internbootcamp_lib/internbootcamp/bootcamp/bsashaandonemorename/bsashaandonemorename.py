"""# 

### 谜题描述
Reading books is one of Sasha's passions. Once while he was reading one book, he became acquainted with an unusual character. The character told about himself like that: \"Many are my names in many countries. Mithrandir among the Elves, Tharkûn to the Dwarves, Olórin I was in my youth in the West that is forgotten, in the South Incánus, in the North Gandalf; to the East I go not.\"

And at that moment Sasha thought, how would that character be called in the East? In the East all names are palindromes. A string is a palindrome if it reads the same backward as forward. For example, such strings as \"kazak\", \"oo\" and \"r\" are palindromes, but strings \"abb\" and \"ij\" are not. 

Sasha believed that the hero would be named after one of the gods of the East. As long as there couldn't be two equal names, so in the East people did the following: they wrote the original name as a string on a piece of paper, then cut the paper minimum number of times k, so they got k+1 pieces of paper with substrings of the initial string, and then unite those pieces together to get a new string. Pieces couldn't be turned over, they could be shuffled.

In this way, it's possible to achive a string abcdefg from the string f|de|abc|g using 3 cuts (by swapping papers with substrings f and abc). The string cbadefg can't be received using the same cuts.

More formally, Sasha wants for the given palindrome s find such minimum k, that you can cut this string into k + 1 parts, and then unite them in such a way that the final string will be a palindrome and it won't be equal to the initial string s. It there is no answer, then print \"Impossible\" (without quotes).

Input

The first line contains one string s (1 ≤ |s| ≤ 5 000) — the initial name, which consists only of lowercase Latin letters. It is guaranteed that s is a palindrome.

Output

Print one integer k — the minimum number of cuts needed to get a new name, or \"Impossible\" (without quotes).

Examples

Input


nolon


Output


2


Input


otto


Output


1


Input


qqqq


Output


Impossible


Input


kinnikkinnik


Output


1

Note

In the first example, you can cut the string in those positions: no|l|on, and then unite them as follows on|l|no. It can be shown that there is no solution with one cut.

In the second example, you can cut the string right in the middle, and swap peaces, so you get toot.

In the third example, you can't make a string, that won't be equal to the initial one.

In the fourth example, you can cut the suffix nik and add it to the beginning, so you get nikkinnikkin.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
#! /usr/bin/python

s = raw_input().strip()

a = s[:len(s)/2]

if a == \"\".join([a[0] for _ in range(len(a))]):
	print \"Impossible\"
	exit()

def isPalindrome(s):
	for i in range(len(s)/2):
		if s[i] != s[-1-i]:
			return False
	return True

for i in range(0, len(a) + 1):
	if s != s[i:] + s[:i]:
		if isPalindrome(s[i:] + s[:i]):
			print 1
			exit()

print 2
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import string
import re
from bootcamp import Basebootcamp

class Bsashaandonemorenamebootcamp(Basebootcamp):
    def __init__(self, case_type=None, min_length=3, max_length=20):
        super().__init__()
        self.case_type = case_type
        self.min_length = min_length
        self.max_length = max_length

    def case_generator(self):
        case_type = self.case_type if self.case_type in ['impossible', 'k1', 'k2'] else random.choice(['impossible', 'k1', 'k2'])
        
        while True:
            if case_type == 'impossible':
                s = self._generate_impossible_case()
                if len(set(s)) == 1:
                    return {'s': s}
            elif case_type == 'k1':
                s = self._generate_k1_case()
                if any(self._is_valid_rotation(s, i) for i in range(1, len(s))):
                    return {'s': s}
            else:
                s = self._generate_k2_case()
                if not any(self._is_valid_rotation(s, i) for i in range(1, len(s))):
                    return {'s': s}

    def _generate_impossible_case(self):
        n = random.randint(self.min_length, self.max_length)
        c = random.choice(string.ascii_lowercase)
        return c * n

    def _generate_k1_case(self):
        while True:
            base = ''.join(random.choices(string.ascii_lowercase, k=random.randint(2, self.max_length//2)))
            if base != base[::-1]:
                s = base + base[::-1]
                if s != s[::-1]:
                    continue
                return s

    def _generate_k2_case(self):
        while True:
            s = self._generate_complex_palindrome()
            if all(c == s[0] for c in s):
                continue
            return s

    def _generate_complex_palindrome(self):
        """生成无法通过简单旋转得到不同解的复杂回文"""
        while True:
            left = []
            for _ in range(random.randint(2, self.max_length//2)):
                candidates = [c for c in string.ascii_lowercase if not left or c != left[-1]]
                left.append(random.choice(candidates))
            left_str = ''.join(left)
            right_str = left_str[::-1]
            
            if random.choice([True, False]) and len(left_str) > 1:
                mid = random.choice(string.ascii_lowercase.replace(left_str[-1], ''))
            else:
                mid = ''
            s = left_str + mid + right_str
            
            if not any(self._is_valid_rotation(s, i) for i in range(1, len(s))):
                return s

    @staticmethod
    def _is_valid_rotation(s, i):
        rotated = s[i:] + s[:i]
        return rotated != s and rotated == rotated[::-1]

    @staticmethod
    def prompt_func(question_case):
        s = question_case['s']
        return f"""根据神秘东方命名规则，你需要将一个回文字符串{s}分割重组为不同的回文：
1. 只能通过切割并重组子串来形成新回文
2. 新回文必须与原字符串不同
3. 需要找到最小切割次数k

请按照以下格式给出答案：[answer]答案[/answer]
示例：
输入：nolon → [answer]2[/answer]
输入：otto → [answer]1[/answer]
输入：qqqq → [answer]Impossible[/answer]
"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL|re.IGNORECASE)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        s = identity['s']
        solution = str(solution).strip().lower()
        
        # Impossible条件判断
        if len(set(s)) == 1:
            return solution == "impossible"
        
        # 检查是否存在k=1的解
        has_k1_solution = any(cls._is_valid_rotation(s, i) for i in range(1, len(s)))
        
        # 正确答案逻辑
        if has_k1_solution:
            return solution == "1"
        else:
            # 当原始字符串为双字符回文时特殊处理（如aa）
            return solution in ("2", "impossible") if len(s) == 2 else solution == "2"
