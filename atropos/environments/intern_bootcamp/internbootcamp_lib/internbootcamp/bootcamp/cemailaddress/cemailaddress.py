"""# 

### 谜题描述
Sometimes one has to spell email addresses over the phone. Then one usually pronounces a dot as dot, an at sign as at. As a result, we get something like vasyaatgmaildotcom. Your task is to transform it into a proper email address (vasya@gmail.com). 

It is known that a proper email address contains only such symbols as . @ and lower-case Latin letters, doesn't start with and doesn't end with a dot. Also, a proper email address doesn't start with and doesn't end with an at sign. Moreover, an email address contains exactly one such symbol as @, yet may contain any number (possible, zero) of dots. 

You have to carry out a series of replacements so that the length of the result was as short as possible and it was a proper email address. If the lengths are equal, you should print the lexicographically minimal result. 

Overall, two variants of replacement are possible: dot can be replaced by a dot, at can be replaced by an at. 

Input

The first line contains the email address description. It is guaranteed that that is a proper email address with all the dots replaced by dot an the at signs replaced by at. The line is not empty and its length does not exceed 100 symbols.

Output

Print the shortest email address, from which the given line could be made by the described above replacements. If there are several solutions to that problem, print the lexicographically minimal one (the lexicographical comparison of the lines are implemented with an operator < in modern programming languages).

In the ASCII table the symbols go in this order: . @ ab...z

Examples

Input

vasyaatgmaildotcom


Output

vasya@gmail.com


Input

dotdotdotatdotdotat


Output

dot..@..at


Input

aatt


Output

a@t

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys

file = sys.stdin
#file = open(\"test\", \"r\")
line = file.readline().rstrip()
ori = line
line = ori[0] + line[1: len(line) - 1].replace(\"at\", \"@\", 1)+ ori[-1]
line = ori[0] + line[1: len(line) - 1].replace(\"dot\", \".\")+ ori[-1]
print line
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import string
import re
from bootcamp import Basebootcamp

class Cemailaddressbootcamp(Basebootcamp):
    def __init__(self, min_local_length=3, max_local_length=10, min_domain_length=3, max_domain_length=10):
        self.min_local_length = min_local_length
        self.max_local_length = max_local_length
        self.min_domain_length = min_domain_length
        self.max_domain_length = max_domain_length
    
    def _generate_part(self, part_type):
        """生成符合规范的邮箱部分（local或domain）"""
        while True:
            length = random.randint(self.min_local_length if part_type == "local" else self.min_domain_length,
                                   self.max_local_length if part_type == "local" else self.max_domain_length)
            
            # 首尾必须是小写字母
            first = random.choice(string.ascii_lowercase)
            last = random.choice(string.ascii_lowercase)
            
            # 中间字符生成（避免连续点）
            middle = []
            for _ in range(length-2):
                choices = string.ascii_lowercase
                if middle and middle[-1] != '.':
                    choices += '.'
                middle.append(random.choice(choices))
            
            # 拼接并验证
            candidate = first + ''.join(middle) + last
            if '.' in candidate:
                candidate = re.sub(r'\.{2,}', '.', candidate)  # 移除连续点
            if (candidate[0] not in ('.', '@') and 
                candidate[-1] not in ('.', '@') and 
                '@' not in candidate):
                return candidate
    
    def _generate_email(self):
        """生成合法邮箱并确保对应输入字符串具有唯一最优解"""
        while True:
            local = self._generate_part("local")
            domain = self._generate_part("domain")
            email = f"{local}@{domain}"
            
            # 生成输入字符串并验证唯一最优解
            input_str = (
                email[0] +
                email[1:-1].replace('@', 'at').replace('.', 'dot') +
                email[-1]
            )
            
            # 确保输入字符串中仅包含一个at（对应邮箱中的@）
            if input_str.count('at') == 1 and 'at' not in [input_str[:2], input_str[-2:]]:
                return email, input_str
    
    def case_generator(self):
        email, input_str = self._generate_email()
        return {
            'input': input_str,
            'answer': email
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        input_str = question_case['input']
        return f"""Convert the phone-spelled email address to proper format. Rules:
1. Replace 'at' with @ and 'dot' with . where possible
2. Result must be valid (exactly one @, no invalid start/end)
3. Choose the SHORTEST possible result
4. If same length, choose lexicographically smallest

Input: {input_str}
Put your final answer within [answer][/answer] tags."""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['answer']
