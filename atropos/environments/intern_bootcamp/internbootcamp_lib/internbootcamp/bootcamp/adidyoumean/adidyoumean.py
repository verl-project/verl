"""# 

### 谜题描述
Beroffice text editor has a wide range of features that help working with text. One of the features is an automatic search for typos and suggestions of how to fix them.

Beroffice works only with small English letters (i.e. with 26 letters from a to z). Beroffice thinks that a word is typed with a typo if there are three or more consonants in a row in the word. The only exception is that if the block of consonants has all letters the same, then this block (even if its length is greater than three) is not considered a typo. Formally, a word is typed with a typo if there is a block of not less that three consonants in a row, and there are at least two different letters in this block.

For example:

  * the following words have typos: \"hellno\", \"hackcerrs\" and \"backtothefutttture\"; 
  * the following words don't have typos: \"helllllooooo\", \"tobeornottobe\" and \"oooooo\". 



When Beroffice editor finds a word with a typo, it inserts as little as possible number of spaces in this word (dividing it into several words) in such a way that each of the resulting words is typed without any typos.

Implement this feature of Beroffice editor. Consider the following letters as the only vowels: 'a', 'e', 'i', 'o' and 'u'. All the other letters are consonants in this problem.

Input

The only line contains a non-empty word consisting of small English letters. The length of the word is between 1 and 3000 letters.

Output

Print the given word without any changes if there are no typos.

If there is at least one typo in the word, insert the minimum number of spaces into the word so that each of the resulting words doesn't have any typos. If there are multiple solutions, print any of them.

Examples

Input

hellno


Output

hell no 


Input

abacaba


Output

abacaba 


Input

asdfasdf


Output

asd fasd f 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
s = raw_input().strip()
w = []
t = []
con = set()
k = 0
for c in s:
    if c in 'aieuo':
        t.append(c)
        con = set()
        k = 0
    else:
        if k >= 2 and (len(con) > 1 or c not in con):
            w.append(''.join(t))
            t = []
            con = set()
            k = 0
        t.append(c)
        con.add(c)
        k += 1
w.append(''.join(t))
print ' '.join(w)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
import string
from bootcamp import Basebootcamp

class Adidyoumeanbootcamp(Basebootcamp):
    def __init__(self, max_length=3000):
        self.max_length = max_length
    
    def _generate_valid_part(self, vowels, consonants, min_len=1, max_len=8):
        """生成合法字符串片段（不会触发typo）"""
        part_type = random.choice(['vowel', 'single_consonant', 'mixed_consonant'])
        
        if part_type == 'vowel':
            return ''.join(random.choices(vowels, k=random.randint(min_len, max_len)))
        elif part_type == 'single_consonant':
            c = random.choice(consonants)
            return c * random.randint(1, max_len)
        else:  # 混合辅音但长度≤2
            length = random.randint(1, 2)
            return ''.join(random.choices(consonants, k=length))
    
    def _generate_typo_block(self, consonants):
        """生成至少3个辅音且至少2个不同字符的块"""
        while True:
            length = random.randint(3, 6)
            candidates = random.choices(consonants, k=length)
            if len(set(candidates)) >= 2:
                return ''.join(candidates)
    
    def case_generator(self):
        # 50%概率生成合法或非法案例
        case_type = random.choice(['valid', 'invalid'])
        vowels = ['a', 'e', 'i', 'o', 'u']
        consonants = list(set(string.ascii_lowercase) - set(vowels))
        
        if case_type == 'valid':
            # 生成完全合法的字符串
            parts = []
            while len(''.join(parts)) < self.max_length:
                parts.append(self._generate_valid_part(vowels, consonants))
                if random.random() < 0.3:  # 30%概率插入元音间隔
                    parts.append(random.choice(vowels))
            s = ''.join(parts)[:self.max_length]
        else:
            # 生成至少包含一个typo的字符串
            core = self._generate_typo_block(consonants)
            prefix = self._generate_valid_part(vowels, consonants)
            suffix = self._generate_valid_part(vowels, consonants)
            s = prefix + core + suffix
            s = s[:self.max_length]  # 确保总长度不超限
        
        return {'input': s}
    
    @staticmethod
    def prompt_func(question_case):
        input_str = question_case['input']
        return f"""Adidyoumean文本编辑器会检测单词中的typo。当单词中出现三个及以上连续辅音（非a/e/i/o/u），且其中至少有两个不同字符时视为typo。你需要插入最少的空格分割单词，消除所有typo。

输入：{input_str}

要求：
1. 无typo时保持原样输出
2. 必须使用最少空格数分割
3. 答案可能有多个正确解，任选其一

请将最终答案放在[answer]和[/answer]之间。例如：
[answer]hel lo[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\]\s*(.*?)\s*\[/answer\]', output, re.DOTALL)
        return matches[-1] if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        original = identity['input']
        if not solution or original.replace(' ', '') != solution.replace(' ', ''):
            return False
        
        for part in solution.split():
            if cls._has_typo(part):
                return False
        
        ref_answer = cls._process_via_reference(identity['input'])
        return len(solution.split()) == len(ref_answer.split())
    
    @staticmethod
    def _has_typo(s):
        vowels = {'a', 'e', 'i', 'o', 'u'}
        consonant_count = 0
        seen_chars = set()
        
        for c in s:
            if c not in vowels:
                consonant_count += 1
                seen_chars.add(c)
                if consonant_count >= 3 and len(seen_chars) >= 2:
                    return True
            else:
                consonant_count = 0
                seen_chars.clear()
        return False
    
    @staticmethod
    def _process_via_reference(s):
        w = []
        current_part = []
        consonants = set()
        count = 0
        
        for c in s:
            if c in 'aeiou':
                current_part.append(c)
                consonants = set()
                count = 0
            else:
                if count >= 2 and (len(consonants) > 1 or c not in consonants):
                    w.append(''.join(current_part))
                    current_part = []
                    consonants = set()
                    count = 0
                current_part.append(c)
                consonants.add(c)
                count += 1
        w.append(''.join(current_part))
        return ' '.join(w)
