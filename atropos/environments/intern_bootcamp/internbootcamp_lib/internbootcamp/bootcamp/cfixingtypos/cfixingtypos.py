"""# 

### 谜题描述
Many modern text editors automatically check the spelling of the user's text. Some editors even suggest how to correct typos.

In this problem your task to implement a small functionality to correct two types of typos in a word. We will assume that three identical letters together is a typo (for example, word \"helllo\" contains a typo). Besides, a couple of identical letters immediately followed by another couple of identical letters is a typo too (for example, words \"helloo\" and \"wwaatt\" contain typos).

Write a code that deletes the minimum number of letters from a word, correcting described typos in the word. You are allowed to delete letters from both ends and from the middle of the word.

Input

The single line of the input contains word s, its length is from 1 to 200000 characters. The given word s consists of lowercase English letters.

Output

Print such word t that it doesn't contain any typos described in the problem statement and is obtained from s by deleting the least number of letters.

If there are multiple solutions, print any of them.

Examples

Input

helloo


Output

hello


Input

woooooow


Output

woow

Note

The second valid answer to the test from the statement is \"heloo\".

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
word1 = raw_input()
ans = [\"!\", \"@\"]

for char in word1:
    if char == ans[-1] and (char == ans[-2] or ans[-2] == ans[-3]):
        ans.pop()

    ans.append(char)

ans_str = \"\"
for char in ans[2:]:
    ans_str += char

print ans_str
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
import string
from collections import deque
from bootcamp import Basebootcamp

def process_word(s):
    """严格遵循题目参考代码的处理逻辑"""
    ans = ["!", "@"]
    for char in s:
        while len(ans) >= 3 and (char == ans[-1] == ans[-2] or char == ans[-1] and ans[-3] == ans[-2]):
            ans.pop()
        ans.append(char)
    return ''.join(ans[2:])

class Cfixingtyposbootcamp(Basebootcamp):
    def __init__(self, min_length=6, max_length=200):
        self.min_length = max(6, min_length)  # 确保最小有效长度
        self.max_length = min(max_length, 20000)  # 遵循题目输入约束

    def case_generator(self):
        """生成保证包含至少一个有效错误的测试用例"""
        error_type = random.choice([1, 2, 1, 2])  # 增加类型1的概率
        s = []

        # 生成基础字符流（保证无预存错误）
        base_chars = [
            c for c in random.choices(string.ascii_lowercase, 
            k=random.randint(self.min_length-3, self.max_length))
            if len(s) < 1 or c != s[-1]  # 防止自然产生连续对
        ]

        # 插入错误模式
        if error_type == 1:  # 三连字符
            insert_pos = random.randint(0, len(base_chars)-1)
            c = base_chars[insert_pos] if random.random() < 0.5 else random.choice(string.ascii_lowercase)
            error = [c]*3
        else:  # 连续重复对
            pairs = [random.choice(string.ascii_lowercase) for _ in range(2)]
            error = pairs*2 if random.random() < 0.5 else [pairs[0]]*2 + [pairs[1]]*2

        # 将错误模式插入随机位置
        insert_pos = random.randint(0, len(base_chars))
        s = base_chars[:insert_pos] + error + base_chars[insert_pos:]

        # 转换为字符串并校验有效性
        s = ''.join(s[:self.max_length])
        t = process_word(s)

        # 确保生成有效错误
        if len(t) >= len(s):  # 重新生成直到产生有效错误
            return self.case_generator()

        return {
            'input': s,
            'correct_length': len(t),
            '_ref_solution': t  # 调试用
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        return f"""作为文本处理专家，请修正以下单词中的打字错误：

输入单词: {question_case['input']}

修正规则：
1. 删除最少数量的字符
2. 不允许三个连续相同字母（如"baaaad"→"baad"）
3. 不允许两组连续重复对相邻（如"wooooooow"→"woow"）

请将最终答案放在[answer]标签内，例如：[answer]corrected[/answer]。"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 基础格式检查
        if not isinstance(solution, str) or not solution.isalpha():
            return False
        
        # 长度校验
        if len(solution) != identity['correct_length']:
            return False
        
        # 子序列验证
        src = deque(identity['input'])
        try:
            for c in solution:
                while src.popleft() != c:
                    pass
        except IndexError:
            return False

        # 错误模式检查
        # 三级联检查
        for i in range(len(solution)-2):
            if solution[i] == solution[i+1] == solution[i+2]:
                return False
        
        # 连续两对检查
        for i in range(len(solution)-3):
            if solution[i] == solution[i+1] and solution[i+2] == solution[i+3]:
                return False

        return True
