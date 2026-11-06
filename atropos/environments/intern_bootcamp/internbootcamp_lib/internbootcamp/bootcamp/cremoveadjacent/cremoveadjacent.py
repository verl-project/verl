"""# 

### 谜题描述
You are given a string s consisting of lowercase Latin letters. Let the length of s be |s|. You may perform several operations on this string.

In one operation, you can choose some index i and remove the i-th character of s (s_i) if at least one of its adjacent characters is the previous letter in the Latin alphabet for s_i. For example, the previous letter for b is a, the previous letter for s is r, the letter a has no previous letters. Note that after each removal the length of the string decreases by one. So, the index i should satisfy the condition 1 ≤ i ≤ |s| during each operation.

For the character s_i adjacent characters are s_{i-1} and s_{i+1}. The first and the last characters of s both have only one adjacent character (unless |s| = 1).

Consider the following example. Let s= bacabcab.

  1. During the first move, you can remove the first character s_1= b because s_2= a. Then the string becomes s= acabcab. 
  2. During the second move, you can remove the fifth character s_5= c because s_4= b. Then the string becomes s= acabab. 
  3. During the third move, you can remove the sixth character s_6='b' because s_5= a. Then the string becomes s= acaba. 
  4. During the fourth move, the only character you can remove is s_4= b, because s_3= a (or s_5= a). The string becomes s= acaa and you cannot do anything with it. 



Your task is to find the maximum possible number of characters you can remove if you choose the sequence of operations optimally.

Input

The first line of the input contains one integer |s| (1 ≤ |s| ≤ 100) — the length of s.

The second line of the input contains one string s consisting of |s| lowercase Latin letters.

Output

Print one integer — the maximum possible number of characters you can remove if you choose the sequence of moves optimally.

Examples

Input


8
bacabcab


Output


4


Input


4
bcda


Output


3


Input


6
abbbbb


Output


5

Note

The first example is described in the problem statement. Note that the sequence of moves provided in the statement is not the only, but it can be shown that the maximum possible answer to this test is 4.

In the second example, you can remove all but one character of s. The only possible answer follows.

  1. During the first move, remove the third character s_3= d, s becomes bca. 
  2. During the second move, remove the second character s_2= c, s becomes ba. 
  3. And during the third move, remove the first character s_1= b, s becomes a. 

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
import sys
input = sys.stdin.readline

n = int(input())
s = input().strip()
ans = 0
while True:
    ansn = ans
    for i in xrange(25):
        while True:
            new_s = ''
            ansnn = ans
            for j in xrange(len(s)):
                if s[j] == chr(122-i) and ((j>0 and s[j-1] == chr(122-i-1)) or (j<len(s)-1 and s[j+1]  == chr(122-i-1))):
                    ans += 1
                else:
                    new_s += s[j]
            s = new_s
            if ansnn == ans:
                break
    if ansn == ans:
        break
print(ans)
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import string
import re
from bootcamp import Basebootcamp

class Cremoveadjacentbootcamp(Basebootcamp):
    def __init__(self, min_length=1, max_length=100):
        self.min_length = min_length
        self.max_length = max_length

    def case_generator(self):
        """生成包含随机小写字母字符串的案例，覆盖全长度范围"""
        n = random.randint(self.min_length, self.max_length)
        s = ''.join(random.choices(string.ascii_lowercase, k=n))
        return {'s': s}

    @staticmethod
    def prompt_func(question_case):
        """生成包含明确格式要求的完整问题描述"""
        s = question_case['s']
        n = len(s)
        return f"""Given a string s of length {n}: {s}
Find the maximum number of removable characters according to the rules:
1. Remove a character if adjacent to its previous Latin letter
2. Choose optimal removal sequence

Put your final answer within [answer][/answer] tags like [answer]4[/answer]"""

    @staticmethod
    def extract_output(output):
        """严格提取最后一个[answer]标签内容"""
        matches = re.findall(r'\[answer\]\s*(\d+)\s*\[/answer\]', output)
        return int(matches[-1]) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """严格验证答案的正確性"""
        return solution == cls.compute_max_removals(identity['s'])

    @staticmethod
    def compute_max_removals(s):
        """精确实现题目要求的贪心算法"""
        ans = 0
        while True:
            update_flag = False
            # 从z到a依次处理每个字符
            for char_code in range(122, 96, -1):
                current_char = chr(char_code)
                if current_char == 'a':
                    continue  # a不可删除
                
                prev_char = chr(char_code-1)
                while True:
                    new_s = []
                    removed = 0
                    for i in range(len(s)):
                        if s[i] == current_char:
                            left_ok = (i > 0 and s[i-1] == prev_char)
                            right_ok = (i < len(s)-1 and s[i+1] == prev_char)
                            if left_ok or right_ok:
                                ans += 1
                                removed += 1
                            else:
                                new_s.append(s[i])
                        else:
                            new_s.append(s[i])
                    
                    if removed > 0:
                        s = ''.join(new_s)
                        update_flag = True
                    else:
                        break
            if not update_flag:
                break
        return ans
