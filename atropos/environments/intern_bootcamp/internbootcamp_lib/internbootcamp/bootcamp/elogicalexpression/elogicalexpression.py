"""# 

### 谜题描述
You are given a boolean function of three variables which is defined by its truth table. You need to find an expression of minimum length that equals to this function. The expression may consist of: 

  * Operation AND ('&', ASCII code 38) 
  * Operation OR ('|', ASCII code 124) 
  * Operation NOT ('!', ASCII code 33) 
  * Variables x, y and z (ASCII codes 120-122) 
  * Parentheses ('(', ASCII code 40, and ')', ASCII code 41) 



If more than one expression of minimum length exists, you should find the lexicographically smallest one.

Operations have standard priority. NOT has the highest priority, then AND goes, and OR has the lowest priority. The expression should satisfy the following grammar:

E ::= E '|' T | T

T ::= T '&' F | F

F ::= '!' F | '(' E ')' | 'x' | 'y' | 'z'

Input

The first line contains one integer n — the number of functions in the input (1 ≤ n ≤ 10 000).

The following n lines contain descriptions of functions, the i-th of them contains a string of length 8 that consists of digits 0 and 1 — the truth table of the i-th function. The digit on position j (0 ≤ j < 8) equals to the value of the function in case of <image>, <image> and <image>.

Output

You should output n lines, the i-th line should contain the expression of minimum length which equals to the i-th function. If there is more than one such expression, output the lexicographically smallest of them. Expressions should satisfy the given grammar and shouldn't contain white spaces.

Example

Input

4
00110011
00000111
11110000
00011111


Output

y
(y|z)&amp;x
!x
x|y&amp;z

Note

The truth table for the second function:

<image>

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
t = input()
ans = ['!x&x', 'x&y&z', '!z&x&y', 'x&y', '!y&x&z', 'x&z', '!y&x&z|!z&x&y', '(y|z)&x', '!y&!z&x', '!y&!z&x|x&y&z', '!z&x', '!z&x|x&y', '!y&x', '!y&x|x&z', '!(y&z)&x', 'x', '!x&y&z', 'y&z', '!x&y&z|!z&x&y', '(x|z)&y', '!x&y&z|!y&x&z', '(x|y)&z', '!x&y&z|!y&x&z|!z&x&y', '(x|y)&z|x&y', '!x&y&z|!y&!z&x', '!y&!z&x|y&z', '!x&y&z|!z&x', '!z&x|y&z', '!x&y&z|!y&x', '!y&x|y&z', '!(y&z)&x|!x&y&z', 'x|y&z', '!x&!z&y', '!x&!z&y|x&y&z', '!z&y', '!z&y|x&y', '!x&!z&y|!y&x&z', '!x&!z&y|x&z', '!y&x&z|!z&y', '!z&y|x&z', '!(!x&!y|x&y|z)', '!(!x&!y|x&y|z)|x&y&z', '!z&(x|y)', '!z&(x|y)|x&y', '!x&!z&y|!y&x', '!x&!z&y|!y&x|x&z', '!y&x|!z&y', '!z&y|x', '!x&y', '!x&y|y&z', '!(x&z)&y', 'y', '!x&y|!y&x&z', '!x&y|x&z', '!(x&z)&y|!y&x&z', 'x&z|y', '!x&y|!y&!z&x', '!x&y|!y&!z&x|y&z', '!x&y|!z&x', '!z&x|y', '!x&y|!y&x', '!x&y|!y&x|x&z', '!(x&z)&y|!y&x', 'x|y', '!x&!y&z', '!x&!y&z|x&y&z', '!x&!y&z|!z&x&y', '!x&!y&z|x&y', '!y&z', '!y&z|x&z', '!y&z|!z&x&y', '!y&z|x&y', '!(!x&!z|x&z|y)', '!(!x&!z|x&z|y)|x&y&z', '!x&!y&z|!z&x', '!x&!y&z|!z&x|x&y', '!y&(x|z)', '!y&(x|z)|x&z', '!y&z|!z&x', '!y&z|x', '!x&z', '!x&z|y&z', '!x&z|!z&x&y', '!x&z|x&y', '!(x&y)&z', 'z', '!(x&y)&z|!z&x&y', 'x&y|z', '!x&z|!y&!z&x', '!x&z|!y&!z&x|y&z', '!x&z|!z&x', '!x&z|!z&x|x&y', '!x&z|!y&x', '!y&x|z', '!(x&y)&z|!z&x', 'x|z', '!(!y&!z|x|y&z)', '!(!y&!z|x|y&z)|x&y&z', '!x&!y&z|!z&y', '!x&!y&z|!z&y|x&y', '!x&!z&y|!y&z', '!x&!z&y|!y&z|x&z', '!y&z|!z&y', '!y&z|!z&y|x&y', '!(!x&!y|x&y|z)|!x&!y&z', '!(!x&!y|x&y|z)|!x&!y&z|x&y&z', '!x&!y&z|!z&(x|y)', '!x&!y&z|!z&(x|y)|x&y', '!x&!z&y|!y&(x|z)', '!x&!z&y|!y&(x|z)|x&z', '!y&(x|z)|!z&y', '!y&z|!z&y|x', '!x&(y|z)', '!x&(y|z)|y&z', '!x&z|!z&y', '!x&z|y', '!x&y|!y&z', '!x&y|z', '!(x&y)&z|!z&y', 'y|z', '!x&(y|z)|!y&!z&x', '!x&(y|z)|!y&!z&x|y&z', '!x&(y|z)|!z&x', '!x&z|!z&x|y', '!x&(y|z)|!y&x', '!x&y|!y&x|z', '!x&y|!y&z|!z&x', 'x|y|z', '!(x|y|z)', '!(x|y|z)|x&y&z', '!(!x&y|!y&x|z)', '!(x|y|z)|x&y', '!(!x&z|!z&x|y)', '!(x|y|z)|x&z', '!(!x&y|!y&x|z)|!y&x&z', '!(x|y|z)|(y|z)&x', '!y&!z', '!y&!z|x&y&z', '!(!x&y|z)', '!y&!z|x&y', '!(!x&z|y)', '!y&!z|x&z', '!(!x&y|z)|!y&x', '!y&!z|x', '!(!y&z|!z&y|x)', '!(x|y|z)|y&z', '!(!x&y|!y&x|z)|!x&y&z', '!(x|y|z)|(x|z)&y', '!(!x&z|!z&x|y)|!x&y&z', '!(x|y|z)|(x|y)&z', '!(!x&y|!y&x|z)|!x&y&z|!y&x&z', '!(x|y|z)|(x|y)&z|x&y', '!x&y&z|!y&!z', '!y&!z|y&z', '!(!x&y|z)|!x&y&z', '!(!x&y|z)|y&z', '!(!x&z|y)|!x&y&z', '!(!x&z|y)|y&z', '!(!x&y|z)|!x&y&z|!y&x', '!y&!z|x|y&z', '!x&!z', '!x&!z|x&y&z', '!(!y&x|z)', '!x&!z|x&y', '!x&!z|!y&x&z', '!x&!z|x&z', '!(!y&x|z)|!y&x&z', '!(!y&x|z)|x&z', '!(x&y|z)', '!(x&y|z)|x&y&z', '!z', '!z|x&y', '!x&!z|!y&x', '!(x&y|z)|x&z', '!y&x|!z', '!z|x', '!(!y&z|x)', '!x&!z|y&z', '!(!y&x|z)|!x&y', '!x&!z|y', '!(!y&z|x)|!y&x&z', '!(!y&z|x)|x&z', '!(!y&x|z)|!x&y|!y&x&z', '!x&!z|x&z|y', '!x&y|!y&!z', '!(x&y|z)|y&z', '!x&y|!z', '!z|y', '!(!x&!y&z|x&y)', '!x&!z|!y&x|y&z', '!x&y|!y&x|!z', '!z|x|y', '!x&!y', '!x&!y|x&y&z', '!x&!y|!z&x&y', '!x&!y|x&y', '!(!z&x|y)', '!x&!y|x&z', '!(!z&x|y)|!z&x&y', '!(!z&x|y)|x&y', '!(x&z|y)', '!(x&z|y)|x&y&z', '!x&!y|!z&x', '!(x&z|y)|x&y', '!y', '!y|x&z', '!y|!z&x', '!y|x', '!(!z&y|x)', '!x&!y|y&z', '!(!z&y|x)|!z&x&y', '!(!z&y|x)|x&y', '!(!z&x|y)|!x&z', '!x&!y|z', '!(!z&x|y)|!x&z|!z&x&y', '!x&!y|x&y|z', '!x&z|!y&!z', '!(x&z|y)|y&z', '!(!x&!z&y|x&z)', '!x&!y|!z&x|y&z', '!x&z|!y', '!y|z', '!x&z|!y|!z&x', '!y|x|z', '!(x|y&z)', '!(x|y&z)|x&y&z', '!x&!y|!z&y', '!(x|y&z)|x&y', '!x&!z|!y&z', '!(x|y&z)|x&z', '!(!y&!z&x|y&z)', '!x&!y|!z&y|x&z', '!((x|y)&z|x&y)', '!((x|y)&z|x&y)|x&y&z', '!x&!y|!z', '!x&!y|!z|x&y', '!x&!z|!y', '!x&!z|!y|x&z', '!y|!z', '!y|!z|x', '!x', '!x|y&z', '!x|!z&y', '!x|y', '!x|!y&z', '!x|z', '!x|!y&z|!z&y', '!x|y|z', '!x|!y&!z', '!x|!y&!z|y&z', '!x|!z', '!x|!z|y', '!x|!y', '!x|!y|z', '!(x&y&z)', '!x|x']
for tt in range(t):
	s = raw_input()
	print ans[eval('0b' + s)]
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import re
import html

class Elogicalexpressionbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__(**params)
    
    def case_generator(self):
        ans = self._get_ans_list()
        import random
        index = random.randint(0, len(ans)-1)
        truth_table = bin(index)[2:].zfill(8)
        return {'truth_table': truth_table}
    
    @staticmethod
    def _get_ans_list():
        return [
            '!x&x', 'x&y&z', '!z&x&y', 'x&y', '!y&x&z', 'x&z', '!y&x&z|!z&x&y', '(y|z)&x', '!y&!z&x', '!y&!z&x|x&y&z',
            '!z&x', '!z&x|x&y', '!y&x', '!y&x|x&z', '!(y&z)&x', 'x', '!x&y&z', 'y&z', '!x&y&z|!z&x&y', '(x|z)&y',
            '!x&y&z|!y&x&z', '(x|y)&z', '!x&y&z|!y&x&z|!z&x&y', '(x|y)&z|x&y', '!x&y&z|!y&!z&x', '!y&!z&x|y&z',
            '!x&y&z|!z&x', '!z&x|y&z', '!x&y&z|!y&x', '!y&x|y&z', '!(y&z)&x|!x&y&z', 'x|y&z', '!x&!z&y',
            '!x&!z&y|x&y&z', '!z&y', '!z&y|x&y', '!x&!z&y|!y&x&z', '!x&!z&y|x&z', '!y&x&z|!z&y', '!z&y|x&z',
            '!(!x&!y|x&y|z)', '!(!x&!y|x&y|z)|x&y&z', '!z&(x|y)', '!z&(x|y)|x&y', '!x&!z&y|!y&x',
            '!x&!z&y|!y&x|x&z', '!y&x|!z&y', '!z&y|x', '!x&y', '!x&y|y&z', '!(x&z)&y', 'y', '!x&y|!y&x&z',
            '!x&y|x&z', '!(x&z)&y|!y&x&z', 'x&z|y', '!x&y|!y&!z&x', '!x&y|!y&!z&x|y&z', '!x&y|!z&x', '!z&x|y',
            '!x&y|!y&x', '!x&y|!y&x|x&z', '!(x&z)&y|!y&x', 'x|y', '!x&!y&z', '!x&!y&z|x&y&z', '!x&!y&z|!z&x&y',
            '!x&!y&z|x&y', '!y&z', '!y&z|x&z', '!y&z|!z&x&y', '!y&z|x&y', '!(!x&!z|x&z|y)', '!(!x&!z|x&z|y)|x&y&z',
            '!x&!y&z|!z&x', '!x&!y&z|!z&x|x&y', '!y&(x|z)', '!y&(x|z)|x&z', '!y&z|!z&x', '!y&z|x', '!x&z',
            '!x&z|y&z', '!x&z|!z&x&y', '!x&z|x&y', '!(x&y)&z', 'z', '!(x&y)&z|!z&x&y', 'x&y|z', '!x&z|!y&!z&x',
            '!x&z|!y&!z&x|y&z', '!x&z|!z&x', '!x&z|!z&x|x&y', '!x&z|!y&x', '!y&x|z', '!(x&y)&z|!z&x', 'x|z',
            '!(!y&!z|x|y&z)', '!(!y&!z|x|y&z)|x&y&z', '!x&!y&z|!z&y', '!x&!y&z|!z&y|x&y', '!x&!z&y|!y&z',
            '!x&!z&y|!y&z|x&z', '!y&z|!z&y', '!y&z|!z&y|x&y', '!(!x&!y|x&y|z)|!x&!y&z', '!(!x&!y|x&y|z)|!x&!y&z|x&y&z',
            '!x&!y&z|!z&(x|y)', '!x&!y&z|!z&(x|y)|x&y', '!x&!z&y|!y&(x|z)', '!x&!z&y|!y&(x|z)|x&z', '!y&(x|z)|!z&y',
            '!y&z|!z&y|x', '!x&(y|z)', '!x&(y|z)|y&z', '!x&z|!z&y', '!x&z|y', '!x&y|!y&z', '!x&y|z', '!(x&y)&z|!z&y',
            'y|z', '!x&(y|z)|!y&!z&x', '!x&(y|z)|!y&!z&x|y&z', '!x&(y|z)|!z&x', '!x&z|!z&x|y', '!x&(y|z)|!y&x',
            '!x&y|!y&x|z', '!x&y|!y&z|!z&x', 'x|y|z', '!(x|y|z)', '!(x|y|z)|x&y&z', '!(!x&y|!y&x|z)', '!(x|y|z)|x&y',
            '!(!x&z|!z&x|y)', '!(x|y|z)|x&z', '!(!x&y|!y&x|z)|!y&x&z', '!(x|y|z)|(y|z)&x', '!y&!z', '!y&!z|x&y&z',
            '!(!x&y|z)', '!y&!z|x&y', '!(!x&z|y)', '!y&!z|x&z', '!(!x&y|z)|!y&x', '!y&!z|x', '!(!y&z|!z&y|x)',
            '!(x|y|z)|y&z', '!(!x&y|!y&x|z)|!x&y&z', '!(x|y|z)|(x|z)&y', '!(!x&z|!z&x|y)|!x&y&z', '!(x|y|z)|(x|y)&z',
            '!(!x&y|!y&x|z)|!x&y&z|!y&x&z', '!(x|y|z)|(x|y)&z|x&y', '!x&y&z|!y&!z', '!y&!z|y&z', '!(!x&y|z)|!x&y&z',
            '!(!x&y|z)|y&z', '!(!x&z|y)|!x&y&z', '!(!x&z|y)|y&z', '!(!x&y|z)|!x&y&z|!y&x', '!y&!z|x|y&z', '!x&!z',
            '!x&!z|x&y&z', '!(!y&x|z)', '!x&!z|x&y', '!x&!z|!y&x&z', '!x&!z|x&z', '!(!y&x|z)|!y&x&z', '!(!y&x|z)|x&z',
            '!(x&y|z)', '!(x&y|z)|x&y&z', '!z', '!z|x&y', '!x&!z|!y&x', '!(x&y|z)|x&z', '!y&x|!z', '!z|x',
            '!(!y&z|x)', '!x&!z|y&z', '!(!y&x|z)|!x&y', '!x&!z|y', '!(!y&z|x)|!y&x&z', '!(!y&z|x)|x&z',
            '!(!y&x|z)|!x&y|!y&x&z', '!x&!z|x&z|y', '!x&y|!y&!z', '!(x&y|z)|y&z', '!x&y|!z', '!z|y',
            '!(!x&!y&z|x&y)', '!x&!z|!y&x|y&z', '!x&y|!y&x|!z', '!z|x|y', '!x&!y', '!x&!y|x&y&z', '!x&!y|!z&x&y',
            '!x&!y|x&y', '!(!z&x|y)', '!x&!y|x&z', '!(!z&x|y)|!z&x&y', '!(!z&x|y)|x&y', '!(x&z|y)', '!(x&z|y)|x&y&z',
            '!x&!y|!z&x', '!(x&z|y)|x&y', '!y', '!y|x&z', '!y|!z&x', '!y|x', '!(!z&y|x)', '!x&!y|y&z',
            '!(!z&y|x)|!z&x&y', '!(!z&y|x)|x&y', '!(!z&x|y)|!x&z', '!x&!y|z', '!(!z&x|y)|!x&z|!z&x&y',
            '!x&!y|x&y|z', '!x&z|!y&!z', '!(x&z|y)|y&z', '!(!x&!z&y|x&z)', '!x&!y|!z&x|y&z', '!x&z|!y', '!y|z',
            '!x&z|!y|!z&x', '!y|x|z', '!(x|y&z)', '!(x|y&z)|x&y&z', '!x&!y|!z&y', '!(x|y&z)|x&y', '!x&!z|!y&z',
            '!(x|y&z)|x&z', '!(!y&!z&x|y&z)', '!x&!y|!z&y|x&z', '!((x|y)&z|x&y)', '!((x|y)&z|x&y)|x&y&z', '!x&!y|!z',
            '!x&!y|!z|x&y', '!x&!z|!y', '!x&!z|!y|x&z', '!y|!z', '!y|!z|x', '!x', '!x|y&z', '!x|!z&y', '!x|y',
            '!x|!y&z', '!x|z', '!x|!y&z|!z&y', '!x|y|z', '!x|!y&!z', '!x|!y&!z|y&z', '!x|!z', '!x|!z|y', '!x|!y',
            '!x|!y|z', '!(x&y&z)', '!x|x'
        ]
    
    @staticmethod
    def prompt_func(question_case):
        truth_table = question_case['truth_table']
        prompt = f"""你是一位逻辑电路专家，需要解决一个布尔函数的最简表达式问题。给定的布尔函数有三个变量x、y、z，其真值表如下：

真值表：{truth_table}

真值表的每一位对应输入的三变量的二进制组合的结果。其中，第0位对应x=0,y=0,z=0时的结果；第1位对应x=0,y=0,z=1时的结果；依此类推，直到第7位对应x=1,y=1,z=1时的结果。

任务要求：
1. 找出与该真值表对应的布尔表达式，该表达式必须由变量x、y、z，运算符&（AND）、|（OR）、!（NOT）以及括号构成。
2. 表达式必须尽可能短，即使用最少的字符数。如果有多个最短表达式，选择字典序最小的一个。
3. 运算符的优先级为：NOT > AND > OR，必要时使用括号改变优先级。
4. 表达式必须符合特定的语法规则，不允许有多余的空格或其他字符。

请根据上述规则，给出该布尔函数的最简表达式，并将答案放在[answer]和[/answer]标签之间。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        solution = matches[-1].strip()
        solution = html.unescape(solution)  # 处理HTML转义字符
        return solution
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        ans = cls._get_ans_list()
        truth_table = identity['truth_table']
        index = int(truth_table, 2)
        if index >= len(ans):
            return False
        expected = html.unescape(ans[index])
        solution = html.unescape(solution)
        return solution == expected
