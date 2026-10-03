"""# 

### 谜题描述
In Python, code blocks don't have explicit begin/end or curly braces to mark beginning and end of the block. Instead, code blocks are defined by indentation.

We will consider an extremely simplified subset of Python with only two types of statements.

Simple statements are written in a single line, one per line. An example of a simple statement is assignment.

For statements are compound statements: they contain one or several other statements. For statement consists of a header written in a separate line which starts with \"for\" prefix, and loop body. Loop body is a block of statements indented one level further than the header of the loop. Loop body can contain both types of statements. Loop body can't be empty.

You are given a sequence of statements without indentation. Find the number of ways in which the statements can be indented to form a valid Python program.

Input

The first line contains a single integer N (1 ≤ N ≤ 5000) — the number of commands in the program. N lines of the program follow, each line describing a single command. Each command is either \"f\" (denoting \"for statement\") or \"s\" (\"simple statement\"). It is guaranteed that the last line is a simple statement.

Output

Output one line containing an integer - the number of ways the given sequence of statements can be indented modulo 109 + 7. 

Examples

Input

4
s
f
f
s


Output

1


Input

4
f
s
f
s


Output

2

Note

In the first test case, there is only one way to indent the program: the second for statement must be part of the body of the first one.
    
    
      
    simple statement  
    for statement  
        for statement  
            simple statement  
    

In the second test case, there are two ways to indent the program: the second for statement can either be part of the first one's body or a separate statement following the first one.
    
    
      
    for statement  
        simple statement  
        for statement  
            simple statement  
    

or
    
    
      
    for statement  
        simple statement  
    for statement  
        simple statement  
    

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
mod = 10**9+7
n = input()
x = [1]
for i in range(n):
	s = raw_input()
	if s=='f':
		x.insert(0,0)
	else:
		for i in range(len(x)-2,-1,-1):
			x[i] = (x[i]+x[i+1])%mod
print x[0]
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class Cpythonindentationbootcamp(Basebootcamp):
    def __init__(self, min_n=1, max_n=10):
        if min_n < 1:
            raise ValueError("min_n must be at least 1")
        if max_n > 5000:
            raise ValueError("max_n cannot exceed 5000")
        self.min_n = max(1, min_n)
        self.max_n = min(5000, max(max_n, self.min_n))
    
    def case_generator(self):
        n = random.randint(self.min_n, self.max_n)
        commands = [random.choice(['f', 's']) for _ in range(n-1)]
        commands.append('s')  # Ensure last command is 's'
        return {
            'n': n,
            'commands': commands
        }
    
    @staticmethod
    def prompt_func(question_case):
        input_lines = [str(question_case['n'])] + question_case['commands']
        input_example = '\n'.join(input_lines)
        prompt = f"""你是编程专家，需要解决一个关于Python缩进规则的谜题。请仔细阅读问题描述，并给出正确的答案。

问题描述：
在Python中，代码块由缩进定义，而没有显式的开始/结束符号。我们考虑一个极简化的Python子集，只有两种语句：简单语句（s）和for语句（f）。简单语句占据单独一行。for语句是复合语句，包含一个头部和一个循环体。循环体必须比for语句的头部缩进一级，且不能为空。

给定一个由's'和'f'组成的命令序列，最后一个命令一定是's'。你需要计算所有可能的有效缩进方式的数量，结果对10^9+7取模。

输入格式：
第一行包含整数N（命令数）。随后N行，每行是'f'或's'。

输出格式：
输出一个整数，表示有效方式数取模后的结果。

示例输入1：
4
s
f
f
s

示例输出1：
1

示例输入2：
4
f
s
f
s

示例输出2：
2

现在，请你解决以下输入案例：

输入：
{input_example}

请将你的最终答案放在[answer]标签内，例如：[answer]123[/answer]。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            return int(last_match)
        except ValueError:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        mod = 10**9 + 7
        commands = identity['commands']
        dp = [1]
        for stmt in commands:
            if stmt == 'f':
                dp.insert(0, 0)
            else:
                # Propagate the sum backwards
                for i in range(len(dp)-2, -1, -1):
                    dp[i] = (dp[i] + dp[i+1]) % mod
        correct_answer = dp[0] % mod if dp else 0
        try:
            return int(solution) == correct_answer
        except (ValueError, TypeError):
            return False
