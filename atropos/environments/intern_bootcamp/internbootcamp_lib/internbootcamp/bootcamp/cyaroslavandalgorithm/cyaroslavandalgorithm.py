"""# 

### 谜题描述
Yaroslav likes algorithms. We'll describe one of his favorite algorithms.

  1. The algorithm receives a string as the input. We denote this input string as a. 
  2. The algorithm consists of some number of command. Сommand number i looks either as si >> wi, or as si <> wi, where si and wi are some possibly empty strings of length at most 7, consisting of digits and characters \"?\". 
  3. At each iteration, the algorithm looks for a command with the minimum index i, such that si occurs in a as a substring. If this command is not found the algorithm terminates. 
  4. Let's denote the number of the found command as k. In string a the first occurrence of the string sk is replaced by string wk. If the found command at that had form sk >> wk, then the algorithm continues its execution and proceeds to the next iteration. Otherwise, the algorithm terminates. 
  5. The value of string a after algorithm termination is considered to be the output of the algorithm. 



Yaroslav has a set of n positive integers, he needs to come up with his favorite algorithm that will increase each of the given numbers by one. More formally, if we consider each number as a string representing the decimal representation of the number, then being run on each of these strings separately, the algorithm should receive the output string that is a recording of the corresponding number increased by one.

Help Yaroslav.

Input

The first line contains integer n (1 ≤ n ≤ 100) — the number of elements in the set. The next n lines contains one positive integer each. All the given numbers are less than 1025.

Output

Print the algorithm which can individually increase each number of the set. In the i-th line print the command number i without spaces.

Your algorithm will be launched for each of these numbers. The answer will be considered correct if: 

  * Each line will a correct algorithm command (see the description in the problem statement). 
  * The number of commands should not exceed 50. 
  * The algorithm will increase each of the given numbers by one. 
  * To get a respond, the algorithm will perform no more than 200 iterations for each number. 

Examples

Input

2
10
79


Output

10&lt;&gt;11
79&lt;&gt;80

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
print \"\"\"

0??<>1
1??<>2
2??<>3
3??<>4
4??<>5
5??<>6
6??<>7
7??<>8
8??<>9
9??>>??0
??<>1

?0>>0?
?1>>1?
?2>>2?
?3>>3?
?4>>4?
?5>>5?
?6>>6?
?7>>7?
?8>>8?
?9>>9?

0?<>1
1?<>2
2?<>3
3?<>4
4?<>5
5?<>6
6?<>7
7?<>8
8?<>9
9?>>??0

>>?

\"\"\".replace('\n\n', '\n').strip()
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class Cyaroslavandalgorithmbootcamp(Basebootcamp):
    def __init__(self, n=2, max_digits=25):
        self.n = n
        self.max_digits = max_digits
    
    def case_generator(self):
        numbers = []
        for _ in range(self.n):
            if random.random() < 0.1:
                length = random.randint(1, self.max_digits)
                num_str = '9' * length
            else:
                length = random.randint(1, self.max_digits)
                first_digit = str(random.randint(1, 9))
                rest = ''.join(str(random.randint(0, 9)) for _ in range(length - 1))
                num_str = first_digit + rest
            numbers.append(num_str)
        return {'n': self.n, 'numbers': numbers}
    
    @staticmethod
    def prompt_func(question_case) -> str:
        numbers = question_case['numbers']
        n = question_case['n']
        problem = (
            "Yaroslav需要设计一个算法，该算法由一系列命令组成。每个命令的格式为si>>wi或si<>wi，其中si和wi是由数字和'?'组成的字符串，长度不超过7。算法的规则如下：\n\n"
            "1. 算法接收一个字符串作为输入a。\n"
            "2. 按命令顺序逐条检查，找到第一个si在a中出现的命令。\n"
            "3. 将a中第一个出现的si替换为wi。如果命令是si>>wi，则继续执行；如果命令是si<>wi，则终止。\n"
            "4. 算法必须在处理每个输入数字后，输出该数加一的结果。\n\n"
            f"给定{n}个数字，请设计符合条件的命令列表：\n"
        )
        for num in numbers:
            problem += f"{num}\n"
        problem += (
            "\n要求：\n"
            "- 每行一个命令，格式为si>>wi或si<>wi。\n"
            "- 命令数不超过50条。\n"
            "- 每个数字的处理必须在200次迭代内完成。\n"
            "将答案置于[answer]和[/answer]之间。"
        )
        return problem
    
    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        last_answer = answer_blocks[-1].strip()
        lines = [line.strip() for line in last_answer.split('\n') if line.strip()]
        command_pattern = re.compile(r'^\S+?(>>|<>)\S+?$')
        valid_commands = []
        for line in lines:
            if command_pattern.fullmatch(line):
                valid_commands.append(line)
        return '\n'.join(valid_commands) if valid_commands else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        commands = []
        command_pattern = re.compile(r'^(\S+?)(>>|<>)(\S+?)$')
        lines = solution.strip().split('\n')
        if len(lines) > 50:
            return False
        for line in lines:
            line = line.strip()
            match = command_pattern.fullmatch(line)
            if not match:
                return False
            si, op, wi = match.groups()
            if len(si) > 7 or len(wi) > 7:
                return False
            commands.append((si, op, wi))
        for num in identity['numbers']:
            expected = cls.add_one(num)
            result, iterations = cls.apply_commands(num, commands)
            if iterations > 200 or result != expected:
                return False
        return True
    
    @staticmethod
    def add_one(s):
        chars = list(s)
        carry = 1
        i = len(chars) - 1
        while i >= 0 and carry:
            digit = int(chars[i])
            new_digit = digit + carry
            if new_digit == 10:
                chars[i] = '0'
                carry = 1
            else:
                chars[i] = str(new_digit)
                carry = 0
            i -= 1
        if carry:
            chars = ['1'] + chars
        return ''.join(chars)
    
    @classmethod
    def apply_commands(cls, original, commands):
        a = original
        iterations = 0
        max_iter = 200
        while iterations < max_iter:
            found = False
            for si, op, wi in commands:
                pos = a.find(si)
                if pos != -1:
                    a = a[:pos] + wi + a[pos+len(si):]
                    iterations += 1
                    if op == '<>':
                        return a, iterations
                    else:
                        found = True
                        break
            if not found:
                break
        return a, iterations
