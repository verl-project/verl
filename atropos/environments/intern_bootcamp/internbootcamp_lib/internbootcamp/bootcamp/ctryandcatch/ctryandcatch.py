"""# 

### 谜题描述
Vasya is developing his own programming language VPL (Vasya Programming Language). Right now he is busy making the system of exceptions. He thinks that the system of exceptions must function like that.

The exceptions are processed by try-catch-blocks. There are two operators that work with the blocks:

  1. The try operator. It opens a new try-catch-block. 
  2. The catch(<exception_type>, <message>) operator. It closes the try-catch-block that was started last and haven't yet been closed. This block can be activated only via exception of type <exception_type>. When we activate this block, the screen displays the <message>. If at the given moment there is no open try-catch-block, then we can't use the catch operator.



The exceptions can occur in the program in only one case: when we use the throw operator. The throw(<exception_type>) operator creates the exception of the given type.

Let's suggest that as a result of using some throw operator the program created an exception of type a. In this case a try-catch-block is activated, such that this block's try operator was described in the program earlier than the used throw operator. Also, this block's catch operator was given an exception type a as a parameter and this block's catch operator is described later that the used throw operator. If there are several such try-catch-blocks, then the system activates the block whose catch operator occurs earlier than others. If no try-catch-block was activated, then the screen displays message \"Unhandled Exception\".

To test the system, Vasya wrote a program that contains only try, catch and throw operators, one line contains no more than one operator, the whole program contains exactly one throw operator.

Your task is: given a program in VPL, determine, what message will be displayed on the screen.

Input

The first line contains a single integer: n (1 ≤ n ≤ 105) the number of lines in the program. Next n lines contain the program in language VPL. Each line contains no more than one operator. It means that input file can contain empty lines and lines, consisting only of spaces.

The program contains only operators try, catch and throw. It is guaranteed that the program is correct. It means that each started try-catch-block was closed, the catch operators aren't used unless there is an open try-catch-block. The program has exactly one throw operator. The program may have spaces at the beginning of a line, at the end of a line, before and after a bracket, a comma or a quote mark.

The exception type is a nonempty string, that consists only of upper and lower case english letters. The length of the string does not exceed 20 symbols. Message is a nonempty string, that consists only of upper and lower case english letters, digits and spaces. Message is surrounded with quote marks. Quote marks shouldn't be printed. The length of the string does not exceed 20 symbols.

Length of any line in the input file does not exceed 50 symbols. 

Output

Print the message the screen will show after the given program is executed.

Examples

Input

8
try
    try
        throw ( AE ) 
    catch ( BE, \"BE in line 3\")

    try
    catch(AE, \"AE in line 5\") 
catch(AE,\"AE somewhere\")


Output

AE somewhere


Input

8
try
    try
        throw ( AE ) 
    catch ( AE, \"AE in line 3\")

    try
    catch(BE, \"BE in line 5\") 
catch(AE,\"AE somewhere\")


Output

AE in line 3


Input

8
try
    try
        throw ( CE ) 
    catch ( BE, \"BE in line 3\")

    try
    catch(AE, \"AE in line 5\") 
catch(AE,\"AE somewhere\")


Output

Unhandled Exception

Note

In the first sample there are 2 try-catch-blocks such that try operator is described earlier than throw operator and catch operator is described later than throw operator: try-catch(BE,\"BE in line 3\") and try-catch(AE,\"AE somewhere\"). Exception type is AE, so the second block will be activated, because operator catch(AE,\"AE somewhere\") has exception type AE as parameter and operator catch(BE,\"BE in line 3\") has exception type BE.

In the second sample there are 2 try-catch-blocks such that try operator is described earlier than throw operator and catch operator is described later than throw operator: try-catch(AE,\"AE in line 3\") and try-catch(AE,\"AE somewhere\"). Exception type is AE, so both blocks can be activated, but only the first one will be activated, because operator catch(AE,\"AE in line 3\") is described earlier than catch(AE,\"AE somewhere\")

In the third sample there is no blocks that can be activated by an exception of type CE.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def _check(tokens, ex, msg):
    prev = tokens.pop()
    if prev == ex:
        print msg
        exit(0)
    elif prev != TRY:
        _check(tokens, ex, msg)
        tokens.append(prev)
    return
TRY = _check
n = int(raw_input())
lines = [s for s in (raw_input().strip() for i in xrange(n)) if s != '']
stack = []
for line in lines:
    if line == 'try':
        stack.append(TRY)
    elif line.startswith('throw'):
        stack.append(line.split('(')[1].split(')')[0].strip())
    elif line.startswith('catch'):
        ex, msg = [s.strip().strip('\"') for s in line.split('(')[1].split(')')[0].split(',')]
        _check(stack, ex, msg)

print 'Unhandled Exception'
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import string
import re
from bootcamp import Basebootcamp

class Ctryandcatchbootcamp(Basebootcamp):
    def __init__(self, max_depth=3, case_type=None):
        self.max_depth = max_depth
        self.case_type = case_type  # 0: valid, 1: multiple valid, 2: unhandled
    
    def case_generator(self):
        case_type = self.case_type
        if case_type is None:
            case_type = random.choice([0, 1, 2])
        
        ex_type = self._random_string(5)
        correct_msg = self._random_string(10)
        other_ex = self._random_string(5)
        while other_ex == ex_type:
            other_ex = self._random_string(5)
        
        lines = []
        if case_type == 0:
            lines = [
                'try',
                f'throw({ex_type})',
                f'catch({ex_type}, "{correct_msg}")'
            ]
        elif case_type == 1:
            lines = [
                'try',
                'try',
                f'throw({ex_type})',
                f'catch({ex_type}, "{correct_msg}")',
                f'catch({other_ex}, "wrong")',
                f'catch({ex_type}, "later")'
            ]
        elif case_type == 2:
            lines = [
                'try',
                f'throw({ex_type})',
                f'catch({other_ex}, "wrong")'
            ]
            correct_msg = "Unhandled Exception"
        
        program = []
        for line in lines:
            line = self._add_random_spaces(line)
            program.append(line)
        
        answer = self._compute_answer(program)
        return {
            'program': program,
            'answer': answer
        }
    
    @staticmethod
    def prompt_func(question_case):
        program = question_case['program']
        program_text = '\n'.join(program)
        prompt = f"""你是Vasya编程语言（VPL）的测试员，请根据程序确定执行后的输出消息。

规则：
- 每个try必须对应一个catch。
- catch仅在异常类型匹配时触发，且其必须出现在throw之后。
- 多个符合条件的catch选择最内层且最早出现的。

程序：
{program_text}

请将答案放在[answer]和[/answer]之间。"""
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['answer']
    
    def _random_string(self, length):
        chars = string.ascii_letters
        return ''.join(random.choice(chars) for _ in range(length))
    
    def _add_random_spaces(self, line):
        parts = line.split('(', 1)
        operator = parts[0].strip()
        if len(parts) == 1:
            return f"{' ' * random.randint(0,2)}{operator}{' ' * random.randint(0,2)}"
        params = parts[1].rstrip(')').strip()
        params = re.sub(r'\s*,\s*', ', ', params)
        return f"{' ' * random.randint(0,2)}{operator}( {params} ){' ' * random.randint(0,2)}"
    
    def _compute_answer(self, program):
        class CheckExit(Exception):
            def __init__(self, msg):
                self.msg = msg
        
        def _check(tokens, target_ex, msg):
            if not tokens:
                return
            prev = tokens.pop()
            if prev == target_ex:
                raise CheckExit(msg)
            elif prev != 'TRY':
                _check(tokens, target_ex, msg)
                tokens.append(prev)
            else:
                tokens.append(prev)
        
        stack = []
        throw_ex = None
        for line in program:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped == 'try':
                stack.append('TRY')
            elif stripped.startswith('throw'):
                ex = stripped.split('(')[1].split(')')[0].strip()
                throw_ex = ex
                stack.append(ex)
            elif stripped.startswith('catch'):
                content = stripped.split('(', 1)[1].split(')', 1)[0].strip()
                ex, msg_part = content.split(',', 1)
                ex = ex.strip()
                msg = msg_part.strip().strip('"')
                temp_stack = stack.copy()
                try:
                    _check(temp_stack, ex, msg)
                except CheckExit as e:
                    return e.msg
                stack = temp_stack
        return "Unhandled Exception"
