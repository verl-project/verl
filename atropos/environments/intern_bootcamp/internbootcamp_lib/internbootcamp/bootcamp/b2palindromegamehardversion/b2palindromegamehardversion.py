"""# 

### 谜题描述
The only difference between the easy and hard versions is that the given string s in the easy version is initially a palindrome, this condition is not always true for the hard version.

A palindrome is a string that reads the same left to right and right to left. For example, \"101101\" is a palindrome, while \"0101\" is not.

Alice and Bob are playing a game on a string s of length n consisting of the characters '0' and '1'. Both players take alternate turns with Alice going first.

In each turn, the player can perform one of the following operations: 

  1. Choose any i (1 ≤ i ≤ n), where s[i] = '0' and change s[i] to '1'. Pay 1 dollar. 
  2. Reverse the whole string, pay 0 dollars. This operation is only allowed if the string is currently not a palindrome, and the last operation was not reverse. That is, if Alice reverses the string, then Bob can't reverse in the next move, and vice versa. 



Reversing a string means reordering its letters from the last to the first. For example, \"01001\" becomes \"10010\" after reversing.

The game ends when every character of string becomes '1'. The player who spends minimum dollars till this point wins the game and it is a draw if both spend equal dollars. If both players play optimally, output whether Alice wins, Bob wins, or if it is a draw.

Input

The first line contains a single integer t (1 ≤ t ≤ 10^3). Then t test cases follow.

The first line of each test case contains a single integer n (1 ≤ n ≤ 10^3).

The second line of each test case contains the string s of length n, consisting of the characters '0' and '1'. It is guaranteed that the string s contains at least one '0'.

Note that there is no limit on the sum of n over test cases.

Output

For each test case print a single word in a new line: 

  * \"ALICE\", if Alice will win the game, 
  * \"BOB\", if Bob will win the game, 
  * \"DRAW\", if the game ends in a draw. 

Example

Input


3
3
110
2
00
4
1010


Output


ALICE
BOB
ALICE

Note

In the first test case of example, 

  * in the 1-st move, Alice will use the 2-nd operation to reverse the string, since doing the 1-st operation will result in her loss anyway. This also forces Bob to use the 1-st operation. 
  * in the 2-nd move, Bob has to perform the 1-st operation, since the 2-nd operation cannot be performed twice in a row. All characters of the string are '1', game over. 

Alice spends 0 dollars while Bob spends 1 dollar. Hence, Alice wins.

In the second test case of example, 

  * in the 1-st move Alice has to perform the 1-st operation, since the string is currently a palindrome. 
  * in the 2-nd move Bob reverses the string. 
  * in the 3-rd move Alice again has to perform the 1-st operation. All characters of the string are '1', game over. 

Alice spends 2 dollars while Bob spends 0 dollars. Hence, Bob wins.

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
for _ in range(input()):
    n=input()
    s=raw_input()
    awayfrom=0
    for i in range(n/2):
        r=s[i]
        p=s[n-1-i]
        if r!=p:
            awayfrom+=1
    zer=0
    for i in s:
        if i=='0':
            zer+=1
    if zer==0:
        print \"DRAW\"
        continue
    if awayfrom==0:
        
        if zer==0:
            print \"DRAW\"
            continue
        if n%2==0:
            print \"BOB\"
            continue
        else:
            if zer==1:
                print \"BOB\"
                continue
            else:
                k=n/2
                if s[k]=='0':
                    print \"ALICE\"
                else:
                    print \"BOB\"
    elif awayfrom>=2:
        print \"ALICE\"
    elif awayfrom==1 and n%2==0:
        print \"ALICE\"
    elif awayfrom==1 and s[n/2]==\"1\":
        print \"ALICE\"
    else:
        if awayfrom==1 and zer==2:
            print \"DRAW\"
        else:
            print \"ALICE\"
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def solve_puzzle(n, s):
    awayfrom = 0
    for i in range(n // 2):
        if s[i] != s[n - 1 - i]:
            awayfrom += 1
    zer = sum(1 for c in s if c == '0')
    if zer == 0:
        return "DRAW"
    
    if awayfrom == 0:  # 初始是回文的情况
        if n % 2 == 0:
            return "BOB"
        else:
            if zer == 1:
                return "BOB"
            else:
                mid = n // 2
                if s[mid] == '0':
                    return "ALICE"
                else:
                    return "BOB"
    elif awayfrom >= 2:
        return "ALICE"
    elif awayfrom == 1:
        if n % 2 == 0:
            return "ALICE"
        else:
            if s[n//2] == '1':
                return "ALICE"
            else:
                return "DRAW" if zer == 2 else "ALICE"
    else:
        return "ALICE"

class B2palindromegamehardversionbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__(**params)
        self.max_t = params.get('max_t', 3)
        self.max_n = params.get('max_n', 10)
        self.min_t = params.get('min_t', 1)
        self.pal_ratio = params.get('pal_ratio', 0.4)  # 回文案例比例
    
    def case_generator(self):
        t = random.randint(self.min_t, self.max_t)
        test_cases = []
        expected_outputs = []
        
        for _ in range(t):
            n = random.randint(1, self.max_n)
            is_palindrome = random.random() < self.pal_ratio
            
            # 生成字符串逻辑
            while True:
                if is_palindrome:
                    # 构造回文字符串
                    half = n // 2
                    first_half = ''.join(random.choices('01', k=half))
                    # 确保至少一个0
                    if '0' not in first_half and half > 0:
                        first_half = list(first_half)
                        first_half[random.randint(0, half-1)] = '0'
                        first_half = ''.join(first_half)
                    
                    # 构造完整回文
                    if n % 2:
                        mid = random.choice('01')
                        s = first_half + mid + first_half[::-1]
                    else:
                        s = first_half + first_half[::-1]
                else:
                    s = ''.join(random.choices('01', k=n))
                
                # 有效性检查
                if '0' in s:
                    test_result = solve_puzzle(n, s)
                    # 过滤无效案例（当结果超预期时重新生成）
                    if test_result in {'ALICE', 'BOB', 'DRAW'}:
                        break
            
            test_cases.append({'n': n, 's': s})
            expected_outputs.append(solve_puzzle(n, s))
        
        return {
            't': t,
            'test_cases': test_cases,
            'expected_outputs': expected_outputs
        }
    
    @staticmethod
    def prompt_func(question_case):
        input_lines = [str(question_case['t'])]
        for tc in question_case['test_cases']:
            input_lines.append(str(tc['n']))
            input_lines.append(tc['s'])
        
        return f"""## 游戏规则
Alice和Bob在二进制字符串上进行博弈：
1. 初始字符串包含至少一个'0'
2. 玩家轮流执行以下操作之一：
   - 花费$1将任意'0'变为'1'
   - 免费反转字符串（仅当字符串非回文且上步未反转时）
3. 所有字符变为'1'时游戏结束，花费少者胜，平局当花费相同

## 输入格式
{question_case['t']}
{chr(10).join([f"{tc['n']}{chr(10)}{tc['s']}" for tc in question_case['test_cases']])}

## 输出要求
对每个测试案例输出ALICE/BOB/DRAW，每个结果占一行，置于[answer]标签内：

[answer]
结果1
结果2
...[/answer]"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        
        # 取最后一个答案块并标准化格式
        lines = [line.upper().strip() 
                for line in matches[-1].split('\n') 
                if line.strip()]
        
        valid = {'ALICE', 'BOB', 'DRAW'}
        if not all(line in valid for line in lines):
            return None
        return lines
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity.get('expected_outputs', [])
        if not solution or len(solution) != len(expected):
            return False
        return all(s == e for s, e in zip(solution, expected))
