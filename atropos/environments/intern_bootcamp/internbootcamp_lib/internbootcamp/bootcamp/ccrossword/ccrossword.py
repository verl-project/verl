"""# 

### 谜题描述
Vasya trains to compose crossword puzzles. He can only compose crosswords of a very simplе type so far. All of them consist of exactly six words; the words can be read only from top to bottom vertically and from the left to the right horizontally. The words are arranged in the form of a rectangular \"eight\" or infinity sign, not necessarily symmetrical.

The top-left corner of the crossword coincides with the top-left corner of the rectangle. The same thing is correct for the right-bottom corners. The crossword can't degrade, i.e. it always has exactly four blank areas, two of which are surrounded by letters. Look into the output for the samples for clarification.

Help Vasya — compose a crossword of the described type using the given six words. It is allowed to use the words in any order.

Input

Six lines contain the given words. Every word consists of no more than 30 and no less than 3 uppercase Latin letters. 

Output

If it is impossible to solve the problem, print Impossible. Otherwise, print the sought crossword. All the empty squares should be marked as dots.

If there can be several solutions to that problem, print the lexicographically minimum one. I.e. the solution where the first line is less than the first line of other solutions should be printed. If the two lines are equal, compare the second lines and so on. The lexicographical comparison of lines is realized by the < operator in the modern programming languages.

Examples

Input

NOD
BAA
YARD
AIRWAY
NEWTON
BURN


Output

BAA...
U.I...
R.R...
NEWTON
..A..O
..YARD


Input

AAA
AAA
AAAAA
AAA
AAA
AAAAA


Output

AAA..
A.A..
AAAAA
..A.A
..AAA


Input

PTC
JYNYFDSGI
ZGPPC
IXEJNDOP
JJFS
SSXXQOFGJUZ


Output

JJFS....
Y..S....
N..X....
Y..X....
F..Q....
D..O....
S..F....
G..G....
IXEJNDOP
...U...T
...ZGPPC

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
from itertools import permutations
s = []
answers = []
for i in range(6):
    s.append(raw_input().rstrip())
for t in permutations(s):
    if (len(t[0]) + len(t[2]) - 1 == len(t[1])) and (len(t[3]) + len(t[5]) - 1 == len(t[4])):
        if t[0][0] == t[3][0] and t[3][-1] == t[1][0] and t[0][-1] == t[4][0] and t[2][0]==t[4][-1] and t[2][-1] == t[5][-1] and t[1][-1] == t[5][0] and t[1][len(t[0])-1] == t[4][len(t[3])-1]:
            a = len(t[1])
            b = len(t[4])
        
            ans = t[3]+'.'*(b-len(t[3])) + '\n'
            for i in range(1,len(t[0])-1):
                ans += t[0][i] + '.' * (len(t[3])-2) + t[1][i] + '.'*(b-len(t[3])) + '\n'
            ans += t[4] + '\n'
            for i in range(1,len(t[2])-1):
                ans += '.'*(len(t[3])-1) + t[1][i-1+len(t[0])] + '.' * (len(t[5])-2) + t[2][i] + '\n'
            ans += '.'*(b-len(t[5])) + t[5] + '\n'
            answers.append(ans)
if len(answers)>0:
    print(min(answers))
else:
    print 'Impossible'
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from itertools import permutations
import re
from bootcamp import Basebootcamp

class Ccrosswordbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = params
        
    def case_generator(self):
        predefined_cases = [
            ["NOD", "BAA", "YARD", "AIRWAY", "NEWTON", "BURN"],
            ["AAA", "AAA", "AAAAA", "AAA", "AAA", "AAAAA"],
            ["PTC", "JYNYFDSGI", "ZGPPC", "IXEJNDOP", "JJFS", "SSXXQOFGJUZ"]
        ]
        selected_case = random.choice(predefined_cases).copy()
        random.shuffle(selected_case)
        return {"words": selected_case}

    @staticmethod
    def prompt_func(question_case) -> str:
        words = question_case["words"]
        words_str = "\n".join(words)
        return f"""Vasya需要制作包含6个单词的特殊填字游戏，填字结构呈无限符号形状。规则如下：

1. 必须使用所有6个单词，每个单词只能用一次
2. 单词只能水平或垂直排列，形成两个交叉的矩形
3. 输出必须用点号填充空白，若无法解出则输出Impossible
4. 存在多解时选择字典序最小的解（逐行比较）

输入单词：
{words_str}

请将最终答案放在[answer]和[/answer]之间，例如：

[answer]
AAA..
A.A..
AAAAA
..A.A
..AAA
[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        return matches[-1].strip()

    @classmethod
    def _verify_correction(cls, solution, identity):
        input_words = identity["words"]
        answers = []

        for t in permutations(input_words):
            if (len(t[0]) + len(t[2]) - 1 != len(t[1])) or (len(t[3]) + len(t[5]) - 1 != len(t[4])):
                continue
            
            try:
                if not all([
                    t[0][0] == t[3][0],
                    t[3][-1] == t[1][0],
                    t[0][-1] == t[4][0],
                    t[2][0] == t[4][-1],
                    t[2][-1] == t[5][-1],
                    t[1][-1] == t[5][0],
                    t[1][len(t[0])-1] == t[4][len(t[3])-1]
                ]):
                    continue
            except IndexError:
                continue

            b = len(t[4])
            ans = f"{t[3]}{'.'*(b-len(t[3]))}\n"
            for i in range(1, len(t[0])-1):
                ans += f"{t[0][i]}{'.'*(len(t[3])-2)}{t[1][i]}{'.'*(b-len(t[3]))}\n"
            ans += f"{t[4]}\n"
            for i in range(1, len(t[2])-1):
                ans += f"{'.'*(len(t[3])-1)}{t[1][i-1+len(t[0])]}{'.'*(len(t[5])-2)}{t[2][i]}\n"
            ans += f"{'.'*(b-len(t[5]))}{t[5]}\n"
            answers.append(ans)

        if not answers:
            return solution.strip() == "Impossible"
        else:
            correct = min(answers)
            return cls._normalize(solution) == cls._normalize(correct)

    @staticmethod
    def _normalize(s):
        return [line.rstrip() for line in s.strip().split('\n') if line.strip()]
