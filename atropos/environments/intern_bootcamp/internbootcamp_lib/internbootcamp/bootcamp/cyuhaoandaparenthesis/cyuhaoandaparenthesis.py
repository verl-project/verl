"""# 

### 谜题描述
One day, Yuhao came across a problem about checking if some bracket sequences are correct bracket sequences.

A bracket sequence is any non-empty sequence of opening and closing parentheses. A bracket sequence is called a correct bracket sequence if it's possible to obtain a correct arithmetic expression by inserting characters \"+\" and \"1\" into this sequence. For example, the sequences \"(())()\", \"()\" and \"(()(()))\" are correct, while the bracket sequences \")(\", \"(()\" and \"(()))(\" are not correct.

Yuhao found this problem too simple for him so he decided to make the problem harder. You are given many (not necessarily correct) bracket sequences. The task is to connect some of them into ordered pairs so that each bracket sequence occurs in at most one pair and the concatenation of the bracket sequences in each pair is a correct bracket sequence. The goal is to create as many pairs as possible.

This problem unfortunately turned out to be too difficult for Yuhao. Can you help him and solve it?

Input

The first line contains one integer n (1 ≤ n ≤ 10^5) — the number of bracket sequences.

Each of the following n lines contains one bracket sequence — a non-empty string which consists only of characters \"(\" and \")\".

The sum of lengths of all bracket sequences in the input is at most 5 ⋅ 10^5.

Note that a bracket sequence may appear in the input multiple times. In this case, you can use each copy of the sequence separately. Also note that the order in which strings appear in the input doesn't matter.

Output

Print a single integer — the maximum number of pairs which can be made, adhering to the conditions in the statement.

Examples

Input


7
)())
)
((
((
(
)
)


Output


2


Input


4
(
((
(((
(())


Output


0


Input


2
(())
()


Output


1

Note

In the first example, it's optimal to construct two pairs: \"(( )())\" and \"( )\".

Here is a reference code to solve this task. You can use this to help you genereate cases or validate the solution.
```python
def process(s):
    stack = []
    for c in s:
        if c == \")\":
            if len(stack) and stack[-1] == \"(\":
                stack.pop()
                continue
        stack.append(c)
    if len(stack) == 0:
        return 0, True
    has_opening = \"(\" in stack
    has_closing = \")\" in stack
    if has_opening and has_closing:
        return 0, False
    return (len(stack) if has_opening else -len(stack)), True
        

n = int(raw_input())

opening = []
closing = []
balanced = 0

for _ in xrange(n):
    s = str(raw_input())
    score, is_valid = process(s)
    #print s, score, is_valid
    if is_valid:
        if score == 0:
            balanced += 1
        elif score > 0:
            opening.append(score)
        else:
            closing.append(-score)

opening.sort()
closing.sort()
i = 0
j = 0
result = 0
while i < len(opening) and j < len(closing):
    if opening[i] == closing[j]:
        result += 1
        i += 1
        j += 1
    elif opening[i] < closing[j]:
        i += 1
    else:
        j += 1
result += balanced // 2
        
print result
```


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class Cyuhaoandaparenthesisbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = {
            'max_opening_pairs': 3,
            'max_balanced_pairs': 2,
            'max_invalid_sequences': 2,
            'max_unmatched_sequences': 2,
        }
        self.params.update(params)
        super().__init__(**params)  # 修复：添加基类初始化
    
    def case_generator(self):
        max_op = self.params['max_opening_pairs']
        max_bal = self.params['max_balanced_pairs']
        max_inv = self.params['max_invalid_sequences']
        max_unm = self.params['max_unmatched_sequences']

        while True:
            m = random.randint(0, max_op)
            k = random.randint(0, max_bal)
            i = random.randint(0, max_inv)
            j = random.randint(0, max_unm)
            
            sequences = []
            # 生成成对的可匹配序列
            for _ in range(m):
                x = random.randint(1, 3)
                sequences.append('(' * x)
                sequences.append(')' * x)
            
            # 生成平衡序列对
            for _ in range(k):
                sequences.append(self.generate_balanced())
                sequences.append(self.generate_balanced())
            
            # 生成无效序列（同时含有开闭括号残留）
            for _ in range(i):
                sequences.append(self.generate_invalid())
            
            # 生成大长度单边序列
            for _ in range(j):
                x = random.randint(4, 5)
                sequences.append(random.choice(['(', ')']) * x)
            
            if sequences:
                break
            else:  # Fallback机制
                sequences.append(self.generate_balanced())

        random.shuffle(sequences)
        return {'n': len(sequences), 'sequences': sequences}

    def generate_balanced(self):
        k = random.randint(1, 3)
        if random.choice([True, False]):
            return '()' * k  # 扁平结构
        return '(' * k + ')' * k  # 嵌套结构

    def generate_invalid(self):
        # 保证处理后同时含有两种括号的无效序列
        invalid_pool = [
            ')(', 
            '())(',
            '(()))(',
            ')((()',
            ')()(',
            ')))(((', 
            ')())(('
        ]
        return random.choice(invalid_pool)

    @staticmethod
    def prompt_func(question_case) -> str:
        sequences = "\n".join(question_case['sequences'])
        return f"""请解决括号配对问题。给定n个括号序列，找出最大可配对数量。
        
输入格式：
第一行：n
接下来n行：各括号序列

示例：
输入：
7
)())
)
((
((
(
)
输出：
2

当前输入数据：
n = {question_case['n']}
{sequences}

将答案放入[answer]标签内，如：[answer]3[/answer]。"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        return int(matches[-1].strip()) if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 引用题目给出的参考算法进行验证
        def process(s):
            stack = []
            for c in s:
                if c == ")":
                    if stack and stack[-1] == "(":
                        stack.pop()
                        continue
                stack.append(c)
            if not stack: return 0, True
            has_open = '(' in stack
            has_close = ')' in stack
            if has_open and has_close: return 0, False
            return len(stack) if has_open else -len(stack), True

        opening = []
        closing = []
        balanced = 0
        
        for s in identity['sequences']:
            score, valid = process(s)
            if not valid: continue
            if score == 0:
                balanced += 1
            elif score > 0:
                opening.append(score)
            else:
                closing.append(-score)
        
        # 匹配开闭序列
        opening.sort()
        closing.sort()
        pairs = 0
        i = j = 0
        while i < len(opening) and j < len(closing):
            if opening[i] == closing[j]:
                pairs += 1
                i += 1
                j += 1
            elif opening[i] < closing[j]:
                i += 1
            else:
                j += 1
        
        return solution == (pairs + balanced // 2)
