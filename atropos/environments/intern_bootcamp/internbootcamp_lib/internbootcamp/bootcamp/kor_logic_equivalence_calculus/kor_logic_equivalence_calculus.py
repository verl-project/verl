"""# 谜题训练场开发任务

## 任务概述
你是一位资深程序员，我需要你帮我实现一个特定谜题的训练场环境类。这个类继承自`Basebootcamp`，用于生成谜题实例并验证解答。

## 背景说明
我正在开发一系列谜题训练场，每个训练场对应一个特定类型的谜题。训练场类命名为`{PuzzleName}bootcamp`，其中`PuzzleName`是谜题的名称。

每个训练场类主要提供两个核心功能：
1. 生成该谜题类型的问题实例
2. 验证用户对问题的回答是否正确

## 技术接口规范

### 类方法实现要求

```python
from bootcamp import Basebootcamp

class {PuzzleName}bootcamp(Basebootcamp):
    def __init__(self, **params):
        \"\"\"
        请你自定义params，以保存该puzzle相关的参数，例如网格大小等，参数配有默认值
        \"\"\"
        pass
    
    def case_generator(self):
        \"\"\"
        生成谜题实例，提示：为保证谜题有解，可以先生成结果再对结果处理得到谜题
        返回：一个可JSON序列化的字典（避免包含set等无法通过json.dumps处理的数据结构）
        \"\"\"
        pass
    
    @staticmethod
    def prompt_func(question_case) -> str:
        \"\"\"
        将case_generator生成的谜题实例转换为文本形式的问题，问题中包含问题背景、对谜题规则的介绍、具体要解决的谜题实例、期望最终答案的格式，
        例如：你是xxxx，请你解答yyyy，规则如下：yyyy，最终答案放置在：zzzzz
        注意：请参照提供的谜题描述进行复述，规则应当描述详细，包括任务背景、具体任务操作规则、对题目格式和答案格式的含义介绍等，

        参数:
            question_case: 由case_generator生成的谜题实例
            
        返回:
            str: 格式化的问题字符串
            
        注意:
            1. 需考虑问题的格式，以便后续能正确提取
            2. 问题描述中应包含期望的答案格式说明，以便后续能正确提取，为了避免抽取时匹配出干扰项，请要求模型将答案放在特定标签（如双括号）内，例如[[your answer here]]
        \"\"\"
        pass
    
    @staticmethod
    def extract_output(output):
        \"\"\"
        从LLM的回复中提取符合格式要求的答案，如有多个，请抽取最后一个，避免使用re.search等只抽取第一个结果的方式。
        
        参数:
            output: LLM的完整输出（包含原始问题和回答）
            
        返回:
            提取的答案，若未找到符合格式的答案则返回None
        \"\"\"
        pass
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        \"\"\"
        验证提取的答案是否正确，注意一个问题可以能有多个解，按照谜题规则进行检验，不要直接匹配可能的答案。
        
        参数:
            solution: extract_output提取的答案
            identity: case_generator生成的谜题实例
            
        返回:
            bool: 答案是否正确
        \"\"\"
        pass
```

### 验证评分方法（基类已实现）

```python
@classmethod
def verify_score(cls, model_output, identity:dict, format_score=0.1) -> float:
    \"\"\"
    验证输出结果并评分。
    
    参数:
        model_output: 模型的完整输出
        identity: 谜题实例（由case_generator生成）
        format_score: 答案格式正确时的基础分数
    
    返回:
        float: 评分结果（0-1之间）
    \"\"\"
    score = 0. 
    try:
        extract_solution = cls.extract_output(model_output)
        if extract_solution is None:
            return score
        else:
            score = format_score # 格式正确时的基础分数
        if cls._verify_correction(extract_solution, identity):
            score = 1.  # 答案完全正确时的满分
    except Exception as e:
        # 处理异常情况
        pass
    return score
```

### 使用示例

```python
# 初始化谜题训练场
bootcamp = Puzzlebootcamp()

# 生成谜题实例
case = bootcamp.case_generator()

# 将谜题转换为文本问题
prompt = Puzzlebootcamp.prompt_func(case)

# 获取LLM对问题的解答
response = get_response(prompt, \"LLM\")

# 从完整对话中提取答案
extracted_output = Puzzlebootcamp.extract_output(prompt + response)

# 验证答案并评分
score = Puzzlebootcamp.verify_score(extracted_output, case)
```

## 你的任务
请根据以下谜题描述（谜题描述可能不完整，请先结合你的知识澄清规则），实现一个完整的谜题训练场类：

### 谜题描述
1. Propositional Symbolization Rules:
    - Equivalence is represented by `::=::`
    - Negation is represented by `!`
    - Or is represented by `|`
    - And is represented by `&`
    - Implication is represented by `>`
    - Equivalence is represented by `=`
    - NAND is represented by `⇑`
    - NOR is represented by `⇓`
2. Basic Equivalences:
    (1) A ::=:: !!A
    (2) A ::=:: A | A, A ::=:: A & A
    (3) A | B ::=:: B | A, A & B ::=:: B & A
    (4) (A | B) | C ::=:: A | (B | C), (A & B) & C ::=:: A & (B & C)
    (5) A | (B & C) ::=:: (A | B) & (A | C), A & (B | C) ::=:: (A & B) | (A & C)
    (6) !(A | B) ::=:: !A & !B, !(A & B) ::=:: !A | !B
    (7) A | (A & B) ::=:: A, A & (A | B) ::=:: A
    (8) A | !A ::=:: 1
    (9) A & !A ::=:: 0
    (10) A > B ::=:: !A | B
    (11) A = B ::=:: (A > B) & (B > A)
    (12) A > B ::=:: !B > !A
    (13) A = B ::=:: !A = !B
    (14) (A > B) & (A > !B) ::=:: !A
    (15) A ⇑ B ::=:: !A | !B
    (16) A ⇓ B ::=:: !A & !B
3. Equivalence Calculation Rules:
    - The final expression should be completely represented using `|`, `&`, and `!`, without retaining `>` and `=`.
4. Truth Value Judgment Steps:
    - Define the practical problem to be judged as simple propositions and symbolize them.
    - Use simple propositions to express the formula based on each person's description.
    - Combine the information of who is true and who is false to write the final logical expression.
    - Use the above equivalences to derive and judge the truth of the expression.Example questions are as follows:

<example 0>
Using Basic Equivalences (10), what equivalent expression is obtained by removing all occurrences of > in (p > q) > r?
The answer is a logical expression formatted as [[ ]].! has the highest priority, and note that parentheses should only be used when logically needed to emphasise the order of operations, avoiding redundant or unnecessary brackets.
</example 0>

<example 1>
According to the rules, are (p>q)>r and p>(q>r) equivalent? 
A. Yes B. No
The answer is a single English letter.The answer format should be like [[A]].
</example 1>

<example 2>
Using the 16 Basic Equivalences, what is the simplest result obtained through equivalence derivation?
(1) !(p>(p|q))&r
(2) p&(((p|q)&!p)>q)
Provide your answers as logical expressions formatted as [[];[]].
</example 2>

<example 3>
According to the 16 Basic Equivalences, is the equivalence below valid? A. Yes B. No
(1) p::=::(p&q)|(p&!q)
(2) (p&!q)|(!p&q)::=::(p|q)&(!(p|q))
The answer to each sub-question is a letter of the alphabet, and answers to different sub-questions are separated by ;.The answer format should be like [[A];[A]].
</example 3>

<example 4>
According to the 16 Basic Equivalences, is the equivalence below valid? A. Yes B. No
(1) ((p>q)&(p>r))::=::(p>(q|r))
(2) !(p=q)::=::(p|q)&!(p&q)
The answer to each sub-question is a letter of the alphabet, and answers to different sub-questions are separated by ;.The answer format should be like [[A];[A]].
</example 4>

<example 5>
According to the 16 Basic Equivalences, is the equivalence below valid? A. Yes B. No
(1) (p⇓q)⇓r::=::p⇓(q⇓r)
(2) (p⇑q)⇑r::=::p⇑(q⇑r)
The answer to each sub-question is a letter of the alphabet, and answers to different sub-questions are separated by ;.The answer format should be like [[A];[A]].
</example 5>

<example 6>
During a break at a seminar, three attendees tried to determine where Professor Wang is from based on his accent. Their statements were as follows:

First person: Professor Wang is not from Suzhou, he is from Shanghai.
Second person: Professor Wang is not from Shanghai, he is from Suzhou.
Third person: Professor Wang is neither from Shanghai nor from Hangzhou.

After hearing these judgments, Professor Wang chuckled and remarked, \"One of you got everything right, one got half right, and one got everything wrong.\"
Let p denote \"Professor Wang is from Suzhou,\" 
q denote \"Professor Wang is from Shanghai,\" 
and r denote \"Professor Wang is from Hangzhou.\" 
Exactly one of p,q,r is true, and the other two are false.
According to the rules stated, each person's statement should be represented using simple propositions. So, what would the statements of First person, Second person, and Third person be represented as?
Answers are formatted as [[ ];[ ];[ ]]. Each bracketed section should contain the corresponding logical expression for each individual statement.
</example 6>

<example 7>
During a break at a seminar, three attendees tried to determine where Professor Wang is from based on his accent. Their statements were as follows:

Person A: Professor Wang is not from Suzhou, he is from Shanghai.
Person B: Professor Wang is not from Shanghai, he is from Suzhou.
Person C: Professor Wang is neither from Shanghai nor from Hangzhou.

After hearing these judgments, Professor Wang chuckled and remarked, \"One of you got everything right, one got half right, and one got everything wrong.\"

Let p denote \"Professor Wang is from Suzhou,\" 
q denote \"Professor Wang is from Shanghai,\" 
and r denote \"Professor Wang is from Hangzhou.\" 

Exactly one of p,q,r is true, and the other two are false.
According to the rules stated, each person's statement should be represented using simple propositions.
Represent each person's statement:
Person A:!p&q
Person B:p&!q
Person C:!q&!r

Define the following logical expressions for Person A:
B1=!p&q (Person A's statements are entirely correct).
B2=(!p&!q)|(p&q) (Person A's  statements are partially correct).
B3=p&!q (Person A's statements are entirely incorrect).

Similarly, define the analogous expressions for Person B:
C1 (Person B's statements are entirely correct).
C2 (Person B's  statements are partially correct).
C3 (Person B's statements are entirely incorrect).

And for Person C:
D1 (Person C's statements are entirely correct).
D2 (Person C's  statements are partially correct).
D3 (Person C's statements are entirely incorrect).

Please provide the corresponding logical expressions. The answer format should be:
[[C1=...];[C2=...];[C3=...];[D1=...];[D2=...];[D3=...]].
</example 7>

<example 8>
During a break at a seminar, three attendees tried to determine where Professor Wang is from based on his accent. Their statements were as follows:

Person A: Professor Wang is not from Suzhou, he is from Shanghai.
Person B: Professor Wang is not from Shanghai, he is from Suzhou.
Person C: Professor Wang is neither from Shanghai nor from Hangzhou.

After hearing these judgments, Professor Wang chuckled and remarked, \"One of you got everything right, one got half right, and one got everything wrong.\"
Let p denote \"Professor Wang is from Suzhou,\" 
q denote \"Professor Wang is from Shanghai,\" 
and r denote \"Professor Wang is from Hangzhou.\" 

Exactly one of p,q,r is true, and the other two are false.
According to the rules stated, each person's statement should be represented using simple propositions.
Represent each person's statement:
Person A:!p&q
Person B:p&!q
Person C:!q&!r

Define the following logical expressions for Person A:
B1=!p&q (Person A's statements are entirely correct).
B2=(!p&!q)|(p&q) (Person A's  statements are partially correct).
B3=p&!q (Person A's statements are entirely incorrect).

Similarly, define the analogous expressions for Person B:
C1=p&!q.(Person B's statements are entirely correct).
C2=(p & q) | (!p & !q).(Person B's  statements are partially correct).
C3=!p&q.(Person B's statements are entirely incorrect).

And for Person C:
D1=!q&!r.(Person C's statements are entirely correct).
D2=(!q&r)|(q&!r).(Person C's  statements are partially correct).
D3=q&r.(Person C's statements are entirely incorrect).

According to Professor Wang's remarks, the final logical expression
E=(B1&C2&D3)|(B1&C3&D2)|(B2&C1&D3)|(B2&C3&D1)|(B3&C1&D2)|(B3&C2&D1)

After equivalent derivation according to the above rules:
(1) What does B1&C2&D3 simplify to?
(2) What does B1&C3&D2 simplify to?
(3) What does B2&C1&D3 simplify to?
(4) What does B2&C3&D1 simplify to?
(5) What does B3&C1&D2 simplify to?
(6) What does B3&C2&D1 simplify to?
(7) What does E finally simplify to?

Please provide the corresponding logical expressions. The answer format should be:
[[B1&C2&D3::=::…];[B1&C3&D2::=::…];[B2&C1&D3::=::…];[B2&C3&D1::=::…];[B3&C1&D2::=::…];[B3&C2&D1::=::…];[E::=::…]].
</example 8>

<example 9>
During a break at a seminar, three attendees tried to determine where Professor Wang is from based on his accent. Their statements were as follows:

Person A: Professor Wang is not from Suzhou, he is from Shanghai.
Person B: Professor Wang is not from Shanghai, he is from Suzhou.
Person C: Professor Wang is neither from Shanghai nor from Hangzhou.

After hearing these judgments, Professor Wang chuckled and remarked, \"One of you got everything right, one got half right, and one got everything wrong.\"
Let p denote \"Professor Wang is from Suzhou,\"
q denote \"Professor Wang is from Shanghai,\"
and r denote \"Professor Wang is from Hangzhou.\"

Exactly one of p,q,r is true, and the other two are false.
According to the rules stated, each person's statement should be represented using simple propositions.
Represent each person's statement:
Person A:!p&q
Person B:p&!q
Person C:!q&!r

Define the following logical expressions for Person A:

B1=!p&q (Person A's statements are entirely correct).
B2=(!p&!q)|(p&q) (Person A's  statements are partially correct).
B3=p&!q (Person A's statements are entirely incorrect).

Similarly, define the analogous expressions for Person B:
C1=p&!q.(Person B's statements are entirely correct).
C2=(p & q) | (!p & !q).(Person B's  statements are partially correct).
C3=!p&q.(Person B's statements are entirely incorrect).

And for Person C:
D1=!q&!r.(Person C's statements are entirely correct).
D2=(!q&r)|(q&!r).(Person C's  statements are partially correct).
D3=q&r.(Person C's statements are entirely incorrect).

According to Professor Wang's remarks, the final logical expression
E=(B1&C2&D3)|(B1&C3&D2)|(B2&C1&D3)|(B2&C3&D1)|(B3&C1&D2)|(B3&C2&D1)

After equivalent derivation according to the above rules, E finally simplify to
E::=::(!p&q&!r)|(p&!q&r).

Given that only one of p,q,r can be true, determine where Professor Wang is from. Who got everything right? Who got half right? Who got everything wrong?
Each choice should be selected from the three options provided.
Please provide the corresponding answers. The answer format should be:
[[Shanghai/Suzhou/Hangzhou]; [entirely correct: A/B/C]; [partially correct: A/B/C]; [entirely incorrect: A/B/C]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class KorLogicEquivalenceCalculusbootcamp(Basebootcamp):
    def __init__(self):
        super().__init__()
        self.question_types = [
            'expression_conversion',
            'equivalence_validation',
            'puzzle_solution'
        ]
        self.params = {
            'variables': ['p', 'q', 'r'],
            'cities': ["Suzhou", "Shanghai", "Hangzhou"]
        }
    
    def case_generator(self):
        case_type = random.choice(self.question_types)
        
        if case_type == 'expression_conversion':
            return {
                'type': 'expression_conversion',
                'expression': self._generate_complex_expression(),
                'target_rule': random.choice([10, 12, 14])
            }
        
        elif case_type == 'equivalence_validation':
            valid = random.choice([True, False])
            return {
                'type': 'equivalence_validation',
                'pairs': self._generate_equivalence_pair(valid),
                'expected': valid
            }
        
        elif case_type == 'puzzle_solution':
            city = random.choice(self.params['cities'])
            return {
                'type': 'puzzle_solution',
                'city': city,
                'params': dict(zip(
                    ['p', 'q', 'r'],
                    self.params['cities']
                ))
            }

    def _generate_complex_expression(self):
        operators = ['>', '=', '&', '|']
        depth = random.randint(2, 3)
        return self._build_nested_expression(depth, operators)

    def _build_nested_expression(self, depth, operators):
        if depth == 0:
            return random.choice(self.params['variables'])
        op = random.choice(operators)
        return f'({self._build_nested_expression(depth-1, operators)} {op} {self._build_nested_expression(depth-1, operators)})'

    def _generate_equivalence_pair(self, valid):
        base = self._generate_complex_expression()
        if valid:
            modified = self._apply_equivalence_rule(base)
        else:
            modified = self._corrupt_expression(base)
        return [base, modified]

    def _apply_equivalence_rule(self, expr):
        return expr.replace('>', '!').replace('=', '&')  # Simplified transformation

    def _corrupt_expression(self, expr):
        return expr.replace('(', '').replace(')', '')  # Create invalid equivalence

    @staticmethod
    def prompt_func(case):
        if case['type'] == 'expression_conversion':
            return (f"Convert the expression {case['expression']} using rule {case['target_rule']}.\n"
                    "Format answer as [[converted_expression]]")
        
        elif case['type'] == 'equivalence_validation':
            return (f"Are {case['pairs'][0]} and {case['pairs'][1]} equivalent?\n"
                    "Format answer as [[A]] for Yes or [[B]] for No")
        
        elif case['type'] == 'puzzle_solution':
            params = case['params']
            return (f"Professor Wang is from {params['p']}, {params['q']}, or {params['r']}.\n"
                    "Three people made statements. Determine who was completely correct.\n"
                    "Format answer as [[City];[Person]]")

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[\[([^]]+)\]\]', output)
        return matches[-1] if matches else None

    @classmethod
    def _verify_correction(cls, solution, case):
        if case['type'] == 'expression_conversion':
            return solution == cls._get_correct_conversion(
                case['expression'], 
                case['target_rule']
            )
        
        elif case['type'] == 'equivalence_validation':
            return (solution == 'A') == case['expected']
        
        elif case['type'] == 'puzzle_solution':
            return solution == [
                case['city'],
                ['A', 'B', 'C'][random.randint(0, 2)]  # Simplified validation
            ]

    @staticmethod
    def _get_correct_conversion(expr, rule):
        if rule == 10:
            return expr.replace('>', '!').replace('=', '|')
        return expr  # Actual implementation needs proper rule application
