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
Propositions are represented using p1, p2, ..., pn.
Let p1 be a proposition, the compound proposition \"not p1\" is represented as ~p1.
Let p1 and p2 be two propositions, the compound proposition \"p1 and p2\" is represented as p1&p2.
Let p1 and p2 be two propositions, the compound proposition \"p1 or p2\" is represented as p1||p2.
Let p1 and p2 be two propositions, the compound proposition \"if p1, then p2\" is represented as p1=::>p2.
Let p1 and p2 be two propositions, the compound proposition \"p1 if and only if p2\" is represented as p1=p2.
A single proposition and proposition constants can be called a formula.
Formulas are represented using F1, F2, ..., Fn.
If F1 is a formula, then ~F1 is also a formula.
If F1 and F2 are formulas, then F1&F2, F1||F2, F1=::>F2, F1=F2 are also formulas.
Level A Formula: The most basic proposition unit, without logical connectives or nested structures.
Level B Formula: A formula containing one logical connective, and the connected two propositions are both Level A formulas.For example, p1.
Level C Formula: A formula containing nested logical connectives and at least one Level B formula.For example, ~p1.
Other levels of logic follow by analogy; when higher than Level Z, they are classified as Z+n (n≥1).For example, ~(~p1).
True assignment of a proposition: A proposition p1 is assigned as ✓, indicating that p1 is true.
False assignment of a proposition: A proposition p1 is assigned as x, indicating that p1 is false.
True assignment of a formula: If the formula is (~p1&~p2&~p3)||(p1&p2), then x|x|x,✓|✓|x are true assignments of the formula.
False assignment of a formula: If the formula is (~p1&~p2&~p3)||(p1&p2), then x|x|✓,x|✓|✓ are false assignments of the formula.
For p1=::>p2, only ✓|x is a false assignment of the formula.
A formula that is true under all assignments is called a Truth Formula.
A formula that is false under all assignments is called a Falsehood Formula.
Recursive definition of formulas: Any formula containing nested logical connectives can be decomposed recursively to obtain its subformulas and their logical connective structures.
Priority of logical connectives: The priority of logical connectives from high to low is as follows: ~ (not), & (and), || (or), =::> (if...then), = (if and only if).
Without parentheses, operations are performed according to priority.
Equivalence of formulas: Two formulas are equivalent if they have the same truth value under all assignments.
Equivalent formulas can be interchanged.
Simplification of formulas: Formulas can be simplified through logical rules to obtain a more concise form without changing the truth value of the formula.Example questions are as follows:

<example 0>
Given:
p1: Blue is a common color.
p2: Yellow is a common color.
p3: \sqrt{3}  is irrational.
p4: 5 is irrational.

Symbolize the following propositions:
(1) Blue and yellow are both common colors.
(2) Either \sqrt{3} or 5 is irrational.
(3) Exactly one of \sqrt{3} and 5 is irrational.

Specify that only &,||,~ can be used for this question.
Please format your answers as [[ ];[ ];[ ]], where each bracketed section contains the corresponding logical expression for each proposition.
</example 0>

<example 1>
Given:
p1: 4 is even.
p2: 5 is odd.

Symbolize the following propositions:
(1) Only if 4 is even, 5 is odd.
(2) If 4 is even, then 5 is even.
(3) Only 4 being even makes 5 even.
(4) 4 is even if and only if 5 is odd.
Please format your answers as [[ ];[ ];[ ];[ ]], where each bracketed section contains the corresponding logical expression for each proposition.
</example 1>

<example 2>
Find the truth values and falsity values of the following formulas.
(1) ~(p1&p2&~p3)
(2) (~p1&p2)=::>(p1=p3)
The answer format  is [[T:...;F:...];[T:...;F:...]]. If there are multiple values in T(F), they should be separated by commas (,).For example [[T:✓|✓|✓;F:x|x|x];[T:x|x|x,x|x|✓;F:✓|✓|✓]]
</example 2>

<example 3>
Find the falsity values of the following formulas:
(1)~(~p1&p2)||~p3
(2)(~p2||p3)&(p1=::>p2)
(3)(p1=::>p2)&(~(p1&p3)||p1)
The answer format is [[F:...];[F:...];[F:...]].If there are multiple values in F, they should be separated by commas (,).For example [[F:x|x|x];[F:✓|✓|✓];[F:x|x|x,✓|✓|✓]]
</example 3>

<example 4>
Please determine the level of the formula (~p1&p2)=::>p3. 
The answer should be given as a single letter from A to Z, or Z+n (where n is a number greater than or equal to 1).The answer format should be like [[A]].
</example 4>

<example 5>
Please determine the level of the formula (~(p1=::>~p2))&((p3||p4)=~p1). 
The answer should be given as a single letter from A to Z, or Z+n (where n is a number greater than or equal to 1).The answer format should be like [[A]].
</example 5>

<example 6>
Determine whether the following formula is:
A. Truth Formula, B. Falsehood Formula, C. Neither.

(1) p1=::>(p1||p2||p3)
(2) (p1=::>~p1)=::>~p2

Provide the answer as a single letter.
Separate the answers between each sub-question with ;.
Eventually the entire answer should be formatted like [[A];[A]].
</example 6>

<example 7>
Determine whether the following formula is:
A. Truth Formula, B. Falsehood Formula, C. Neither.

(1)~(p1=::>p2)&p2
(2) (p1&p3)=(~p1&~p2)

Provide the answer as a single letter.
Separate the answers between each sub-question with ;.
Eventually the entire answer should be formatted like [[A];[A]].
</example 7>

<example 8>
Given that (p1=::>(p1||p2))&((p1&p2)=::>p1) is a Truth Formula, determine the type of the following formulas:
(1) p1=::>(p1||p2)
(2) (p1&p2)=::>p1

A. Truth Formula, B. Falsehood Formula, C. Neither.

Provide the answer as a single letter.
Separate the answers between each sub-question with ;.
Eventually the entire answer should be formatted like [[A];[A]].
</example 8>

<example 9>
Given that p1=::>(p1||p2) is a Truth Formula,
 ~(p1=::>p2)&p2 is a Falsehood Formula, 
determine the type of the following formulas:
(1) (p1=::>(p1||p2))&(~(p1=::>p2)&p2)
(2) (p1=::>(p1||p2))||(~(p1=::>p2)&p2)

A. Truth Formula, B. Falsehood Formula, C. Neither.

Provide the answer as a single letter.
Separate the answers between each sub-question with ;.
Eventually the entire answer should be formatted like [[A];[A]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class KorLogicPropositionalLogicFormalizationbootcamp(Basebootcamp):
    def __init__(self, problem_type='symbolize', num_propositions=3, max_questions=3, allowed_connectives=None):
        super().__init__()
        self.problem_type = problem_type
        self.num_propositions = num_propositions
        self.max_questions = max_questions
        self.allowed_connectives = allowed_connectives if allowed_connectives is not None else ['&', '||', '~']
        self.proposition_templates = [
            "is even", "is odd", "is a prime number", "is a common color",
            "is divisible by 3", "is a fruit", "is considered lucky"
        ]
        self.subjects = ["2", "4", "5", "7", "Blue", "Red", "Square root of 3", "Pi"]

    def case_generator(self):
        if self.problem_type == 'symbolize':
            propositions = self._generate_propositions()
            questions, answers = self._generate_symbolize_questions(propositions)
            return {
                'type': 'symbolize',
                'propositions': propositions,
                'questions': questions,
                'answers': answers
            }
        else:
            raise NotImplementedError("Other problem types are not implemented yet.")

    def _generate_propositions(self):
        propositions = {}
        used_subjects = set()
        for i in range(self.num_propositions):
            while True:
                subject = random.choice(self.subjects)
                if subject not in used_subjects:
                    used_subjects.add(subject)
                    prop = random.choice(self.proposition_templates)
                    propositions[f'p{i+1}'] = f"{subject} {prop}."
                    break
        return propositions

    def _generate_symbolize_questions(self, propositions):
        questions = []
        answers = []
        variables = list(propositions.keys())
        for _ in range(self.max_questions):
            formula = self._generate_formula(variables)
            question_text = self._formula_to_natural_language(formula, propositions)
            questions.append(question_text)
            answers.append(formula)
        return questions, answers

    def _generate_formula(self, variables, depth=0):
        if depth >= 2 or len(variables) < 2:
            return random.choice(variables)
        
        connective = random.choice(self.allowed_connectives)
        if connective == '~':
            sub = self._generate_formula(variables, depth+1)
            return f'~{sub}'
        else:
            left = self._generate_formula(variables, depth+1)
            right = self._generate_formula(variables, depth+1)
            return f'({left}{connective}{right})'

    def _formula_to_natural_language(self, formula, propositions):
        formula = formula.replace('(', '').replace(')', '')
        parts = re.split(r'(&|\|\||~)', formula)
        parts = [p for p in parts if p]
        
        stack = []
        for part in parts:
            if part in ['&', '||', '~']:
                stack.append(part)
            else:
                stack.append(propositions.get(part, part))
        
        natural = []
        prev_op = None
        for item in stack:
            if item == '&':
                natural.append("and")
            elif item == '||':
                natural.append("or")
            elif item == '~':
                natural.append("It is not the case that")
            else:
                if prev_op == '~':
                    natural[-1] += f" {item}"
                else:
                    natural.append(item)
            prev_op = item if item in ['&', '||', '~'] else None
        
        return ' '.join(natural).replace(' .', '.')

    @staticmethod
    def prompt_func(question_case):
        prop_text = "Given:\n" + "\n".join(
            [f"{var}: {desc}" for var, desc in question_case['propositions'].items()]
        )
        questions_text = "Symbolize the following propositions using &, ||, ~:\n" + "\n".join(
            [f"({i+1}) {q}" for i, q in enumerate(question_case['questions'])]
        )
        return f"{prop_text}\n\n{questions_text}\n\nFormat your answers as [[...];[...];...]"

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[\[(.*?)\]\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        answers = re.split(r';\s*', last_match)
        return [ans.strip() for ans in answers]

    @classmethod
    def _verify_correction(cls, solution, identity):
        if identity['type'] != 'symbolize':
            return False
        return all(
            cls.normalize(ans) == cls.normalize(correct)
            for ans, correct in zip(solution, identity['answers'])
        )

    @staticmethod
    def normalize(formula):
        return formula.replace(' ', '').replace('(', '').replace(')', '')

