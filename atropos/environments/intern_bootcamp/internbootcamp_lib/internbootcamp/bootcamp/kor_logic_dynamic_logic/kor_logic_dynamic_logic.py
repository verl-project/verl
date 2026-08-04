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
1.Symbol Definitions
- Command: `c` represents a basic operation within a program.
- Proposition: `φ` represents a statement or condition.
- Program State: Represents the system state after the execution of a command.

2.Dynamic Operators
- Necessity Operator: `[ c ]φ` indicates that after executing command `c`, the proposition `φ` will necessarily be true.
- Possibility Operator: `⟨ c ⟩φ` indicates that after executing command `c`, the proposition `φ` may be true.

3.Axioms and Rules
- Substitution Rule:If `c` behaves the same as `d`, then `[ c ]φ` is equivalent to `[ d ]φ`.
- Sequence Rule:`[ c_1; c_2 ]φ` is equivalent to `[ c_1 ][ c_2 ]φ`.
- Choice Rule:`[ c_1 + c_2 ]φ` is equivalent to `([ c_1 ]φ ∨ [ c_2 ]φ)`.
- Loop Rule:For the loop command `c*`, `[ c* ]φ` is equivalent to `φ ∨ ([ c ][ c* ]φ)`.
- Concurrent Rule:If `c_1` and `c_2` can be executed concurrently, then `⟨ c_1 || c_2 ⟩φ` is equivalent to `⟨ c_1 ⟩⟨ c_2 ⟩φ ∨ ⟨ c_2 ⟩⟨ c_1 ⟩φ`.
- Interruption Rule:If `c_1` can interrupt `c_2`, then `⟨ c_1; c_2 ⟩φ` is equivalent to `⟨ c_1 ⟩φ`.
- Exception Rule:If `c` may trigger an exception `e`, then `[ c ]φ` is equivalent to `([ c ]φ ∧ [ e ]φ)`.
- Resource Limitation Rule:If the command `c` is subject to resource limitation `R`, then `[ c ]φ` is equivalent to `(R ∧ [ c ]φ)`.
- Dependency Rule:If the execution of `c_1` depends on `c_2`, then `[ c_1 ]φ` is equivalent to `[ c_2 ][ c_1 ]φ`.
- Priority Rule:If `c_1` has higher priority than `c_2`, then `⟨ c_1; c_2 ⟩φ` is equivalent to `⟨ c_1 ⟩⟨ c_2 ⟩φ`.
- History Rule:If the execution of `c` depends on the historical command `h`, then `[ c ]φ` is equivalent to `[ h ][ c ]φ`.
- Prediction Rule:If the system can predict the outcome of `c`, then `[ c ]φ` is equivalent to `[ predict(c) ]φ`.Example questions are as follows:

<example 0>
Express using a logical expression that after executing the command sequence c1; c2, the proposition φ will necessarily be true.
Please provide your answer in the format of [[]].
</example 0>

<example 1>
Write out a logical expression that represents the possibility of the proposition φ being true after executing the command c.
Please provide your answer in the format of [[]].
</example 1>

<example 2>
Write out a logical expression that represents the proposition φ necessarily being true after the selection of executing command c1 or c2.
Please provide your answer in the format of [[]].In all expressions, the simplest form after equivalence must be used, i.e., have the fewest occurrences of [] and <>.
</example 2>

<example 3>
If Alice is convinced that the loop command c* will continue to execute until the proposition φ is true, what logical expression should be used to represent her belief?
Please provide your answer in the format of [[]].In all expressions, the simplest form after equivalence must be used, i.e., have the fewest occurrences of [] and <>.
</example 3>

<example 4>
If Alice considers that executing the command c results in the library's open state being represented by the proposition open, 
and she believes that after executing c, it is certain that open will be true, 
how would you express Alice's belief in logical terms?

Please provide your answer in the format of [[]].
</example 4>

<example 5>
If Alice is convinced that the loop command c* will persist in execution until the proposition φ is true, what logical expression should be used to represent her belief?
Please provide your answer in the format of [[]].
</example 5>

<example 6>
If the commands c and d are equivalent according to the Substitution Rule, with what logical expression is [c]φ equivalent?
Please provide your answer in the format of [[]].
</example 6>

<example 7>
According to the Concurrent Rule,
if two commands c1 and c2 can be executed simultaneously, 
and neither affects the truth value of the proposition φ, 
please write out the logical expression.

Please provide your answer in the format of [[]].In all expressions, the simplest form after equivalence must be used, i.e., have the fewest occurrences of [] and <>.
</example 7>

<example 8>
Which of the following rules applies to the situation where an exception e may be triggered after the execution of the command c1?

A. Substitution Rule
B. Sequence Rule
C. Choice Rule
D. Loop Rule
E. Concurrent Rule
F. Interruption Rule
G. Exception Rule
H. Resource Limitation Rule
I. Dependency Rule
J. Priority Rule
K. History Rule
L. Prediction Rule

Please provide your answer in the format of [[A/B/C/D/E/F/G/H/I/J/K/L]].
</example 8>

<example 9>
If Alice is certain that once the resource limitation R is satisfied, the execution of command c will inevitably result in the outcome result being true, to which of the following rules does this belong?

A. Substitution Rule
B. Sequence Rule 
C. Choice Rule 
D. Loop Rule   
E. Concurrent Rule  
F. Interruption Rule
G. Exception Rule
H. Resource Limitation Rule
I. Dependency Rule
J. Priority Rule
K. History Rule
L. Prediction Rule
    
Please provide your answer in the format of [[A/B/C/D/E/F/G/H/I/J/K/L]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class KorLogicDynamicLogicbootcamp(Basebootcamp):
    RULE_MAP = {
        'A': 'Substitution Rule',
        'B': 'Sequence Rule',
        'C': 'Choice Rule',
        'D': 'Loop Rule',
        'E': 'Concurrent Rule',
        'F': 'Interruption Rule',
        'G': 'Exception Rule',
        'H': 'Resource Limitation Rule',
        'I': 'Dependency Rule',
        'J': 'Priority Rule',
        'K': 'History Rule',
        'L': 'Prediction Rule'
    }
    
    def __init__(self, **params):
        self.max_attempts = params.get('max_attempts', 3)  # 可配置参数示例
        super().__init__(**params)
    
    def case_generator(self):
        problem_type = random.choice(['expression', 'multiple_choice'])
        return self._generate_expression_problem() if problem_type == 'expression' \
            else self._generate_multiple_choice_problem()

    def _generate_expression_problem(self):
        cases = [
            # 基础规则集
            {
                'rule': 'Sequence',
                'input': '[c1;c2]φ',
                'answer': '[c1][c2]φ',
                'variants': ['[c2;c1]φ']  # 无效变体示例
            },
            {
                'rule': 'Choice',
                'input': '[c1 + c2]φ',
                'answer': '([c1]φ ∨ [c2]φ)',
                'variants': ['([c2]φ ∨ [c1]φ)']
            },
            {
                'rule': 'Loop',
                'input': '[c*]φ',
                'answer': '(φ ∨ [c][c*]φ)',
                'variants': ['(φ∨[c][c*]φ)']
            },
            # 扩展规则集
            {
                'rule': 'Concurrent',
                'input': '⟨c1||c2⟩φ',
                'answer': '(⟨c1⟩⟨c2⟩φ ∨ ⟨c2⟩⟨c1⟩φ)',
                'variants': ['⟨c2||c1⟩φ']
            },
            {
                'rule': 'Interruption',
                'input': '⟨c1;c2⟩φ',
                'answer': '⟨c1⟩φ',
                'variants': []
            }
        ]
        case = random.choice(cases)
        return {
            'type': 'expression',
            'rule': case['rule'],
            'problem': f"Convert '{case['input']}' to equivalent simplest form",
            'expected': case['answer'],
            'valid_variants': [case['answer']] + case['variants']
        }

    def _generate_multiple_choice_problem(self):
        cases = [
            {
                'scenario': "Exception e may be triggered during command execution",
                'correct': 'G',
                'distractors': ['F', 'L']
            },
            {
                'scenario': "Execution depends on historical command h",
                'correct': 'K',
                'distractors': ['I', 'J']
            },
            {
                'scenario': "Commands c1 and c2 can be executed in parallel",
                'correct': 'E',
                'distractors': ['J', 'F']
            }
        ]
        case = random.choice(cases)
        options = random.sample(
            [chr(65+i) for i in range(12) if chr(65+i) != case['correct']],
            3
        ) + [case['correct']]
        random.shuffle(options)
        
        return {
            'type': 'multiple_choice',
            'scenario': case['scenario'],
            'correct': case['correct'],
            'options': options
        }

    @staticmethod
    def prompt_func(question_case):
        if question_case['type'] == 'expression':
            return f"""Apply logical rules to simplify the expression:
            
{question_case['problem']}

Rules available:
- {question_case['rule']} Rule

Format your answer within [[double brackets]]."""
        else:
            options = '\n'.join(
                [f"{opt}: {KorLogicDynamicLogicbootcamp.RULE_MAP[opt]}" 
                 for opt in question_case['options']]
            )
            return f"""Which rule applies to this scenario?

Scenario: {question_case['scenario']}

Options:
{options}

Answer format: [[LETTER]]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[\[(.*?)\]\]', output)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if identity['type'] == 'multiple_choice':
            return solution.upper() == identity['correct']
        else:
            return cls._normalize(solution) in identity['valid_variants']

    @staticmethod
    def _normalize(expr):
        # 统一表达式格式处理
        return re.sub(r'\s+', '', expr).replace('⟨', '[').replace('⟩', ']')
