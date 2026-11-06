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
Custom Formal Fallacy Naming Rules:
- NegAnt Method: If P, then Q. Not P, erroneously concludes Not Q.
- AffCons Method: If P, then Q. Q is true, erroneously concludes P.
- CondSwap Method: If P then Q, erroneously believes that if Q then P.
- IncorrNeg Method: If P then Q, erroneously concludes that if Not P then Not Q.
- DisjSyl Method: Either P or Q. Knowing Q, erroneously concludes Not P.
- QuantSwitch Method: ∀x∃y R(x, y), therefore, ∃y∀x R(x, y). Erroneously changes the order of quantifiers, leading to an invalid conclusion.
- IllTrans Method: ∀x (Sx → Px), therefore, ∀x (Px → Sx). It is erroneous to infer \"all P are S\" from \"all S are P\". Similarly, from ∃x (Sx ∧ ¬Px), it is erroneous to infer ∃x (Px ∧ ¬Sx). Erroneously converts the terms in the proposition, leading to an invalid conclusion.
- IncorrInf Method: From ∃x (Sx ∧ Px) infer ∃x (Sx ∧ ¬Px), and from ∃x (Sx ∧ ¬Px) infer ∃x (Sx ∧ Px). It is erroneous to infer \"some S are not P\" from \"some S are P\" and vice versa. An invalid inference is made about propositions with existential quantifiers.
- InvSubError Method: `K(x, y)` indicates that individual x knows that y is true. `R(x, y, z)` indicates that x has a relationship z with y. `SubError(x, y, z)` indicates a substitution error when incorrectly applying knowledge or attributes about y to z.
- LetClauseShift Method: When the structure of a statement is incorrectly adjusted or interpreted, causing the original intent or logical relationship to be misrepresented. For example, a shift in the structure of a let clause leads to an invalid inference.Example questions are as follows:

<example 0>
If Li Gua murdered his boss, then he is an evil person. Li Gua did not murder his boss, so Li Gua is not an evil person. This reasoning is obviously unsound. The act of murder (regardless of whether it is the boss) can indeed make a person an evildoer, but evildoers are not limited to murderers; there are many other forms of wrongdoing. Therefore, it cannot be concluded that \"Li Gua is not an evil person\" from \"Li Gua did not murder someone.\"

What type of formal fallacy is this?

A. NegAnt Method
B. AffCons Method
C. CondSwap Method
D. IncorrNeg Method
E. DisjSyl Method
F. QuantSwitch Method
G. IllTrans Method
H. IncorrInf Method
I. InvSubError Method
J. LetClauseShift Method

Please give your answer in the format [[A/B/C/D/E/F/G/H/I/J]].
</example 0>

<example 1>
If Wang Meng is an internet enthusiast, then he will spend a long time online. Wang Meng does indeed spend a long time online, so Wang Meng must be an internet enthusiast. This reasoning is invalid. Even if the premises are true, the conclusion can be false. For example, Wang Meng spends a long time online because it is his job. He has started to hate his job because he is always dealing with the virtual world of the internet, which has made him a bit confused about reality and truth, losing a sense of security, and not as real and substantial as interacting with real people.

What type of formal fallacy is this?

A. NegAnt Method
B. AffCons Method
C. CondSwap Method
D. IncorrNeg Method
E. DisjSyl Method
F. QuantSwitch Method
G. IllTrans Method
H. IncorrInf Method
I. InvSubError Method
J. LetClauseShift Method

Please give your answer in the format [[A/B/C/D/E/F/G/H/I/J]].
</example 1>

<example 2>
If x is a positive even number, then x is a natural number, so, if x is a natural number, then x is a positive even number. Everyone who has been to elementary school understands that this reasoning is incorrect.

What type of formal fallacy is this?

A. NegAnt Method
B. AffCons Method
C. CondSwap Method
D. IncorrNeg Method
E. DisjSyl Method
F. QuantSwitch Method
G. IllTrans Method
H. IncorrInf Method
I. InvSubError Method
J. LetClauseShift Method

Please give your answer in the format [[A/B/C/D/E/F/G/H/I/J]].
</example 2>

<example 3>
If all countries in the Middle East disarm, it will bring peace to the region, so if the countries in the Middle East have not disarmed, there will be no peace in the region. The premise is true, but the conclusion is obviously not valid, because it is impossible for all countries in the Middle East to completely disarm. According to this conclusion, there will never be peace in the Middle East, but the real situation will not be so.

What type of formal fallacy is this?

A. NegAnt Method
B. AffCons Method
C. CondSwap Method
D. IncorrNeg Method
E. DisjSyl Method
F. QuantSwitch Method
G. IllTrans Method
H. IncorrInf Method
I. InvSubError Method
J. LetClauseShift Method

Please give your answer in the format [[A/B/C/D/E/F/G/H/I/J]].
</example 3>

<example 4>
Du Fu is either a great poet or a person from the Tang Dynasty, and Du Fu is a world-renowned great poet, so Du Fu is not a person from the Tang Dynasty. Since the disjunctive proposition as the premise is compatible, each branch proposition can be true at the same time, this reasoning is incorrect.

What type of formal fallacy is this?

A. NegAnt Method
B. AffCons Method
C. CondSwap Method
D. IncorrNeg Method
E. DisjSyl Method
F. QuantSwitch Method
G. IllTrans Method
H. IncorrInf Method
I. InvSubError Method
J. LetClauseShift Method

Please give your answer in the format [[A/B/C/D/E/F/G/H/I/J]].
</example 4>

<example 5>
Considering the domain of individuals as natural numbers and R representing the \"less than\" relationship, ∀x∃yR(x, y) states that for any natural number, you can find another natural number greater than it, meaning there is no largest natural number. However, ∃y∀xR(x, y) suggests that there is a natural number greater than any other natural number, implying the existence of a largest natural number. Here, the premise is true, but the conclusion is false, making the reasoning invalid.

What type of formal fallacy is this?

A. NegAnt Method
B. AffCons Method
C. CondSwap Method
D. IncorrNeg Method
E. DisjSyl Method
F. QuantSwitch Method
G. IllTrans Method
H. IncorrInf Method
I. InvSubError Method
J. LetClauseShift Method

Please give your answer in the format [[A/B/C/D/E/F/G/H/I/J]].
</example 5>

<example 6>
\"All Chinese billionaires are Chinese people,\" so \"all Chinese people are Chinese billionaires.\" The premise is true, but the conclusion is false, making the reasoning invalid.

What type of formal fallacy is this?

A. NegAnt Method
B. AffCons Method
C. CondSwap Method
D. IncorrNeg Method
E. DisjSyl Method
F. QuantSwitch Method
G. IllTrans Method
H. IncorrInf Method
I. InvSubError Method
J. LetClauseShift Method

Please give your answer in the format [[A/B/C/D/E/F/G/H/I/J]].
</example 6>

<example 7>
Given: Some students are doctors. Erroneous inference: Therefore, some students are not doctors.

What type of formal fallacy is this?

A. NegAnt Method
B. AffCons Method
C. CondSwap Method
D. IncorrNeg Method
E. DisjSyl Method
F. QuantSwitch Method
G. IllTrans Method
H. IncorrInf Method
I. InvSubError Method
J. LetClauseShift Method

Please give your answer in the format [[A/B/C/D/E/F/G/H/I/J]].
</example 7>

<example 8>
Xiao Qiang knows that Lu Xun is Lu Xun, and Lu Xun is the brother of the biologist Zhou Jianren, so Xiao Qiang knows that Lu Xun is the brother of the biologist Zhou Jianren. This reasoning is invalid; it incorrectly infers a proposition about Xiao Qiang's knowledge from a proposition in the real world, creating a logical fallacy.

What type of formal fallacy is this?

A. NegAnt Method
B. AffCons Method
C. CondSwap Method
D. IncorrNeg Method
E. DisjSyl Method
F. QuantSwitch Method
G. IllTrans Method
H. IncorrInf Method
I. InvSubError Method
J. LetClauseShift Method

Please give your answer in the format [[A/B/C/D/E/F/G/H/I/J]].
</example 8>

<example 9>
Suppose a company manager (let's call him Manager M) announces a new policy: \"All employees (E) will receive a bonus (B) after completing a project (P).\" However, an employee (let's call him Employee A) misunderstands this statement, thinking that \"only when an employee receives a bonus (B) have they completed a project (P).\"

What type of formal fallacy is this?

A. NegAnt Method
B. AffCons Method
C. CondSwap Method
D. IncorrNeg Method
E. DisjSyl Method
F. QuantSwitch Method
G. IllTrans Method
H. IncorrInf Method
I. InvSubError Method
J. LetClauseShift Method

Please give your answer in the format [[A/B/C/D/E/F/G/H/I/J]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import OrderedDict
from bootcamp import Basebootcamp

class KorLogicFormalFallaciesbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.fallacy_map = OrderedDict([
            ('A', 'NegAnt Method'),
            ('B', 'AffCons Method'),
            ('C', 'CondSwap Method'),
            ('D', 'IncorrNeg Method'),
            ('E', 'DisjSyl Method'),
            ('F', 'QuantSwitch Method'),
            ('G', 'IllTrans Method'),
            ('H', 'IncorrInf Method'),
            ('I', 'InvSubError Method'),
            ('J', 'LetClauseShift Method')
        ])
        self.params = params
        
    def case_generator(self):
        correct_key = random.choice(list(self.fallacy_map.keys()))
        question_text, analysis = self.generate_question(correct_key)
        return {
            'question': question_text,
            'analysis': analysis,
            'correct_answer': correct_key,
            'options': self.fallacy_map.copy()
        }

    def generate_question(self, key):
        generators = {
            'A': self._gen_negant,
            'B': self._gen_affcons,
            'C': self._gen_condswap,
            'D': self._gen_incorrneg,
            'E': self._gen_disjsyl,
            'F': self._gen_quantswitch,
            'G': self._gen_illtrans,
            'H': self._gen_incorrinf,
            'I': self._gen_invsuberror,
            'J': self._gen_letclauseshift
        }
        return generators[key]()

    # 实现所有缺失的生成方法
    def _gen_negant(self):
        templates = [
            ("If {A} then {B}. Not {A}, therefore not {B}.", 
            "否定前件错误：通过否定条件命题的前件来错误否定后件")
        ]
        return self._fill_template(templates, 
            {'A': ['P', 'Q', 'X'], 'B': ['Q', 'R', 'Y']})

    def _gen_affcons(self):
        templates = [
            ("If {A} then {B}. {B} is true, so {A} must be true.",
            "肯定后件错误：通过肯定条件命题的后件来错误肯定前件")
        ]
        return self._fill_template(templates,
            {'A': ['P', 'Q'], 'B': ['Q', 'R']})

    def _gen_condswap(self):
        templates = [
            ("If {A} then {B}, therefore if {B} then {A}.",
            "条件倒置错误：错误交换条件命题的前后件")
        ]
        return self._fill_template(templates,
            {'A': ['P', 'Q'], 'B': ['Q', 'R']})

    def _gen_incorrneg(self):
        templates = [
            ("If {A} then {B}, therefore if ¬{A} then ¬{B}.",
            "错误否定推演：错误地将原命题的否定作为结论")
        ]
        return self._fill_template(templates,
            {'A': ['P', 'Q'], 'B': ['Q', 'R']})

    def _gen_disjsyl(self):
        templates = [
            ("Either {A} or {B}. {B} is true, so {A} is false.",
            "析取谬误：错误否定相容析取命题的另一选项")
        ]
        return self._fill_template(templates,
            {'A': ['P', 'X'], 'B': ['Q', 'Y']})

    def _gen_quantswitch(self):
        templates = [
            ("∀x∃y R(x,y) therefore ∃y∀x R(x,y)",
            "量词换序错误：错误交换全称量词和存在量词的位置")
        ]
        return random.choice(templates)

    def _gen_illtrans(self):
        templates = [
            ("All {S} are {P}, therefore all {P} are {S}.",
            "非法换位：错误转换全称命题的主谓项位置")
        ]
        elements = {'S': ['S', 'A'], 'P': ['P', 'B']}
        return self._fill_template(templates, elements)

    def _gen_incorrinf(self):
        templates = [
            ("Some {S} are {P}, therefore some {S} are not {P}.",
            "存在量词谬误：错误转换存在命题的肯定与否定")
        ]
        return self._fill_template(templates,
            {'S': ['S', 'A'], 'P': ['P', 'B']})

    def _gen_invsuberror(self):
        templates = [
            ("Knowing {X} is {Y}, therefore {X} knows {Z}.",
            "无效替换错误：错误替换认知命题中的嵌套内容")
        ]
        return self._fill_template(templates, {
            'X': ['A', 'B'], 
            'Y': ['P', 'Q'], 
            'Z': ['R', 'S']
        })

    def _gen_letclauseshift(self):
        templates = [
            ("Original statement: {A}, Misinterpretation: {B}",
            "条款结构篡改：错误解释逻辑连接词的辖域范围")
        ]
        return self._fill_template(templates, {
            'A': ["∀x(P(x)→Q(x))", "∃x(S(x)∧T(x))"],
            'B': ["∀x(P(x)∧Q(x))", "∃x(S(x)→T(x))"]
        })

    def _fill_template(self, templates, elements):
        template, analysis = random.choice(templates)
        filled = template.format(**{
            k: random.choice(v) for k, v in elements.items()
        })
        return filled, analysis

    @staticmethod
    def prompt_func(question_case) -> str:
        options = "\n".join([f"{k}. {v}" for k, v in question_case['options'].items()])
        return f"""请分析以下逻辑谬误类型：

{question_case['question']}

{question_case['analysis']}

备选类型：
{options}

请将答案用双括号包裹，例如[[A]]。"""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[\[([A-J])]]', output)
        return matches[-1].upper() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            return solution == identity['correct_answer']
        except:
            return False
