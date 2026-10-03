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
1. GB Paradox

(1) Definition:
The use of the same observation to draw contradictory predictive conclusions through different inductive reasoning.

(2) Rules:
- Premise: An observed phenomenon p is noted.
- If hypothesis q_1 is assumed, then p.
- If hypothesis q_2 is assumed, then p.
- Since q_1 and q_2 are contradictory, they cannot both be true.
- Conclusion: In such cases, it is not possible to determine which hypothesis is correct, and further rules are needed to distinguish which inductions are reasonable.

(3) Symbolic Representation:
- p
- q_1 → p
- q_2 → p
- q_1 ∧ q_2 → ⊥

2. BC Paradox

(1) Definition:
The paradox that arises from the intuitive contradiction in the confirmation of a universal hypothesis through equivalence conditions and confirmation standards.

(2) Rules:
- Premise: Universal hypothesis ∀x(R(x)→B(x)).
- According to the confirmation standard, R(a)∧B(a) confirms ∀x(R(x)→B(x)).
- According to the equivalence condition, ∀x(R(x)→B(x)) is equivalent to ∀x(¬B(x)→¬R(x)).
- According to the equivalence condition, ¬B(b)∧¬R(b) also confirms ∀x(R(x)→B(x)).
- Conclusion: In such cases, care must be taken when determining the confirmation standards to avoid contradiction.

(3) Symbolic Representation:
- ∀x(R(x)→B(x))
- R(a)∧B(a)→∀x(R(x)→B(x))
- ∀x(R(x)→B(x))≡∀x(¬B(x)→¬R(x))
- ¬B(b)∧¬R(b)→∀x(R(x)→B(x))

3. LS Paradox

(1) Definition:
The paradox that arises when multiple hypotheses are reasonably accepted, and the logical consequence derived from these hypotheses contradicts another reasonably accepted hypothesis.

(2) Rules:
- Premise: Hypotheses r_1, r_2, ..., r_n are all reasonable.
- The logical consequence of these hypotheses, r_1 ∧ r_2 ∧ ... ∧ r_n, is also reasonable.
- If the conjunction of the propositions r_1 ∧ r_2 ∧ ... ∧ r_n contradicts hypothesis r_n+1, then the reasonableness of these hypotheses needs to be reassessed.
- Conclusion: In such cases, a balance point for the conditions of reasonableness needs to be found to avoid contradiction.

(3) Symbolic Representation:
- r_1, r_2, ..., r_n (reasonable hypotheses)
- r_1 ∧ r_2 ∧ ... ∧ r_n → Reasonable
- If r_1 ∧ r_2 ∧ ... ∧ r_n ∧ r_n+1 → ⊥, then reassess the reasonableness.Example questions are as follows:

<example 0>
The scientist observed that rats escaped whenever the lights in the lab were on. He proposes two hypotheses: 
(1) the rats escape because the light is on; 
(2) the rats escape because they hear a sound. 
If the light and the sound contradict each other, which paradox is this?

A. GB Paradox
B. BC Paradox
C. LS Paradox

Give your answer in [[A/B/C]] format.
</example 0>

<example 1>
Suppose we have a holomorphic hypothesis: 
\"All swans are white\" (∀x(S(x) → W(x))), 
where S(x) means that x is a swan and W(x) means that x is white.

- According to the corroboration criterion, if we see a white swan, this corroborates the holomorphic hypothesis.
- However, if we see a white object that is not a swan, this also confirms the holomorphic hypothesis, according to the inverse proposition.

Which paradox is this? 

A. GB Paradox
B. BC Paradox
C. LS Paradox

Please give your answer in [[A/B/C]] format.
</example 1>

<example 2>
Suppose we accept the following two hypotheses:
(A) All birds fly;
(B) Penguins are birds.

However, penguins do not fly.

Which paradox does this belong to?

A. GB Paradox
B. BC Paradox
C. LS Paradox

Give your answer in [[A/B/C]] format.
</example 2>

<example 3>
Given the following logical expression, which paradox does this belong to?

- Expression: (q_1 → p) ∧ (q_2 → p) ∧ (q_1 ∧ q_2 → ⊥)

A. GB Paradox
B. BC Paradox
C. LS Paradox

Please give your answer in [[A/B/C]] format.
</example 3>

<example 4>
Consider the following holomorphic hypothesis and corroboration conditions, which paradox does this belong to?

- ∀x(R(x) → B(x)): for all x, if R(x) then B(x)
- R(a) ∧ B(a): R(a) and B(a) hold simultaneously
- Expression:
    - ∀x(R(x)→B(x)) ≡ ∀x(¬B(x)→¬R(x))
    - ¬B(b) ∧ ¬R(b) → ∀x(R(x) → B(x))

A. GB Paradox
B. BC Paradox
C. LS Paradox

Please give your answer in [[A/B/C]] format.
</example 4>

<example 5>
Suppose we have the following plausible hypothesis:

- r_1: All apples are fruits
- r_2: All fruits are rich in vitamins
- r_3: Apples contain vitamin C

If we find a hypothesis r_4 that contradicts r_3, we need to:

A. Ignore r_4 
B. Reevaluate r_1, r_2, r_3 
C. Accept r_4 as the new truth

Please give your answer in [[A/B/C]] format.
</example 5>

<example 6>
Consider the following logical expression, which paradox does it represent?

- p: an event
- q: a hypothesis
- Expression: (p → q) ∧ (¬p → q)

A. GB Paradox
B. BC Paradox
C. LS Paradox

Please give your answer in [[A/B/C]] format.
</example 6>

<example 7>
If the following logical expression is true, does it represent the BC Paradox?

- ∀x(R(x) → B(x)): for all x, if R(x) then B(x)
- ¬B(b) ∧ ¬R(b): for some b, ¬B(b) and ¬R(b) hold simultaneously
- Expression: ¬B(b) ∧ ¬R(b) → ∀x(R(x) → B(x))

A. Yes
B. No.

Please give your answer in [[A/B]] format.
</example 7>

<example 8>
Suppose we have the following set of plausible hypotheses:

{r_1, r_2, ... , r_n}

If we add a new hypothesis r_n+1 which contradicts the ensemble proposition r_1 ∧ r_2 ∧ ... ∧ r_n contradicts, we need:

A. accept r_n+1 as the new truth 
B. reassess the plausibility of all hypotheses 
C. ignore r_n+1

Please give your answer in [[A/B/C]] format.
</example 8>

<example 9>
Consider the following logical expression, does it represent the LS Paradox?

- r_1: Assumption 1
- r_2: Assumption 2
- Expression: ((r_1 ∧ r_2) → ⊥)

A. Yes
B. No.

Please give your answer in [[A/B]] format.
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class KorLogicInductionParadoxbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__(**params)
        self.example_pool = [
            {
                "phenomenon": "实验室灯亮时老鼠逃跑",
                "hypotheses": ["灯光导致逃跑", "声音导致逃跑"],
                "contradiction": "灯光与声音开启条件互斥"
            },
            {
                "phenomenon": "火山喷发前动物躁动",
                "hypotheses": ["地震前兆假说", "气压变化假说"],
                "contradiction": "两种地质现象不会同时发生"
            }
        ]
        self.symbolic_templates = [
            "({q1} → {p}) ∧ ({q2} → {p}) ∧ ({q1} ⊻ {q2})",
            "{p} ⇒ ({h1} ∨ {h2}), 其中{h1}与{h2}矛盾"
        ]

    def case_generator(self):
        if random.random() < 0.5:
            return self._generate_example_case()
        else:
            return self._generate_symbolic_case()

    def _generate_example_case(self):
        case = random.choice(self.example_pool)
        return {
            "type": "example",
            "phenomenon": case["phenomenon"],
            "hypotheses": case["hypotheses"],
            "contradiction": case["contradiction"],
            "correct_answer": "A"
        }

    def _generate_symbolic_case(self):
        template = random.choice(self.symbolic_templates)
        elements = {
            'p': random.choice(["现象X", "观测结果Y", "事件Z"]),
            'q1': random.choice(["假设α", "理论Q1", "推论A"]),
            'q2': random.choice(["假设β", "理论Q2", "推论B"]),
            'h1': random.choice(["H₁", "理论Γ"]),
            'h2': random.choice(["H₂", "理论Δ"])
        }
        return {
            "type": "symbolic",
            "expression": template.format(**elements),
            "correct_answer": "A"
        }

    @staticmethod
    def prompt_func(question_case):
        if question_case["type"] == "example":
            desc = question_case
            return f'''观察到现象：{desc["phenomenon"]}
提出的两个互斥假设：
1. {desc["hypotheses"][0]}
2. {desc["hypotheses"][1]}
已知：{desc["contradiction"]}

这属于哪个逻辑悖论？
A. GB Paradox（矛盾假设归纳悖论）
B. BC Paradox（等价确认悖论）
C. LS Paradox（多重假设冲突悖论）
答案请用[[A/B/C]]格式给出'''
        else:
            return f'''请分析以下逻辑表达式对应的悖论类型：
{question_case["expression"]}

选项：
A. GB Paradox（具有矛盾假设的归纳悖论）
B. BC Paradox（基于等价转换的验证悖论） 
C. LS Paradox（多重合理假设的冲突悖论）
答案格式：[[答案字母]]'''

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[\[([A-Ca-c])]]', output)
        return matches[-1].upper() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        # 基础格式验证
        if solution != identity["correct_answer"]:
            return False
        
        # 语义结构验证
        if identity["type"] == "example":
            return (
                len(identity["hypotheses"]) == 2 and
                identity["contradiction"] != "" and
                identity["phenomenon"] != ""
            )
        else:
            expr = identity["expression"]
            return (
                "→" in expr and 
                "∧" in expr and 
                ("⊻" in expr or "矛盾" in expr)
            )
