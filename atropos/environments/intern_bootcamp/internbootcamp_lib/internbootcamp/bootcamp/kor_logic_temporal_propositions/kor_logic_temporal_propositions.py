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
Time Propositions:
1. Symbol \"H\" represents \"past point in time\".
2. Symbol \"A\" represents \"past period of time\".
3. Symbol \"F\" represents \"future point in time\".
4. Symbol \"G\" represents \"future period of time\".
5. Symbol \"T\" represents \"present\".

Time Proposition Relationships:
(1) ※ Relationship:
- Pairs: Ap and H¬p; A¬p and Hp; Gp and F¬p; G¬p and Fp
- Properties: They cannot both be true, nor both false.

(2) ↦ Relationship:
- Pairs: Ap and A¬p; Gp and G¬p
- Properties: They cannot both be true, but can both be false.

(3) ⚭ Relationship:
- Pairs: Hp and H¬p; Fp and F¬p
- Properties: They cannot both be false, but can both be true.

(4) ⁂ Relationship:
- Pairs: Ap and Hp, A¬p and H¬p; Gp and Fp, G¬p and F¬p
- Properties: They can both be true, or both be false.

Time Proposition Inference Formulas:
(1) Ap ↔ H¬p
(2) A¬p ↔ ¬Hp
(3) Hp ↔ ¬A¬p
(4) H¬p ↔ ¬Ap
(5) Ap → ¬A¬p
(6) A¬p → ¬Ap
(7) ¬Hp → H¬p
(8) ¬H¬p → Hp
(9) Ap → Hp
(10) A¬p → H¬p
(11) ¬Hp → ¬Ap
(12) ¬H¬p → ¬A¬p
(13) Gp ↔ F¬p
(14) G¬p ↔ ¬Fp
(15) Fp ↔ ¬G¬p
(16) F¬p ↔ ¬Gp
(17) Gp → ¬G¬p
(18) G¬p → ¬Gp
(19) ¬Fp → F¬p
(20) ¬F¬p → Fp
(21) Gp → Fp
(22) G¬p → F¬p
(23) ¬Fp → ¬Gp
(24) ¬F¬p → ¬G¬pExample questions are as follows:

<example 0>
Symbolize the following propositions:
(1) Wang Qiang worked in Beijing for one year in the past.
(2) Lin Min has lived in Ningbo in the past.

Use p to represent the ordinary propositions.

Please provide the answers in the format [[];[]].
</example 0>

<example 1>
Symbolize the following propositions:
(1) Xiao Jin will go to England to study abroad next year.
(2) Xiao Qian will permanently settle in England.

Use p to represent the ordinary propositions.

Please provide the answers in the format [[];[]].
</example 1>

<example 2>
What relationships do the following sentences have?

(1) \"Old Li's health was good in the past\" and \"At some point in the past, Old Li's health was not very good\"
(2) \"Aunt Wang never won a major award in the past\" and \"Allow the execution of contracts\"

A. ※ Relationship       B. ↦ Relationship     C. ⚭ Relationship    D. ⁂ Relationship

Please provide the answer in the format [[A/B/C/D];[A/B/C/D]].
</example 2>

<example 3>
What relationships do the following sentences have?

(1) \"Xiao Lin will win the computer competition championship trophy\" and \"It is not true that Xiao Lin will never win the computer competition championship trophy\"
(2) \"Xiao Bai will permanently settle in the United States\" and \"Xiao Bai will settle in the United States\"

A. ※ Relationship       B. ↦ Relationship     C. ⚭ Relationship    D. ⁂ Relationship

Please provide the answer in the format [[A/B/C/D];[A/B/C/D]].
</example 3>

<example 4>
\"Old Zhao did not work in Ningbo at some point in the past\" can be inferred from \"It is not the case that Old Zhao worked in Ningbo all the time in the past.\" 
Conversely, \"It is not the case that Old Zhao worked in Ningbo all the time in the past\" can be inferred from \"Old Zhao did not work in Ningbo at some point in the past.\"

Which reasoning formulas does this correspond to?

Please give your answer in [[number]] format.
</example 4>

<example 5>
\"Dr Lee has been working on farms in the past\" leads to: \"Dr Lee has been working on farms at some time in the past\".

Which of these correspond to the inference formulae?

Please give your answer in [[number]] format.
</example 5>

<example 6>
According to reasoning formula 5, what can be inferred from \"Lao Chen has always worked diligently in the past\"?

A.It is not that Mr Chan has not been working seriously in the past.
B. Mr Chen has always been serious about his work in the future.
C. Mr Chen will start working seriously in March next year.
D. Mr Chan has not been working seriously in the past.

Please provide the answer in the format [[A/B/C/D]].
</example 6>

<example 7>
According to Reasoning Equation 21, what can be deduced from \"I will always keep on painting\"?

A. I used to stick to painting.
B. I keep on painting.
C. I will start painting tomorrow.
D. I will keep on painting.

Please provide the answer in the format [[A/B/C/D]].
</example 7>

<example 8>
Reasoning Formula 7 is consistent with what following what relationship?

A. ∗ relationship 
B. ↦ relationship 
C. ⚭ relationship 
D. ⁂ relationship

Please provide the answer in the format [[A/B/C/D]].
</example 8>

<example 9>
Reasoning Formula 17 is consistent with what following what relationship?

A. ∗ relation 
B. ↦ relation 
C. ⚭ relation 
D. ⁂ relation

Please provide the answer in the format [[A/B/C/D]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class KorLogicTemporalPropositionsbootcamp(Basebootcamp):
    def __init__(self, **params):
        default_params = {
            'symbolize_prob': 0.4,
            'relationship_prob': 0.4,
            'formula_inference_prob': 0.2,
            'action_words': ['work in Beijing', 'have good health', 'study abroad', 'settle permanently'],
            'subjects': ['Wang Qiang', 'Lin Min', 'Xiao Jin', 'Old Zhao', 'Dr Lee']
        }
        self.params = {**default_params, **params}
        
    def case_generator(self):
        case_type = random.choices(
            ['symbolize', 'relationship', 'formula_inference'],
            weights=[
                self.params['symbolize_prob'],
                self.params['relationship_prob'],
                self.params['formula_inference_prob']
            ],
            k=1
        )[0]

        if case_type == 'symbolize':
            return self._generate_symbolize_case()
        elif case_type == 'relationship':
            return self._generate_relationship_case()
        else:
            return self._generate_formula_inference_case()

    def _generate_symbolize_case(self):
        symbols = ['A', 'H']
        negation = ['', '¬']
        propositions = []
        answers = []
        
        for _ in range(2):
            sym = random.choice(symbols)
            neg = random.choice(negation)
            p = f"{sym}{neg}p"
            propositions.append(self._symbol_to_sentence(p))
            answers.append(p)
        
        return {
            'type': 'symbolize',
            'propositions': propositions,
            'answers': answers
        }

    def _generate_relationship_case(self):
        rel_defs = [
            ('※', [('Ap', 'H¬p'), ('Gp', 'F¬p')], 'A'),
            ('↦', [('Ap', 'A¬p'), ('Gp', 'G¬p')], 'B'),
            ('⚭', [('Hp', 'H¬p'), ('Fp', 'F¬p')], 'C'),
            ('⁂', [('Ap', 'Hp'), ('Gp', 'Fp')], 'D')
        ]
        rel_type = random.choice(rel_defs)
        pairs = [random.choice(rel_type[1]) for _ in range(2)]
        
        return {
            'type': 'relationship',
            'pairs': [(self._symbol_to_sentence(p[0]), self._symbol_to_sentence(p[1])) for p in pairs],
            'correct_options': [rel_type[2]]*2
        }

    def _generate_formula_inference_case(self):
        formula_map = [
            (7, '¬Hp', 'H¬p'),
            (8, '¬H¬p', 'Hp'),
            (21, 'Gp', 'Fp')
        ]
        formula = random.choice(formula_map)
        return {
            'type': 'formula_inference',
            'premise': self._symbol_to_sentence(formula[1]),
            'conclusion': self._symbol_to_sentence(formula[2]),
            'correct_formula': formula[0]
        }

    def _symbol_to_sentence(self, symbol):
        subject = random.choice(self.params['subjects'])
        action = random.choice(self.params['action_words'])
        base = {
            'Ap': f"{subject} always {action}ed in the past",
            'A¬p': f"{subject} never {action}ed during the entire past period",
            'Hp': f"At some point in the past, {subject} {action}ed",
            'H¬p': f"At some point in the past, {subject} did not {action}",
            'Gp': f"{subject} will {action} permanently in the future",
            'G¬p': f"{subject} will not {action} permanently in the future",
            'Fp': f"Next year, {subject} will {action}",
            'F¬p': f"In the future, {subject} will not {action}",
            '¬Hp': f"It is not true that {subject} {action}ed at some past time",
            '¬H¬p': f"It is not true that {subject} did not {action} at some past time"
        }
        return base.get(symbol, "Invalid symbol")

    @staticmethod
    def prompt_func(question_case):
        if question_case['type'] == 'symbolize':
            prompt = """Time Proposition Symbolization Rules:
1. H: Past point in time (e.g., "at some point")
2. A: Past period of time (e.g., "throughout the past")
3. F: Future point in time
4. G: Future period of time
5. T: Present (not used in these examples)

Symbolize these propositions:
"""
            for i, p in enumerate(question_case['propositions'], 1):
                prompt += f"({i}) {p}\n"
            prompt += "\nFormat your answer as [[symbol1];[symbol2]]"
            return prompt
        
        elif question_case['type'] == 'relationship':
            prompt = """Relationship Types:
※: Cannot be both true nor both false
↦: Cannot both be true but can both be false
⚭: Cannot both be false but can both be true
⁂: Can both be true or both be false

Analyze these pairs:
"""
            for i, (p1, p2) in enumerate(question_case['pairs'], 1):
                prompt += f"({i}) '{p1}' vs '{p2}'\n"
            prompt += "Options:\nA. ※\nB. ↦\nC. ⚭\nD. ⁂\nAnswer format: [[A/B/C/D];[A/B/C/D]]"
            return prompt
        
        else:
            prompt = f"""Given the inference rule:
Premise: {question_case['premise']}
Conclusion: {question_case['conclusion']}

Which inference formula number does this correspond to? Answer with [[number]]"""
            return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[\[(.*?)\]\]', output)
        if not matches:
            return None
        last_match = matches[-1]
        return [x.strip() for x in last_match.split(';')]

    @classmethod
    def _verify_correction(cls, solution, identity):
        if identity['type'] == 'symbolize':
            return solution == identity['answers']
        elif identity['type'] == 'relationship':
            return solution == identity['correct_options']
        elif identity['type'] == 'formula_inference':
            return solution == [str(identity['correct_formula'])]
        return False
