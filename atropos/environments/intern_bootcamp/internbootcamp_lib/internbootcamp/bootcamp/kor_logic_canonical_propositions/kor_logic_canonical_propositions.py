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
1. The symbols are defined as follows: ¶ represents obligation modality, § represents permission modality, and ‽ represents prohibition modality.
2. The four types of normative propositions relationships are:

(1) * relationship:
- Propositional pairs: ¶p and §¬p; ‽p and §p
- Nature: Both cannot be true and cannot be false.

(2) x relationship:
- Propositional pairs: ¶p and ‽p
- Nature: Both cannot be true, but can both be false.

(3) @ relationship:
- Propositional pairs: §p and §¬p
- Nature: Both cannot be false, but can both be true.

(4) % relationship:
- **Propositional pairs**: ¶p and §p; ‽p and §¬p
- **Nature**: Both can be true, and both can be false.

3. Normative reasoning formulas
(1) ¶p←→¬§¬p
(2) ‽p←→¬§p
(3) §p←→¬‽p
(4) §¬p←→¬¶p
(5) ¶p→¬‽p
(6) ‽p→¬¶p
(7) ¬§p→§¬p
(8) ¬§¬p→§p
(9) ¶p→§p
(10) ‽p→§¬p
(11) ¬§p→¬¶p
(12) ¬§¬p→¬‽pExample questions are as follows:

<example 0>
Symbolize the following proposition:

\"Private enterprises are permitted to operate legally for reasonable profits, but price fraud is prohibited.\"

Let p represents: \"Private enterprises operate legally for reasonable profits\"; 
q represents: \"There is price fraud\". 

Please give the answer in the format [[]].
</example 0>

<example 1>
Symbolize the following proposition:
\"Abuse of the elderly, women, and children is not allowed.\"
 
Let p represents: \"abuse the elderly\"; 
q represents: \"abuse women\"; 
r represents: \"abuse children\". 

Please give the answer in the format [[]].
</example 1>

<example 2>
Symbolize the following proposition:
\"Playing football allows reasonable collisions, but watching movies prohibits collisions.\"
 
Let p represents: \"reasonable collisions in football\"; 
q represents: \"collisions in movies\".

Please give the answer in the format [[]].
</example 2>

<example 3>
What is the relationship between each of the following sentences?

(1) \"In public places, smoking is prohibited.\" and \"In public places, smoking is mandatory.\"
(2) \"Not fulfilling the contract is allowed.\" and \"Fulfilling the contract is allowed.\"

A. * Relation     B. x Relation    C. @ Relation   D. % Relation

Please provide the answer in the format [[A/B/C/D];[A/B/C/D]].
</example 3>

<example 4>
What is the relationship between the following sentences?

(1) \"Citizens' rights must be protected.\" and \"Not protecting citizens' rights is allowed.\"
(2) \"Caring for the growth of the next generation is mandatory.\" and \"Caring for the growth of the next generation is allowed.\"

A. x Relation   B. * Relation   C. @ Relation   D. % Relation

Please provide the answer in the format [[A/B/C/D];[A/B/C/D]].
</example 4>

<example 5>
\"Elderly abuse is prohibited\" can lead to: \"Elderly abuse is not allowed\".
Conversely, \"Elder abuse is not allowed\" can also lead to: \"Elder abuse is prohibited.\"

Which canonical reasoning formulas does this correspond to?

Please give your answer in [[number]] format.
</example 5>

<example 6>
\"Widows are allowed to remarry\" can be followed by: \"Widows are not prevented from remarrying\". 
Conversely, \"Widows are not prevented from remarrying\" also follows: \"Widows are allowed to remarry\". 

Which normative reasoning formulas does this correspond to?

Please give your answer in [[number]] format.
</example 6>

<example 7>
According to reasoning formula 9, what can be inferred from \"Citizens all have the duty to protect national property\"?

A: Allow citizens not to protect national property.
B: Prohibit citizens from protecting national property.
C: Allow citizens to protect national property.
D: Prohibit citizens from not protecting national property.

Please provide the answer in the format [[A/B/C/D]].
</example 7>

<example 8>
According to Modal Reasoning Formula 10, what can be deduced from \"loud noises are prohibited in the reading room\"?

A:Disallow not making loud noises in the reading room.
B:Allow not making loud noises in the reading room.
C:Prohibit not making loud noises in the reading room.
D:Prohibit making loud noises in the reading room.

Please provide the answer in the format [[A/B/C/D]].
</example 8>

<example 9>
Modal Reasoning Formula 7 conforms to what following what relationship?

A.* relationship 
B.x relationship 
C.@ relationship 
D.% relationship

Please provide the answer in the format [[A/B/C/D]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from itertools import permutations
from bootcamp import Basebootcamp

class KorLogicCanonicalPropositionsbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__(**params)
        self.variable_pool = params.get('variable_pool', [
            "operate legally for reasonable profits",
            "price fraud occurs",
            "abuse of vulnerable groups",
            "reasonable sports collisions", 
            "disruptive behavior in cultural venues",
            "tobacco use in public areas",
            "contract fulfillment",
            "citizens' rights protection",
            "youth development support",
            "national property protection",
            "noise in quiet zones",
            "remarriage rights"
        ])
        self.formula_pool = params.get('formula_pool', [
            (2, "‽p←→¬§p", "Prohibition-Permission Negation"),
            (3, "§p←→¬‽p", "Permission-Prohibition Duality"),
            (5, "¶p→¬‽p", "Obligation-Prohibition Exclusion"),
            (7, "¬§p→§¬p", "Permission Negation Implication"),
            (9, "¶p→§p", "Obligation-Permission Entailment"),
            (10, "‽p→§¬p", "Prohibition-Permission Consequence")
        ])
        self.relation_definitions = {
            '*': "Cannot be true/false together",
            'x': "Cannot both be true",
            '@': "Cannot both be false",
            '%': "No mutual exclusion"
        }

    def case_generator(self):
        case_type = random.choices(
            population=['relationship', 'symbolization', 'formula'],
            weights=[0.4, 0.3, 0.3],
            k=1
        )[0]
        
        if case_type == 'relationship':
            return self._gen_relationship_case()
        elif case_type == 'symbolization':
            return self._gen_symbolization_case()
        else:
            return self._gen_formula_case()

    def _gen_relationship_case(self):
        case = {
            "type": "relationship",
            "pairs": [],
            "options": ["A.*", "B.x", "C.@", "D.%"],
            "key": [],
            "relation_definitions": self.relation_definitions.copy()  # 修复1：添加定义信息
        }
        
        for _ in range(2):
            rel_type, symbol_pair = self._random_relation_pair()
            context = random.choice(self.variable_pool)
            
            statements = [
                self._symbol_to_natural(symbol_pair[0], context),
                self._symbol_to_natural(symbol_pair[1], context)
            ]
            
            case["pairs"].append(statements)
            case["key"].append(rel_type)
        
        return case

    def _gen_symbolization_case(self):
        variables = random.sample(self.variable_pool, 2)
        structure = random.choice([
            ("permitted", "prohibited"),
            ("mandatory", "prohibited"),
            ("prohibited", "permitted not"),
            ("mandatory", "permitted not")  # 修复3：新增结构类型
        ])
        
        return {
            "type": "symbolization",
            "sentence": f"{variables[0]} is {structure[0]}, but {variables[1]} is {structure[1]}.",
            "mapping": {chr(97+i): var for i, var in enumerate(variables)},
            "solution": self._structure_to_symbols(structure)
        }

    def _gen_formula_case(self):
        formula = random.choice(self.formula_pool)
        context_var = random.choice(self.variable_pool)
        
        premise = self._apply_formula_context(formula[1].split('←→')[0], context_var)
        options = self._generate_formula_options(formula, context_var)
        
        return {
            "type": "formula",
            "question": f"Which formula corresponds to: {premise}?",
            "options": options,
            "correct": formula[0]
        }

    @staticmethod
    def _random_relation_pair():
        relations = {
            '*': [('¶p', '§¬p'), ('‽p', '§p')],
            'x': [('¶p', '‽p')],
            '@': [('§p', '§¬p')],
            '%': [('¶p', '§p'), ('‽p', '§¬p')]
        }
        rel_type = random.choice(list(relations.keys()))
        return rel_type, random.choice(relations[rel_type])

    def _symbol_to_natural(self, symbol, context):
        modality_map = {
            '¶': ['must', 'is obligatory for'],
            '§': ['may', 'is permitted for'],
            '‽': ['must not', 'is prohibited for']
        }
        
        operator, proposition = symbol[0], symbol[1:]
        negation = "not " if '¬' in proposition else ""
        clean_prop = proposition.replace('¬', '')
        
        modality = random.choice(modality_map[operator])
        return f"{modality} {negation}{context}"

    def _structure_to_symbols(self, structure):
        conversion = {
            ('permitted', 'prohibited'): ('§p', '‽q'),
            ('mandatory', 'prohibited'): ('¶p', '‽q'),
            ('prohibited', 'permitted not'): ('‽p', '§¬q'),
            ('mandatory', 'permitted not'): ('¶p', '§¬q')  # 修复3：新增转换规则
        }
        return ' ∧ '.join(conversion[structure])

    def _apply_formula_context(self, formula_part, context):
        replacements = {
            'p': context,
            '§': 'permitted',
            '¶': 'mandatory',
            '‽': 'prohibited'
        }
        for k, v in replacements.items():
            formula_part = formula_part.replace(k, v)
        return formula_part.capitalize()

    def _generate_formula_options(self, formula, context):
        options = {chr(65): f"Formula {formula[0]}: {formula[2]}"}
        
        # 生成干扰项（排除正确公式）
        distractors = [f for f in self.formula_pool if f[0] != formula[0]]
        random.shuffle(distractors)
        
        for i in range(1, 4):
            if i-1 < len(distractors):
                options[chr(65+i)] = f"Formula {distractors[i-1][0]}: {distractors[i-1][2]}"
            else:  # 如果公式池不够，补充通用干扰项
                options[chr(65+i)] = f"Formula {random.randint(1,12)}: Generic Principle"
        
        return options

    @staticmethod
    def prompt_func(case):
        if case["type"] == "relationship":
            prompt = [
                "Analyze the normative relationships between these statement pairs:",
                "\n".join([f"Pair {i+1}:\n   a. {pair[0]}\n   b. {pair[1]}" 
                          for i, pair in enumerate(case["pairs"])]),
                "\nRelationship Types:",
                *[f"{opt}: {case['relation_definitions'][opt[2:]]}" 
                  for opt in case["options"]],
                "Answer format: [[SELECTION;SELECTION]] (e.g. [[A;C]])"
            ]
            return "\n".join(prompt)
        
        elif case["type"] == "symbolization":
            return (
                f"Symbolize the following regulation:\n{case['sentence']}\n\n"
                f"Variable definitions:\n" + 
                '\n'.join(f"{k}) {v}" for k,v in case["mapping"].items()) + 
                "\n\nUse modalities: ¶ (must), § (may), ‽ (prohibited)\n" 
                "Format answer: [[EXPRESSION]] (e.g. [[§p ∧ ‽q]])"
            )
        
        else:  # formula case
            return (
                f"{case['question']}\n\nOptions:\n" +
                '\n'.join(f"{k}) {v}" for k,v in case["options"].items()) +
                "\n\nAnswer with [[FORMULA_NUMBER]] (e.g. [[7]])"
            )

    @staticmethod
    def extract_output(text):
        matches = re.findall(r'\[\[(.*?)\]\]', text, flags=re.DOTALL)
        if matches:
            last_match = matches[-1].strip()
            # 清理可能的换行符
            return last_match.replace('\n', ' ')
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            if identity["type"] == "relationship":
                answers = [ans.strip().upper() for ans in solution.split(';')]
                correct = [cls._rel_type_to_option(r) for r in identity["key"]]
                return answers == correct
            
            elif identity["type"] == "symbolization":
                # 允许逻辑等价的不同顺序
                user_ans = re.sub(r'\s+', '', solution).upper()
                expected = re.sub(r'\s+', '', identity["solution"]).upper()
                
                # 生成所有可能的排列组合
                parts = expected.split('∧')
                permutations_set = {
                    '∧'.join(p).strip() 
                    for p in permutations(parts)
                }
                return user_ans in permutations_set
            
            else:  # formula case
                return str(identity["correct"]) in re.findall(r'\d+', solution)
        
        except Exception as e:
            return False

    @staticmethod
    def _rel_type_to_option(rel_type):
        return {'*':'A', 'x':'B', '@':'C', '%':'D'}[rel_type]
