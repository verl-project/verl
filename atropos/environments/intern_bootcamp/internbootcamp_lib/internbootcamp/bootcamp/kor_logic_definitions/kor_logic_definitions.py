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
1. Intensional Definitions

Intensional definitions reveal the intension of a term or concept. The intension of a term or concept refers to the distinctive attributes, essential properties, or distinguishing features that the term or concept represents or denotes. Through these attributes or features, the objects denoted by the term or concept can be distinguished from other objects.

(1) ⚯ Definition:

This is the most common form of intensional definition. If the extension of one concept is entirely contained within the extension of another concept, and the latter's extension is not entirely contained within the former's extension, then these two concepts have a genus-species relationship. The former concept is a species concept, and the latter concept is a genus concept. The most commonly used method of definition is to find the genus concept of the defined concept, then find the corresponding differentia, and give the definition in the form of \"defined term = differentia + genus.\"

(2) ⌘ Definition:

This form of definition reveals the differentia from the origin or source of the entity represented or denoted by the defined concept.

(3) ⚒ Definition:

This form of definition uses the specific function or purpose of an entity as the differentia.

(4) Ϟ Definition:

This form of definition uses the special relationship between entities as the differentia, defining the concept in relation to other concepts.

(5) ☖ Definition:

This form of definition defines the term by describing a set of related operational procedures.

(6) Ѭ Definition:

This form of definition, not belonging to the genus-differentia form of definition, is often used, sometimes necessarily, for defining relational concepts. This may involve using logical expressions for definition.

2. Extensional Definitions

Extensional definitions, through enumerating the extension of a concept, enable people to obtain a certain understanding and recognition of the concept, thereby clarifying the meaning and scope of application of the concept. Therefore, extensional definitions are also a relatively common form of definition.

(1) ￥ Definition:

If the number of objects referred to by a concept is very small, or the types are limited, an exhaustive extensional definition can be given.

(2) ! Definition:

When the number of objects belonging to a concept is large, or the types are numerous and cannot be exhaustively listed, some examples are provided to help people gain some understanding of the objects referred to by the concept.

3. Lexical Definitions

Lexical definitions deal with words, often involving the origin, meaning, usage, etc., of the word, and not the objects and entities that the word represents or denotes.

(1) ℵ Definition:

This type of definition reports or describes the existing usage of the defined word. Most definitions in language dictionaries are of this type.

(2) ℓ Definition:

This type of definition explains the meaning of a word by describing its origin and evolution.

(3) ∇ Definition:

In scientific research and daily interactions, sometimes for confidentiality, more often for simplicity and practicality, and sometimes to avoid the interference of certain unrelated meanings often associated with familiar terms, it is necessary to invent new words or redefine the meaning of existing words.\"Example questions are as follows:

<example 0>
\"Sociology is a comprehensive discipline that studies social relations and social behavior to explore the conditions and laws of social coordination, development, and good functioning, providing knowledge and methods for understanding, managing, and transforming society.\"

Which of the Intensional Definitions does the definition above belong to?

A.⚯ Definition
B.⌘ Definition
C.⚒ Definition
D.Ϟ Definition
E.☖ Definition
F.Ѭ Definition

Please give your answer in [[A/B/C/D/E/F]] format.
</example 0>

<example 1>
\"Nuclear energy, also known as atomic energy, refers to the energy released during nuclear reactions when the structure of atomic nuclei changes.\"

Which of the Intensional Definitions does the definition above belong to?

A.⚯ Definition
B.⌘ Definition
C.⚒ Definition
D.Ϟ Definition
E.☖ Definition
F.Ѭ Definition

Please give your answer in [[A/B/C/D/E/F]] format.
</example 1>

<example 2>
\"A particle collider is an experimental device that increases the effective interaction energy of particles by colliding two beams of particles moving in opposite directions.\"

Which of the Intensional Definitions does the definition above belong to?

A.⚯ Definition
B.⌘ Definition
C.⚒ Definition
D.Ϟ Definition
E.☖ Definition
F.Ѭ Definition

Please give your answer in [[A/B/C/D/E/F]] format.
</example 2>

<example 3>
Trademark registration refers to the legal activity of the user applying for trademark registration with the trademark management authority according to the conditions and procedures specified in the Trademark Law and its implementing rules, where the application is reviewed and approved by the Trademark Office, recorded in the trademark register, a registration certificate is issued, and an announcement is made, granting the registrant the exclusive right to use the trademark.\"

Which of the Intensional Definitions does the definition above belong to?

A.⚯ Definition
B.⌘ Definition
C.⚒ Definition
D.Ϟ Definition
E.☖ Definition
F.Ѭ Definition

Please give your answer in [[A/B/C/D/E/F]] format.
</example 3>

<example 4>
\"(A→B) = df (¬A∨B).\"

Which of the Intensional Definitions does the definition above belong to?

A.⚯ Definition
B.⌘ Definition
C.⚒ Definition
D.Ϟ Definition
E.☖ Definition
F.Ѭ Definition

Please give your answer in [[A/B/C/D/E/F]] format.
</example 4>

<example 5>
\"The oxygen group elements refer to oxygen (O), sulfur (S), selenium (Se), tellurium (Te), and polonium (Po).\"

Which of the Extensional Definitions does the above definition belong to?

A. ¥ Definition
B. ! Definition

Please give your answer in [[A/B]] format.
</example 5>

<example 6>
\"China's ethnic minorities include Tibetans, Uighurs, Mongols, Hui, Zhuang, Tujia, and Miao, among others.\"

Which of the Extensional Definitions does the above definition belong to?

A. ¥ Definition
B. ! Definition

Please give your answer in [[A/B]] format.
</example 6>

<example 7>
\"Father can be used both as a noun and a verb. As a noun, it has the following meanings: ① Father; father-in-law; stepfather; adoptive father. ② [often ~s] Ancestors; predecessors; elders. ③ Founder, initiator; creator; inventor; designer; source. ④ [often ~s] Elders; members of the council; senators of ancient Rome. ⑤ [the F-] God; the Holy Father; early Christian writers who clarified doctrines; priests addressed as Father. ⑥ [often F-] Used as a term of respect for elderly men. ⑦ The father of a plant or animal. (New English-Chinese Dictionary).\"


Which of the Lexical Definitions does the above definition belong to?

A. ℵ Definition
B. ℓ Definition
C. ∇ Definition

Please give your answer in [[A/B/C]] format.
</example 7>

<example 8>
Taiyi, an ancient Chinese philosophical term. \"Tai\" means supreme and extreme, \"yi\" means absolutely unique. In \"Zhuangzi·Tianxia,\" it is said that Laozi's teachings are \"based on Taiyi.\" \"Taiyi\" is another name for the \"Dao\" mentioned in Laozi's \"Dao De Jing.\"

Which of the Lexical Definitions does the above definition belong to?

A. ℵ Definition
B. ℓ Definition
C. ∇ Definition

Please give your answer in [[A/B/C]] format.
</example 8>

<example 9>
\"Internet: This is a transliteration and partial translation of the English word \"Internet.\" The partial translation refers to the network, and the transliteration refers to the international network. It is an international computer communication network that connects thousands of different sizes and types of computer networks worldwide.\"

Which of the Lexical Definitions does the above definition belong to?

A. ℵ Definition
B. ℓ Definition
C. ∇ Definition

Please give your answer in [[A/B/C]] format.
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class KorLogicDefinitionsbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.intensional_ratio = params.get('intensional_ratio', 0.5)
        self.extensional_ratio = params.get('extensional_ratio', 0.3)
        self.lexical_ratio = params.get('lexical_ratio', 0.2)
        self.terms_pool = [
            "Sociology", "Nuclear energy", "Particle collider",
            "Trademark registration", "Oxygen group elements"
        ]

    def case_generator(self):
        category = random.choices(
            ['Intensional', 'Extensional', 'Lexical'],
            weights=[self.intensional_ratio, self.extensional_ratio, self.lexical_ratio],
            k=1
        )[0]
        
        if category == 'Intensional':
            return self._generate_intensional_case()
        elif category == 'Extensional':
            return self._generate_extensional_case()
        else:
            return self._generate_lexical_case()

    def _generate_intensional_case(self):
        subtypes = ['A', 'B', 'C', 'D', 'E', 'F']
        subtype = random.choice(subtypes)
        templates = {
            'A': '"{term}" is a type of {genus} distinguished by {differentia}.',
            'B': '"{term}" is defined by its origin in {origin}.',
            'C': '"{term}" is primarily used for {function}.',
            'D': '"{term}" is characterized by its relationship with {relation}.',
            'E': '"{term}" involves these steps: {steps}.',
            'F': '"{term}" is logically defined as {expression}.'
        }
        params = {
            'term': random.choice(self.terms_pool),
            'genus': random.choice(["biological classification", "mathematical concept"]),
            'differentia': random.choice(["unique genetic markers", "distinct properties"]),
            'origin': random.choice(["ancient philosophy", "quantum physics"]),
            'function': random.choice(["energy conversion", "information processing"]),
            'relation': random.choice(["other entities", "dependent variables"]),
            'steps': random.choice(["assembly, testing, deployment", "input, processing, output"]),
            'expression': random.choice(["¬P∨Q", "X⊆Y"])
        }
        return {
            'question': templates[subtype].format(**params),
            'options': [f"{k}. {self._get_intensional_label(k)}" for k in 'ABCDEF'],
            'correct_answer': subtype,
            'category': 'Intensional'
        }

    def _generate_extensional_case(self):
        subtype = random.choice(['A', 'B'])
        examples = {
            'A': ['oxygen (O)', 'sulfur (S)', 'selenium (Se)'],
            'B': ['Tibetans', 'Uighurs', 'Mongols', 'and 53 other ethnic groups']
        }[subtype]
        return {
            'question': f'"The elements include: {", ".join(examples)}."',
            'options': ["A. ¥ (Exhaustive)", "B. ! (Partial)"],
            'correct_answer': 'A' if subtype == 'A' else 'B',
            'category': 'Extensional'
        }

    def _generate_lexical_case(self):
        subtype = random.choice(['A', 'B', 'C'])
        templates = {
            'A': '"Father" means: ① Male parent ② Ancestor (Oxford Dictionary).',
            'B': '"Taiyi" originates from ancient Chinese philosophy meaning "supreme unity".',
            'C': '"Internet" is redefined as a global computer network.'
        }
        return {
            'question': templates[subtype],
            'options': [f"{k}. {self._get_lexical_label(k)}" for k in 'ABC'],
            'correct_answer': subtype,
            'category': 'Lexical'
        }

    @staticmethod
    def _get_intensional_label(key):
        labels = {
            'A': '⚯ (Genus-Differentia)',
            'B': '⌘ (Origin)',
            'C': '⚒ (Function)',
            'D': 'Ϟ (Relation)',
            'E': '☖ (Procedure)',
            'F': 'Ѭ (Logical)'
        }
        return labels[key]

    @staticmethod
    def _get_lexical_label(key):
        labels = {
            'A': 'ℵ (Usage)',
            'B': 'ℓ (Etymology)',
            'C': '∇ (Redefinition)'
        }
        return labels[key]

    @staticmethod
    def prompt_func(question_case):
        category = question_case['category']
        rules = {
            'Intensional': "Intensional definitions specify essential attributes:\nA. Genus-Differentia\nB. Origin\nC. Function\nD. Relation\nE. Procedure\nF. Logical",
            'Extensional': "Extensional definitions enumerate examples:\nA. ¥ (Exhaustive)\nB. ! (Partial)",
            'Lexical': "Lexical definitions describe word usage:\nA. ℵ (Usage)\nB. ℓ (Etymology)\nC. ∇ (Redefinition)"
        }[category]
        
        options_str = "\n".join(question_case['options'])
        
        return f"""{rules}

Question: {question_case['question']}

Options:
{options_str}

Answer in [[LETTER]] format:"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[\[([A-Fa-f])]]', output)
        return matches[-1].upper() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['correct_answer']
