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
Direct Propositions:
Reflect propositions that assert whether something does or does not possess a certain property, also known as property propositions or subject-predicate propositions.

Examples:
- [1] All metals are conductive.
- [2] Some products of labor are not commodities.

Composition of Direct Propositions:
- Subject (S): The term in the proposition that denotes the object being discussed. Examples include \"metals\" in [1] and \"products of labor\" in [2].
- Predicate (P): The term in the proposition that denotes the property of the object. Examples include \"conductive\" in [1] and \"commodities\" in [2].
- Connectives(C): Words that connect the subject and predicate.
    - Affirmative connective (e.g., \"are\"): Asserts that the subject possesses the property.
    - Negative connective (e.g., \"are not\"): Asserts that the subject does not possess the property.
- Quantifiers(Q): Words that indicate the quantity of objects referred to by the subject.
    - Universal quantifier (e.g., \"all\"): Indicates all members.
    - Particular quantifier (e.g., \"some\"): Indicates at least one member.

Logical Forms of Direct Propositions:
- Universal Affirmative (A): All S are P, abbreviated as SAP.
- Universal Negative (E): No S are P, abbreviated as SEP.
- Particular Affirmative (I): Some S are P, abbreviated as SIP.
- Particular Negative (O): Some S are not P, abbreviated as SOP.
- Singular Affirmative: a is P.
- Singular Negative: a is not P.

Relationships: 
The relationships between declarative propositions are based on the premise that the subject and predicate are identical. 
This identity is referred to as having the same subject (S) and predicate (P). 
There are four types of relationships as follows:
- * Relation:
    - Between A propositions and O propositions, E propositions and I propositions.
    - If one is true, the other is false; if one is false, the other is true.
- # Relation:
    - Between A propositions and E propositions.
    - If one is true, the other is false; if one is false, the other may be true or false.
- & Relation:
    - Between I propositions and O propositions.
    - If one is false, the other is true; if one is true, the other may be false or true.
- % Relation:
    - Between A propositions and I propositions, E propositions and O propositions.
    - If the universal proposition is true, the particular proposition is true; if the particular proposition is false, the universal proposition is false.Example questions are as follows:

<example 0>
All mammals are warm-blooded animals.

1. S is what?
2. P is what?
3. C is what?
4. Q is what?

A.all  B. mammals  C.are  D.warm-blooded animals  

Please answer in the format of [[A/B/C/D];[A/B/C/D];[A/B/C/D];[A/B/C/D]].
</example 0>

<example 1>
Some students do not like mathematics.

1. S is what?
2. P is what?
3. C is what?
4. Q is what?

A. students  B.like mathematics  C.some   D.do not 

Please answer in the format of [[A/B/C/D];[A/B/C/D];[A/B/C/D];[A/B/C/D]].
</example 1>

<example 2>
[1] All products are qualified.
[2] All products are not qualified.
[3] All products are not unqualified.
[4] Some products are unqualified.

Only when S and P are completely identical do they have a relationship. 
Do [1] and [2] have a relationship? 
Do [1] and [3] have a relationship? 
Do [3] and [4] have a relationship?

A. Yes B. No

Please answer in the format of [[A/B];[A/B];[A/B]].
</example 2>

<example 3>
[1] All products are qualified.
[2] All products are unqualified.
[3] No products are unqualified.
[4] Some products are unqualified.

What is the relationship between [1] and [2]? 

What is the relationship between [3] and [4]? 

Choose from the following four types:
A. *  B. #  C. &  D. %

Please answer in the format of [[A/B/C/D];[A/B/C/D]].
</example 3>

<example 4>
What type of proposition is the following statement?
\"Some stars are planets.\"

Please answer in the format of [[SAP/SEP/SIP/SOP]].
</example 4>

<example 5>
What type of proposition is the following statement?
\"All pencils are not pens.\"

Please answer in the format of [[SAP/SEP/SIP/SOP]].
</example 5>

<example 6>
If the proposition SAP is true, then the proposition SOP is what?
If the proposition SIP is true, then the proposition SEP is what?
If the proposition SIP is false, then the proposition SEP is what?
If the proposition SOP is false, then the proposition SAP is what?

Please answer in the format of [[true/false];[true/false];[true/false];[true/false]].
</example 6>

<example 7>
If the proposition SIP is false, then the proposition SOP is what?
If the proposition SOP is false, then the proposition SIP is what?
If the proposition SAP is true, then the proposition SEP is what?
If the proposition SEP is true, then the proposition SAP is what?

Please answer in the format of [[true/false];[true/false];[true/false];[true/false]].
</example 7>

<example 8>
In Class A, with a total of 40 students, they discussed the situation regarding computer typing. Three students, A, B, and C, each expressed their opinions:

- Student A said: \"Li Cong from Class A has not learned how to type on a computer.\"
- Student B said: \"Some students in Class A have learned how to type on a computer.\"
- Student C said: \"Some students in Class A have not learned how to type on a computer.\"

What is the relationship between the statements made by Students B and C among the four types?

Please answer in the format of [[*/#/&/%]].
</example 8>

<example 9>
After a certain tax inspection, four tax inspectors came to the following conclusions:

- Inspector A: All individual businesses did not pay taxes.
- Inspector B: The individual business owner, Mr. Chen, did not pay taxes.
- Inspector C: Some individual businesses have paid taxes.
- Inspector D: Some individual businesses have not paid taxes.

What is the relationship between what Inspector A and Inspector C said among the four types?

Please answer in the format of [[*/#/&/%]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import json
import random
from bootcamp import Basebootcamp

class KorLogicPropositionalLogicConceptsbootcamp(Basebootcamp):
    def __init__(self, s_list=None, p_list=None, problem_types=None, **params):
        super().__init__(**params)
        self.s_list = s_list or ["metals", "products", "students", "mammals", "pencils", "stars", "individual businesses"]
        self.p_list = p_list or ["conductive", "qualified", "like mathematics", "warm-blooded animals", 
                                "pens", "planets", "paid taxes"]
        self.problem_types = problem_types or ["components", "proposition_type", "relationship_exists", 
                                              "relationship_type", "truth_value"]

    def case_generator(self):
        problem_type = random.choice(self.problem_types)
        if problem_type == "components":
            return self._generate_components_case()
        elif problem_type == "proposition_type":
            return self._generate_proposition_type_case()
        elif problem_type == "relationship_exists":
            return self._generate_relationship_exists_case()
        elif problem_type == "relationship_type":
            return self._generate_relationship_type_case()
        elif problem_type == "truth_value":
            return self._generate_truth_value_case()
        else:
            return self._generate_components_case()

    @staticmethod
    def prompt_func(question_case) -> str:
        pt = question_case["problem_type"]
        if pt == "components":
            return KorLogicPropositionalLogicConceptsbootcamp._components_prompt(question_case)
        elif pt == "proposition_type":
            return KorLogicPropositionalLogicConceptsbootcamp._proposition_type_prompt(question_case)
        elif pt == "relationship_exists":
            return KorLogicPropositionalLogicConceptsbootcamp._relationship_exists_prompt(question_case)
        elif pt == "relationship_type":
            return KorLogicPropositionalLogicConceptsbootcamp._relationship_type_prompt(question_case)
        elif pt == "truth_value":
            return KorLogicPropositionalLogicConceptsbootcamp._truth_value_prompt(question_case)
        else:
            return "Unknown problem type"

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[\[([^]]+)\]\]', output)
        if not matches:
            return None
        last_match = matches[-1].strip()
        return [item.strip() for item in last_match.split(';')]

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        pt = identity["problem_type"]
        if pt == "components":
            return solution == identity["correct_answer"]
        elif pt == "proposition_type":
            return solution == [identity["correct_type"]]
        elif pt == "relationship_exists":
            return solution == identity["correct_answers"]
        elif pt == "relationship_type":
            return solution == identity["correct_relations"]
        elif pt == "truth_value":
            return solution == identity["correct_values"]
        return False

    # region Case Generators
    def _generate_components_case(self):
        s = random.choice(self.s_list)
        p = random.choice([x for x in self.p_list if x != s])
        prop_type = random.choice(["A", "E", "I", "O"])
        
        if prop_type == "E" and random.random() < 0.5:
            q, c = "all", "are not"
        else:
            qc_map = {
                "A": ("all", "are"), 
                "E": ("no", "are"), 
                "I": ("some", "are"), 
                "O": ("some", "are not")
            }
            q, c = qc_map[prop_type]
        
        proposition = f"{q} {s} {c} {p}." if prop_type != "E" or q == "no" else f"{q} {s} {c} {p}."
        elements = [q, s, c, p]
        random.shuffle(elements)
        
        correct = []
        for key in [s, p, c, q]:
            correct.append(chr(65 + elements.index(key)))
        
        return {
            "problem_type": "components",
            "proposition": proposition,
            "shuffled_elements": elements,
            "correct_answer": correct
        }

    def _generate_proposition_type_case(self):
        s = random.choice(self.s_list)
        p = random.choice([x for x in self.p_list if x != s])
        prop_type = random.choice(["A", "E", "I", "O"])
        
        qc_map = {
            "A": ("all", "are"), 
            "E": ("no", "are"), 
            "I": ("some", "are"), 
            "O": ("some", "are not")
        }
        q, c = qc_map[prop_type]
        
        if prop_type == "E" and random.random() < 0.5:
            q, c = "all", "are not"
        
        proposition = f"{q} {s} {c} {p}." if prop_type != "E" or q == "no" else f"{q} {s} {c} {p}."
        type_map = {
            "A": "SAP", 
            "E": "SEP", 
            "I": "SIP", 
            "O": "SOP"
        }
        return {
            "problem_type": "proposition_type",
            "proposition": proposition,
            "correct_type": type_map[prop_type]
        }

    def _generate_relationship_exists_case(self):
        s = random.choice(self.s_list)
        p = random.choice([x for x in self.p_list if x != s])
        
        prop1 = self._generate_proposition(s, p)
        prop2 = self._generate_proposition(s, p)
        prop3 = self._generate_proposition(
            random.choice(self.s_list), 
            random.choice(self.p_list)
        )
        
        return {
            "problem_type": "relationship_exists",
            "propositions": [
                prop1["proposition"], 
                prop2["proposition"], 
                prop3["proposition"]
            ],
            "correct_answers": ["A", "A", "B"]
        }

    def _generate_relationship_type_case(self):
        s = random.choice(self.s_list)
        p = random.choice(self.p_list)
        
        prop1 = self._generate_proposition(s, p)
        prop2 = self._generate_proposition(s, p)
        
        rel_map = {
            ("A", "O"): "*", 
            ("E", "I"): "*",
            ("A", "E"): "#", 
            ("I", "O"): "&",
            ("A", "I"): "%", 
            ("E", "O"): "%"
        }
        
        type1 = prop1["type"]
        type2 = prop2["type"]
        key = tuple(sorted([type1, type2]))
        correct = rel_map.get(key, "B")
        
        return {
            "problem_type": "relationship_type",
            "propositions": [
                prop1["proposition"], 
                prop2["proposition"]
            ],
            "correct_relations": [correct]
        }

    def _generate_truth_value_case(self):
        base_prop = self._generate_proposition_type_case()
        base_type = base_prop["correct_type"]
        
        relations = {
            "SAP": [("SOP", "false")],
            "SEP": [("SIP", "false")],
            "SIP": [("SOP", None)],
            "SOP": [("SIP", None)]
        }
        
        questions = []
        correct = []
        for _ in range(4):
            related_type = random.choice(["SAP", "SEP", "SIP", "SOP"])
            questions.append(f"If {base_type} is true, then {related_type} is ___?")
            if (base_type, related_type) in [("SAP", "SOP"), ("SEP", "SIP")]:
                correct.append("false")
            else:
                correct.append(random.choice(["true", "false"]))
        
        return {
            "problem_type": "truth_value",
            "base_proposition": base_prop["proposition"],
            "questions": questions,
            "correct_values": correct
        }

    def _generate_proposition(self, s, p):
        prop_type = random.choice(["A", "E", "I", "O"])
        qc_map = {
            "A": ("all", "are"), 
            "E": ("no", "are"), 
            "I": ("some", "are"), 
            "O": ("some", "are not")
        }
        q, c = qc_map[prop_type]
        return {
            "proposition": f"{q} {s} {c} {p}.",
            "type": prop_type,
            "S": s,
            "P": p
        }
    # endregion

    # region Prompt Templates
    @staticmethod
    def _components_prompt(case):
        options = "\n".join([
            f"{chr(65+i)}. {case['shuffled_elements'][i]}" 
            for i in range(4)
        ])
        return (
            f"Analyze the components of this proposition:\n"
            f"{case['proposition']}\n\n"
            "1. Subject (S)\n2. Predicate (P)\n"
            "3. Connective (C)\n4. Quantifier (Q)\n"
            f"Options:\n{options}\n"
            "Answer format: [[A/B/C/D;A/B/C/D;A/B/C/D;A/B/C/D]]"
        )

    @staticmethod
    def _proposition_type_prompt(case):
        return (
            f"Classify this proposition:\n"
            f"{case['proposition']}\n"
            "Choose from: [[SAP/SEP/SIP/SOP]]"
        )

    @staticmethod
    def _relationship_exists_prompt(case):
        props = "\n".join([
            f"[{i+1}] {p}" 
            for i, p in enumerate(case['propositions'])
        ])
        return (
            "Determine if these propositions share S and P:\n"
            f"{props}\n"
            "Answer format: [[A/B;A/B;A/B]] (Yes/No)"
        )

    @staticmethod
    def _relationship_type_prompt(case):
        return (
            "Determine the logical relationship:\n"
            f"1. {case['propositions'][0]}\n"
            f"2. {case['propositions'][1]}\n"
            "Choose from: [[*/#/&/%]]"
        )

    @staticmethod
    def _truth_value_prompt(case):
        questions = "\n".join([
            f"Q{i+1}: {q}" 
            for i, q in enumerate(case['questions'])
        ])
        return (
            f"Given: {case['base_proposition']}\n"
            f"Answer these:\n{questions}\n"
            "Format: [[true/false;...]]"
        )
    # endregion
