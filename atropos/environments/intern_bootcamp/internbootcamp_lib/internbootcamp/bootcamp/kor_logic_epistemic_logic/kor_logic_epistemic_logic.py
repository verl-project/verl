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
Symbol Definitions:
- Belief (`B_p`): Indicates that an individual firmly believes the proposition `p` is true.
- Common Belief (`G_p`): Indicates that the proposition `p` is a common belief within the group `G`, meaning all members collectively believe `p`.
- Doubt (`H_p`): Indicates that an individual harbors doubt about the truth of the proposition `p`.

Cognitive Logic Model:
Cognitive logic simulates the interaction between different worlds and beliefs through the construction of models:
- Model (`M`): Composed of three parts:
    - Set of Worlds (`W`): All possible worlds.
    - Accessibility Relation (`R`): If `iRj`, it means individual `i` can recognize the belief state of individual `j`.
    - Truth Value Function of Propositions (`V`): Defines the truth value of each proposition in each world.

Definition of Common Belief:
If `p` is a common belief in the group `G`, then for every member `i` of the group, `B_ip` holds.

Cognitive Logic Axioms:

1. Basic Axioms
- Belief Axiom: `B_p → p`, indicating that if someone firmly believes `p` is true, it can be inferred that `p` is indeed true.

2. Advanced Axioms
- Axiom of Reflexivity: `B_p → BB_p`, indicating that if an individual firmly believes the proposition `p`, they also believe they believe `p`.
- Axiom of Transitivity: If `iRj` and `B_ip`, then `B_jp`, indicating that if individual `i` firmly believes the proposition `p` and can recognize individual `j`, then `j` also believes `p`.
- Axiom of Consistent Belief: `B_p ∧ B_{¬p} → ⊥`, indicating that an individual cannot simultaneously believe in a proposition `p` and its negation `¬p`, as this would lead to a logical contradiction.

3. Axioms of Doubt
- Introduction of Doubt: `H_p → ¬B_p`, indicating that if an individual doubts the proposition `p`, they do not firmly believe `p`.
- Spread of Doubt: If `iRj` and `H_ip`, then `H_jp`, indicating that if individual `i` doubts the proposition `p` and can recognize individual `j`, then `j` may also start to doubt `p`.Example questions are as follows:

<example 0>
Based on the Belief Axiom, if Alice firmly believes that the sun rises in the east (`B_Alice(The sun rises in the east)`), we can conclude the following:

A. Alice may doubt that the sun rises in the east.
B. It is true that the sun rises in the east.
C. Alice is unaware that the sun rises in the east.

Please provide the answers in the format [[A/B/C]].
</example 0>

<example 1>
According to the Axiom of Reflexivity, what does it mean if Alice firmly believes a certain proposition to be true?

A. Alice may have doubts about this proposition.
B. Alice is convinced that she herself is convinced of this proposition.
C. Alice and other people are all aware of this proposition.

Please provide the answers in the format [[A/B/C]].
</example 1>

<example 2>
If both Alice and Bob firmly believe that 2 plus 2 equals 4, according to the definition of common belief, what does this mean?

A. Alice and Bob both know that 2 plus 2 equals 4.
B. Only Alice firmly believes that 2 plus 2 equals 4.
C. Bob doubts that 2 plus 2 equals 4.

Please provide the answers in the format [[A/B/C]].
</example 2>

<example 3>
According to the Axiom of Transitivity, if Alice is certain that Bob is certain of a certain proposition, and Alice is also certain of this proposition, what is Bob's attitude towards this proposition?

A. Bob might suspect this proposition.
B. Bob is convinced of this proposition.
C. Bob's attitude towards this proposition is uncertain.

Please provide the answers in the format [[A/B/C]].
</example 3>

<example 4>
According to the Axiom of Consistent Belief, what does it mean if Alice firmly believes in a proposition and its negation at the same time?

A. Alice's beliefs are coherent.
B. There exists an inconsistency within Alice's beliefs.
C. This scenario is not possible.

Please provide the answers in the format [[A/B/C]].
</example 4>

<example 5>
If Alice harbors doubts that the library is open today, what is Alice convinced of according to the Introduction of Doubt axiom?

A. That the library is open today.
B. That the library is not open today.
C. That she is not certain whether the library is open today.

Please provide the answers in the format [[A/B/C]].
</example 5>

<example 6>
If Alice is skeptical about the library being open today, and Bob can acknowledge Alice's skepticism, what is Bob likely to be convinced of, based on the Spread of Doubt axiom?

A. That the library is open today.
B. That the library is closed today.
C. That he may also begin to doubt whether the library is open today.

Please provide the answers in the format [[A/B/C]].
</example 6>

<example 7>
If there exists an accessibility relation between Alice and Bob, and Alice harbors doubts about a certain proposition, what is Bob likely to be convinced of, based on the Spread of Doubt axiom?

A. That the proposition is true.
B. That the proposition is false.
C. That he might also harbor doubts about the proposition.

Please provide the answers in the format [[A/B/C]].
</example 7>

<example 8>
If a proposition p is the consensus of the group G, 
but the individual Alice doubts this proposition, 
what logical expression can be written according to the definition of consensus?

Please give your answer in the format [[]].
</example 8>

<example 9>
If Alice is sure that the library is open today (proposition p), 
and she is sure that she is sure of this (according to the axiom of self-reflexivity), 
what logical expression is written?

Please give your answer in the format [[]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class KorLogicEpistemicLogicbootcamp(Basebootcamp):
    RULE_DESCRIPTIONS = {
        "Belief": "信念公理（Belief Axiom）：如果某人坚信命题p成立（B_p），那么p是真实的。",
        "Reflexivity": "自反性公理（Axiom of Reflexivity）：如果某人坚信命题p（B_p），那么他也坚信自己坚信p（BB_p）。",
        "Transitivity": "传递性公理（Axiom of Transitivity）：如果个体i可以识别个体j的信念状态（iRj），并且i坚信命题p（B_i p），那么j也将坚信p（B_j p）。",
        "Common Belief": "共同信念定义（Common Belief）：如果命题p是群体G的共同信念（G_p），那么群体内的每个成员都坚信p。",
        "Consistent Belief": "一致性信念公理（Axiom of Consistent Belief）：个体不能同时相信命题p及其否定¬p，否则会导致逻辑矛盾。",
        "Doubt Introduction": "怀疑引入公理（Introduction of Doubt）：如果个体怀疑命题p（H_p），则他不坚信p（¬B_p）。",
        "Doubt Spread": "怀疑传播公理（Spread of Doubt）：如果个体i可以识别j的信念状态（iRj）且i怀疑p（H_i p），则j也怀疑p（H_j p）。"
    }
    
    def __init__(self, names=None, propositions=None, groups=None):
        super().__init__()
        self.names = names or ['Alice', 'Bob', 'Charlie']
        self.propositions = propositions or [
            '太阳从东方升起', '2+2=4', '图书馆今天开放', 
            '地球是圆的', '水在0℃结冰', '人类需要氧气'
        ]
        self.groups = groups or ['G', 'GroupA', 'GroupB']
        self.templates = self._load_templates()

    def _load_templates(self):
        return [
            # 信念公理选择题模板
            {
                "type": "multiple_choice",
                "axiom": "Belief",
                "template": {
                    "scenario": "根据信念公理，如果{name}坚信{proposition}（B_{name}({proposition})），我们可以得出以下哪个结论？",
                    "options": [
                        {"text": "{name}可能怀疑{proposition}。", "is_correct": False},
                        {"text": "{proposition}是真实的。", "is_correct": True},
                        {"text": "{name}不知道{proposition}。", "is_correct": False}
                    ]
                }
            },
            # 自反性公理选择题模板
            {
                "type": "multiple_choice",
                "axiom": "Reflexivity",
                "template": {
                    "scenario": "根据自反性公理，如果{name}坚信某个命题是真的，这意味着什么？",
                    "options": [
                        {"text": "{name}可能对该命题产生怀疑。", "is_correct": False},
                        {"text": "{name}确信自己坚信这个命题。", "is_correct": True},
                        {"text": "{name}和其他人全都知道这个命题。", "is_correct": False}
                    ]
                }
            },
            # 传递性公理选择题模板
            {
                "type": "multiple_choice",
                "axiom": "Transitivity",
                "requires_two_names": True,
                "template": {
                    "scenario": "根据传递性公理，如果{name1}可以识别{name2}的信念状态（{name1}R{name2}），并且{name1}坚信{proposition}（B_{name1}({proposition})），那么{name2}对该命题的态度是什么？",
                    "options": [
                        {"text": "{name2}可能怀疑该命题。", "is_correct": False},
                        {"text": "{name2}坚信该命题。", "is_correct": True},
                        {"text": "{name2}的态度无法确定。", "is_correct": False}
                    ]
                }
            },
            # 共同信念选择题模板
            {
                "type": "multiple_choice",
                "axiom": "Common Belief",
                "template": {
                    "scenario": "如果命题{proposition}是群体{group}的共同信念，这意味着什么？",
                    "options": [
                        {"text": "{group}中的每个成员都坚信{proposition}。", "is_correct": True},
                        {"text": "只有部分成员坚信{proposition}。", "is_correct": False},
                        {"text": "{group}的成员都怀疑{proposition}。", "is_correct": False}
                    ]
                }
            },
            # 怀疑引入公理选择题模板
            {
                "type": "multiple_choice",
                "axiom": "Doubt Introduction",
                "template": {
                    "scenario": "根据怀疑引入公理，如果{name}怀疑{proposition}（H_{name}({proposition})），这意味着什么？",
                    "options": [
                        {"text": "{name}坚信{proposition}。", "is_correct": False},
                        {"text": "{name}不坚信{proposition}。", "is_correct": True},
                        {"text": "{name}知道{proposition}是假的。", "is_correct": False}
                    ]
                }
            },
            # 共同信念表达式模板
            {
                "type": "expression",
                "axiom": "Common Belief",
                "template": {
                    "scenario": "如果命题{proposition}是群体{group}的共同信念，但个体{name}怀疑该命题，根据共同信念的定义，对应的逻辑表达式是什么？",
                    "correct_expression": "G_{proposition} ∧ H_{name}_{proposition}"
                }
            },
            # 自反性公理表达式模板
            {
                "type": "expression",
                "axiom": "Reflexivity",
                "template": {
                    "scenario": "如果{name}确信{proposition}（B_{name}({proposition})），并且根据自反性公理确信自己确信此事，对应的逻辑表达式是什么？",
                    "correct_expression": "B_{name}_{proposition} ∧ B_{name}(B_{name}_{proposition})"
                }
            }
        ]

    def case_generator(self):
        template = random.choice(self.templates)
        return self._fill_template(template)

    def _fill_template(self, template):
        params = {}
        
        # 处理需要两个不同名字的情况
        if template.get('requires_two_names', False):
            names = random.sample(self.names, 2)
            params['name1'] = names[0]
            params['name2'] = names[1]
        else:
            params['name'] = random.choice(self.names)
        
        params['proposition'] = random.choice(self.propositions)
        params['group'] = random.choice(self.groups)
        
        filled = {
            "type": template["type"],
            "axiom": template["axiom"],
            "scenario": template["template"]["scenario"].format(**params)
        }
        
        if template["type"] == "multiple_choice":
            options = []
            correct_answer = None
            for idx, opt in enumerate(template["template"]["options"]):
                option_text = opt["text"].format(**params)
                letter = chr(65 + idx)
                options.append(f"{letter}. {option_text}")
                if opt["is_correct"]:
                    correct_answer = letter
            filled["options"] = options
            filled["correct_answer"] = correct_answer
        elif template["type"] == "expression":
            filled["correct_expression"] = template["template"]["correct_expression"].format(**params).replace(" ", "")
        
        return filled

    @staticmethod
    def prompt_func(question_case) -> str:
        rule_desc = KorLogicEpistemicLogicbootcamp.RULE_DESCRIPTIONS.get(question_case["axiom"], "")
        prompt = f"{rule_desc}\n\n{question_case['scenario']}\n"
        
        if question_case["type"] == "multiple_choice":
            prompt += "\n请选择正确的结论：\n" + "\n".join(question_case["options"])
            prompt += "\n\n请将答案用大写字母放在双括号内，例如[[A]]。"
        elif question_case["type"] == "expression":
            prompt += "\n请将逻辑表达式（使用命题符号，无需自然语言）放在双括号内，例如[[G_p ∧ H_Alice_p]]。"
        
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[\[(.*?)\]\]', output)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
            
        if identity["type"] == "multiple_choice":
            return solution.upper() == identity["correct_answer"]
        elif identity["type"] == "expression":
            # 标准化比较：移除所有空格
            return solution.replace(" ", "") == identity["correct_expression"]
        return False
