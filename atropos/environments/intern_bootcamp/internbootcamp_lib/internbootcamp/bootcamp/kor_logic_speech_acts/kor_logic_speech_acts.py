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
Custom Speech Act Classification Rules

1. Assertives

- Purpose: To commit the speaker to the truth of the expressed proposition.
- Adaptation Direction: From words to the world (*)
- Psychological State: Belief (♥)
- Formula: *♥(p)
- Common Verbs: Assert, affirm, deny, state, notify, remind, oppose, forecast, report.

2. Directives

- Purpose: To get the hearer to do something.
- Adaptation Direction: From the world to words (→)
- Psychological State: Want or desire (✧)
- Formula: →✧(H do A)
- Common Verbs: Command, ask, plead, request, pray, instruct, advise, prohibit.

3. Commissives

- Purpose: To commit the speaker to do something in the future.
- Adaptation Direction: From the world to words (→)
- Psychological State: Intention (✊)
- Formula: →✊(S do A)
- Common Verbs: Promise, agree, ensure, adopt.

4. Expressives

- Purpose: To express the mental state of the speaker.
- Adaptation Direction: Use the wavy sign (~) to indicate no specific direction.
- Formula: ~(p)
- Common Verbs: Congratulate, apologize, express sympathy, thank, praise, protest.

5. Declarations

- Purpose: to change the state of reality through the speech act itself.
- Direction of adaptation: use of double arrows (↔) to indicate bi-directionality.
- Formula: ↔(p)
- Common verbs: declare, announce, appoint, dismiss, approve.Example questions are as follows:

<example 0>
\"I ♥ that the plan is feasible.\" 

Which category of speech act does this sentence belong to?

A. Assertives
B. Directives
C. Commissives
D. Expressives

Please give your answer in the format [[A/B/C/D]].
</example 0>

<example 1>
\"Please →✧ help me get that item.\"

 Which category of speech act does this sentence belong to?

A. Assertives
B. Directives
C. Commissives
D. Expressives

Please give your answer in the format [[A/B/C/D]].
</example 1>

<example 2>
\"I →✊ will complete the task tomorrow.\" Which category of speech act does this sentence belong to?

Which category of speech act does this sentence belong to?

A. Assertives
B. Directives
C. Commissives
D. Expressives

Please give your answer in the format [[A/B/C/D]].
</example 2>

<example 3>
\"I ~ am very grateful for your help.\" Which category of speech act does this sentence belong to?

Which category of speech act does this sentence belong to?

A. Assertives
B. Directives
C. Commissives
D. Expressives

Please give your answer in the format [[A/B/C/D]].
</example 3>

<example 4>
\"I ↔ the meeting is now open.\" 

Which category of speech act does this sentence belong to?

A. Assertives
B. Directives
C. Commissives
D. Declarations

Please give your answer in the format [[A/B/C/D]].
</example 4>

<example 5>
\"I →✧ want you to complete this task as soon as possible.\" 

Which category of speech act does this sentence belong to?

A. Assertives
B. Directives
C. Commissives
D. Expressives

Please give your answer in the format [[A/B/C/D]].
</example 5>

<example 6>
\"I →✊ promise to bring up this topic at the meeting.\" 

Which category of speech act does this sentence belong to?

A. Assertives
B. Directives
C. Commissives
D. Expressives

Please give your answer in the format [[A/B/C/D]].
</example 6>

<example 7>
\"I ↔ You have been appointed as the new project manager.\" 

This statement belongs to which category of speech-order behaviour?

A. Assertives
B. Directives
C. Commissives
D. Declarations

Please give your answer in the format [[A/B/C/D]].
</example 7>

<example 8>
Based on the adaptation direction and psychological state, 
which description best fits the sentence \"I →✊ will submit the report on time tomorrow\"?

A. Adaptation Direction: From words to the world (*), Psychological State: Belief (♥).
B. Adaptation Direction: From the world to words (→), Psychological State: Want or desire (✧).
C. Adaptation Direction: From the world to words (→), Psychological State: Intention (✊) .
D. Adaptation Direction: None (Φ), Psychological State: Gratitude (Quality).

Please give your answer in the format [[A/B/C/D]].
</example 8>

<example 9>
What are the adaptation direction and psychological state in the sentence \"Please →✧ ensure you arrive before the meeting starts\"?

A. Adaptation Direction: From words to the world (*), Psychological State: Intention (✊).
B. Adaptation Direction: From the world to words (→), Psychological State: Want or desire (✧).
C. Adaptation Direction: None (Φ), Psychological State: Gratitude (Quality).
D. Adaptation Direction: Both from words to the world and from the world to words (↔), Psychological State: Resolve (✊).

Please give your answer in the format [[A/B/C/D]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from typing import Dict, List
from bootcamp import Basebootcamp

class KorLogicSpeechActsbootcamp(Basebootcamp):
    # 增强型属性映射表
    CATEGORY_META = {
        'Assertives': {
            'symbol': '♥', 
            'direction': '*', 
            'psych': 'Belief (♥)',
            'examples': ['the data is accurate', 'the decision is final']
        },
        'Directives': {
            'symbol': '→✧',
            'direction': '→',
            'psych': 'Want (✧)',
            'examples': ['submit the report', 'revise the document']
        },
        'Commissives': {
            'symbol': '→✊',
            'direction': '→',
            'psych': 'Intention (✊)',
            'examples': ['deliver goods tomorrow', 'submit feedback soon']
        },
        'Expressives': {
            'symbol': '~',
            'direction': '~',
            'psych': 'Emotion',
            'examples': ['grateful for your help', 'sorry for the delay']
        },
        'Declarations': {
            'symbol': '↔',
            'direction': '↔',
            'psych': 'Authority',
            'examples': ['meeting adjourned', 'policy enacted']
        }
    }

    def __init__(self, problem_type_ratio: float = 0.5):
        self.problem_type_ratio = problem_type_ratio

    def case_generator(self) -> Dict:
        """增强型实例生成器"""
        return (
            self._generate_category_case() 
            if random.random() < self.problem_type_ratio 
            else self._generate_property_case()
        )

    def _generate_category_case(self) -> Dict:
        """优化后的分类问题生成"""
        candidates = list(self.CATEGORY_META.keys())
        correct = random.choice(candidates)
        
        # 构建动态选项池（确保唯一性）
        distractors = random.sample([c for c in candidates if c != correct], 3)
        options = distractors + [correct]
        random.shuffle(options)
        
        return {
            "type": "category",
            "question": self._build_sentence(correct),
            "options": [{"option": chr(65+i), "text": opt} for i, opt in enumerate(options)],
            "correct_answer": chr(65 + options.index(correct))
        }

    def _generate_property_case(self) -> Dict:
        """增强型属性问题生成"""
        candidates = list(self.CATEGORY_META.keys())
        correct = random.choice(candidates)
        target = self.CATEGORY_META[correct]
        
        # 生成干扰项（带唯一性校验）
        seen = set()
        options = []
        while len(options) < 4:
            cat = random.choice(candidates)
            meta = self.CATEGORY_META[cat]
            entry = (meta['direction'], meta['psych'])
            if entry not in seen:
                seen.add(entry)
                options.append(entry)
        
        # 确保正确答案存在
        correct_entry = (target['direction'], target['psych'])
        if correct_entry not in options:
            options[-1] = correct_entry
        random.shuffle(options)
        
        return {
            "type": "property",
            "question": self._build_sentence(correct),
            "options": [{
                "option": chr(65+i),
                "direction": opt[0],
                "psych_state": opt[1]
            } for i, opt in enumerate(options)],
            "correct_answer": chr(65 + options.index(correct_entry))
        }

    def _build_sentence(self, category: str) -> str:
        """动态语句生成"""
        meta = self.CATEGORY_META[category]
        return f"I {meta['symbol']} {random.choice(meta['examples'])}."

    @staticmethod
    def prompt_func(case: Dict) -> str:
        """优化提示模板"""
        if case["type"] == "category":
            options = "\n".join(
                f"{opt['option']}. {opt['text']}" 
                for opt in case["options"]
            )
            return f'''Analyze the speech act symbol and select the correct category:

"{case['question']}"

Options:
{options}

Format your answer as [[LETTER]].'''
        
        options = "\n".join(
            f"{opt['option']}. Dir: {opt['direction']}, Psy: {opt['psych_state']}"
            for opt in case["options"]
        )
        return f'''Match the directional symbol and psychological state:

"{case['question']}"

Options:
{options}

Respond with [[LETTER ONLY]].'''

    @staticmethod
    def extract_output(output: str) -> str:
        """增强型答案提取"""
        matches = re.findall(r'\[\[([A-D])]]', output, re.IGNORECASE)
        return matches[-1].upper() if matches else None

    @classmethod
    def _verify_correction(cls, solution: str, identity: Dict) -> bool:
        return solution == identity["correct_answer"]
