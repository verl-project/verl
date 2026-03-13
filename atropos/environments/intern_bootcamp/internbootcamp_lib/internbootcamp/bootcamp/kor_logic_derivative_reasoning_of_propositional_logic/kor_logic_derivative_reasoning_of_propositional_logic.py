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
Method A
- Definition: Change the quality of the original proposition (affirmative to negative or negative to affirmative) and replace the predicate of the original proposition with its contrary.
- Applicable Propositions:
    1. Universal Affirmative Proposition (A): All S is P, can be converted to All S is not non-P.
    2. Universal Negative Proposition (E): All S is not P, can be converted to All S is non-P.
    3. Particular Affirmative Proposition (I): Some S is P, can be converted to Some S is not non-P.
    4. Particular Negative Proposition (O): Some S is not P, can be converted to Some S is non-P.

Method B
- Definition: Exchange the positions of the subject and predicate of the original proposition to form a new proposition.
- Applicable Propositions:
    1. Universal Negative Proposition (E): All S is not P, can be converted to All P is not S.
    2. Particular Affirmative Proposition (I): Some S is P, can be converted to Some P is S.

Method C
- Definition: First perform obversion, and then perform conversion to transform the proposition.
- Applicable Propositions:
    1. Universal Affirmative Proposition (A): All S is P, can be converted to All non-P is not S.
    2. Universal Negative Proposition (E): All S is not P, can be converted to Some non-P is S.
    3. Particular Negative Proposition (O): Some S is not P, can be converted to Some non-P is S.Example questions are as follows:

<example 0>
What is the result of executing method A for
\"Some products are not top-grade.\"?

Please output the result in [[]] format. 
Be careful to maintain consistency with the original sentence.
</example 0>

<example 1>
What is the result of executing method A for
\"Every natural number is a rational number.\"?

Please output the result in [[]] format. 
Be careful to maintain consistency with the original sentence.
</example 1>

<example 2>
What is the result of executing method B for
\"All thoroughgoing materialists are not theists\"?

Please output the result in [[]] format. 
Be careful to maintain consistency with the original sentence.
</example 2>

<example 3>
What is the result of executing method B for
\"Some college students are patriots.\"?

Please output the result in [[]] format. 
Be careful to maintain consistency with the original sentence.
</example 3>

<example 4>
What is the result of executing method C for
\"All genuine scientific theories are theories that have been tested by practice.\"?

Please output the result in [[]] format. 
Be careful to maintain consistency with the original sentence.
</example 4>

<example 5>
What is the result of executing method C for
\"Some young people are not early risers.\"?

Please output the result in [[]] format. 
Be careful to maintain consistency with the original sentence.
</example 5>

<example 6>
Person A: Hey, speaking is really an art, and not being able to speak well can easily offend people.
Person B: Oh? How so?
Person A: My grandfather is a case in point. He speaks very straightforwardly and often does good deeds with bad intentions. I remember one time he invited some friends over for dinner.
Person B: That's nice, everyone can get together.
Person A: The appointment was for six in the evening, and by half past five, three people had arrived, but one important guest hadn't shown up yet.
Person B: Then wait a bit longer, if they don't come, start eating without them.
Person A: My grandfather is a punctual person. When it was half past six and that guest still hadn't arrived, he got anxious and said to himself, \"The ones who should come are not those who come.\" As a result, one of the guests got upset, thinking they were the one who shouldn't have come, and left.
Person B: That's really a misunderstanding.
Person A: With two guests left, my grandfather was still waiting, and then he said, \"The ones who have left were the ones who should have stayed.\" Another guest felt there was no sincerity and also left.
Person B: How could he say that, another one left.
Person A: The last one was my grandfather's old friend, who advised my grandfather to be careful with his words. My grandfather explained that he wasn't talking about him, but the old friend also felt that they were the one who shouldn't stay, and left too.
Person B: This dinner really was, everyone left.

\"The ones who should come are not those who come.\" by method B, we get? 

Please output the result in [[]] format. 
Be careful to maintain consistency with the original sentence.
</example 6>

<example 7>
Person A: Hey, speaking is really an art, and not being able to speak well can easily offend people.
Person B: Oh? How so?
Person A: My grandfather is a case in point. He speaks very straightforwardly and often does good deeds with bad intentions. I remember one time he invited some friends over for dinner.
Person B: That's nice, everyone can get together.
Person A: The appointment was for six in the evening, and by half past five, three people had arrived, but one important guest hadn't shown up yet.
Person B: Then wait a bit longer, if they don't come, start eating without them.
Person A: My grandfather is a punctual person. When it was half past six and that guest still hadn't arrived, he got anxious and said to himself, \"The ones who should come are not those who come.\" As a result, one of the guests got upset, thinking they were the one who shouldn't have come, and left.
Person B: That's really a misunderstanding.
Person A: With two guests left, my grandfather was still waiting, and then he said, \"The ones who have left were the ones who should have stayed.\" Another guest felt there was no sincerity and also left.
Person B: How could he say that, another one left.
Person A: The last one was my grandfather's old friend, who advised my grandfather to be careful with his words. My grandfather explained that he wasn't talking about him, but the old friend also felt that they were the one who shouldn't stay, and left too.
Person B: This dinner really was, everyone left.

\"The ones who have left were the ones who should have stayed.\" by method C, we get? 

Please output the result in [[]] format. 
Be careful to maintain consistency with the original sentence.
</example 7>

<example 8>
What method is used to infer \"All people who are not upright and open are not true revolutionaries\" from \"All true revolutionaries are upright and open\"?

Please answer in the format of [[A/B/C]].
</example 8>

<example 9>
What method should be used to infer a proposition with \"formally correct reasoning\" as the subject from \"All correct reasoning is formally correct reasoning\"?

Please answer in the format of [[A/B/C]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from bootcamp import Basebootcamp

class KorLogicDerivativeReasoningOfPropositionalLogicbootcamp(Basebootcamp):
    def __init__(self, apply_prob=0.7, methods=None, subjects=None, predicates=None):
        super().__init__()
        self.apply_prob = apply_prob
        self.methods = methods if methods is not None else ['A', 'B', 'C']
        self.subjects = subjects if subjects is not None else [
            "natural numbers", "products", "college students", 
            "true revolutionaries", "formally correct reasoning"
        ]
        self.predicates = predicates if predicates is not None else [
            "rational numbers", "top-grade", "patriots",
            "upright and open", "tested by practice"
        ]
        self.type_templates = {
            'A': "All {s} are {p}",
            'E': "All {s} are not {p}",
            'I': "Some {s} are {p}",
            'O': "Some {s} are not {p}"
        }
        self.method_applicable_types = {
            'A': ['A', 'E', 'I', 'O'],
            'B': ['E', 'I'],
            'C': ['A', 'E', 'O']
        }

    def case_generator(self):
        if random.random() < self.apply_prob:
            return self._generate_apply_case()
        else:
            return self._generate_infer_case()

    def _generate_apply_case(self):
        for _ in range(100):  # 增加重试机制
            method = random.choice(self.methods)
            original_type = random.choice(self.method_applicable_types[method])
            s = random.choice(self.subjects)
            p = random.choice(self.predicates)
            
            # 确保生成有效命题
            if 'non-' in s and 'non-' in p:
                continue
                
            original = self.type_templates[original_type].format(s=s, p=p)
            converted = self._apply_conversion(method, original_type, s, p)
            if converted and converted != original:
                return {
                    "question_type": "apply_method",
                    "method": method,
                    "original_proposition": original,
                    "correct_answer": converted
                }
        return self._generate_infer_case()  # 重试失败后备方案

    def _apply_conversion(self, method, original_type, s, p):
        method_map = {
            'A': {
                'A': f'All {s} are not non-{p}',
                'E': f'All {s} are non-{p}',
                'I': f'Some {s} are not non-{p}',
                'O': f'Some {s} are non-{p}'
            },
            'B': {
                'E': f'All {p} are not {s}',
                'I': f'Some {p} are {s}'
            },
            'C': {
                'A': f'All non-{p} are not {s}',  # 对应示例8的正确转换
                'E': f'Some non-{p} are {s}',     # 对应示例7的转换逻辑
                'O': f'Some non-{p} are {s}'      # 保持示例5的转换一致性
            }
        }
        return method_map.get(method, {}).get(original_type)

    def _generate_infer_case(self):
        for _ in range(100):
            # 生成唯一可识别的方法案例
            method = random.choice(self.methods)
            original_type = random.choice(self.method_applicable_types[method])
            s, p = self._get_valid_terms()
            
            original = self.type_templates[original_type].format(s=s, p=p)
            converted = self._apply_conversion(method, original_type, s, p)
            
            # 验证转换结果的唯一性
            if converted and all(
                self._apply_conversion(m, original_type, s, p) != converted
                for m in self.methods if m != method
            ):
                return {
                    "question_type": "infer_method",
                    "original_proposition": original,
                    "converted_proposition": converted,
                    "correct_method": method
                }
        return self._generate_apply_case()

    def _get_valid_terms(self):
        """生成具有逻辑关联性的主谓项"""
        s = random.choice(self.subjects)
        p = random.choice(self.predicates)
        # 避免无意义的组合（如"natural numbers are patriots"）
        while ('numbers' in s and 'grade' in p) or \
              ('reasoning' in s and 'numbers' in p):
            p = random.choice(self.predicates)
        return s, p

    @staticmethod
    def prompt_func(question_case):
        if question_case['question_type'] == "apply_method":
            return (
                f"Execute Method {question_case['method']} on:\n"
                f"\"{question_case['original_proposition']}\"\n\n"
                "Rules:\n"
                "A: Change quality and replace predicate with its contrary\n"
                "B: Swap subject and predicate\n"
                "C: Apply A then B\n\n"
                "Output format: [[converted proposition]]"
            )
        else:
            return (
                "Identify the conversion method used:\n"
                f"Original: \"{question_case['original_proposition']}\"\n"
                f"Converted: \"{question_case['converted_proposition']}\"\n\n"
                "Options:\n"
                "A) Quality change + contrary predicate\n"
                "B) Subject-predicate swap\n"
                "C) Combine A and B methods\n\n"
                "Answer format: [[METHOD]]"
            )

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[\[([ABC]|".*?")\]\]', output)
        return matches[-1].strip('"') if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if identity['question_type'] == "apply_method":
            # 允许冠词和空格差异（如"All S are P" vs "All S are P."）
            return solution.lower().replace(' ', '').replace('.','') == \
                   identity['correct_answer'].lower().replace(' ', '').replace('.','')
        else:
            return solution.upper() == identity['correct_method']
