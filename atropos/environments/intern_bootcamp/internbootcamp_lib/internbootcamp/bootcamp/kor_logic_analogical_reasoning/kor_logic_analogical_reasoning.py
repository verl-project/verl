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
There are two types of analogical reasoning:

1. Ψ Method:
    Object A has attributes a, b, c, d;
    Object B has attributes a, b, c;
    Therefore, Object B also has attribute d.
    Here, attributes a, b, c are referred to as #Ψ attributes, and d is referred to as the +Ψ attribute.
    
2. ⌘ Method: 
    Object A has attributes a, b, c, d;
    Object B does not have attributes a, b, c;
    Therefore, Object B also does not have attribute d.
    Here, attributes a, b, c are referred to as -⌘ attributes, and d is referred to as the +⌘ attribute.Example questions are as follows:

<example 0>
In the campus of University A, every observed crow a, b, c, etc., has feathers that are black. The bird in the campus of University B is a crow; therefore, the feathers of the bird in the campus of University B might be black.

Which method of reasoning does this argument follow:
A. Ψ Method             B. ⌘ Method

Please provide the answer in the format [[A/B]].
</example 0>

<example 1>
Professor Van Emelen always sends me a gift, and it is always a book that he has written or edited. This gift is from Professor Van Emelen and is a book, therefore, all gifts that Professor Van Emelen sends to me might be books that he has written or edited.

Which method of reasoning does this argument follow:
A. Ψ Method             B. ⌘ Method

Please provide the answer in the format [[A/B]].
</example 1>

<example 2>
Scientists, after observing and analyzing the Moon and comparing it with the Earth, concluded long before humans set foot on the Moon that the Moon could not possibly harbor life as naturally as Earth does. The scientists reasoned in this way: Earth has an atmosphere, water, moderate temperatures, and not too large a temperature difference between day and night, which is why Earth harbors natural life; the Moon has no atmosphere, water, and a large temperature difference between day and night; therefore, the Moon could not possibly harbor life as naturally as Earth does.

Which method of reasoning does this argument follow:
A. Ψ Method             B. ⌘ Method

Please provide the answer in the format [[A/B]].
</example 2>

<example 3>
H University can reach a new level after reform. Since F University is an old school with strong faculty and a good school spirit, it has reached a new level after reform; and H University is also an old school with strong faculty and a good school spirit as well, the basic conditions of the two schools are the same.

Which method of reasoning does this argument follow:
A. Ψ Method             B. ⌘ Method

Please provide the answer in the format [[A/B]].
</example 3>

<example 4>
When humans first explored the deep sea, scientists conducted detailed observations and analyses of the life forms on the seabed and compared them with life on land. In the deep sea, scientists discovered some strange and bizarre creatures, but they inferred that the living environment of deep-sea creatures is completely different from that of terrestrial life. Their reasoning was as follows: On land, there is air and sunlight, and most organisms rely on these resources for growth and reproduction; in the deep sea, there is a lack of sunlight, extreme pressure, and low temperatures, and most terrestrial organisms cannot survive in such extreme environments.

Which method of reasoning does this argument follow:
A. Ψ Method             B. ⌘ Method

Please provide the answer in the format [[A/B]].
</example 4>

<example 5>
In the campus of University A, every observed crow a, b, c, etc., has feathers that are black. The bird in the campus of University B is a crow, therefore, the feathers of the bird in the campus of University B might be black.

This reasoning follows the Ψ Method, where \"the bird's feathers being black\" is what kind of attribute?
A. #Ψ attribute   B. +Ψ attribute

Please provide the answer in the format [[A/B]].
</example 5>

<example 6>
Professor Van Emelen always sends me a gift, and it is always a book that he has written or edited. This gift is from Professor Van Emelen and is a book, therefore, all gifts that Professor Van Emelen sends to me might be books that he has written or edited.

This reasoning follows the Ψ Method, where \"the gift being from Professor Van Emelen\" is what kind of attribute?
A. #Ψ attribute   B. +Ψ attribute

Please provide the answer in the format [[A/B]].
</example 6>

<example 7>
After observing and analyzing the Moon and comparing it with Earth, scientists concluded before humans set foot on the Moon that the Moon could not possibly harbor life as naturally as Earth does. The scientists reasoned in this way: Earth has an atmosphere, water, moderate temperatures, and not too large a temperature difference between day and night, which is why Earth harbors natural life; the Moon lacks an atmosphere, water, and has a large temperature difference between day and night; therefore, the Moon could not possibly harbor life as naturally as Earth does.

This reasoning follows the ⌘ Method, where \"the absence of life\" is what kind of attribute?
A. -⌘ attribute   B. +⌘ attribute

Please provide the answer in the format [[A/B]].
</example 7>

<example 8>
H University can reach a new level after reform. Since F University is an established school with strong faculty and a good academic atmosphere, it has reached a new level after reform; and H University is also an established school with strong faculty and a good academic atmosphere as well, the basic conditions of the two schools are the same.

This reasoning follows the Ψ Method, where \"being an established school\" is what kind of attribute?
A. #Ψ attribute   B. +Ψ attribute
Please provide the answer in the format [[A/B]].
</example 8>

<example 9>
When humans first explored the deep sea, scientists conducted detailed observations and analyses of the life forms on the seabed and compared them with life on land. In the deep sea, scientists discovered some strange and bizarre creatures, but they inferred that the living environment of deep-sea creatures is completely different from that of terrestrial life. Their reasoning was as follows: On land, there is air and sunlight, most organisms rely on these resources for growth and reproduction; in the deep sea, there is a lack of sunlight, extreme pressure, and low temperatures, most terrestrial organisms cannot survive in such extreme environments.

This reasoning follows the ⌘ Method, where \"the presence of air and sunlight\" is what kind of attribute?
A. -⌘ attribute  B. +⌘ attribute

Please provide the answer in the format [[A/B]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class KorLogicAnalogicalReasoningbootcamp(Basebootcamp):
    def __init__(self, attribute_pools=None, **params):
        super().__init__(**params)
        self.attribute_pools = attribute_pools or {
            'university': {
                'attributes': ['an established school', 'strong faculty', 'good academic atmosphere'],
                'derived_positive': 'reached a new level after reform',
                'derived_negative': 'cannot reach a new level after reform',
                'objects': ['University F', 'University H']
            },
            'ornithology': {
                'attributes': ['black feathers', 'large beaks', 'carnivorous diet'],
                'derived_positive': 'nocturnal activity',
                'derived_negative': 'non-nocturnal activity',
                'objects': ['crows in University A', 'a bird in University B']
            },
            'astronomy': {
                'attributes': ['an atmosphere', 'liquid water', 'moderate temperatures'],
                'derived_positive': 'natural life',
                'derived_negative': 'no natural life',
                'objects': ['Earth', 'the Moon']
            },
            'marine': {
                'attributes': ['sunlight', 'stable pressure', 'moderate temperatures'],
                'derived_positive': 'terrestrial organisms',
                'derived_negative': 'no terrestrial organisms',
                'objects': ['land', 'the deep sea']
            }
        }
        self.params = params

    def case_generator(self):
        # Select category and method
        category_key = random.choice(list(self.attribute_pools.keys()))
        category = self.attribute_pools[category_key]
        method = random.choice(['Ψ', '⌘'])
        question_type = random.choice(['method', 'attribute'])
        
        # Generate attributes
        common_attrs = random.sample(category['attributes'], 3)
        derived_attr = category['derived_positive'] if method == 'Ψ' else category['derived_negative']
        
        # Determine correct answer
        if question_type == 'method':
            correct_answer = 'A' if method == 'Ψ' else 'B'
        else:
            # Randomly select which attribute to question
            target_is_common = random.choice([True, False])
            if method == 'Ψ':
                correct_answer = 'A' if target_is_common else 'B'
                target_attr = common_attrs[0] if target_is_common else derived_attr
            else:
                correct_answer = 'A' if target_is_common else 'B'
                target_attr = common_attrs[0] if target_is_common else derived_attr

        return {
            'question_type': question_type,
            'method': method,
            'category': category_key,
            'common_attrs': common_attrs,
            'derived_attr': derived_attr,
            'objects': category['objects'],
            'correct_answer': correct_answer,
            'target_attr': target_attr if question_type == 'attribute' else None
        }

    @staticmethod
    def prompt_func(case):
        # Contextual templates
        method_descriptions = {
            'Ψ': {
                'premise': "{objA} has {attrs}. {objB} has {common_attrs}.",
                'conclusion': "Therefore, {objB} also has {derived_attr}."
            },
            '⌘': {
                'premise': "{objA} has {attrs}. {objB} does not have {common_attrs}.",
                'conclusion': "Therefore, {objB} does not have {derived_attr}."
            }
        }
        
        if case['question_type'] == 'method':
            template = method_descriptions[case['method']]
            premise = template['premise'].format(
                objA=case['objects'][0],
                attrs=', '.join(case['common_attrs'] + [case['derived_attr']]),
                objB=case['objects'][1],
                common_attrs=', '.join(case['common_attrs'])
            )
            conclusion = template['conclusion'].format(
                objB=case['objects'][1],
                derived_attr=case['derived_attr']
            )
            
            return f"""{premise}
{conclusion}

Which method of reasoning does this argument follow?
A. Ψ Method (shared attributes lead to positive conclusion)
B. ⌘ Method (missing attributes lead to negative conclusion)

Please answer with [[A]] or [[B]]."""
        
        else:
            role_map = {
                'Ψ': {'A': '#Ψ attribute (shared)', 'B': '+Ψ attribute (inferred)'},
                '⌘': {'A': '-⌘ attribute (missing)', 'B': '+⌘ attribute (excluded)'}
            }
            roles = role_map[case['method']]
            
            return f"""In the following {case['method']} Method example:
"{case['target_attr']}" is which type of attribute?
A. {roles['A']}
B. {roles['B']}

Please answer with [[A]] or [[B]]."""

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[\[([AB])\]\]', output)
        return matches[-1] if matches else None

    @classmethod
    def _verify_correction(cls, solution, case):
        return solution == case['correct_answer']
