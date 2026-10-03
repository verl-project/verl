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
1. * Induction
(1) Definition:* induction involves inferring a general conclusion based on observing specific instances within a class.
(2) Symbolic Representation:
- `e_i` represents the ith instance.
- `P(e_i)` denotes that instance `e_i` has property `P`.
- `forall e` indicates \"for all instances `e`\".
- The conclusion `forall e, P(e)` signifies that all instances `e` possess property `P`.
(3) Rules:
- Premise: Observations of instances `e_1, e_2, ..., e_k` all possessing property `P`, where these instances are part of class `S`.
    - Symbolically: `P(e_1), P(e_2), ..., P(e_k)`
- Conclusion: Based on limited observation, it is inferred that all instances `e` in class `S` possess property `P`.
    - Symbolically: `forall e in S, P(e)` (this is a conjecture).

2. Φ Induction

(1) Definition:Φ induction derives a general conclusion about all members of a class based on examining the properties of every individual in that class.
(2) Symbolic Representation:
- `E` represents the set of all individuals in the class.
- `P(E)` denotes that every individual in set `E` possesses property `P`.
(3) Rules:
- Premise: Every individual `e_i` in set `E` possesses property `P`, where `e_1, e_2, ..., e_n` are all members of class `S`.
    - Symbolically: `P(e_1), P(e_2), ..., P(e_n)`
- Conclusion: All members of class `S` possess property `P`.
    - Symbolically: `P(E)`

3. Key Differences
- * Induction:
    - Premise: Based on observations of some instances.
    - Conclusion: Inferred for all instances.
    - Symbolic Representation: `P(e_1), P(e_2), ..., P(e_k) -> forall e in S, P(e)`.
- Φ Induction:
    - Premise: Based on observations of all instances.
    - Conclusion: Determined for all instances.
    - Symbolic Representation: `P(e_1), P(e_2), ..., P(e_n) -> P(E)`.Example questions are as follows:

<example 0>
Premise: We observed five different oranges, each of which was sweet.
conclusion: All oranges are sweet.

Is this * inductive reasoning or **Φ** inductive reasoning?
A. * inductive reasoning B. **Φ** inductive reasoning

Please give your answer in [[A/B]] format.
</example 0>

<example 1>
Premise: We examined every known element in the periodic table and found that they all have atomic numbers.
Conclusion: all elements have atomic numbers.

Is this * inductive reasoning or **Φ** inductive reasoning?
A. * inductive reasoning B. **Φ** inductive reasoning

Please give your answer in [[A/B]] format.
</example 1>

<example 2>
Premise: In one class, we found that the first ten students enjoyed maths.
Conclusion: All the students in this class like maths.

Is this * inductive reasoning or **Φ** inductive reasoning?
A. * inductive reasoning B. **Φ** inductive reasoning

Please give your answer in [[A/B]] format.
</example 2>

<example 3>
Premise: We have examined all known birds and found that they can fly.
Conclusion: All birds can fly.

Is this * inductive reasoning or **Φ** inductive reasoning?
A. * inductive reasoning B. **Φ** inductive reasoning

Please give your answer in [[A/B]] format.
</example 3>

<example 4>
Premise: We observe six different apples, each of which is red.
Conclusion: All apples are red.

Is this * inductive reasoning or **Φ** inductive reasoning?
A. * inductive reasoning B. **Φ** inductive reasoning

Please give your answer in [[A/B]] format.
</example 4>

<example 5>
Premise: The observed instances a1, a2, a3 all have property P, and a1, a2, a3 are partial individuals in the S class.
Conclusion: Based on finite observations, it is conjectured that all instances a of class S have property P.

Please symbolise the premises and conclusion above.

Follow [[premise symbolisation];[conclusion symbolisation]] to give your answer.
</example 5>

<example 6>
Premise: Each individual a1, a2, a3, a4 in the set A has the property P and a1, a2, a3, a4 are all individuals in the class S.
Conclusion: All members of the S class have property P.

Please symbolise the premises and conclusion above.

Follow [[premise symbolisation];[conclusion symbolisation]] to give your answer.
</example 6>

<example 7>
Premise: We observe that e1, e2, e3, e4, e5 are all green and that these are some of the individuals in the T class.
Conclusion: All instances of the T class are green.

Please symbolise the premises and conclusion above.

Follow [[premise symbolisation];[conclusion symbolisation]] to give your answer.
</example 7>

<example 8>
Premise: In a set of sample S, the observed instances s1, s2, s3, s4 all have the property Q, and these are all the individuals in sample S.
Conclusion: All members of class S have property Q.

Please symbolise the premises and conclusion above.

Follow [[premise symbolisation];[conclusion symbolisation]] to give your answer.
</example 8>

<example 9>
Premise: After looking at b1, b2, b3, it turns out that they are all blue, and that these are some of the individuals in the B class.
Conclusion: All instances of the B class are blue.

Please symbolise the premises and conclusion above.

Follow [[premise symbolisation];[conclusion symbolisation]] to give your answer.
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re
from collections import defaultdict

class KorLogicEnumerativeInductiveReasoningbootcamp(Basebootcamp):
    def __init__(self, class_names=None, properties=None, type_prob=0.5, question_types=None):
        super().__init__()
        # 扩展默认数据
        self.class_names = class_names or [
            '苹果', '橙子', '元素', '学生', '鸟类', '样本S', '类别T', '类别B',
            '行星', '微生物', '化合物', '历史事件', '编程语言', '几何图形',
            '国家', '化学反应', '文学作品', '数学函数'
        ]
        self.properties = properties or [
            '红色', '甜', '有原子数', '喜欢数学', '会飞', '绿色',
            '有属性Q', '蓝色', '导电', '可降解', '有历史记载',
            '面向对象', '可迭代', '可导', '有韵律', '可逆'
        ]
        self.type_prob = type_prob
        self.question_types = question_types or {
            'choice': 0.6,  # 选择题比例
            'symbolic': 0.4  # 符号题比例
        }

    def case_generator(self):
        # 随机选择问题类型
        q_type = random.choices(
            list(self.question_types.keys()),
            weights=list(self.question_types.values()),
            k=1
        )[0]

        # 公共参数生成
        class_name = random.choice(self.class_names)
        prop = random.choice(self.properties)
        total = random.randint(5, 20)  # 统一总量范围
        
        # 根据问题类型生成不同结构
        if q_type == 'choice':
            case = self._generate_choice_case(class_name, prop, total)
        else:
            case = self._generate_symbolic_case(class_name, prop, total)
        
        case['question_type'] = q_type
        return case

    def _generate_choice_case(self, class_name, prop, total):
        problem_type = 'A' if random.random() < self.type_prob else 'B'
        
        if problem_type == 'A':
            observed = random.randint(3, max(3, total-1))  # 确保观察数合理
            premise = (
                f"在{class_name}类别中，研究人员随机选取了{observed}个不同个体进行观察，"
                f"发现这些样本均具有「{prop}」特征。"
            )
        else:
            observed = total
            premise = (
                f"经过全面核查，确认当前{class_name}类别下所有{total}个注册个体，"
                f"每一个都符合「{prop}」的标准。"
            )

        return {
            "type": problem_type,
            "premise": premise,
            "conclusion": f"由此推断：所有{class_name}都具有「{prop}」特征。",
            "class": class_name,
            "property": prop,
            "total": total,
            "observed": observed
        }

    def _generate_symbolic_case(self, class_name, prop, total):
        problem_type = 'A' if random.random() < self.type_prob else 'B'
        instances = [f'e{i+1}' for i in range(total)]
        sampled = random.sample(instances, k=3) if problem_type == 'A' else instances

        premise_desc = {
            'A': (
                f"观察到{sampled}都具有属性P，"
                f"这些是{class_name}类中的部分实例"
            ),
            'B': (
                f"每个实例{instances}都具有属性P，"
                f"这些构成{class_name}类的完整集合"
            )
        }[problem_type]

        conclusion_desc = {
            'A': f"所有{class_name}类的实例都具有属性P",
            'B': f"{class_name}类整体具有属性P"
        }[problem_type]

        return {
            "type": problem_type,
            "premise": premise_desc,
            "conclusion": conclusion_desc,
            "instances": instances,
            "sampled": sampled,
            "class": class_name
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        if question_case['question_type'] == 'choice':
            return KorLogicEnumerativeInductiveReasoningbootcamp._choice_prompt(question_case)
        return KorLogicEnumerativeInductiveReasoningbootcamp._symbolic_prompt(question_case)

    @staticmethod
    def _choice_prompt(case):
        return (
            "## 归纳推理类型判断\n"
            "**定义说明**\n"
            "A. *归纳推理：基于部分实例的观察得出结论\n"
            "   - 例：检查50辆共享单车→所有车辆都完好\n"
            "B. Φ归纳推理：基于全部实例的检查得出结论\n"
            "   - 例：核验所有参会人员→全部完成注册\n\n"
            "**题目描述**\n"
            f"{case['premise']}\n"
            f"{case['conclusion']}\n\n"
            "**请选择正确的推理类型**\n"
            "将答案用[[A]]或[[B]]标记"
        )

    @staticmethod
    def _symbolic_prompt(case):
        return (
            "## 逻辑符号化练习\n"
            "**符号约定**\n"
            "- e_i: 第i个实例\n"
            "- P(e_i): 实例具有属性P\n"
            "- ∀e∈S: S类的所有实例\n"
            "- P(S): 类S整体具有属性P\n\n"
            "**题目要求**\n"
            f"请将以下陈述转换为标准符号表示：\n"
            f"前提：{case['premise']}\n"
            f"结论：{case['conclusion']}\n\n"
            "**格式要求**\n"
            "按照[[前提符号];[结论符号]]格式作答\n"
            "示例：[[P(e1)∧P(e2);∀e∈S,P(e)]]"
        )

    @staticmethod
    def extract_output(output):
        # 处理两种题型
        choice_match = re.findall(r'\[\[([AB])\]\]', output)
        if choice_match:
            return choice_match[-1]
        
        symbolic_match = re.search(r'\[\[(.+?);(.+?)\]\]', output)
        if symbolic_match:
            return [symbolic_match.group(1), symbolic_match.group(2)]
        
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if identity['question_type'] == 'choice':
            return solution == identity['type']
        
        # 符号题验证逻辑
        expected_premise = {
            'A': '∧'.join([f'P({e})' for e in identity['sampled']]),
            'B': '∧'.join([f'P({e})' for e in identity['instances']])
        }[identity['type']]
        
        expected_conclusion = {
            'A': f'∀e∈{identity["class"]},P(e)',
            'B': f'P({identity["class"]})'
        }[identity['type']]
        
        return (
            solution[0].replace(' ', '') == expected_premise and
            solution[1].replace(' ', '') == expected_conclusion
        )

    @property
    def params(self):
        return {
            'class_names': self.class_names,
            'properties': self.properties,
            'type_prob': self.type_prob,
            'question_types': self.question_types
        }
