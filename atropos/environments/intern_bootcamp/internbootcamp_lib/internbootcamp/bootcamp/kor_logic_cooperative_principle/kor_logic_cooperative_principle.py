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
Custom Cooperation Principles

1. C* Principle

(1) Speaker's Criterion: Do not let your statement be weaker in information than what your knowledge allows, unless a stronger statement conflicts with the Information Principle.
(2) Hearer's Inference:
    - CQ1: If the speaker says A(w), and <s, w> brackets the words in order of information strength with s (strong) followed by w (weak), A(s) entails A(w), then it can be inferred that K~(A(s)), meaning the speaker knows that the stronger information cannot be established.
    - CQ2: The speaker states A(w), which does not entail the content of the embedded sentence Q, but the content of Q is entailed by the stronger information A(s), and {s, w} form a contrast set, then it can be deduced that ~K(Q), meaning the speaker does not know whether Q can be established.

2. C% Principle

(1) Speaker's Criterion: Minimalization Criterion - Speak as little as possible, only speak to the minimum extent necessary to achieve the purpose of communication.
(2) Hearer's Inference:
    - CI1: Assume that the relationship between the objects and time in the sentence follows the convention unless there is clear evidence to the contrary.
    - CI2: If a certain existence or fact exactly matches the confirmed situation, it is set that this is what the sentence is saying. The Information Principle actually refers to the speaker striving to \"speak as little as possible,\" while the hearer strives to \"expand the information\" until fully grasping the intention of the speech.

3. C! Principle

(1) Speaker's Criterion: Do not use lengthy, obscure, or marked expressions without reason.
(2) Hearer's Inference: If the speaker uses a lengthy marked expression, their meaning is different from what they could have expressed with an unmarked expression, especially they should try to avoid conventional associations or derive meanings using the Information Principle.Example questions are as follows:

<example 0>
A: \"Do you have tickets for tonight's movie?\"
B: \"I bought two tickets.\" 
C: \"I managed to get two tickets.\" 
C uses \"managed to get\" instead of directly saying \"bought,\" implying that \"getting the tickets was not easy and took some effort.\"

Which of the following principles does this conform to?

A. C* Principle     
B. C% Principle 
C. C! Principle

Please give your answer in the format [[A/B/C]].
</example 0>

<example 1>
A: \"Do you love Xiao Hong? Please tell me.\"
B: \"I like her.\"
Here, the pair <love, like> forms a hierarchy. 
B answered with the weaker information, thus implying that the stronger statement \"I love her\" does not hold. 
Therefore, which principle did B use to tactfully reveal the truth?

A. C* Principle     
B. C% Principle 
C. C! Principle

Please give your answer in the format [[A/B/C]].
</example 1>

<example 2>
A: All soccer players are on the field.
B: Some soccer players are on the field.
Here, the pair <all, some> forms a hierarchy. 
Therefore, if the speaker says B, it indicates that they know saying A does not match the facts. 

Which principle did the speaker use to reveal the truth?

A. C* Principle     
B. C% Principle 
C. C! Principle

Please give your answer in the format [[A/B/C]].
</example 2>

<example 3>
A: I believe you are a college student.
B: I know you are a college student.

The speaker says A, which does not entail the clause \"you are a college student,\" but B can entail \"you are a college student,\" because the pair <know, believe> forms a hierarchy. 
That is to say, when the speaker says A, they do not actually know whether \"you are a college student\" is established.

Which of the following principles does this conform to?

A. C* Principle     
B. C% Principle 
C. C! Principle

Please give your answer in the format [[A/B/C]].
</example 3>

<example 4>
Xiao Ma opens the food box, and the beer is still warm → Beer is part of the food in the food box.

Which of the following principles does this conform to?

A. C* Principle     
B. C% Principle 
C. C! Principle

Please give your answer in the format [[A/B/C]].
</example 4>

<example 5>
A: \"Can we complete this project on time? I need a definite answer.\"
B: \"We have finished most of the work, with only a few details left to address.\"
C: \"We have essentially wrapped up the project, with just some minor finishing touches remaining.\"

In this dialogue, A is requesting a definite answer about whether the project can be completed on time. B's response provides some information but does not directly answer A's question, instead implying that the project is nearly finished but there is still work to be done. C's response is similar to B's but uses the expression \"essentially wrapped up the project,\" which may suggest that the project is substantially complete with only minor steps left. C's response might lead one to understand that although there are still parts of the project unfinished, the main work has been accomplished, and C has chosen a more euphemistic and optimistic way of expression, implying a high likelihood of project success while also leaving some openness, indicating there might be some unforeseen work left.

Which of the following principles does this conform to?

A. C* Principle     
B. C% Principle 
C. C! Principle

Please give your answer in the format [[A/B/C]].
</example 5>

<example 6>
The baby lying in bed cries, and the mother picks her up. → The mother is the baby's mother. (Attributive inference type)

Which of the following principles does this conform to?

A. C* Principle     
B. C% Principle 
C. C! Principle

Please give your answer in the format [[A/B/C]].
</example 6>

<example 7>
Zhang San bought a new car, but the door won't close → Zhang San's new car has doors. (Connection inference type)

Which of the following principles does this conform to?

A. C* Principle     
B. C% Principle 
C. C! Principle

Please give your answer in the format [[A/B/C]].
</example 7>

<example 8>
Xiao Wang gives flowers to a nurse. → Xiao Wang gives flowers to a female. (Common sense inference type)

Which of the following principles does this conform to?

A. C* Principle     
B. C% Principle 
C. C! Principle

Please give your answer in the format [[A/B/C]].
</example 8>

<example 9>
A: \"Can you help me borrow the materials for tomorrow's meeting?\"
B: \"I borrowed the materials.\"
C: \"I managed to get the materials.\"

In this dialogue, B's response \"I borrowed the materials\" is a direct and conventional answer, indicating that B has successfully completed the action of borrowing the materials. However, C's response \"I managed to get the materials\" uses the word \"managed,\" which may imply that the process of obtaining the materials was not simple and may involve additional effort or the use of some special methods. C's response may lead people to understand that the process of obtaining the materials was \"quite troublesome,\" or C encountered some obstacles or difficulties in obtaining the materials.

Which of the following principles does this conform to?

A. C* Principle     
B. C% Principle 
C. C! Principle

Please give your answer in the format [[A/B/C]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from bootcamp import Basebootcamp

class KorLogicCooperativePrinciplebootcamp(Basebootcamp):
    RULE_DESCRIPTIONS = {
        'A': [
            "C*原则 - 信息强度准则",
            "1. 说话者准则：除非强陈述违反信息原则，否则不应使用信息量更弱的陈述",
            "2. 听者推理：",
            "   - 若使用弱信息A(w)，且存在<s,w>强度序列，则暗示K¬A(s)",
            "   - 若A(w)不蕴含Q，但A(s)蕴含Q且{s,w}形成对比，则暗示¬KQ"
        ],
        'B': [
            "C%原则 - 最小化准则",
            "1. 说话者准则：仅传达必要的最小信息量",
            "2. 听者推理：",
            "   - 默认按常规关系理解（CI1）",
            "   - 精确匹配已知事实时优先采用（CI2）"
        ],
        'C': [
            "C!原则 - 标记性准则",
            "1. 说话者准则：避免使用复杂标记性表达",
            "2. 听者推理：当使用标记性表达时，其含义应与非标记表达不同"
        ]
    }

    def __init__(self, case_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.lexicon = {
            'strength_pairs': [
                ('love', 'like'), 
                ('all', 'some'),
                ('know', 'believe'),
                ('finished', 'managed to get'),
                ('perfect', 'good enough')
            ],
            'inference_types': [
                ('buy car', 'has doors', '连接推理'),
                ('mother and baby', 'parent-child', '属性推理'),
                ('nurse', 'female', '常识推理')
            ],
            'marked_phrases': [
                ('essentially wrapped up', 'finished'),
                ('secured tickets', 'bought tickets'),
                ('persuaded to join', 'asked to join')
            ]
        }
        self.weights = case_weights or [1, 1, 1]

    def case_generator(self):
        principle = random.choices(['A', 'B', 'C'], weights=self.weights, k=1)[0]
        case = {'correct': principle}
        
        if principle == 'A':
            s, w = random.choice(self.lexicon['strength_pairs'])
            case.update({
                'type': 'strength_hierarchy',
                'dialogue': [
                    f"你是否{s}这个？请如实回答。",
                    f"我{w}它。"
                ],
                'explanation': f"使用弱项'{w}'暗示强项'{s}'不成立"
            })
        elif principle == 'B':
            context, inference, i_type = random.choice(self.lexicon['inference_types'])
            case.update({
                'type': i_type,
                'scenario': f"{context} → {inference}",
                'explanation': f"{i_type}类型推理"
            })
        else:
            marked, plain = random.choice(self.lexicon['marked_phrases'])
            case.update({
                'type': 'marked_expression',
                'dialogue': [
                    "项目完成了吗？",
                    f"我们已经{marked}。" if random.random() > 0.5 else 
                    f"我们{marked}。"
                ],
                'contrast': plain,
                'explanation': f"使用标记表达'{marked}'代替常规'{plain}'"
            })
        return case

    @staticmethod
    def prompt_func(question_case):
        prompt = ["请根据对话分析适用的协作原则（答案格式：[[A/B/C]]）\n"]
        
        if 'dialogue' in question_case:
            prompt.append("对话情景：")
            prompt.extend([f"- {line}" for line in question_case['dialogue']])
        elif 'scenario' in question_case:
            prompt.append(f"场景描述：{question_case['scenario']}")
        
        prompt.append("\n规则说明：")
        prompt.extend(KorLogicCooperativePrinciplebootcamp.RULE_DESCRIPTIONS[question_case['correct']])
        prompt.append("\n补充说明：" + question_case['explanation'])
        prompt.append("\n正确答案是：[[ ]]")
        
        return '\n'.join(prompt)

    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[\[([ABC])\]\]', output, re.IGNORECASE)
        return matches[-1].upper() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return str(solution).upper() == identity['correct']
