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
Five Methods for Exploring Causal Relationships

1. Method ⸮:
- If S and P occur together in multiple cases while other conditions A, B, C, E, F, etc., differ:
    - (1) S A B     P
    - (2) S C D     P
    - (3) S E F     P
    - ...
    - Therefore, S and P may have a causal relationship.

2. Method ؆:
- If P occurs when S is present and does not occur when S is absent:
    - (1) S A B     P
    - (2) - A B     P
    - Therefore, S and P may have a causal relationship.

3. Method ꙮ:
- Positive group: S and P occur together, while other conditions A, B, C, D, E, F, etc., differ:
    - Positive group
        - (1) S A B     P
        - (2) S C D    P
        - (3) S E F     P
    - Negative group: S is absent, P is also absent, and other conditions A, C, D, E, F, etc., differ:
        - Negative group
            - (1') - A C    -
            - (2') - D E    -
            - (3') - B F    -
    - Therefore, S and P may have a causal relationship.

4. Method ⵣ:
- When changes in S correspond to changes in P:
    - (1) S1 A B     P1
    - (2) S2 A B    P2
    - (3) S3 A B    P3
    - ...
    - Therefore, S and P may have a causal relationship.

5. Method ⚘:
- When S, A, B, C have causal relationships with P, X, Y, Z, and the causal relationships between A and X, B and Y, C and Z are known:
    - (1) A has a causal relationship with X
    - (2) B has a causal relationship with Y
    - (3) C has a causal relationship with Z
    - Therefore, S and P may have a causal relationship.Example questions are as follows:

<example 0>
People rub their frozen hands together, and their hands become warm; 
people strike cold stones, and the stones can spark; 
people continuously hammer an iron block, and the iron block can also become red-hot. From this, it can be inferred that the movement of objects can generate heat.

The causal derivation of this discourse fits which method:
A.⸮ method   B.؆ method   C.ꙮ method    D.ⵣ method   E.⚘ method

Please give your answer in [[A/B/C/D/E]] format.
</example 0>

<example 1>
At the Southern Experiment Station of the University of California, USA, Chinese hybrid rice varieties were tested against American rice varieties twice, in 1980 and 1981. 
The temperature, fertiliser, water, soil and management methods were the same, but the only difference was the sub-seed. 
The results of the trial planting: in 1980, the average harvest of hybrid rice in China was 737 mcm per mu, while that of the U.S. variety rice was 279.25 mcm per mu; in 1981, the average harvest of hybrid rice in China was 783.15 mcm per mu, while that of the U.S. variety rice was 279.35 mcm per mu. 
The use of Chinese hybrid rice varieties was found to be the cause of the high yield of rice in the course of comparative trials.

Which of the above approaches is consistent with the derivation of causality in this discourse:
A. ⸮ method B. ؆ method C. ꙮ method D. ⵣ method E. ⚘ method

Please give your answer in [[A/B/C/D/E]] format.
</example 1>

<example 2>
In examining the relationship between regular physical activity and lung size, a group of people of different ages, genders, and occupations who were regularly physically active were first examined, followed by another group of people of different ages, genders, and occupations who were infrequently physically active; when comparing the lung sizes of these two groups, it was found that those who were regularly physically active had significantly larger lungs than those who were infrequently physically active. 
When comparing the sizes of the lungs of these two groups, it was found that those who were regularly physically active had significantly larger lung volumes than those who were rarely physically active. It was then concluded that regular physical activity resulted in an increase in lung capacity.

The derivation of cause and effect in this passage is consistent with which of the approaches:

A. ⸮ Method  B. ؆ Method   C. ꙮ Method   D. ⵣ Method   E. ⚘ Method

Please give your answer in [[A/B/C/D/E]] format.
</example 2>

<example 3>
One year, a symposium was held in London, England, on the question of how long a person who has been shipwrecked and fallen into the water can hold out in the water. 
The researchers found that the average person can last 15 minutes at 0°C, 30 minutes at 2.5°C, 1 hour at 5°C, 3 hours at 10°C, and more than a day and night at 25°C.
 These data are important and provide a good example of how long people can survive in the water. 
These data are important as they provide a quantitative basis for research and improvement of various insulating swimsuits so that people can stay in cold water for longer periods of time. 
From this we can establish that there is a causal link between changes in water temperature and changes in the length of time that people stay in water.

The derivation of causality in this discourse is consistent with which approach:
A. ⸮ method   B. ؆ method   C. ꙮ method   D. ⵣ method   E. ⚘ method

Please give your answer in [[A/B/C/D/E]] format.
</example 3>

<example 4>
The discovery of Neptune in 1846 has always been considered a prime example of the use of the residual method. 
Based on the law of gravitation of Marcel van Gogh, scientists were able to calculate the effects of the various objects known at the time on Uranus, and thus calculate the orbit of Uranus. 
However, based on astronomical observations, the actual orbit of Uranus deviated significantly from the theoretically calculated orbit. 
As a result, scientists deduced that the gravitational force of a then-undiscovered object might have caused Uranus to deviate. 
The scientists calculated the position of this possible object and later found the new star, Neptune, in that position.

The derivation of cause and effect in this discourse is consistent with which of the methods above:
A. ⸮ method   B. ؆ method   C. ꙮ method   D. ⵣ method   E. ⚘ method

Please give your answer in [[A/B/C/D/E]] format.
</example 4>

<example 5>
In 1973, the comrades of the Shanghai Water Geology Team carried out extensive investigations in order to find out the main cause of the subsidence of the city of Shanghai. During the investigation, it was found that the ground subsidence was more serious in several working areas in the east and west of the city. 
The conditions of these work areas were different, such as the layout of workplaces, geographical conditions, and so on, which were totally different. 
However, they later found that in the several workplaces with different conditions, there was one common situation, i.e. \"the amount of subsidence was greater in areas with a relatively higher concentration of textiles\". 
Subsequently, further investigation revealed that although many of the conditions in the textile factories were different, there was a common thread in the fact that the textile factories had \"a high number of deep wells and a high volume of underground water use\".
In the end, they concluded that this common situation was the main reason for the sinking of the city's surface.

Which method of deducing cause and effect in this discourse is appropriate:
A. ⸮ Method   B. ؆ Method   C. ꙮ Method   D. ⵣ Method   E. ⚘ Method

Please give your answer in [[A/B/C/D/E]] format.
</example 5>

<example 6>
For a long time, it has been noted that animals that have been injured tend to hide in quiet places and repeatedly lick their mouths with their heads. 
Some people believe that this is a response to pain, while others believe that animals use this method to keep away from pain. 
In response, scientists conducted an experiment. 
They divided a number of wheat and nickel white animals into two groups: one with salivary glands that had been removed manually, and one with normal wheat and nickel white, which differed only in this case, but were identical in all other cases. Then bruise it. 
The result was that the wound healed much faster in the normal millets than in those whose salivary glands had been removed.

The derivation of cause and effect in this discourse is consistent with which method:
A. ⸮ method   B. ؆ method   C. ꙮ method   D. ⵣ method   E. ⚘ method

Please give your answer in [[A/B/C/D/E]] format.
</example 6>

<example 7>
Having previously used the method of seeking common ground to clarify that the cause of the ground subsidence in Shanghai is \"the large number of deep wells drilled and the large amount of underground water used,\" the comrades of the Shanghai Municipal Water Geology Team further investigated the history and current status of deep wells and the use of underground water in Shanghai, and created a \"file\" for each well. \"The first wells in Shanghai were dug in the early 1970s. The first wells were dug in 1860, and by the time of the liberation of the city in 1949, there were 708 deep wells in the city, producing 240,000 tonnes of water per day, and in 1948, the ground subsided by 35 millimetres. During the Great Leap Forward movement of 1958-1960, the number of deep wells increased to 1,183, with a water output of 560,000 tonnes per day, and the amount of surface subsidence increased to 98 millimetres per year. Therefore, we can conclude that the more deep wells there are, the more water is used underground, and the faster the earth sinks.

The derivation of cause and effect in this passage of the discourse is consistent with which method:
A. ⸮ method   B. ؆ method   C. ꙮ method   D. ⵣ method   E. ⚘ method

Please give your answer in [[A/B/C/D/E]] format.
</example 7>

<example 8>
In 1789, Klaprozli, a German man, experimented with a blackish, leach-like mineral and obtained a lustrous, blackish substance with an appearance very similar to that of the metal, which he considered to be a new element and named uranium. Later, Mrs Curie and her husband Pierre Curie experimentally measured the uranium content of a sample of leachite to determine whether it contained uranium worth refining. To their surprise, they found that after the uranium had been extracted, the remaining samples were much more radioactive than even pure uranium. This meant that the radioactivity could not be explained by the presence of uranium. Therefore, it must contain another radioactive element. After further research, they finally realised that this remaining radioactivity was a new element emitting radioactivity and isolated the elements radium and polonium from the leachate.

The causal derivation of this passage of the discourse is consistent with which method:
A. ⸮ method   B. ؆ method   C. ꙮ method   D. ⵣ method   E. ⚘ method

Please give your answer in [[A/B/C/D/E]] format.
</example 8>

<example 9>
Research has shown that the impact of family education styles on children's emotional intelligence varies across cultures. Through an analysis of similarities and differences, we first examined a group of families whose parents adopted an educational style that actively participated and encouraged the expression of emotions. Children in these families demonstrated higher levels of emotional intelligence, showing greater emotional expression and problem-solving skills. In contrast, the other group of parents adopted a passive and negative educational style with little involvement in their children's emotional expression and problem-solving processes. In these families, the children had significantly lower EQ development and showed emotional suppression and inadequate problem-solving skills. Therefore, we conclude that a family education style that actively participates in and encourages the expression of emotions significantly contributes to the development of children's emotional intelligence.

The causal derivation of this discourse is consistent with which approach:
A. ⸮ method   B. ؆ method   C. ꙮ method   D. ⵣ method   E. ⚘ method

Please give your answer in [[A/B/C/D/E]] format.
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class KorLogicLogicalMethodsForExploringCauseAndEffectRelationshipsbootcamp(Basebootcamp):
    _METHOD_TEMPLATES = [
        {  # 方法⸮ (求同法)
            'code': 'A',
            'templates': [
                "在多个案例中，{S}和{P}同时出现，而其他条件各不相同：\n(1) {S}与条件A、B同时存在时出现{P}\n(2) {S}与条件C、D同时存在时也出现{P}\n这符合哪种方法？",
                "研究发现{S}与{P}在以下不同情境中总是一起出现：\n- 情境1：条件X/Y存在时\n- 情境2：条件M/N存在时\n其他条件均不同，这符合："
            ],
            'factors': [
                {'S': '摩擦生热', 'P': '温度上升'},
                {'S': '广告曝光', 'P': '销量增长'}
            ]
        },
        {  # 方法؆ (差异法)
            'code': 'B',
            'templates': [
                "当{S}存在时{P}发生，当{S}不存在时{P}不发生（其他条件相同）：\n实验组：{S} + 条件A → {P}\n对照组：条件A → 无{P}\n这符合：",
                "对比实验显示：\n- 添加{S}时出现{P}\n- 移除{S}时{P}消失\n其他变量保持相同，这属于："
            ],
            'factors': [
                {'S': '新药物X', 'P': '症状缓解'},
                {'S': '特定基因', 'P': '疾病表现'}
            ]
        },
        {  # 方法ꙮ (求同求异并用法)
            'code': 'C',
            'templates': [
                "正组案例({S}存在)：\n- 案例1：条件A/B → {P}\n- 案例2：条件C/D → {P}\n负组案例({S}缺失)：\n- 案例1'：条件A/C → 无{P}\n- 案例2'：条件D/E → 无{P}\n这符合：",
                "研究包含两组对比：\n阳性组({S}存在)：不同条件下均出现{P}\n阴性组({S}缺失)：不同条件下均无{P}\n这属于："
            ],
            'factors': [
                {'S': '教育培训', 'P': '技能提升'},
                {'S': '定期维护', 'P': '设备故障减少'}
            ]
        },
        {  # 方法ⵣ (共变法)
            'code': 'D',
            'templates': [
                "当{S}从S1变化到S2时，{P}相应从P1变为P2（其他条件不变），例如：\n{S}=低强度 → {P}=轻微\n{S}=高强度 → {P}=显著",
                "跟踪数据显示{S}与{P}呈正相关：\n- 时期1：{S}↑20% → {P}↑15%\n- 时期2：{S}↓30% → {P}↓25%"
            ],
            'factors': [
                {'S': '温度变化', 'P': '材料膨胀'},
                {'S': '学习时长', 'P': '成绩提高'}
            ]
        },
        {  # 方法⚘ (剩余法)
            'code': 'E',
            'templates': [
                "在排除已知因素（A、B）的影响后，剩余现象只能由{S}解释：\n总效应 = 已知效应 + {S}的影响",
                "当已知因素解释部分现象，剩余未解释部分与{S}的存在相关"
            ],
            'factors': [
                {'S': '未知行星', 'P': '轨道偏差'},
                {'S': '新型元素', 'P': '异常辐射'}
            ]
        }
    ]

    def __init__(self, **params):
        super().__init__(**params)
        self.random_seed = params.get('random_seed')
        if self.random_seed:
            random.seed(self.random_seed)

    def case_generator(self):
        method_info = random.choice(self._METHOD_TEMPLATES)
        template = random.choice(method_info['templates'])
        factors = random.choice(method_info['factors'])
        
        # 严格验证模板变量
        required_keys = set(re.findall(r'{(\w+)}', template))
        assert required_keys.issubset(factors.keys()), f"模板变量{required_keys}与参数{factors.keys()}不匹配"
        
        return {
            'method': method_info['code'],
            'description': template.format(**factors),
            'variables': factors
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        methods_intro = """请根据因果推断方法论选择正确选项：

1. ⸮方法（求同法）: 不同案例中唯一共同因素
2. ؆方法（差异法）: 单一变量存在与否的对比
3. ꙮ方法（并用法）: 正负组的双重验证
4. ⵣ方法（共变法）: 因素与现象的同步变化
5. ⚘方法（剩余法）: 排除已知后的剩余归因

答案请使用[[大写字母]]格式，示例：[[A]]

当前案例描述：
"""
        return f"{methods_intro}{question_case['description']}"

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[\[\s*([A-Ea-e])\s*\]\]', output)
        return matches[-1].upper() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        return solution == identity['method']
