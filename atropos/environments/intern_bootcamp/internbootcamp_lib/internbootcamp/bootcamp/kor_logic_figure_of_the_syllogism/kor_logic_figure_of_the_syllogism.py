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
Between propositions p1 and p2, the representation is defined as follows:
A: ∀p1⇒p2
E: ∀p1⇒¬p2
I: ∃p1⇒p2
O: ∃p1⇒¬p2

The figures and moods of the syllogism are as follows:
1.Figure I
Form:
M()P
S()M
∴S()P
The parentheses can be filled in with the following Valid Moods.
Valid Moods:
- AAA
- EAE
- AII
- EIO

2.Figure II
Form: 
P()M
S()M
∴S()P
The parentheses can be filled in with the following Valid Moods.
Valid Moods:
- AEE
- EAE
- EIO
- AOO

3.Figure III
Form:
M()P
M()S
∴S()P
The parentheses can be filled in with the following Valid Moods.
Valid Moods:
- AII
- EIO
- IAI
- OAO

4.Figure IV
Form: 
P()M
M()S
∴S()P
The parentheses can be filled in with the following Valid Moods.
Valid Moods:
- AEE
- EIO
- IAIExample questions are as follows:

<example 0>
Given the logical statements:
∀M⇒P, ∀S⇒M ⇒ ∀S⇒P

Determine which figure and mood of syllogism the argument corresponds to,using the format [[I/II/III/IV];[Mood]].
</example 0>

<example 1>
Given the logical statements:
∀M⇒P, ∃M⇒S ⇒ ∃S⇒P.

Determine which figure and mood of syllogism the argument corresponds to, using the format [[I/II/III/IV];[Mood]].
</example 1>

<example 2>
Given the logical statements:
∀P⇒M, ∀M⇒¬S ⇒ ∀S⇒¬P

Determine which figure and mood of syllogism the argument corresponds to, using the format [[I/II/III/IV];[Mood]].
</example 2>

<example 3>
Given the logical statements:
∀P⇒¬M, ∃S⇒M ⇒ ∃S⇒¬P

Determine which figure and mood of syllogism the argument corresponds to, using the format [[I/II/III/IV];[Mood]].
</example 3>

<example 4>
Given the logical statements:
∀P⇒M, ∃S⇒¬M ⇒ ∃S⇒¬P

Determine which figure and mood of syllogism the argument corresponds to, using the format [[I/II/III/IV];[Mood]].
</example 4>

<example 5>
Please fill in the parentheses below
P()M
SEM
∴S()P

Provide the answer in the format of [[];[]].
</example 5>

<example 6>
Please fill in the parentheses below
()O()
()A()
∴SOP

Provide the answer in the format of [[];[];[];[]].
</example 6>

<example 7>
Please fill in the parentheses below
()AP
MI()
∴SIP

Provide the answer in the format of [[];[]].
</example 7>

<example 8>
Please fill in the parentheses below
P()M
M()S
∴SOP

Provide the answer in the format of [[];[]].
</example 8>

<example 9>
Please fill in the parentheses below
MIP
M()S
∴S()P

Provide the answer in the format of [[];[]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re
from itertools import zip_longest

SYLLOGISM_CONFIG = {
    'I': {
        'premises': [('M', 'P'), ('S', 'M')],
        'conclusion': ('S', 'P'),
        'valid': ['AAA', 'EAE', 'AII', 'EIO'],
        'templates': [
            "M({})P",  # 大前提模板
            "S({})M",  # 小前提模板 
            "S({})P"   # 结论模板
        ]
    },
    'II': {
        'premises': [('P', 'M'), ('S', 'M')],
        'conclusion': ('S', 'P'),
        'valid': ['AEE', 'EAE', 'EIO', 'AOO'],
        'templates': [
            "P({})M",
            "S({})M",
            "S({})P"
        ]
    },
    'III': {
        'premises': [('M', 'P'), ('M', 'S')],
        'conclusion': ('S', 'P'),
        'valid': ['AII', 'EIO', 'IAI', 'OAO'],
        'templates': [
            "M({})P",
            "M({})S",
            "S({})P"
        ]
    },
    'IV': {
        'premises': [('P', 'M'), ('M', 'S')],
        'conclusion': ('S', 'P'),
        'valid': ['AEE', 'EIO', 'IAI'],
        'templates': [
            "P({})M",
            "M({})S",
            "S({})P"
        ]
    }
}

MOOD_CONVERSION = {
    'A': lambda s,p: f"∀{s}⇒{p}",
    'E': lambda s,p: f"∀{s}⇒¬{p}",
    'I': lambda s,p: f"∃{s}⇒{p}",
    'O': lambda s,p: f"∃{s}⇒¬{p}"
}

class KorLogicFigureOfTheSyllogismbootcamp(Basebootcamp):
    def __init__(self, *, enable_fill=True, **kwargs):
        super().__init__(**kwargs)
        self.enable_fill = enable_fill
        self.figure_pool = list(SYLLOGISM_CONFIG.keys())

    def case_generator(self):
        # 选择随机图形和有效式
        figure = random.choice(self.figure_pool)
        cfg = SYLLOGISM_CONFIG[figure]
        mood = random.choice(cfg['valid'])
        
        # 50%概率生成填空题型
        if self.enable_fill and random.random() < 0.5:
            return self._make_fill_case(figure, mood, cfg)
        return self._make_standard_case(figure, mood, cfg)

    def _make_standard_case(self, figure, mood, cfg):
        """生成标准推理题型"""
        premises = []
        # 生成前提命题
        for i, (subj, pred) in enumerate(cfg['premises']):
            premises.append(MOOD_CONVERSION[mood[i]](subj, pred))
        # 生成结论命题
        conc_subj, conc_pred = cfg['conclusion']
        conclusion = MOOD_CONVERSION[mood[2]](conc_subj, conc_pred)
        
        return {
            'type': 'standard',
            'figure': figure,
            'mood': mood,
            'premises': premises,
            'conclusion': conclusion
        }

    def _make_fill_case(self, figure, mood, cfg):
        """生成填空题型""" 
        components = []
        # 生成大前提和小前提
        for i in range(2):
            template = cfg['templates'][i]
            # 40%概率显示式字母
            if random.random() < 0.4:
                components.append(template.format(mood[i]))
            else:
                components.append(template.format(''))
        # 结论始终隐藏式字母
        components.append(cfg['templates'][2].format(''))
        
        return {
            'type': 'fill',
            'figure': figure,
            'mood': mood,
            'template': '\n'.join(components),
            'visible_moods': [c for c in mood if c in ''.join(components)]
        }

    @staticmethod
    def prompt_func(case):
        base_rule = """三段论规则：
1. 图形结构：
   I: M-P, S-M → S-P
   II: P-M, S-M → S-P
   III: M-P, M-S → S-P
   IV: P-M, M-S → S-P

2. 有效式：
   I: AAA, EAE, AII, EIO
   II: AEE, EAE, EIO, AOO  
   III: AII, EIO, IAI, OAO
   IV: AEE, EIO, IAI

3. 式字母含义：
   A: 全称肯定 ∀x⇒y
   E: 全称否定 ∀x⇒¬y
   I: 特称肯定 ∃x⇒y
   O: 特称否定 ∃x⇒¬y

请根据以下内容判断三段论的图形和式，答案格式：[[图形;式]]（如[[I;AAA]]）\n"""

        if case['type'] == 'standard':
            problem = "给定命题：\n"
            problem += '\n'.join(case['premises']) 
            problem += f"\n∴ {case['conclusion']}"
            question = "请判断对应的三段论图形和式"
        else:
            problem = "补全三段论结构：\n"
            problem += case['template'].replace('{}', '()')  # 确保显示括号
            question = "根据结构推断完整的图形和式"
        
        return f"{base_rule}{problem}\n\n{question}\n答案格式：[[图形;式]]"

    @staticmethod
    def extract_output(text):
        # 支持多格式匹配：带空格、不同括号等
        patterns = [
            r'\[{2}\s*([IV]{1,3})\s*;\s*([AEIO]{3})\s*\]{2}',  # [[II; EAE]]
            r'\[{1}\s*([IV]{1,3})\s*;\s*([AEIO]{3})\s*\]{1}',  # [III;AII]
            r'图形\s*([IV]{1,3})\s*式\s*([AEIO]{3})'           # 图形IV 式EIO
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                fig, mood = matches[-1]
                return f"{fig.upper()};{mood.upper()}"
        return None

    @classmethod
    def _verify_correction(cls, solution, case):
        try:
            # 规范格式：去除空格/中文符号
            cleaned = solution.replace('；', ';').replace('，', ';').replace(' ', '')
            figure_part, mood_part = cleaned.split(';')
            
            # 验证图形有效性
            if figure_part not in SYLLOGISM_CONFIG:
                return False
                
            # 验证式有效性
            cfg = SYLLOGISM_CONFIG[figure_part]
            if len(mood_part)!=3 or mood_part not in cfg['valid']:
                return False
            
            # 最终验证
            return figure_part == case['figure'] and mood_part == case['mood']
        except:
            return False
