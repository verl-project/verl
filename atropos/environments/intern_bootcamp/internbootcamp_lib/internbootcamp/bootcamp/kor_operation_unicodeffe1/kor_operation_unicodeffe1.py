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
A￡B=(A∪B)−(A∩B).Example questions are as follows:

<example 0>
A={1,2,3}, B={3,4,5}.
Compute A￡B.
Please wrap the answer in double square brackets, like this: [[{the elements in the collection}]].
</example 0>

<example 1>
A={a,b,c}, B={c,d,e}.
Compute A￡B.
Please wrap the answer in double square brackets, like this: [[{the elements in the collection}]].
</example 1>

<example 2>
A={1,2,3,4,5}, B={4,5,6,7,8}.
Compute A￡B.
Please wrap the answer in double square brackets, like this: [[{the elements in the collection}]]
</example 2>

<example 3>
A={m,n}, B={n,o,p}.
Compute A￡B.
Please wrap the answer in double square brackets, like this: [[{the elements in the collection}]]
</example 3>

<example 4>
A={1,3,5}, B={2,4,6}.
Compute A￡B.
Please wrap the answer in double square brackets, like this: [[{the elements in the collection}]]
</example 4>

<example 5>
A={x,y,z}, B={w,x,z}.
Compute A￡B.
Please wrap the answer in double square brackets, like this: [[{the elements in the collection}]]
</example 5>

<example 6>
A={p,q,r}, B={q,r,s}.
Compute A￡B.
Please wrap the answer in double square brackets, like this: [[{the elements in the collection}]].
</example 6>

<example 7>
A={x∈R∣x>0}, B={x∈R∣x<1}.
Compute A￡B.
Use a set like {x∣x<1} to present the answer. Use '≤' for less than or equal to, and '≥' for greater than or equal to. Separate conditions with 'or' if there are multiple conditions.
Please wrap the answer in double square brackets, like this: [[you answer]].
</example 7>

<example 8>
A={x∣x is a real number}, B = {x∣x^2<1}.
Compute A￡B.
Use a set like {x∣x<1} to present the answer. Use '≤' for less than or equal to, and '≥' for greater than or equal to. Separate conditions with 'or' if there are multiple conditions.
Please wrap the answer in double square brackets, like this: [[you answer]].
</example 8>

<example 9>
A={x∣x is a natural number}, B = {x∣x is postive}.
Compute A￡B.
Please wrap the answer in double square brackets, like this: [[{the elements in the collection}]]
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class KorOperationUnicodeffe1bootcamp(Basebootcamp):
    def __init__(self, finite_prob=0.7, interval_prob=0.2, special_prob=0.1, max_size=5, element_type='mixed', **kwargs):
        super().__init__()
        total = finite_prob + interval_prob + special_prob
        if total <= 0:
            total = 1.0
            finite_prob = 0.7
            interval_prob = 0.2
            special_prob = 0.1
        self.finite_prob = finite_prob / total
        self.interval_prob = interval_prob / total
        self.special_prob = special_prob / total
        self.max_size = max_size
        self.element_type = element_type

    def case_generator(self):
        rand_val = random.random()
        if rand_val < self.finite_prob:
            return self.generate_finite_case()
        elif rand_val < self.finite_prob + self.interval_prob:
            return self.generate_interval_case()
        else:
            return self.generate_special_case()

    def generate_finite_case(self):
        element_type = self.element_type
        if element_type == 'mixed':
            element_type = random.choice(['number', 'letter'])
        
        size_A = random.randint(2, self.max_size)
        size_B = random.randint(2, self.max_size)
        
        if element_type == 'number':
            elements = list(range(1, 21))
            A = sorted(random.sample(elements, size_A))
            B = sorted(random.sample(elements, size_B))
        else:
            letters = [chr(ord('a') + i) for i in range(26)]
            A = sorted(random.sample(letters, size_A))
            B = sorted(random.sample(letters, size_B))
        
        A_set = set(A)
        B_set = set(B)
        solution = sorted(list(A_set.symmetric_difference(B_set)))
        return {
            'type': 'finite',
            'A': A,
            'B': B,
            'solution': solution
        }

    def generate_interval_case(self):
        template = random.choice([1, 2, 3])
        if template == 1:  # 非重叠区间
            a = random.randint(-5, 3)
            b = a + random.randint(2, 4)
            while True:
                c = random.randint(b+1, b+3)
                if c > b: break
            A_desc = f'x > {a}'
            B_desc = f'x < {b}'
            solution = f'{{x | x ≤ {a} or x ≥ {b}}}'
        elif template == 2:  # 包含区间
            a = random.randint(2, 5)
            b = random.randint(-3, a-1)
            A_desc = f'x < {a}'
            B_desc = f'x > {b}'
            solution = f'{{x | x ≤ {b} or x ≥ {a}}}'
        else:  # 二次不等式
            c = random.randint(1, 3)
            A_desc = 'x is a real number'
            B_desc = f'x² < {c**2}'
            solution = f'{{x | x ≤ -{c} or x ≥ {c}}}'
        return {
            'type': 'interval',
            'A': A_desc,
            'B': B_desc,
            'solution': solution
        }

    def generate_special_case(self):
        case_type = random.choice([1, 2])
        if case_type == 1:  # 自然数 vs 正整数
            return {
                'type': 'special',
                'A': 'x is a natural number (including 0)',
                'B': 'x is a positive integer',
                'solution': '{0}'
            }
        else:  # 全体实数 vs 空集
            return {
                'type': 'special',
                'A': 'x is a real number',
                'B': 'x is an element of empty set',
                'solution': '{x | x ∈ ℝ}'
            }

    @staticmethod
    def prompt_func(question_case) -> str:
        case_type = question_case.get('type', 'finite')
        if case_type == 'finite':
            A_str = "{" + ", ".join(map(str, question_case['A'])) + "}"
            B_str = "{" + ", ".join(map(str, question_case['B'])) + "}"
            prompt = f"Given two finite sets:\nA = {A_str}\nB = {B_str}\n\nCompute the symmetric difference A£B.\nFormat: [[sorted, comma-separated elements]]"
        elif case_type == 'interval':
            prompt = (
                f"Given:\nA = {{{question_case['A']}}}\n"
                f"B = {{{question_case['B']}}}\n\n"
                "Compute A£B using inequalities with ≤/≥.\n"
                "Format: [[{{x | condition}}]]"
            )
        else:
            prompt = (
                f"Given:\nA = {{{question_case['A']}}}\n"
                f"B = {{{question_case['B']}}}\n\n"
                "Compute A£B considering mathematical definitions.\n"
                "Format: [[set_notation]]"
            )
        
        rules = (
            "Rules:\n"
            "1. A£B = (A∪B) - (A∩B)\n"
            "2. Use comma-separated sorted elements for finite sets\n"
            "3. Use '≤'/'≥' for inequalities\n"
            "4. Answer MUST be within double square brackets\n\n"
        )
        return rules + prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[\[(.*?)\]\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        case_type = identity.get('type', 'finite')
        correct = identity['solution']

        def normalize(s):
            s = re.sub(r'\s+', '', s).lower()
            s = s.replace('< =', '≤').replace('>=', '≥')
            s = s.replace('=<', '≤').replace('=>', '≥')
            return s

        if case_type == 'finite':
            try:
                elements = re.findall(r'[^,{}\s]+', solution)
                parsed = {int(e) if e.isdigit() else e for e in elements}
                return parsed == set(correct)
            except:
                return False
        else:
            return normalize(solution) == normalize(str(correct))
