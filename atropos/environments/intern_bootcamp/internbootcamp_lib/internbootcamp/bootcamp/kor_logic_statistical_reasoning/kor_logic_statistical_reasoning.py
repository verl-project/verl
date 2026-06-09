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
1. Statistical Reasoning Categories and Symbolization

(1) U-Generalization
    - Symbol: `U`
    - Definition: If all individuals in a sample possess a certain attribute, we infer that all individuals in the population may also possess that attribute.

(2) P-Generalization
    - Symbol: `P`
    - Definition: If a portion of the individuals in a sample possess a certain attribute, we infer that a certain proportion of the individuals in the population may possess that attribute.

(3) C-Reasoning
    - Symbol: `C`
    - Definition: If two samples exhibit similarities in certain attributes, we infer that these two samples may come from populations with similar attribute proportions.

2. Statistical Attribute Inference Based on Samples

(1) Rule Description:
- Randomly select a representative sample from the population.
- Observe and record specific attributes of individuals in the sample.
- Depending on the frequency of the attributes and the type of sample, apply the following rules:

(2) U-Generalization Rule:
- If all individuals (denoted as `n`) in the sample possess attribute `A`, then we can infer that all individuals in the population also possess attribute `A`.
- Symbolization: If `U(A, n)`, then `∀x ∈ P, A(x)`.

(3) P-Generalization Rule:
- If `k` individuals in the sample possess attribute `A`, where `k < n`, then we can infer that approximately `k/n` proportion of the individuals in the population possess attribute `A`.
- Symbolization: If `P(A, k, n)`, then `Pr(A) ≈ k/n`.

(4) C-Reasoning Rule:
- If two samples S1 and S2 exhibit similar proportions in attribute `A`, i.e., `P(A, k1, n1)` and `P(A, k2, n2)`, then we can infer that these two samples may come from populations with similar proportions of attribute `A`.
- Symbolization: If `C(A, k1/n1, k2/n2)`, then `Pr(A, P1) ≈ Pr(A, P2)`.Example questions are as follows:

<example 0>
In a class, 10 students were randomly selected to take a maths test and all got an A. According to the U-Generalization Rule, estimate the proportion of the whole class that would get an A if the class had 50 students.

Give your answer in [[number%]] format.
</example 0>

<example 1>
An air quality test was conducted in a city on 5 randomly selected days and it was found that 4 of the days had an air quality index (AQI) below 50.Using the P-Generalization rule, estimate the proportion of days in which the average AQI of the city was below 50. 

Please give your answer in [[number%]] format.
</example 1>

<example 2>
A clinical trial of a new drug showed a positive response in 150 of 200 patients. Using the P-Generalization rule, the effectiveness of the drug in a wider group of patients was estimated.

Please give your answer in [[number%]] format.
</example 2>

<example 3>
In a biodiversity research project, researchers observed birds on an island. They randomly selected 20 bird species endemic to that island to be examined for health status.
If all 20 birds showed good health, with no signs of disease or parasitic infections, using the U-generalisation rule, the researchers could make an estimate of what the proportion of that species on the whole island was healthy.

Please give your answer in [[number%]] format.
</example 3>

<example 4>
A company performs quality testing on its products and randomly selects 50 products from a batch of 1,000, resulting in 2 defective products. Using the P-Generalization rule, estimate the rate of defective products for the entire batch.

Please give your answer in [[number%]] format.
</example 4>

<example 5>
The final class of a high school conducts a mock examination and all 50 students score 90 or more in mathematics. 
Please represent them symbolically according to U-Generalization rule.

The observed attribute is a maths score of 90 or above, denoted by A.
P denotes the maths score of all the students in the final year of high school.

Therefore, the whole can be symbolised to denote why?

Please give the answer in the format [[]].
</example 5>

<example 6>
In a survey of student satisfaction in two different schools, 180 out of 200 students in School X said they were satisfied with the school's facilities, and 210 out of 300 students in School Y said they were satisfied.

Using the C-Reasoning Rule, 
denote the attribute 'student satisfaction' as F.

Therefore, the whole can be symbolised to denote why?

Please give the answer in the format [[]].
</example 6>

<example 7>
In a library's annual report, 1,000 loans are recorded, of which 200 are for science fiction books.
Please represent this symbolically according to the P-Generalization rule .

Denote the attribute science fiction books borrowed as A.

Therefore, the whole can be symbolised to denote why?

Please give the answer in the format [[]].
</example 7>

<example 8>
In two different regional health surveys, 90 out of 100 respondents in Region A and 75 out of 150 respondents in Region B reported exercising daily.
Please denote this symbolically by C-reasoning  rule.

Denote the attribute daily running as S.

Therefore, the whole can be symbolised to denote why?

Please give your answer in the format [[]].
</example 8>

<example 9>
In a survey of traffic violations in a city, 500 cars were randomly selected for observation and 40 cars were found to be speeding.
Please represent them symbolically according to P-Generalization rule .
Denote the property speeding behaviour as A.

Therefore, the whole can be symbolised to denote what?

Please give your answer in the format [[]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class KorLogicStatisticalReasoningbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__(**params)
        self.min_n = params.get('min_n', 5)
        self.max_n = params.get('max_n', 50)
        self.attribute_descriptions = [
            'math score above 90',
            'healthy',
            'satisfied with facilities',
            'defective',
            'daily exercise',
            'speeding behavior',
            'positive response',
        ]
    
    def case_generator(self):
        question_type = random.choice([
            'U_calculation', 'U_symbolization',
            'P_calculation', 'P_symbolization',
            'C_symbolization'
        ])
        
        case = {'question_type': question_type}
        attr_symbol = chr(random.randint(65, 90))  # Random uppercase letter
        case['attribute'] = {
            'symbol': attr_symbol,
            'desc': random.choice(self.attribute_descriptions)
        }
        
        if 'U_' in question_type:
            case['n'] = random.randint(self.min_n, self.max_n)
        elif 'P_' in question_type:
            case['n'] = random.randint(self.min_n, self.max_n)
            case['k'] = random.randint(1, case['n']-1)
        elif question_type == 'C_symbolization':
            case['n1'] = random.randint(self.min_n, self.max_n)
            case['k1'] = random.randint(1, case['n1'])
            case['n2'] = random.randint(self.min_n, self.max_n)
            case['k2'] = random.randint(1, case['n2'])
        
        return case
    
    @staticmethod
    def prompt_func(question_case):
        attr = question_case['attribute']
        qt = question_case['question_type']
        
        if qt == 'U_calculation':
            return (
                f"In a study, {question_case['n']} subjects were randomly selected and all demonstrated "
                f"{attr['desc']} (denoted as {attr['symbol']}). Using U-Generalization Rule, estimate the proportion. "
                "Format your answer as [[number%]]."
            )
        elif qt == 'U_symbolization':
            return (
                f"Represent symbolically: All {question_case['n']} sampled subjects have {attr['desc']} "
                f"(denoted as {attr['symbol']}). Apply U-Generalization Rule. "
                f"Format your answer as [[U({attr['symbol']}, {question_case['n']})]]."
            )
        elif qt == 'P_calculation':
            return (
                f"In a sample of {question_case['n']} subjects, {question_case['k']} demonstrated "
                f"{attr['desc']} (denoted as {attr['symbol']}). Using P-Generalization Rule, estimate the proportion. "
                "Format your answer as [[number%]]."
            )
        elif qt == 'P_symbolization':
            return (
                f"Symbolize: {question_case['k']} out of {question_case['n']} samples show {attr['desc']} "
                f"(denoted as {attr['symbol']}). Apply P-Generalization Rule. "
                f"Format your answer as [[P({attr['symbol']}, {question_case['k']}, {question_case['n']})]]."
            )
        elif qt == 'C_symbolization':
            return (
                f"Two samples show {attr['desc']} (denoted as {attr['symbol']}): "
                f"Sample 1 has {question_case['k1']} out of {question_case['n1']}, "
                f"Sample 2 has {question_case['k2']} out of {question_case['n2']}. Apply C-Reasoning Rule. "
                f"Format your answer as [[C({attr['symbol']}, {question_case['k1']}/{question_case['n1']}, {question_case['k2']}/{question_case['n2']})]]."
            )
        return ""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[\[(.*?)\]\]', output)
        return matches[-1].strip() if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        if not solution:
            return False
        
        try:
            qt = identity['question_type']
            attr = identity['attribute']['symbol']
            solution_clean = re.sub(r'\s+', '', solution)
            
            if qt == 'U_calculation':
                return abs(float(solution_clean.strip('%')) - 100) < 1e-6
            elif qt == 'P_calculation':
                expected = (identity['k'] / identity['n']) * 100
                return abs(float(solution_clean.strip('%')) - expected) < 1e-6
            elif qt == 'U_symbolization':
                expected = f"U({attr},{identity['n']})"
                return solution_clean == re.sub(r'\s+', '', expected)
            elif qt == 'P_symbolization':
                expected = f"P({attr},{identity['k']},{identity['n']})"
                return solution_clean == re.sub(r'\s+', '', expected)
            elif qt == 'C_symbolization':
                expected = f"C({attr},{identity['k1']}/{identity['n1']},{identity['k2']}/{identity['n2']})"
                return solution_clean == re.sub(r'\s+', '', expected)
            return False
        except:
            return False
