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
1.The game gives a formula of letters, each representing a unique number (0-9). 
2.Different letters cannot represent the same number.
3.The first letter of any multi-digit number cannot represent 0.Example questions are as follows:

<example 0>
SEND+MORE=MONEY. 
Please provide your answer in the form: letter=number, and make sure to enclose your answer in double square brackets, like this: [[A=1,B=2,...]].
</example 0>

<example 1>
TO+GO=OUT.
Please provide your answer in the form: letter=number, and make sure to enclose your answer in double square brackets, like this: [[A=1,B=2,...]].
</example 1>

<example 2>
ONE+ONE+TWO=FOUR.
Please provide your answer in the form: letter=number, and make sure to enclose your answer in double square brackets, like this: [[A=1,B=2,...]].
</example 2>

<example 3>
TT+TT=LTC
Please provide your answer in the form: letter=number, and make sure to enclose your answer in double square brackets, like this: [[A=1,B=2,...]].
</example 3>

<example 4>
FORTY+TEN+TEN=SIXTY.
Please provide your answer in the form: letter=number, and make sure to enclose your answer in double square brackets, like this: [[A=1,B=2,...]].
</example 4>

<example 5>
FIFTY+STATES=AMERICA.
Please provide your answer in the form: letter=number, and make sure to enclose your answer in double square brackets, like this: [[A=1,B=2,...]].
</example 5>

<example 6>
I+BB=ILL.
Please provide your answer in the form: letter=number, and make sure to enclose your answer in double square brackets, like this: [[A=1,B=2,...]].
</example 6>

<example 7>
EAT+THAT=APPLE.
Please provide your answer in the form: letter=number, and make sure to enclose your answer in double square brackets, like this: [[A=1,B=2,...]].
</example 7>

<example 8>
HERE+SHE=COMES.
Please provide your answer in the form: letter=number, and make sure to enclose your answer in double square brackets, like this: [[A=1,B=2,...]].
</example 8>

<example 9>
NUMBER+NUMBER=PUZZLE.
Please provide your answer in the form: letter=number, and make sure to enclose your answer in double square brackets, like this: [[A=1,B=2,...]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from itertools import permutations
from bootcamp import Basebootcamp

class KorPuzzleCryptoMathbootcamp(Basebootcamp):
    def __init__(self, **params):
        """
        参数:
            min_terms: 加数最小数量 (默认2)
            max_terms: 加数最大数量 (默认3)
            term_length: 项的数字位数 (默认3)
            result_length: 结果的数字位数 (默认4)
        """
        self.min_terms = params.get('min_terms', 2)
        self.max_terms = params.get('max_terms', 3)
        self.term_length = params.get('term_length', 3)
        self.result_length = params.get('result_length', 4)
    
    def _generate_valid_equation(self):
        """动态生成有效等式"""
        # 生成随机加法结构：A + B + ... = SUM
        num_terms = random.randint(self.min_terms, self.max_terms)
        
        while True:
            # 生成随机数字组合
            digits = random.sample(range(0, 10), self.term_length)
            terms = [random.randint(10**(self.term_length-1), 10**self.term_length-1) 
                    for _ in range(num_terms)]
            total = sum(terms)
            
            if len(str(total)) == self.result_length:
                # 转换为字母模式
                letters = set()
                equation_parts = []
                for term in terms + [total]:
                    term_str = str(term)
                    if len(term_str) < self.term_length:
                        term_str = term_str.zfill(self.term_length)
                    equation_parts.append(term_str)
                    letters.update(term_str)
                
                # 确保结果首位非零
                if equation_parts[-1][0] == '0':
                    continue
                
                # 转换为字母方程
                char_map = {}
                unique_chars = list(letters)
                random.shuffle(unique_chars)
                for c in unique_chars:
                    char_map[c] = chr(65 + len(char_map))  # 映射到不同字母
                
                equation = []
                for part in equation_parts[:-1]:
                    equation.append(''.join([char_map[c] for c in part]))
                result = ''.join([char_map[c] for c in equation_parts[-1]])
                
                return f"{'+'.join(equation)}={result}"

    def case_generator(self):
        """生成动态有效的字母算术方程"""
        equation = self._generate_valid_equation()
        return {"equation": equation}

    @staticmethod
    def prompt_func(question_case) -> str:
        equation = question_case["equation"]
        prompt = f"""Solve this cryptarithmetic puzzle where each letter represents a unique digit (0-9). 
Different letters must have different values. Leading letters cannot be zero. 

Equation: {equation}

Provide your answer as comma-separated letter=number pairs enclosed in double square brackets. 
Example: [[A=5,B=3,...,Z=9]]"""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[\[(.*?)\]\]', output)
        if not matches:
            return None
        
        solution = {}
        for pair in matches[-1].split(','):
            pair = pair.strip()
            if '=' not in pair:
                continue
            letter, value = pair.split('=', 1)
            letter = letter.strip().upper()
            try:
                num = int(value.strip())
                if 0 <= num <= 9:
                    solution[letter] = num
            except ValueError:
                continue
        return solution if solution else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        equation = identity["equation"]
        try:
            left, right = equation.split('=')
            terms = left.split('+')
            sum_terms = 0
            for term in terms:
                if len(term) > 1 and solution[term[0]] == 0:
                    return False
                sum_terms += int(''.join(str(solution[c]) for c in term))
            
            result = right.strip()
            if len(result) > 1 and solution[result[0]] == 0:
                return False
            result_num = int(''.join(str(solution[c]) for c in result))
            
            # 验证唯一性
            values = list(solution.values())
            if len(values) != len(set(values)):
                return False
            
            return sum_terms == result_num
        except (KeyError, ValueError):
            return False
