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
f□g=f'(x) \quad +g'(x) \quad.Example questions are as follows:

<example 0>
f(x)=x^2, g(x)=sin(x) , compute f□g.
Please provide your answer in LaTeX format. 
Wrap the final answer in double square brackets, like this: [[your answer]].
</example 0>

<example 1>
f(x)=e^x, g(x)=ln(x),find the value of f□g when x=1.
Please provide your answer in LaTeX format. 
Wrap the final answer in double square brackets, like this: [[your answer]].
</example 1>

<example 2>
f(x)=cos(x), g(x)=x^3 , compute f□g.
Please provide your answer in LaTeX format. 
Wrap the final answer in double square brackets, like this: [[your answer]].
</example 2>

<example 3>
f(x)=ln(x), g(x)=e^x , compute f□g.
Please provide your answer in LaTeX format. 
Wrap the final answer in double square brackets, like this: [[your answer]].
</example 3>

<example 4>
f(x)=\sqrt{x} ,g(x)=cos(x) , compute f□g.
Please provide your answer in LaTeX format. 
Wrap the final answer in double square brackets, like this: [[your answer]].
</example 4>

<example 5>
f(x)=sin(x), g(x)=ln(x) , compute f□g.
Please provide your answer in LaTeX format. 
Wrap the final answer in double square brackets, like this: [[your answer]].
</example 5>

<example 6>
f(x)=e^x,g(x)=sin(x),find the value of f□g when x=0.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 6>

<example 7>
f(x)=ln(x), g(x)=x^2,find the value of f□g when x=1.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 7>

<example 8>
f(x)=tan(x), g(x)=x,find the value of f□g when x=π/4.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 8>

<example 9>
f(x)=x^3,g(x)=e^x,find the value of f□g when x=1.
Please provide your answer in LaTeX format. 
Wrap the final answer in double square brackets, like this: [[your answer]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import sympy as sp
import random
import math
from bootcamp import Basebootcamp

x = sp.symbols('x')

base_functions = [
    'x**2',
    'x**3',
    'sin(x)',
    'cos(x)',
    'exp(x)',
    'ln(x)',
    'sqrt(x)',
    'tan(x)',
    'x',
]

class KorOperationUnicode25a1bootcamp(Basebootcamp):
    def __init__(self, **params):
        self.params = params.get('params', {})
        self.prob_evaluate = self.params.get('prob_evaluate', 0.5)
        self.x_candidates = self.params.get('x_candidates', [0, 1, 2, math.pi/4, math.pi/2, math.pi])
    
    def case_generator(self):
        f_str = random.choice(base_functions)
        g_str = random.choice(base_functions)
        
        f_expr = sp.parse_expr(f_str)
        g_expr = sp.parse_expr(g_str)
        
        df = f_expr.diff(x)
        dg = g_expr.diff(x)
        answer_expr = df + dg
        
        answer_latex = sp.latex(answer_expr, mul_symbol=None)
        
        problem_type = 'expression'
        correct_answer = answer_latex
        x_value = None
        
        if random.random() < self.prob_evaluate:
            max_attempts = 5
            for _ in range(max_attempts):
                x_val = random.choice(self.x_candidates)
                try:
                    value = answer_expr.subs(x, x_val).evalf()
                    if value.is_real and not value.has(sp.nan, sp.zoo, sp.oo):
                        correct_answer = round(float(value), 4)
                        problem_type = 'evaluate'
                        x_value = x_val
                        break
                except:
                    continue
        
        case = {
            'f': sp.latex(f_expr, mul_symbol=None),
            'g': sp.latex(g_expr, mul_symbol=None),
            'problem_type': problem_type,
            'correct_answer': correct_answer,
            'x_value': x_value
        }
        return case
    
    @staticmethod
    def prompt_func(question_case) -> str:
        f = question_case['f']
        g = question_case['g']
        x_val = question_case['x_value']
        problem_type = question_case['problem_type']
        
        prompt = [
            "You are a calculus expert. Solve the problem using the following rules:",
            "The operation f□g is defined as f'(x) + g'(x), where f' and g' are derivatives with respect to x.",
            "",
            f"Given:",
            f"f(x) = {f}",
            f"g(x) = {g}",
            ""
        ]
        
        if problem_type == 'evaluate':
            prompt.append(f"Find the numerical value of f□g at x = {x_val}.")
            prompt.append("Provide a single number within [[ ]].")
        else:
            prompt.append("Compute the expression for f□g.")
            prompt.append("Present your answer in LaTeX within [[ ]].")
        
        prompt.append("Example: [[2x + \\cos(x)]] or [[3.14]]")
        return '\n'.join(prompt)
    
    @staticmethod
    def extract_output(output):
        import re
        matches = re.findall(r'\[\[(.*?)\]\]', output)
        return matches[-1].strip() if matches else None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        problem_type = identity['problem_type']
        correct = identity['correct_answer']
        
        if problem_type == 'expression':
            return solution == correct
        else:
            try:
                num = round(float(solution), 4)
                return abs(num - correct) < 1e-4
            except:
                return False
