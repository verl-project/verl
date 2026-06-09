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
f△g=(f(g(x)))′.Example questions are as follows:

<example 0>
f(x)=x^2*sin(x), g(x)=x, compute f△g.
Please provide your answer in LaTeX format. 
Wrap the final answer in double square brackets, like this: [[your answer]].
</example 0>

<example 1>
f(x)=e^x, g(x)=ln(x) , compute f△g.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 1>

<example 2>
f(x)=cos(x), g(x)=x^3 , compute f△g.
Please provide your answer in LaTeX format. 
Wrap the final answer in double square brackets, like this: [[your answer]].
</example 2>

<example 3>
f(x)=ln(x), g(x)=e^x , compute f△g.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 3>

<example 4>
f(x)=\sqrt{x} ,g(x)=cos(x) , compute f△g.
Please provide your answer in LaTeX format. 
Wrap the final answer in double square brackets, like this: [[your answer]].
</example 4>

<example 5>
f(x)=sin(x), g(x)=ln(x) , compute f△g.
Please provide your answer in LaTeX format. 
Wrap the final answer in double square brackets, like this: [[your answer]].
</example 5>

<example 6>
f(x)=e^x,g(x)=sin(x) , compute f△g.
Please provide your answer in LaTeX format. 
Wrap the final answer in double square brackets, like this: [[your answer]].
</example 6>

<example 7>
f(x)=ln(x), g(x)=x^2 , compute f△g.
If the answer is a fraction, write it in 'a/b' text format.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 7>

<example 8>
f(x)=x, g(x)=cosx,find the value of f△g when x=π.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 8>

<example 9>
f(x)=x^3,g(x)=e^x,find the value of f△g when x=0.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
import sympy
from sympy import symbols, diff, exp, log, sin, cos, sqrt, pi, simplify, Rational

x = symbols('x')

class KorOperationUnicode25b3bootcamp(Basebootcamp):
    def __init__(self, max_degree=3, allow_x_value=True):
        self.max_degree = max_degree
        self.allow_x_value = allow_x_value

    def generate_function(self):
        func_types = ['poly', 'sin', 'cos', 'exp', 'ln', 'sqrt']
        func_type = random.choice(func_types)
        
        if func_type == 'poly':
            degree = random.randint(1, self.max_degree)
            expr = x**degree
        elif func_type == 'sin':
            expr = sin(x)
        elif func_type == 'cos':
            expr = cos(x)
        elif func_type == 'exp':
            expr = exp(x)
        elif func_type == 'ln':
            expr = log(x)
        elif func_type == 'sqrt':
            expr = sqrt(x)
        
        if random.random() < 0.5:
            expr *= x**random.randint(1, 2)
        return expr

    def case_generator(self):
        f_expr = self.generate_function()
        g_expr = self.generate_function()

        h = f_expr.subs(x, g_expr)
        h_prime = diff(h, x)

        x_value, correct_answer, answer_format = None, None, 'latex'

        if self.allow_x_value and random.random() < 0.2:
            for candidate in [1, 2, pi/2, pi]:
                try:
                    g_val = g_expr.subs(x, candidate)
                    if not (g_val.is_real and g_val.is_finite):
                        continue
                    h_prime_val = h_prime.subs(x, candidate)
                    if not (h_prime_val.is_real and h_prime_val.is_finite):
                        continue

                    simplified = simplify(h_prime_val)
                    if simplified.is_Integer:
                        correct_answer = f"{int(simplified)}"
                        answer_format = 'integer'
                    elif isinstance(simplified, Rational):
                        correct_answer = f"{simplified.p}/{simplified.q}"
                        answer_format = 'fraction'
                    else:
                        num_val = float(simplified.evalf())
                        correct_answer = f"{num_val:.2f}"
                        answer_format = 'float'
                    x_value = candidate
                    break
                except:
                    continue

        if correct_answer is None:
            correct_answer = sympy.latex(simplify(h_prime))
            answer_format = 'latex'

        case = {
            'f': sympy.latex(f_expr),
            'g': sympy.latex(g_expr),
            'x_value': sympy.latex(x_value) if x_value is not None else None,
            'correct_answer': correct_answer,
            'answer_format': answer_format
        }
        return case

    @staticmethod
    def prompt_func(question_case):
        f = question_case['f']
        g = question_case['g']
        x_val = question_case['x_value']
        fmt = question_case['answer_format']

        problem = f"f(x) = {f}, g(x) = {g}. Compute f△g."
        if x_val is not None:
            problem = f"f(x) = {f}, g(x) = {g}. Find f△g at x = {x_val}."

        format_rules = {
            'latex': "Provide your answer in LaTeX enclosed in [[ ]].",
            'fraction': "Write fractions as 'a/b' within [[ ]].",
            'integer': "Provide an integer within [[ ]].",
            'float': "Round to two decimal places within [[ ]]."
        }.get(fmt, "Place your answer within [[ ]].")

        return f"""Define that: f△g=(f(g(x)))′

Solve the calculus puzzle:

**Problem**: {problem}

**Rules**:
- Compute the derivative of f(g(x)).
{f"Substitute x = {x_val}." if x_val else ""}
- {format_rules}

**Answer**: [[your answer here]]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[\[(.*?)\]\]', output, re.DOTALL)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        expected = identity['correct_answer'].strip()
        solution = solution.strip()

        if identity['answer_format'] == 'latex':
            solution = re.sub(r'\s+', '', solution)
            expected = re.sub(r'\s+', '', expected)
            solution = solution.replace(r'\cdot', '').replace('*', '')
            expected = expected.replace(r'\cdot', '').replace('*', '')
        else:
            solution = solution.replace(' ', '')
            expected = expected.replace(' ', '')

        return solution == expected
