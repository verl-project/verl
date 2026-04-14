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
f◆D=\iint_{D} f(x, y) \, dx \, dy.Example questions are as follows:

<example 0>
f(x,y)=1,D:0≤x≤2, 0≤y≤2, compute f◆D.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 0>

<example 1>
f(x,y)=xy,D:1≤x≤2, 0≤y≤1, compute f◆D.
If the answer is a fraction, write it in 'a/b' text format.Decimals are not allowed.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 1>

<example 2>
f(x,y)=x^2+y^2,D:0≤x≤1, 0≤y≤1, compute f◆D.
If the answer is a fraction, write it in 'a/b' text format.Decimals are not allowed.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 2>

<example 3>
f(x,y)=x+y,D:0≤x≤1, 0≤y≤2, compute f◆D.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 3>

<example 4>
f(x,y)=x^2+y^2,D:0≤x≤2, 0≤y≤2, compute f◆D.
If the answer is a fraction, write it in 'a/b' text format.Decimals are not allowed.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 4>

<example 5>
f(x,y)=e^x⋅sin(y),D:0≤x≤1, 0≤y≤π, compute f◆D.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 5>

<example 6>
f(x,y)=x^2y,D:0≤x≤1, 0≤y≤3, compute f◆D.
If the answer is a fraction, write it in the form 'a/b'
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 6>

<example 7>
f(x,y)=xy,D:0≤x≤2, 0≤y≤3, compute f◆D.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 7>

<example 8>
f(x,y)=e^(x+y),D:0≤x≤1, 0≤y≤2, compute f◆D.
If there is a power inside the answer, write it in the form a^b (a is the base, b is the exponent).
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 8>

<example 9>
f(x,y)=sin(x)⋅cos(y),D:0≤x≤π, 0≤y≤π/2, compute f◆D.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import sympy as sp
from sympy import integrate, exp, sin, cos
import random
import re
from bootcamp import Basebootcamp

# 显式定义符号变量
x, y = sp.symbols('x y')

class KorOperationUnicode25c6bootcamp(Basebootcamp):
    def __init__(self, max_power=2, x_bounds=(0, 2), y_bounds=(0, 2)):
        self.max_power = max_power
        self.x_bounds = x_bounds  # (min, max) for x
        self.y_bounds = y_bounds  # (min, max) for y
    
    def case_generator(self):
        # 生成多种函数类型
        func_types = [
            lambda: (x**random.randint(0, self.max_power), y**random.randint(0, self.max_power)),  # 多项式
            lambda: (exp(x), exp(y)),  # 指数函数
            lambda: (sin(x), cos(y)),  # 三角函数
            lambda: (x**random.randint(0,1), y)  # 混合类型
        ]
        
        # 随机选择函数类型
        f_x, f_y = random.choice(func_types)()
        
        # 生成有效积分区间
        a = random.randint(self.x_bounds[0], self.x_bounds[1]-1)
        b = random.randint(a+1, self.x_bounds[1])
        c = random.randint(self.y_bounds[0], self.y_bounds[1]-1)
        d = random.randint(c+1, self.y_bounds[1])
        
        # 计算积分
        integral_x = integrate(f_x, (x, a, b))
        integral_y = integrate(f_y, (y, c, d))
        total = integral_x * integral_y
        
        # 处理特殊符号
        def format_expr(expr):
            s = sp.StrPrinter().doprint(expr)
            return (s.replace('**', '^')
                    .replace('exp', 'e^')
                    .replace('sin', 'sin')
                    .replace('cos', 'cos'))
        
        return {
            'function': f"{format_expr(f_x)}·{format_expr(f_y)}",
            'x_range': (a, b),
            'y_range': (c, d),
            'correct_answer': format_expr(total),
            'format_type': self._determine_format(total)
        }

    def _determine_format(self, expr):
        if expr.is_Integer:
            return 'integer'
        if expr.has(sp.Rational) and not expr.is_Integer:
            return 'fraction'
        if expr.has(exp) or any(c in str(expr) for c in ['^', 'sin', 'cos']):
            return 'symbolic'
        return 'decimal'

    @staticmethod
    def prompt_func(case):
        format_map = {
            'integer': "答案应为整数",
            'fraction': "请使用分数格式 a/b",
            'symbolic': "请保留符号表达式（如e^2-1）",
            'decimal': "可用小数或整数"
        }
        return f"""计算二重积分 $$f◆D=\iint_{D} f(x, y) \, dx \, dy$$.
        
函数：f(x,y) = {case['function']}
区域：x ∈ [{case['x_range'][0]}, {case['x_range'][1]}], y ∈ [{case['y_range'][0]}, {case['y_range'][1]}]

{format_map[case['format_type']]}
答案格式示例：[[答案]]"""

    @staticmethod
    def extract_output(text):
        matches = re.findall(r'\[\[(.*?)\]\]', text.replace('\n', ' '))
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, case):
        try:
            # 标准化表达式
            def normalize(s):
                return (s.strip().replace('^', '**')
                        .replace(' ', '')
                        .replace('·','*')
                        .lower())
            
            user = sp.sympify(normalize(solution))
            correct = sp.sympify(normalize(case['correct_answer']))
            return user.equals(correct)
        except:
            return False
