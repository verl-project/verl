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
f■g=\frac{\partial f}{\partial x}+\frac{\partial g}{\partial x}.Example questions are as follows:

<example 0>
f(x,y)=x^2+y,g(x,y)=sinx+cosy, compute f■g.
Please provide your answer in LaTeX format. 
Wrap the final answer in double square brackets, like this: [[your answer]].
</example 0>

<example 1>
f(x,y)=x^2+y,g(x,y)=3x+y^2, compute f■g.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 1>

<example 2>
f(x,y)=sinx+y^2,g(x,y)=cosx-y, compute f■g.
Please provide your answer in LaTeX format. 
Wrap the final answer in double square brackets, like this: [[your answer]].
</example 2>

<example 3>
f(x,y)=e^x,g(x,y)=x^3+y^3, compute f■g.
Please provide your answer in LaTeX format. 
Wrap the final answer in double square brackets, like this: [[your answer]].
</example 3>

<example 4>
f(x,y)=x^2+xy,g(x,y)=2x+y, compute f■g.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 4>

<example 5>
f(x,y)=sin^2(x)+y,g(x,y)=cos^2(x)+y, compute f■g.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 5>

<example 6>
f(x,y)=x^2+y^2,g(x,y)=e^x+sin(y), compute f■g.
If there is a power inside the answer, write it in the form a^b (a is the base, b is the exponent).
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 6>

<example 7>
f(x,y)=x^3+y^3,g(x,y)=x^2+y^2, compute f■g.
If there is a power inside the answer, write it in the form a^b (a is the base, b is the exponent).
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 7>

<example 8>
f(x,y)=x⋅sin(y),g(x,y)=e^x+y^2,compute f■g.
Please provide your answer in LaTeX format. 
Wrap the final answer in double square brackets, like this: [[your answer]].
</example 8>

<example 9>
f(x,y)=x/y,g(x,y)=x^3+y^3, compute f■g.
Please provide your answer in LaTeX format. 
Wrap the final answer in double square brackets, like this: [[your answer]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
import sympy
from bootcamp import Basebootcamp

x, y = sympy.symbols('x y')

class KorOperationUnicode25a0bootcamp(Basebootcamp):
    def __init__(self, max_terms=3, max_degree=3, **kwargs):
        self.max_terms = max_terms
        self.max_degree = max_degree
        super().__init__(**kwargs)
    
    def _generate_term(self):
        term_types = [
            # 多项式项
            lambda: x**random.randint(1, self.max_degree),
            lambda: y**random.randint(1, self.max_degree),
            # 三角函数
            lambda: sympy.sin(random.choice([x, y])),
            lambda: sympy.cos(random.choice([x, y])),
            # 指数函数
            lambda: sympy.exp(x),
            # 分式项
            lambda: sympy.Mul(
                sympy.Poly(random.randint(1, 3)*x**random.randint(0,2), x), 
                sympy.Pow(y, -random.randint(1,2)), 
                evaluate=False
            ),
            # 常数项
            lambda: sympy.Integer(random.randint(1, 5))
        ]
        return random.choice(term_types)()
    
    def _generate_expression(self):
        num_terms = random.randint(1, self.max_terms)
        expr = sympy.Integer(0)
        for _ in range(num_terms):
            term = self._generate_term()
            # 确保不生成全零表达式
            if expr == 0:
                expr = term
            else:
                expr += term
        return expr
    
    def case_generator(self):
        while True:
            try:
                f_expr = self._generate_expression()
                g_expr = self._generate_expression()
                
                df_dx = sympy.diff(f_expr, x)
                dg_dx = sympy.diff(g_expr, x)
                ans_expr = sympy.simplify(df_dx + dg_dx)
                
                # 过滤无效表达式
                if ans_expr.is_number:
                    continue
                    
                return {
                    'f_latex': sympy.latex(f_expr),
                    'g_latex': sympy.latex(g_expr),
                    '_f_sympy': str(f_expr),
                    '_g_sympy': str(g_expr),
                    '_answer_sympy': str(ans_expr)
                }
            except:
                continue
    
    @staticmethod
    def prompt_func(question_case) -> str:
        return f"""请计算以下函数的偏导数之和：
        
给定：
$$f(x, y) = {question_case['f_latex']}$$
$$g(x, y) = {question_case['g_latex']}$$

其中运算符■定义为：
$$f■g = \\frac{{\\partial f}}{{\\partial x}} + \\frac{{\\partial g}}{{\\partial x}}$$

要求：
1. 结果必须使用LaTeX公式表示
2. 指数使用^符号（如x²写作x^2）
3. 分式使用\\frac{{分子}}{{分母}}格式
4. 将最终答案包裹在双方括号中，例如：[[2x + \\cos x]]

请直接给出最终答案："""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[\[(.*?)\]\]', output, re.DOTALL)
        if not matches:
            return None
        solution = matches[-1].strip()
        # 清理多余空格和换行
        return re.sub(r'\s+', '', solution)
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            # 转换用户答案
            user_clean = solution.replace('\\', '').replace('{','').replace('}','')
            user_expr = sympy.parse_expr(user_clean, transformations='all')
            
            # 转换标准答案
            ans_expr = sympy.parse_expr(identity['_answer_sympy'])
            
            # 符号等价验证
            diff = sympy.simplify(user_expr - ans_expr)
            return diff.equals(0)
        except Exception as e:
            return False
