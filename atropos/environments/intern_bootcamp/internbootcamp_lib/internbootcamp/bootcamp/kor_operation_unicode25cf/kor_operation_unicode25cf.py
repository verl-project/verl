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
a●b=\int_{a}^{b} f(x) \, dx+6.Example questions are as follows:

<example 0>
Given f(x)=2x, compute 1●3.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 0>

<example 1>
Given f(x)=sin(x), compute 0●π.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 1>

<example 2>
Given f(x)=x^2, compute 0●2.
If the answer is a fraction, write it in 'a/b' text format. Decimals are not allowed.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 2>

<example 3>
Given f(x)=1/x, compute 1●e.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 3>

<example 4>
Given f(x)=x^3, compute -1●1.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 4>

<example 5>
Given f(x)=cos(x), Compute 0●π/2.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 5>

<example 6>
Given f(x)=x-1, compute -2●2.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 6>

<example 7>
Given f(x)=x a★3=10, find a.
If there is more than one answer, please separate them with 'or',e.g.[[1or2]].
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 7>

<example 8>
Given f(x)=1/x 1★a=ln(2)+6, find a.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 8>

<example 9>
Given f(x)=x^3 a★1=6, find a.
The answer may be negative, if so write it in a format such as '-5'.
If there is more than one answer, please separate them with 'or',e.g.[[1or2]].
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
import sympy as sp
from sympy.abc import x, a, b
from bootcamp import Basebootcamp

class KorOperationUnicode25cfbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.operator_symbols = params.get('operator_symbols', ['●', '★', '◆'])
        self.compute_prob = params.get('compute_prob', 0.7)
        self.function_list = params.get('function_list', [
            'm*x', 'x**n', 'sin(x)', 'cos(x)', '1/x'
        ])
        self.default_a_range = params.get('a_range', (-5, 5))
        self.default_b_range = params.get('b_range', (-5, 5))

    def case_generator(self):
        if random.random() < self.compute_prob:
            return self._generate_compute_case()
        else:
            return self._generate_solve_case()

    def _generate_compute_case(self):
        func_info = self._generate_random_function()
        f_expr, f_str = func_info['expr'], func_info['str']
        
        if func_info['str'] == '1/x':
            a_val, b_val = self._generate_valid_interval_for_reciprocal()
        else:
            a_val, b_val = self._generate_valid_interval()
        
        integral = sp.integrate(f_expr, (x, a_val, b_val))
        result = self._safe_eval(integral + 6)
        
        return {
            'problem_type': 'compute',
            'f': f_str,
            'a': a_val,
            'b': b_val,
            'operator': random.choice(self.operator_symbols),
            'correct_answer': result
        }

    def _generate_solve_case(self):
        func_info = self._generate_random_function(solve_case=True)
        f_expr, f_str = func_info['expr'], func_info['str']
        target_var = random.choice(['a', 'b'])
        known_var = 'b' if target_var == 'a' else 'a'
        
        # Handle special case for 1/x
        if f_str == '1/x':
            sign_constraint = 'positive'
        else:
            sign_constraint = None

        if target_var == 'a':
            b_val = self._generate_value(exclude_zero=f_str == '1/x', sign=sign_constraint)
            a_sample = self._generate_value(exclude=b_val, sign=sign_constraint)
        else:
            a_val = self._generate_value(exclude_zero=f_str == '1/x', sign=sign_constraint)
            b_sample = self._generate_value(exclude=a_val, sign=sign_constraint)

        # Generate equation with explicit result value
        simple_case = func_info.get('simple', False)
        result_val = random.randint(5, 15) if simple_case else random.randint(3, 20)
        
        if target_var == 'a':
            integral = sp.integrate(f_expr, (x, sp.Symbol('a'), b_val))
        else:
            integral = sp.integrate(f_expr, (x, a_val, sp.Symbol('b')))
            
        equation = integral + 6 - result_val
        
        solutions = self._solve_equation(equation, target_var)
        if not solutions:
            return self.case_generator()
            
        return {
            'problem_type': 'solve',
            'f': f_str,
            'known_var': known_var,
            'known_value': b_val if target_var == 'a' else a_val,
            'target_var': target_var,
            'operator': random.choice(self.operator_symbols),
            'result': result_val,  # Add result field
            'correct_answers': solutions
        }

    @staticmethod
    def prompt_func(question_case):
        operator = question_case['operator']
        if question_case['problem_type'] == 'compute':
            return (
                f"Given f(x) = {question_case['f']}, compute {question_case['a']}{operator}{question_case['b']}.\n"
                f"The operation a{operator}b is defined as ∫_a^b f(x)dx + 6. "
                "Calculate the result. For fractions, use 'a/b' format. "
                "Put your answer in [[ ]]."
            )
        else:
            return (
                f"Given f(x) = {question_case['f']}, {question_case['target_var']}{operator}{question_case['known_value']} = {question_case['result']}. "
                f"Find {question_case['target_var']}.\n"
                f"The operation a{operator}b is defined as ∫_a^b f(x)dx + 6. "
                "For multiple answers, separate with 'or'. Use fractions if needed. "
                "Put your answer in [[ ]]."
            )

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[\[(.*?)\]\]', output)
        if not matches:
            return None
            
        last_match = matches[-1].strip()
        solutions = []
        for part in last_match.split('or'):
            part = part.strip()
            try:
                if '/' in part:
                    numerator, denominator = map(int, part.split('/'))
                    solutions.append(numerator / denominator)
                else:
                    solutions.append(float(part))
            except:
                continue
        return solutions if len(solutions) > 1 else solutions[0] if solutions else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        if identity['problem_type'] == 'compute':
            return abs(solution - identity['correct_answer']) < 1e-6
        else:
            correct_set = {round(c, 6) for c in identity['correct_answers']}
            user_sols = [solution] if not isinstance(solution, list) else solution
            user_set = {round(s, 6) for s in user_sols}
            return correct_set == user_set

    # Helper methods with type safety
    def _generate_random_function(self, solve_case=False):
        func_type = random.choice(self.function_list)
        params = {}
        
        if func_type == 'm*x':
            params['m'] = random.choice([-2, -1, 1, 2, 3])
            return {'expr': params['m']*x, 'str': f"{params['m']}x", 'simple': True}
        elif func_type == 'x**n':
            params['n'] = random.randint(2, 3) if solve_case else random.randint(2, 4)
            return {'expr': x**params['n'], 'str': f"x^{params['n']}"}
        elif func_type == '1/x':
            return {'expr': 1/x, 'str': "1/x"}
        elif func_type in ('sin(x)', 'cos(x)'):
            return {'expr': sp.__dict__[func_type[:3]](x), 'str': func_type}
        raise ValueError("Invalid function type")

    def _generate_valid_interval(self):
        while True:
            a_val = random.randint(*self.default_a_range)
            b_val = random.randint(*self.default_b_range)
            if a_val != b_val:
                return (a_val, b_val) if a_val < b_val else (b_val, a_val)

    def _generate_valid_interval_for_reciprocal(self):
        while True:
            # Ensure same sign and non-zero
            if random.choice([True, False]):
                a_val = random.randint(1, self.default_a_range[1])
                b_val = random.randint(1, self.default_b_range[1])
            else:
                a_val = random.randint(self.default_a_range[0], -1)
                b_val = random.randint(self.default_b_range[0], -1)
            if a_val != b_val:
                return (a_val, b_val) if a_val < b_val else (b_val, a_val)

    def _generate_value(self, exclude=None, exclude_zero=False, sign=None):
        while True:
            val = random.randint(*self.default_a_range)
            if sign == 'positive' and val <= 0:
                continue
            if sign == 'negative' and val >= 0:
                continue
            if exclude_zero and val == 0:
                continue
            if val == exclude:
                continue
            return val

    def _solve_equation(self, equation, target_var):
        symbol = sp.Symbol(target_var)
        try:
            solutions = sp.solve(equation, symbol)
            real_solutions = [sol.evalf() for sol in solutions if sol.is_real]
            return list({round(float(sol), 6) for sol in real_solutions if sol.is_real})
        except:
            return []

    @staticmethod
    def _safe_eval(expr):
        try:
            return float(expr.evalf())
        except:
            return float(expr)
