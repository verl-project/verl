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
A○B=(A+3B)×(A+B)
A and B are integers.Example questions are as follows:

<example 0>
Compute 2○3.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 0>

<example 1>
Compute 7○2.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 1>

<example 2>
Compute 2○3○4.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 2>

<example 3>
Compute 6○3○2.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 3>

<example 4>
If X○2=60, find X.
The answer should only be given as a number.
The answer may be negative; if so, write it in text format like '-5'.
If there is more than one answer, please separate them with 'or', e.g., 1 or 2.
Ensure that the final answer is wrapped in double square brackets, like this: [[1or2]].
</example 4>

<example 5>
If X○5=200, find X.
The answer should only be given as a number.
The answer may be negative; if so, write it in text format like '-5'.
If there is more than one answer, please separate them with 'or', e.g., 1 or 2.
Ensure that the final answer is wrapped in double square brackets, like this: [[1or2]].
</example 5>

<example 6>
If 8○X=187, find X.
The answer should only be given as a number.
If the answer is a fraction, write it in 'a/b' text format.Decimals are not allowed.
The answer may be negative; if so, write it in text format like '-5'.
If there is more than one answer, please separate them with 'or', e.g., 1 or 2.
Ensure that the final answer is wrapped in double square brackets, like this: [[1or2]].
</example 6>

<example 7>
If 5○(X○3)=63, find X.
The answer should only be given as a number.
If the answer contains a root sign, write it in the form \sqrt{x} (x is the number under the root sign).
The answer may be negative; if so, write it in text format like '-5'.
If there is more than one answer, please separate them with 'or', e.g., 1 or 2.
Ensure that the final answer is wrapped in double square brackets, like this: [[1or2]].
</example 7>

<example 8>
Now we make a little change to the rule. Define that  A ○ B = (xA + yB) × (A + B),x and y are parameters, Given that 7 ○ 2 = 180; 2 ○ 3 = 65, solve x and y.
Your answer should be in the format: x= an integer, y= an integer. 
Please wrap the answer in double square brackets, like this: [[x=value,y=value]].
</example 8>

<example 9>
Now we make a little change to the rule. Define that  A ○ B = (A + 3B) × (xA + yB),x and y are parameters, Given that 7 ○ 2 = 208; 2 ○ 3 = 77 solve x and y.
Your answer should be in the format: x= an integer, y= an integer. 
Please wrap the answer in double square brackets, like this: [[x=value,y=value]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from fractions import Fraction
from math import isclose
from bootcamp import Basebootcamp

class KorOperationUnicode25cbbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__(**params)
        self.x = params.get('x', 1)
        self.y = params.get('y', 3)

    def case_generator(self):
        problem_type = random.choices(
            ['compute', 'solve_x', 'solve_xy', 'nested_solve'],
            weights=[5, 3, 1, 1],
            k=1
        )[0]

        if problem_type == 'compute':
            length = random.choice([2, 3])
            numbers = [random.randint(-5, 5) for _ in range(length)]
            return {
                "type": "compute",
                "expr": numbers,
                "answer": self._compute_chain(numbers)
            }

        elif problem_type == 'solve_x':
            eq_type = random.choice(["X○B=K", "A○X=K"])
            x, y = self.x, self.y
            
            if eq_type == "X○B=K":
                B = random.choice([n for n in range(-5,6) if n != 0])
                X1 = random.randint(-5, 5)
                K = (x*X1 + y*B) * (X1 + B)
                X2 = (-B*(x + y) - X1*y) // x
                return {
                    "type": "solve_x",
                    "equation": f"X○{B} = {K}",
                    "solutions": sorted([X1, X2])
                }
            else:
                A = random.randint(1, 5)
                X1 = random.randint(-5, 5)
                K = (x*A + y*X1) * (A + X1)
                X2 = Fraction(-A*(x + y) - x*A, y)
                return {
                    "type": "solve_x",
                    "equation": f"{A}○X = {K}",
                    "solutions": sorted([X1, float(X2)])
                }

        elif problem_type == 'solve_xy':
            x = random.randint(-3, 3)
            y = random.randint(-3, 3)
            equations = []
            for _ in range(2):
                while True:
                    A, B = random.randint(1,5), random.randint(1,5)
                    if (A + B) != 0:
                        break
                K = (A*x + B*y) * (A + B)
                equations.append(f"{A}○{B} = {K}")
            return {
                "type": "solve_xy",
                "equations": equations,
                "solution": (x, y)
            }

        elif problem_type == 'nested_solve':
            # 生成有效嵌套方程：A○(X○B)=C
            A = random.randint(1, 5)
            B = random.randint(1, 5)
            X = random.randint(-5, 5)
            
            # 计算内部表达式值
            inner_val = self._compute_op(X, B, self.x, self.y)
            # 计算最终结果C
            C = self._compute_op(A, inner_val, self.x, self.y)
            
            return {
                "type": "nested_solve",
                "equation": f"{A}○(X○{B}) = {C}",
                "solution": X,
                "params": (self.x, self.y),
                "B": B,
                "A": A
            }

    def _compute_chain(self, numbers):
        res = numbers[0]
        for n in numbers[1:]:
            res = (self.x*res + self.y*n) * (res + n)
        return res

    @staticmethod
    def _compute_op(a, b, x, y):
        return (x*a + y*b) * (a + b)

    @staticmethod
    def prompt_func(case):
        if case['type'] == 'compute':
            expr = '○'.join(map(str, case['expr']))
            return (
                f"计算表达式：{expr}\n"
                "运算符○定义为：A○B = (xA + yB)(A + B)，其中x={x}，y={y}\n"
                "答案请用[[结果]]包裹，例如[[25]]"
            ).format(x=case.get('x',1), y=case.get('y',3))

        elif case['type'] == 'solve_x':
            return (
                f"解方程：{case['equation']}\n"
                "运算符○定义为：A○B = (xA + yB)(A + B)，其中x={x}，y={y}\n"
                "可能有多个解，答案格式：[[解1 or 解2]]\n"
                "支持分数（如5/3）和负数，示例：[[-3/2 or 4]]"
            ).format(x=case.get('x',1), y=case.get('y',3))

        elif case['type'] == 'solve_xy':
            return (
                "已知以下方程：\n" + 
                '\n'.join(case['equations']) + 
                "\n求参数x和y的值，答案格式：[[x=值,y=值]]"
            )

        elif case['type'] == 'nested_solve':
            return (
                f"解嵌套方程：{case['equation']}\n"
                "运算符○定义为：A○B = (xA + yB)(A + B)，其中x={x}，y={y}\n"
                "答案用[[数值]]包裹，示例：[[5]]"
            ).format(x=case['params'][0], y=case['params'][1])

    @staticmethod
    def extract_output(text):
        patterns = [
            # 参数解
            r'\[\[x\s*=\s*(-?\d+)\s*,\s*y\s*=\s*(-?\d+)\]\]',
            # 带分数/根号的多解
            r'\[\[((?:-?\d+/\d+|\d+|-\d+|\\sqrt{\d+})(?:\s+or\s+[-?\d+/\d+|\d+|\\sqrt{\d+}]+)+)\]\]',
            # 单值解
            r'\[\[(-?\d+/\d+|\d+|-\d+|\\sqrt{\d+})\]\]'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                last_match = matches[-1]
                if isinstance(last_match, tuple):
                    return f"x={last_match[0]},y={last_match[1]}"
                return last_match
        return None

    @classmethod
    def _verify_correction(cls, answer, case):
        try:
            if case['type'] == 'compute':
                return int(answer) == case['answer']
            
            elif case['type'] == 'solve_x':
                expected = set(case['solutions'])
                # 解析答案
                parts = answer.split(' or ')
                solutions = []
                for p in parts:
                    if '/' in p:
                        solutions.append(float(Fraction(p)))
                    elif 'sqrt' in p:
                        solutions.append(eval(p.replace('\\sqrt', 'math.sqrt'))) 
                    else:
                        solutions.append(float(p))
                # 允许浮点误差
                return all(any(isclose(s, e, rel_tol=1e-9) for e in expected) for s in solutions) and len(solutions) == len(expected)
            
            elif case['type'] == 'solve_xy':
                x = int(re.search(r'x=(-?\d+)', answer).group(1))
                y = int(re.search(r'y=(-?\d+)', answer).group(1))
                return (x, y) == case['solution']
            
            elif case['type'] == 'nested_solve':
                X = float(answer)
                # 验证嵌套计算
                inner = cls._compute_op(X, case['B'], *case['params'])
                final = cls._compute_op(case['A'], inner, *case['params'])
                return isclose(final, int(case['equation'].split('=')[-1]), rel_tol=1e-9)
            
            return False
        except:
            return False
