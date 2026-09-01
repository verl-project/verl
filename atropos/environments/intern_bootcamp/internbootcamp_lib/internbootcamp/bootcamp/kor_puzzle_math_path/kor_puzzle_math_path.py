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
1.Give a set of operation symbols.
2.Find the correct number from numbers 0 through 9 to make the equation equal to the given number.
3.Follow the normal order of operations.Example questions are as follows:

<example 0>
?+?*?-?=10
There may be many solutions, end by citing a feasible solution.
Provide the equation with values filled in, and enclose the entire equation in double brackets, like this: [[a+b*c-d=10]].
</example 0>

<example 1>
?-?+?+?=2
There may be many solutions, end by citing a feasible solution.
Provide the equation with values filled in, and enclose the entire equation in double brackets, like this: [[a-b+c+d=2]].
</example 1>

<example 2>
?/?+?+?=12
There may be many solutions, end by citing a feasible solution.
Provide the equation with values filled in, and enclose the entire equation in double brackets, like this: [[a/b+c+d=12]].
</example 2>

<example 3>
?+?+?*?=28
There may be many solutions, end by citing a feasible solution.
Provide the equation with values filled in, and enclose the entire equation in double brackets, like this: [[a+b+c*d=28]].
</example 3>

<example 4>
?/?+?*?+?=14
There may be many solutions, end by citing a feasible solution.
Provide the equation with values filled in, and enclose the entire equation in double brackets, like this: [[a/b+c*d+e=14]].
</example 4>

<example 5>
?-?+?+?/?=6
There may be many solutions, end by citing a feasible solution.
Provide the equation with values filled in, and enclose the entire equation in double brackets, like this: [[a-b+c+d/e=6]].
</example 5>

<example 6>
?/?+?+?+?=17
There may be many solutions, end by citing a feasible solution.
Provide the equation with values filled in, and enclose the entire equation in double brackets, like this: [[a/b+c+d+e=17]].
</example 6>

<example 7>
?*?+?+?/?=46
There may be many solutions, end by citing a feasible solution.
Provide the equation with values filled in, and enclose the entire equation in double brackets, like this: [[a*b+c+d/e=46]].
</example 7>

<example 8>
?/?+?/?-?-?=-5
There may be many solutions, end by citing a feasible solution.
Provide the equation with values filled in, and enclose the entire equation in double brackets, like this: [[a/b+c/d-e-f=-5]].
</example 8>

<example 9>
?*?*?+?-?*?=125
There may be many solutions, end by citing a feasible solution.
Provide the equation with values filled in, and enclose the entire equation in double brackets, like this: [[a*b*c+d-e*f=125]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

def evaluate_expression(numbers, operators):
    nums = numbers.copy()
    ops = operators.copy()
    
    # 处理乘除运算
    i = 0
    while i < len(ops):
        if ops[i] in ('*', '/'):
            a = nums[i]
            b = nums[i+1]
            try:
                if ops[i] == '*':
                    res = a * b
                else:
                    res = a // b if b != 0 else 0
                nums[i] = res
                del nums[i+1]
                del ops[i]
            except:
                return None
        else:
            i += 1
    
    # 处理加减运算
    try:
        result = nums[0]
        for i in range(len(ops)):
            if ops[i] == '+':
                result += nums[i+1]
            else:
                result -= nums[i+1]
        return result
    except:
        return None

class KorPuzzleMathPathbootcamp(Basebootcamp):
    def __init__(self, max_ops=4, allow_division=True, min_target=-50, max_target=100):
        self.params = {
            'max_ops': max_ops,
            'allow_division': allow_division,
            'min_target': min_target,
            'max_target': max_target,
            'max_attempts': 100
        }
    
    def case_generator(self):
        allowed_ops = ['+', '-', '*']
        if self.params['allow_division']:
            allowed_ops.append('/')
        
        for _ in range(self.params['max_attempts']):
            n_ops = random.randint(1, self.params['max_ops'])
            ops = [random.choice(allowed_ops) for _ in range(n_ops)]
            num_vars = n_ops + 1
            numbers = []
            valid = True
            
            numbers.append(random.randint(0, 9))
            for i in range(n_ops):
                op = ops[i]
                if op == '/':
                    prev_num = numbers[i]
                    if prev_num == 0:
                        next_num = random.randint(1, 9)
                    else:
                        possible_divisors = [x for x in range(1, 10) if x != 0 and prev_num % x == 0]
                        if not possible_divisors:
                            valid = False
                            break
                        next_num = random.choice(possible_divisors)
                    numbers.append(next_num)
                else:
                    numbers.append(random.randint(0, 9))
            
            if not valid:
                continue
            
            target = evaluate_expression(numbers, ops)
            if target is None:
                continue
            if not (self.params['min_target'] <= target <= self.params['max_target']):
                continue
            
            return {
                'operators': ops,
                'target': target,
                'num_vars': num_vars
            }
        
        return {
            'operators': ['+', '*'],
            'target': 10,
            'num_vars': 3
        }
    
    @staticmethod
    def prompt_func(question_case):  # 修正此处缩进
        operators = question_case['operators']
        target = question_case['target']
        equation = '?'
        for op in operators:
            equation += f'{op}?'
        equation += f'={target}'
        
        prompt = f"""你是一位数学谜题解答专家，需要解决以下等式问题。请用0到9的数字填入问号，使等式成立。遵循数学中的运算顺序规则（先乘除，后加减）。

等式： {equation}

要求：
- 每个问号必须填入一个0到9之间的整数
- 允许重复使用数字
- 严格按照正确运算顺序计算结果

请提供一个可行的解，并将完整等式用双括号括起来，例如：[[答案填入这里]]。确保将最终答案放置在双括号内。"""
        return prompt
    
    @staticmethod  # 修正此处缩进
    def extract_output(output):
        matches = re.findall(r'\[\[(.*?)\]\]', output)
        return matches[-1] if matches else None
    
    @classmethod  # 修正此处缩进
    def _verify_correction(cls, solution, identity):
        try:
            if '=' not in solution:
                return False
            left, right = solution.split('=', 1)
            target = int(right.strip())
            if target != identity['target']:
                return False
            
            tokens = re.findall(r'(\d+|\+|\-|\*|/)', left)
            if len(tokens) < 1 or len(tokens) % 2 == 0:
                return False
            
            numbers = []
            operators = []
            for i, token in enumerate(tokens):
                if i % 2 == 0:
                    if not token.isdigit():
                        return False
                    num = int(token)
                    if num < 0 or num > 9:
                        return False
                    numbers.append(num)
                else:
                    operators.append(token)
            
            if len(operators) != len(identity['operators']):
                return False
            for op_case, op_user in zip(identity['operators'], operators):
                if op_case != op_user:
                    return False
            
            calculated = evaluate_expression(numbers, operators)
            return calculated == identity['target']
        except:
            return False
