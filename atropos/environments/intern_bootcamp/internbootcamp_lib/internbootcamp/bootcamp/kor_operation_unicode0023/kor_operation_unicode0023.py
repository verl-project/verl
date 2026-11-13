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
a#b is the average of all even numbers between a and b (including a and b).Example questions are as follows:

<example 0>
Compute 3#7.
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 0>

<example 1>
Compute 2#5.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 1>

<example 2>
Compute 4#6.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 2>

<example 3>
Compute 1#5.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 3>

<example 4>
Compute 3#9.
Please ensure the answer is a single number and wrap it in double square brackets, like this: [[your answer]].
</example 4>

<example 5>
If X#6=5, find X.
The answer should only be given as a number.
If there is more than one answer, please separate them with 'or',e.g.[[1or2]].
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 5>

<example 6>
If 3#X=4, find X.
The answer should only be given as a number.
If there is more than one answer, please separate them with 'or',e.g.[[1or2]].
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 6>

<example 7>
If X#5=3, find X.
The answer should only be given as a number.
If there is more than one answer, please separate them with 'or',e.g.[[1or2]].
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 7>

<example 8>
If X#7=6, find X.
The answer should only be given as a number.
If there is more than one answer, please separate them with 'or',e.g.[[1or2]].
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 8>

<example 9>
If 3#X=6, find X.
The answer should only be given as a number.
If there is more than one answer, please separate them with 'or',e.g.[[1or2]].
Please wrap the answer in double square brackets, like this: [[your answer]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from numbers import Number
from typing import Union
from bootcamp import Basebootcamp

class KorOperationUnicode0023bootcamp(Basebootcamp):
    def __init__(self, compute_range=(0, 20), solve_range=(0, 30)):
        self.c_min, self.c_max = compute_range
        self.s_min, self.s_max = solve_range

    def case_generator(self) -> dict:
        """生成两种问题类型：计算类（60%）和方程求解类（40%）"""
        return random.choice([
            self._generate_compute_case,
            self._generate_solve_case
        ])()

    def _generate_compute_case(self) -> dict:
        """生成数值计算问题"""
        while True:
            a, b = sorted([random.randint(self.c_min, self.c_max) for _ in range(2)])
            if (a + b) == 0:
                continue  # 避免除零错误
            
            even_numbers = [x for x in range(a, b+1) if x%2 == 0]
            if not even_numbers:
                continue
                
            avg = sum(even_numbers)/len(even_numbers)
            return {
                'type': 'compute',
                'params': {'a': a, 'b': b},
                'answer': avg
            }

    def _generate_solve_case(self) -> dict:
        """生成方程求解问题"""
        problem_type = random.choice(['solve_left', 'solve_right'])
        return {
            'solve_left': self._generate_left_solve_case,
            'solve_right': self._generate_right_solve_case
        }[problem_type]()

    def _generate_left_solve_case(self) -> dict:
        """生成X#b = target类型问题"""
        while True:
            # 生成合法区间
            b = random.randint(self.s_min, self.s_max)
            valid_x = []
            
            # 遍历所有可能的X值
            for x in range(self.s_min, b+1):
                # 计算x到b闭区间的偶数平均值
                evens = [n for n in range(min(x,b), max(x,b)+1) if n%2 ==0]
                if not evens:
                    continue
                avg = sum(evens)/len(evens)
                valid_x.append( (x, avg) )
            
            if not valid_x:
                continue
            
            # 选择有解的target值
            target_entry = random.choice(valid_x)
            target = target_entry[1]
            
            # 验证所有可能解
            solutions = []
            for x, avg in valid_x:
                if abs(avg - target) < 1e-9:
                    solutions.append(x)
            
            if solutions:
                return {
                    'type': 'solve_left',
                    'params': {'b': b, 'target': target},
                    'answer': solutions
                }

    def _generate_right_solve_case(self) -> dict:
        """生成a#X = target类型问题"""
        while True:
            a = random.randint(self.s_min, self.s_max)
            valid_x = []
            
            for x in range(a, self.s_max+1):
                evens = [n for n in range(min(a,x), max(a,x)+1) if n%2 ==0]
                if not evens:
                    continue
                avg = sum(evens)/len(evens)
                valid_x.append( (x, avg) )
            
            if not valid_x:
                continue
                
            target_entry = random.choice(valid_x)
            target = target_entry[1]
            
            solutions = []
            for x, avg in valid_x:
                if abs(avg - target) < 1e-9:
                    solutions.append(x)
            
            if solutions:
                return {
                    'type': 'solve_right',
                    'params': {'a': a, 'target': target},
                    'answer': solutions
                }

    @staticmethod
    def prompt_func(case: dict) -> str:
        """生成自然语言问题描述"""
        definition = "Define that a#b is the average of all even numbers between a and b (including a and b).\n\n"
        if case['type'] == 'compute':
            a = case['params']['a']
            b = case['params']['b']
            return definition + f"Compute {a}#{b}. All even numbers between (and including) {a} and {b} are considered.\nAnswer must be in [[answer]] format."
        
        elif case['type'].startswith('solve'):
            params = case['params']
            target = case['answer']
            
            # 格式化target显示
            target_value = params['target']
            if isinstance(target_value, float) and target_value.is_integer():
                target_value = int(target_value)
            
            if case['type'] == 'solve_left':
                return definition + f"If X#{params['b']} = {target_value}, find X. "\
                       f"Multiple answers should be separated by 'or'. "\
                       f"Put your answer in [[X]] or [[X1orX2]] format."
            else:
                return definition + f"If {params['a']}#X = {target_value}, find X. "\
                       f"Multiple answers should be separated by 'or'. "\
                       f"Put your answer in [[X]] or [[X1orX2]] format."

    @staticmethod
    def extract_output(text: str) -> Union[str, None]:
        """从回答文本中提取最后一个[[...]]内容"""
        matches = re.findall(r'\[\[(.*?)\]\]', text)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution: str, case: dict) -> bool:
        """验证答案正确性"""
        try:
            if case['type'] == 'compute':
                user_ans = float(solution)
                return abs(user_ans - case['answer']) < 1e-9
            
            else:  # 方程求解类型
                if 'or' in solution:
                    user_answers = set(map(int, solution.split('or')))
                else:
                    user_answers = {int(solution)}
                
                # 处理边界情况：允许answer字段存储为列表或数值
                correct_answers = set(case['answer'] if isinstance(case['answer'], list) else [case['answer']])
                return user_answers == correct_answers
                
        except (ValueError, TypeError):
            return False
