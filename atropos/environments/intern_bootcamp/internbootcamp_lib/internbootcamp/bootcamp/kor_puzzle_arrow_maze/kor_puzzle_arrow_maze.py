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
1.The maze consists of a grid with an arrow in each grid cell pointing in one of eight directions up, down, left, right, or diagonally.
2.The maze has a well-defined start and end point.
3.The player starts at the starting point, moves to the next grid cell in the direction indicated by the arrow, and then continues to move as indicated by the arrow in the new grid.
4.The player must move strictly in the direction indicated by the arrows and cannot go in the opposite direction or choose another path.
5.The game is won when the player successfully reaches the end from the starting point.Example questions are as follows:

<example 0>
→        ↓        
→        ○
The answers are required to point out the position of each inflection point in order, 0 indicates a point not on the path.
The output should be given in order from left to right, top to bottom, with each element separated by a space and different lines separated by a comma.
Ensure that your final answer is wrapped in double square brackets, like this: [[1 0 2,4 0 5,3 0 6]].
</example 0>

<example 1>
↘        ↓        
→         ○
The answers are required to point out the position of each inflection point in order, 0 indicates a point not on the path.
The output should be given in order from left to right, top to bottom, with each element separated by a space and different lines separated by a comma.
Ensure that your final answer is wrapped in double square brackets, like this: [[1 0 2,4 0 5,3 0 6]].
</example 1>

<example 2>
↓         ↓        
→         ○
The answers are required to point out the position of each inflection point in order, 0 indicates a point not on the path.
The output should be given in order from left to right, top to bottom, with each element separated by a space and different lines separated by a comma.
Ensure that your final answer is wrapped in double square brackets, like this: [[1 0 2,4 0 5,3 0 6]].
</example 2>

<example 3>
→        ↓        ↙
→        ↖        ↓
↑        ←        ○
The answers are required to point out the position of each inflection point in order, 0 indicates a point not on the path.
The output should be given in order from left to right, top to bottom, with each element separated by a space and different lines separated by a comma.
Ensure that your final answer is wrapped in double square brackets, like this: [[1 0 5,4 0 2,6 0 3]].
</example 3>

<example 4>
→        ←        ↘        ↘        ↙        ←
→        ↗        ↓        ↗        ↘        ↙
↓        →        ↙        ↘        ↙        ↑
↓        ↓        ↙        ↖        ↓        ↙
↗        ←        ↙        ←        ↖        ↓
↑        ↖        ↗        ↖        →        ○
The answers are required to point out the position of each inflection point in order, 0 indicates a point not on the path.
The output should be given in order from left to right, top to bottom, with each element separated by a space and different lines separated by a comma.
Ensure that your final answer is wrapped in double square brackets, like this: [[1 0 2,4 0 5,3 0 6]].
</example 4>

<example 5>
→	→	↓	←
↓	←	→	↙
→	↑	↓	↓
↗	←	↑	○
The answers are required to point out the position of each inflection point in order, 0 indicates a point not on the path.
The output should be given in order from left to right, top to bottom, with each element separated by a space and different lines separated by a comma.
Ensure that your final answer is wrapped in double square brackets, like this: [[1 0 2,4 0 5,3 0 6]].
</example 5>

<example 6>
↓	→	↘	↓	←
↗	↑	↘	↖	↙
↗	↗	↘	↓	↙
↘	←	↑	↙	↑
↑	↑	↖	←	○
The answers are required to point out the position of each inflection point in order, 0 indicates a point not on the path.
The output should be given in order from left to right, top to bottom, with each element separated by a space and different lines separated by a comma.
Ensure that your final answer is wrapped in double square brackets, like this: [[1 0 2,4 0 5,3 0 6]].
</example 6>

<example 7>
→	↗	↗	↗	↙
↗	↓	↖	↙	↘
→	↓	↘	←	↗
↗	↗	→	↖	↑
↙	→	↑	↘	○
The answers are required to point out the position of each inflection point in order, 0 indicates a point not on the path.
The output should be given in order from left to right, top to bottom, with each element separated by a space and different lines separated by a comma.
Ensure that your final answer is wrapped in double square brackets, like this: [[1 0 2,4 0 5,3 0 6]].
</example 7>

<example 8>
↓	→	→	↘	→	↙	←	←
↗	↗	↓	→	↘	↘	↗	↖
↗	↖	←	↙	↙	↘	←	↖
↗	↗	↖	↘	→	↓	↗	↑
↘	←	↓	→	←	→	↓	↙
↗	←	↑	↘	↘	↗	↓	↓
↘	↑	↑	↖	↓	↗	←	↖
↑	←	↗	↗	↖	←	↖	○
</example 8>

<example 9>
↓	↘	→	↙	←
↗	↓	↑	↓	↙
↗	↓	→	↑	↙
→	↑	↖	↑	←
↗	↑	→	↗	○
The answers are required to point out the position of each inflection point in order, 0 indicates a point not on the path.
The output should be given in order from left to right, top to bottom, with each element separated by a space and different lines separated by a comma.
Ensure that your final answer is wrapped in double square brackets, like this: [[1 0 2,4 0 5,3 0 6]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from itertools import combinations
from bootcamp import Basebootcamp

class KorPuzzleArrowMazebootcamp(Basebootcamp):
    def __init__(self, **params):
        self.vars = params.get('vars', ['p', 'q', 'r', 's'])
        self.max_clause_length = params.get('max_clause_length', 3)
        self.problem_types = params.get('problem_types', ['can_resolve', 'compute_dispel', 'algorithm_output'])
        random.seed(params.get('seed', None))

    def case_generator(self):
        problem_type = random.choice(self.problem_types)
        if problem_type == 'can_resolve':
            return self._generate_can_resolve_case()
        elif problem_type == 'compute_dispel':
            return self._generate_compute_dispel_case()
        elif problem_type == 'algorithm_output':
            return self._generate_algorithm_output_case()
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")

    @staticmethod
    def prompt_func(question_case):
        problem_type = question_case['problem_type']
        if problem_type == 'can_resolve':
            C1_str = ' ∨ '.join(question_case['C1'])
            C2_str = ' ∨ '.join(question_case['C2'])
            return f"Can clauses C1 = {C1_str} and C2 = {C2_str} be resolved?\nA. Yes\nB. No\nAnswer format: [[option]]."
        elif problem_type == 'compute_dispel':
            C1_str = ' ∨ '.join(question_case['C1'])
            C2_str = ' ∨ '.join(question_case['C2'])
            return f"If C1 = {C1_str} and C2 = {C2_str}, what is dispel(C1, C2)?\nProvide answer in format [[result]].\nFor multiple results use [[result1;result2]].\nFor empty clause write [[0]]."
        elif problem_type == 'algorithm_output':
            cnf_str = ' ∧ '.join([f'({" ∨ ".join(clause)})' for clause in question_case['cnf']])
            return f"Apply resolution algorithm to: {cnf_str}\nWhat is the output (Plausible/Implausible) and cycle count?\nAnswer format: [[output];[number]]."
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[\[(.*?)\]\]', output)
        return matches[-1].strip() if matches else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        problem_type = identity['problem_type']
        if problem_type == 'can_resolve':
            expected = identity['expected']
            ans = solution.upper()
            return (ans == 'A' and expected) or (ans == 'B' and not expected)
        
        elif problem_type == 'compute_dispel':
            expected = set(identity['expected'].split(' ∨ ')) if identity['expected'] != '0' else set()
            answers = [a.strip() for a in solution.split(';')]
            for ans in answers:
                ans_set = set(ans.split(' ∨ ')) if ans != '0' else set()
                if ans_set == expected:
                    return True
            return False
        
        elif problem_type == 'algorithm_output':
            try:
                output_part, steps_part = solution.split(';')
                expected_output = identity['expected_output'].lower()
                return (output_part.strip().lower() == expected_output and 
                        int(steps_part) == identity['steps'])
            except:
                return False
        return False

    # Helper methods
    def _generate_can_resolve_case(self):
        if random.random() < 0.5:
            var = random.choice(self.vars)
            C1 = [var] + self._gen_literals(exclude=[var])
            C2 = [f'¬{var}'] + self._gen_literals(exclude=[var])
            expected = True
        else:
            C1, C2 = self._gen_non_resolvable_clauses()
            expected = False
        return {'problem_type': 'can_resolve', 'C1': C1, 'C2': C2, 'expected': expected}

    def _generate_compute_dispel_case(self):
        var = random.choice(self.vars)
        C1 = [var] + self._gen_literals(exclude=[var])
        C2 = [f'¬{var}'] + self._gen_literals(exclude=[var])
        resolvent = list(set([l for l in C1 if l != var] + [l for l in C2 if l != f'¬{var}']))
        expected = ' ∨ '.join(resolvent) if resolvent else '0'
        return {'problem_type': 'compute_dispel', 'C1': C1, 'C2': C2, 'expected': expected}

    def _generate_algorithm_output_case(self):
        cnf = [['p'], ['¬p']] if random.random() < 0.5 else [self._gen_clause()]
        output, steps = self._run_resolution(cnf)
        return {'problem_type': 'algorithm_output', 'cnf': cnf, 
                'expected_output': output, 'steps': steps}

    def _gen_literals(self, exclude=[]):
        return list(set([self._gen_literal(exclude) for _ in range(random.randint(0, self.max_clause_length-1))]))

    def _gen_literal(self, exclude):
        available = [v for v in self.vars if v not in exclude and f'¬{v}' not in exclude]
        var = random.choice(available) if available else random.choice(self.vars)
        return f'¬{var}' if random.random() < 0.5 else var

    def _gen_non_resolvable_clauses(self):
        while True:
            C1 = self._gen_clause()
            C2 = self._gen_clause()
            if not self._can_resolve(C1, C2):
                return C1, C2

    def _gen_clause(self):
        return list(set([self._gen_literal([]) for _ in range(random.randint(1, self.max_clause_length))]))

    def _can_resolve(self, C1, C2):
        return any(('¬'+l in C2 or l[1:] in C2) for l in C1)

    def _run_resolution(self, cnf):
        S0, S1, steps = set(), {frozenset(c) for c in cnf}, 0
        while True:
            S2 = set()
            # Resolve S0 and S1
            for C0 in S0:
                for C1 in S1:
                    if resolvents := self._resolve(C0, C1):
                        if any(not r for r in resolvents):
                            return 'Implausible', steps + 1
                        S2.update(r for r in resolvents if r not in S0 and r not in S1)
            # Resolve S1 with itself
            for C1, C2 in combinations(S1, 2):
                if resolvents := self._resolve(C1, C2):
                    if any(not r for r in resolvents):
                        return 'Implausible', steps + 1
                    S2.update(r for r in resolvents if r not in S0 and r not in S1)
            if not S2:
                return 'Plausible', steps + 1
            S0.update(S1)
            S1 = S2
            steps += 1

    def _resolve(self, C1, C2):
        resolved = []
        C1_set, C2_set = set(C1), set(C2)
        for l in C1_set:
            comp = f'¬{l}' if not l.startswith('¬') else l[1:]
            if comp in C2_set:
                new_clause = (C1_set - {l}) | (C2_set - {comp})
                resolved.append(frozenset(new_clause))
        return resolved
