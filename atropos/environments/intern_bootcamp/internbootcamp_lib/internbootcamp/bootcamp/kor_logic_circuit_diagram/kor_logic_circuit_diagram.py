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
In a simple circuit diagram, logical operators \"negation\", \"conjunction\", and \"disjunction\" function similarly.
When there is one input it is recorded as \"I\", when there is more than 1 all inputs are represented in order as \"I1, I2, ......\".
If powered, represented as \"+\"; if not powered, represented as \"-\".
The output of the circuit diagram is represented as \"O\". Hence, a circuit diagram can be depicted and described like a truth table.Example questions are as follows:

<example 0>
Please provide a simple circuit diagram for a NOT gate, 
formatted as [[input, output]; [output when circuit is powered, output when circuit is not powered]; ...].
</example 0>

<example 1>
Please provide a simple circuit diagram for a AND gate, 
formatted as [[input, output]; [output when circuit is powered, output when circuit is not powered]; ...].
</example 1>

<example 2>
Please provide a simple circuit diagram for a OR gate, 
formatted as [[input, output]; [output when circuit is powered, output when circuit is not powered]; ...].
</example 2>

<example 3>
What is the simple circuit diagram corresponding to the logical expression ¬(p∧q)?
 Please provide the answer in the format [[input, output]; [output when circuit is powered, output when circuit is not powered]; ...].
</example 3>

<example 4>
What is the simple circuit diagram corresponding to the logical expression (p∧q)∨(p∧r)?
Please provide the answer in the format [[input, output]; [output when circuit is powered, output when circuit is not powered]; ...].
</example 4>

<example 5>
Assuming an \"OR logic gate\" 
has one input I1 as \"-\" and the other input I2 as \"+\",
what is the output? 
Please provide the answer in the format [[output]].
</example 5>

<example 6>
Assuming an \"AND logic gate\" 
has one input I1 as \"+\" and the other input I2 as \"-\",
what is the output?
Please provide the answer in the format [[output]].
</example 6>

<example 7>
Assuming a simple circuit diagram 
corresponding to the logical expression (p∧q)∨(¬p∧r), 
with inputs I1 as \"+\", I2 as \"-\", and I3 as \"+\", 
what is the output? 
Please provide the answer in the format [[output]].
</example 7>

<example 8>
Assuming a simple circuit diagram 
corresponding to the logical expression (p∧q)∨(¬p∧(q∨r)), 
with output O as \"+\", 
what are the corresponding inputs? 
Please provide multiple inputs that satisfy this condition in the format [[];[];…].
</example 8>

<example 9>
Assuming a simple circuit diagram 
corresponding to the logical expression (p∧q∧¬r)∨(¬p∧q)∨(p∨r), 
with output O as \"-\", 
what are the corresponding inputs? 
Please provide multiple inputs that satisfy this condition in the format [[];[];…].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
from itertools import product
from bootcamp import Basebootcamp

class KorLogicCircuitDiagrambootcamp(Basebootcamp):
    def __init__(self, gate_types=('NOT', 'AND', 'OR'), max_variables=4):
        self.gate_types = gate_types
        self.max_variables = max_variables
    
    def case_generator(self):
        import random
        problem_type = random.choice([
            'gate_truth_table',
            'compute_gate_output',
            'compute_expression_output',
            'find_inputs'
        ])
        
        if problem_type == 'gate_truth_table':
            return {
                'type': 'gate_truth_table',
                'gate': random.choice(self.gate_types)
            }
        
        elif problem_type == 'compute_gate_output':
            gate = random.choice(self.gate_types)
            num_inputs = 1 if gate == 'NOT' else 2
            return {
                'type': 'compute_gate_output',
                'gate': gate,
                'inputs': [random.choice(['+', '-']) for _ in range(num_inputs)],
                'powered': random.choice([True, False])
            }
        
        elif problem_type == 'compute_expression_output':
            expressions = [
                '¬p', 'p ∧ q', 'p ∨ q', '¬(p ∧ q)', '(p ∧ q) ∨ r',
                'p ∧ q ∧ ¬r', '(p ∨ q) ∧ ¬r', '¬p ∨ (q ∧ r)',
                '(p∧q)∨(¬p∧(q∨r))', '(p∧q∧¬r)∨(¬p∧q)∨(p∨r)'
            ]
            expr = random.choice(expressions)
            variables = self.extract_variables(expr)
            inputs = {var: random.choice(['+', '-']) for var in variables}
            return {
                'type': 'compute_expression_output',
                'expression': expr,
                'inputs': inputs,
                'powered': random.choice([True, False])
            }
        
        elif problem_type == 'find_inputs':
            expressions = [
                'p ∧ q', 'p ∨ q', '¬p', 'p ∧ q ∧ ¬r',
                '(p ∧ q) ∨ ¬r', '¬(p ∨ q) ∧ r',
                '(p∧q)∨(¬p∧(q∨r))', '(p∧q∧¬r)∨(¬p∧q)'
            ]
            expr = random.choice(expressions)
            variables = self.extract_variables(expr)
            return {
                'type': 'find_inputs',
                'expression': expr,
                'output': random.choice(['+', '-']),
                'powered': random.choice([True, False]),  # 修复点：允许非供电状态
                'variables': variables
            }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        problem_type = question_case.get('type')
        
        if problem_type == 'gate_truth_table':
            gate = question_case['gate']
            return (
                f"Provide the complete truth table for a {gate} gate. Format each row as:\n"
                "[[inputs], [powered_output, unpowered_output]].\n"
                "All input combinations must be included. Enclose the entire answer between [[ ]]."
            )
        
        elif problem_type == 'compute_gate_output':
            inputs = ', '.join([f"I{i+1}={val}" for i, val in enumerate(question_case['inputs'])])
            state = "when powered" if question_case['powered'] else "when unpowered"
            return (
                f"Given {inputs} in a {question_case['gate']} gate, what is the output {state}? "
                "Put your final answer within [[ ]]."
            )
        
        elif problem_type == 'compute_expression_output':
            inputs = ', '.join([f"{k}={v}" for k, v in question_case['inputs'].items()])
            state = "when powered" if question_case['powered'] else "when unpowered"
            return (
                f"For the logical expression: {question_case['expression']}\n"
                f"With inputs: {inputs}\n"
                f"What is the output {state}? Put your answer in [[ ]]."
            )
        
        elif problem_type == 'find_inputs':
            power_state = "when powered" if question_case['powered'] else "when unpowered"
            return (
                f"Find all possible input combinations {power_state} for:\n"
                f"Expression: {question_case['expression']}\n"
                f"That produce output: {question_case['output']}\n"
                f"Variables should be ordered as: {', '.join(question_case['variables'])}\n"
                "Format answer as [[val1,val2,...];...] within [[ ]]."
            )
        
        return "Invalid problem type"
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[\[(.*?)\]\]', output, flags=re.DOTALL)
        if matches:
            last_match = matches[-1].strip()
            return re.sub(r'\s+', '', last_match)  # 移除所有空白字符
        return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            problem_type = identity['type']
            
            if problem_type == 'gate_truth_table':
                return cls._verify_gate_table(solution, identity['gate'])
            
            elif problem_type == 'compute_gate_output':
                expected = cls._compute_gate_output(
                    identity['gate'],
                    identity['inputs'],
                    identity['powered']
                )
                return cls._sanitize_answer(solution) == expected
            
            elif problem_type == 'compute_expression_output':
                expected = cls._evaluate_expression(
                    identity['expression'],
                    identity['inputs'],
                    identity['powered']
                )
                return cls._sanitize_answer(solution) == expected
            
            elif problem_type == 'find_inputs':
                return cls._verify_input_combinations(
                    solution,
                    identity['expression'],
                    identity['output'],
                    identity['variables'],
                    identity['powered']
                )
            
            return False
        except Exception as e:
            print(f"Verification error: {str(e)}")
            return False
    
    # Enhanced verification methods
    @classmethod
    def _verify_gate_table(cls, solution, gate):
        try:
            # 处理带换行的格式
            cleaned = solution.replace('\n', '').replace(' ', '')
            rows = [eval(r) for r in cleaned.split(';') if r]
            correct = cls._generate_gate_truth_table(gate)
            return rows == correct
        except SyntaxError:
            return False
    
    @classmethod
    def _verify_input_combinations(cls, solution, expr, target, variables, powered):
        try:
            # 解析所有可能的输入组合
            all_combos = set()
            for combo in product(['+', '-'], repeat=len(variables)):
                inputs = dict(zip(variables, combo))
                if cls._evaluate_expression(expr, inputs, powered) == target:
                    all_combos.add(tuple(combo))
            
            # 解析用户答案
            user_answers = set()
            for entry in solution.split(';'):
                entry = entry.strip("[] ")
                if not entry:
                    continue
                parts = [p.strip("'\" ") for p in entry.split(',')]
                if len(parts) != len(variables):
                    return False
                user_answers.add(tuple(parts))
            
            return user_answers == all_combos
        except Exception as e:
            print(f"Input verification error: {str(e)}")
            return False
    
    @staticmethod
    def _sanitize_answer(answer):
        """统一处理各种格式变体"""
        return answer.strip("[]'\" ").replace(' ', '').upper()
    
    @classmethod
    def _evaluate_expression(cls, expr, inputs, powered):
        """增强表达式解析"""
        if not powered:
            return '-'
        try:
            # 转换为Python表达式
            expr = (
                expr.replace('¬', ' not ')
                .replace('∧', ' and ')
                .replace('∨', ' or ')
                .replace('  ', ' ')
            )
            # 创建评估环境
            env = {k: v == '+' for k, v in inputs.items()}
            # 安全评估
            result = eval(expr, {'__builtins__': None}, env)
            return '+' if result else '-'
        except:
            return '-'
    
    @staticmethod
    def extract_variables(expr):
        """使用正则表达式精确提取变量"""
        return sorted(set(re.findall(r'\b[p-z]\b', expr)))
