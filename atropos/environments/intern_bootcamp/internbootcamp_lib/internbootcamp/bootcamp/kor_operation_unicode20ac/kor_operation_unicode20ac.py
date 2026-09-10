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
A€B=2A+3B
A and B are matrices.Example questions are as follows:

<example 0>
A=
\[
\begin{pmatrix}
  1 & 2 \\
  3 & 4
\end{pmatrix}
\]
B=
\[
\begin{pmatrix}
  5 & 6 \\
  7 & 8
\end{pmatrix}
\]
Compute A€B.
The answer is a matrix, write it in this form:[[((a,b),(c,d))]].
</example 0>

<example 1>
A=
\[
\begin{pmatrix}
  0 & 1 \\
  1 & 0
\end{pmatrix}
\]
B=
\[
\begin{pmatrix}
  2 & 3 \\
  4 & 5
\end{pmatrix}
\]
Compute A€B.
The answer is a matrix, write it in this form:[[((a,b),(c,d))]].
</example 1>

<example 2>
A=
\[
\begin{pmatrix}
  -1 & -2 \\
  -3 & -4
\end{pmatrix}
\]
B=
\[
\begin{pmatrix}
  1 & 2 \\
  3 & 4
\end{pmatrix}
\]
Compute A€B.
The answer is a matrix, write it in this form:[[((a,b),(c,d))]].
</example 2>

<example 3>
A=
\[
\begin{pmatrix}
  1 & 0 \\
  0 & 1
\end{pmatrix}
\]
B=
\[
\begin{pmatrix}
  2 & 2 \\
  2 & 2
\end{pmatrix}
\]
Compute A€B.
The answer is a matrix, write it in this form:[[((a,b),(c,d))]].
</example 3>

<example 4>
A=
\[
\begin{pmatrix}
  2 & 4 \\
  6 & 8
\end{pmatrix}
\]
B=
\[
\begin{pmatrix}
  1 & 3 \\
  5 & 7
\end{pmatrix}
\]
Compute A€B.
The answer is a matrix, write it in this form:[[((a,b),(c,d))]].
</example 4>

<example 5>
A=
\[
\begin{pmatrix}
  1 & 1 \\
  1 & 1
\end{pmatrix}
\]
B=
\[
\begin{pmatrix}
  2 & 2 \\
  2 & 2
\end{pmatrix}
\]
Compute A€B.
The answer is a matrix, write it in this form:[[((a,b),(c,d))]].
</example 5>

<example 6>
A=
\[
\begin{pmatrix}
  3 & 3 \\
  3 & 3
\end{pmatrix}
\]
B=
\[
\begin{pmatrix}
  4 & 4 \\
  4 & 4
\end{pmatrix}
\]
Compute A€B.
The answer is a matrix, write it in this form:[[((a,b),(c,d))]].
</example 6>

<example 7>
A=
\[
\begin{pmatrix}
  1 & 2 & 3\\
  4 & 5 & 6\\
  7 & 8 & 9
\end{pmatrix}
\]
B=
\[
\begin{pmatrix}
  9 & 8 & 7\\
  6 & 5 & 4\\
  3 & 2 & 1
\end{pmatrix}
\]
Compute A€B.
The answer is a matrix, write it in this form:[[((a,b,c),(d,e,f),(g,h,i))]].
</example 7>

<example 8>
A=
\[
\begin{pmatrix}
  0 & 2 \\
  4 & 6
\end{pmatrix}
\]
B=
\[
\begin{pmatrix}
  1 & 3 \\
  5 & 7
\end{pmatrix}
\]
Compute A€B.
The answer is a matrix, write it in this form:[[((a,b),(c,d))]].
</example 8>

<example 9>
A=
\[
\begin{pmatrix}
  1 & 2 \\
  3 & 4
\end{pmatrix}
\]
B=
\[
\begin{pmatrix}
  0 & 1 \\
  2 & 3
\end{pmatrix}
\]
Compute A€B.
The answer is a matrix, write it in this form:[[((a,b),(c,d))]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
import math
import ast
from bootcamp import Basebootcamp

class KorOperationUnicode20acbootcamp(Basebootcamp):
    def __init__(self, matrix_shape=(2,2), min_val=-10, max_val=10):
        if not (len(matrix_shape) == 2 
                and all(isinstance(d, int) and d > 0 for d in matrix_shape)):
            raise ValueError("matrix_shape must be a tuple of two positive integers")
        if min_val > max_val:
            raise ValueError("min_val must be <= max_val")
        self.matrix_shape = matrix_shape
        self.min_val = min_val
        self.max_val = max_val
    
    def case_generator(self):
        rows, cols = self.matrix_shape
        return {
            'A': [[random.randint(self.min_val, self.max_val) for _ in range(cols)] 
                  for _ in range(rows)],
            'B': [[random.randint(self.min_val, self.max_val) for _ in range(cols)] 
                  for _ in range(rows)]
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        def matrix_to_latex(matrix):
            return '\\[\n\\begin{pmatrix}\n' + ' \\\\\n'.join(
                '  ' + ' & '.join(map(str, row)) for row in matrix
            ) + '\n\\end{pmatrix}\n\\]'
        
        rows = len(question_case['A'])
        cols = len(question_case['A'][0]) if rows else 0
        
        # 修正示例格式生成逻辑
        example_rows = [
            '(' + ','.join(['...']*cols) + ')' 
            for _ in range(rows)
        ]
        format_example = f'[[({",".join(example_rows)})]]'
        
        return f"""请计算矩阵运算A€B=2A+3B，其中：

矩阵A：
{matrix_to_latex(question_case['A'])}

矩阵B：
{matrix_to_latex(question_case['B'])}

答案应为{rows}x{cols}矩阵。按照格式要求将最终答案置于双括号内：
示例格式：{format_example}"""

    @staticmethod
    def extract_output(output):
        def safe_parse(s):
            try:
                parsed = ast.literal_eval(s)
                if not isinstance(parsed, tuple) or not all(isinstance(row, tuple) for row in parsed):
                    return None
                return parsed
            except:
                return None

        matches = re.findall(r'\[\[(.*?)\]\]', output, re.DOTALL)
        if not matches:
            return None
            
        clean_str = re.sub(r'\s+', '', matches[-1].strip())
        return safe_parse(clean_str)  # 移除破坏性字符处理

    @classmethod
    def _verify_correction(cls, solution, identity):
        A = identity['A']
        B = identity['B']
        
        try:
            # 维度校验
            if (len(solution) != len(A)) or any(len(s_row) != len(a_row) for s_row, a_row in zip(solution, A)):
                return False
            
            # 元素校验
            for i_row, (a_row, b_row) in enumerate(zip(A, B)):
                for j_col, (a, b) in enumerate(zip(a_row, b_row)):
                    expected = 2 * a + 3 * b
                    actual = solution[i_row][j_col]
                    if not math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-9):
                        return False
            return True
        except (TypeError, IndexError):
            return False
