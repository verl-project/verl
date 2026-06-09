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

        参数:
            question_case: 由case_generator生成的谜题实例
            
        返回:
            str: 格式化的问题字符串
            
        注意:
            1. 需考虑问题的格式，以便后续能正确提取
            2. 问题描述中应包含期望的答案格式说明，以便后续能正确提取，为了避免抽取时匹配出干扰项，请要求模型将答案放在特定标签，如[answer] [/answer]内
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

Kakurasu is a logic puzzle played on a rectangular grid (typically N×N). Each cell in the grid can be either shaded or unshaded. The puzzle provides target values for each row and column, and the goal is to shade cells such that:

1. **Row Constraints**: For every row, the sum of the *column indices* of the shaded cells in that row equals the target value specified for that row.  
   - Example: If a row's target is 7, you might shade cells in columns 3 and 4 (since 3 + 4 = 7).

2. **Column Constraints**: For every column, the sum of the *row indices* of the shaded cells in that column equals the target value specified for that column.  
   - Example: If a column's target is 5, you might shade cells in rows 2 and 3 (since 2 + 3 = 5).

3. **Unique Contributions**: A shaded cell at position (row *i*, column *j*) contributes its *column index* **j** to its row's sum and its *row index* **i** to its column's sum. These dual contributions must satisfy both the row and column targets simultaneously.

4. **No Overlapping Rules**: Unlike Sudoku, there are no region constraints—only row and column sums matter. However, cells cannot be \"partially\" shaded; they are either fully shaded or unshaded.

The puzzle is solved when all row and column targets are satisfied without contradiction.


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
import ast

class Kakurasubootcamp(Basebootcamp):
    def __init__(self, n=5):
        self.n = n
    
    def case_generator(self):
        n = self.n
        grid = [[random.choice([0, 1]) for _ in range(n)] for _ in range(n)]
        
        row_targets = []
        for i in range(n):
            total = sum((j + 1) * cell for j, cell in enumerate(grid[i]))
            row_targets.append(total)
        
        col_targets = []
        for j in range(n):
            total = sum((i + 1) * grid[i][j] for i in range(n))
            col_targets.append(total)
        
        return {
            'n': n,
            'row_targets': row_targets,
            'col_targets': col_targets
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        row_targets = question_case['row_targets']
        col_targets = question_case['col_targets']
        return f"""你正在解决一个Kakurasu谜题。这是一个{n}x{n}的网格谜题，目标是根据行和列的约束条件涂黑单元格。

规则：
1. 每行中被涂黑单元格的列索引（从左到右为1到{n}）之和等于该行的目标值。
2. 每列中被涂黑单元格的行索引（从上到下为1到{n}）之和等于该列的目标值。
3. 每个单元格必须明确涂黑（1）或未涂黑（0）。

当前谜题的行目标值（从上到下）：{row_targets}
当前谜题的列目标值（从左到右）：{col_targets}

请将你的解答格式化为{n}x{n}的二维数组，其中1表示涂黑，0表示未涂黑，并用[answer]和[/answer]标签包裹。例如：
[answer]
[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
[/answer]
"""
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            solution = ast.literal_eval(last_match)
            return solution
        except (SyntaxError, ValueError):
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        row_targets = identity['row_targets']
        col_targets = identity['col_targets']
        
        # 验证答案结构
        if not isinstance(solution, list) or len(solution) != n:
            return False
        for row in solution:
            if not isinstance(row, list) or len(row) != n:
                return False
            for cell in row:
                if cell not in (0, 1):
                    return False
        
        # 验证行约束
        for i in range(n):
            row_sum = sum((j + 1) * cell for j, cell in enumerate(solution[i]))
            if row_sum != row_targets[i]:
                return False
        
        # 验证列约束
        for j in range(n):
            col_sum = sum((i + 1) * solution[i][j] for i in range(n))
            if col_sum != col_targets[j]:
                return False
        
        return True
