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

Nonograms, also called \"Paint by Numbers,\" are logic puzzles where you reveal a hidden image by filling cells in a grid according to numerical clues. Here are the general rules:

1. **Grid Structure**:  
   - The puzzle consists of a rectangular grid (e.g., 10×10, 15×15, etc.).  
   - Each **row** and **column** has a sequence of numbers (clues) at its edge.

2. **Clue Interpretation**:  
   - Clues indicate groups of **consecutively filled cells** in that row/column.  
     Example: A clue of `3 2` means the row/column contains a block of 3 filled cells, followed by **at least one empty cell**, then a block of 2 filled cells.  
   - The **order of clues** matches the order of blocks (left-to-right for rows, top-to-bottom for columns).  
   - Empty cells can be marked with an \"X\" or left blank, depending on the puzzle variant.

3. **Rules for Filling**:  
   - **Exact blocks**: The numbers must correspond **exactly** to the filled cells.  
     Example: If a row has a clue `5`, the entire row must be filled with 5 contiguous cells.  
   - **No overlaps**: Blocks of filled cells cannot overlap unless the clues explicitly allow it (rare).  
   - **Separation**: Blocks in the same row/column must be separated by **at least one empty cell**.  

4. **Solving Logic**:  
   - Use **cross-referencing** between row and column clues to deduce filled cells.  
   - Eliminate impossible configurations using overlaps or forced gaps.  

5. **Victory Condition**:  
   - The puzzle is solved when all filled cells match the clues for every row and column, revealing the hidden image.  

Nonograms require no guessing—only logical deduction based on the clues and grid constraints.  


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re

class Nonogramsbootcamp(Basebootcamp):
    def __init__(self, **params):
        self.rows = params.get('rows', 5)
        self.cols = params.get('cols', 5)
        self.fill_prob = params.get('fill_prob', 0.3)

    def case_generator(self):
        # Generate solution grid
        solution = [
            [random.random() < self.fill_prob for _ in range(self.cols)]
            for _ in range(self.rows)
        ]

        # Calculate clues
        rows_clues = [self._get_clues(row) for row in solution]
        cols_clues = [self._get_clues([solution[r][c] for r in range(self.rows)]) 
                     for c in range(self.cols)]

        return {'rows': rows_clues, 'columns': cols_clues}

    @staticmethod
    def prompt_func(question_case) -> str:
        prompt = """你正在解决一个Nonogram谜题。根据行和列的数字线索填充网格：

规则说明：
1. 数字表示连续填充的单元格块，块间至少间隔一个空格
2. 行线索从左到右排列，列线索从上到下排列
3. 用'X'表示填充，用空格或'.'表示空白

行线索：
""" + "\n".join(
    f"第{i+1}行: {clues if clues else '无'}" 
    for i, clues in enumerate(question_case['rows'])
) + "\n\n列线索：\n" + "\n".join(
    f"第{i+1}列: {clues if clues else '无'}" 
    for i, clues in enumerate(question_case['columns'])
) + """

请将最终答案放在[answer]标签内，每行用'X'和空格表示填充状态：
示例：
[answer]
XX X
 XXX
X  X
[/answer]"""
        return prompt

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not matches:
            return None
        grid_str = matches[-1].strip()
        solution = []
        for line in grid_str.split('\n'):
            line = line.strip()
            if not line:
                continue
            solution.append([c.upper() == 'X' for c in line if not c.isspace() or c == '.'])
        return solution

    @classmethod
    def _verify_correction(cls, solution, identity):
        # Validate grid dimensions
        if len(solution) != len(identity['rows']):
            return False
        if any(len(row) != len(identity['columns']) for row in solution):
            return False

        # Check row clues
        for i, row in enumerate(solution):
            if cls._get_clues(row) != identity['rows'][i]:
                return False

        # Check column clues
        for j in range(len(identity['columns'])):
            col = [solution[i][j] for i in range(len(solution))]
            if cls._get_clues(col) != identity['columns'][j]:
                return False

        return True

    @staticmethod
    def _get_clues(line):
        clues = []
        current = 0
        for cell in line:
            if cell:
                current += 1
            elif current > 0:
                clues.append(current)
                current = 0
        if current > 0:
            clues.append(current)
        return clues
