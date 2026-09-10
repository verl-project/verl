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

Norinori is a grid-based logic puzzle where the goal is to partition the entire grid into non-overlapping **domino regions** (each covering exactly two orthogonally adjacent cells). The rules are as follows:

1. **Grid Division**: The grid must be fully divided into domino-sized regions (2x1 or 1x2 tiles). Every cell belongs to exactly one domino.

2. **Shaded Cell Pairing**: The puzzle starts with some cells pre-shaded. Each shaded cell must be part of a domino region that contains **exactly two shaded cells**. In other words, every shaded cell must form a domino with one (and only one) adjacent shaded cell.

3. **Unshaded Regions**: The remaining unshaded cells are also grouped into domino regions, which may be oriented horizontally or vertically.

4. **No Overlaps/Intersections**: No two domino regions may overlap, and all dominoes must stay within the grid boundaries.

In summary, the challenge lies in pairing shaded cells into dominoes while ensuring the entire grid is covered, with no conflicting or isolated regions.


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
import ast

class Norinoribootcamp(Basebootcamp):
    def __init__(self, rows=6, cols=6, num_shaded_dominoes=3):
        if rows % 2 != 0 or cols % 2 != 0:
            raise ValueError("rows and cols must be even numbers.")
        total_dominoes = (rows * cols) // 2
        if num_shaded_dominoes < 0 or num_shaded_dominoes > total_dominoes:
            raise ValueError(f"num_shaded_dominoes must be between 0 and {total_dominoes}.")
        self.rows = rows
        self.cols = cols
        self.num_shaded_dominoes = num_shaded_dominoes

    def case_generator(self):
        dominoes = self._generate_tiling()
        shaded_dominoes = random.sample(dominoes, self.num_shaded_dominoes)
        shaded_cells = []
        for domino in shaded_dominoes:
            shaded_cells.append(tuple(domino[0]))
            shaded_cells.append(tuple(domino[1]))
        return {
            'shaded_cells': shaded_cells,
            'rows': self.rows,
            'cols': self.cols
        }

    def _generate_tiling(self):
        dominoes = []
        for i in range(0, self.rows, 2):
            for j in range(0, self.cols, 2):
                if random.choice([True, False]):
                    dominoes.append(((i, j), (i, j + 1)))
                    dominoes.append(((i + 1, j), (i + 1, j + 1)))
                else:
                    dominoes.append(((i, j), (i + 1, j)))
                    dominoes.append(((i, j + 1), (i + 1, j + 1)))
        return dominoes

    @staticmethod
    def prompt_func(question_case):
        shaded = question_case['shaded_cells']
        rows = question_case['rows']
        cols = question_case['cols']
        shaded_str = ', '.join([f"({r},{c})" for r, c in shaded])
        return f"""你正在解决一个Norinori谜题。规则如下：

1. 将{rows}x{cols}网格划分为1x2或2x1的骨牌，覆盖所有单元格。
2. 所有预着色单元格必须成对组成骨牌。
3. 未着色单元格也需组成骨牌，且不包含任何着色单元格。

预着色单元格的坐标为：{shaded_str}

请输出骨牌划分方案，将答案放在[answer]和[/answer]之间。格式示例：
[answer]
[[(0,0),(0,1)], [(1,0),(1,1)], ...]
[/answer]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, flags=re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            solution = ast.literal_eval(last_match)
            if not isinstance(solution, list):
                return None
            for domino in solution:
                if len(domino) != 2:
                    return None
                for cell in domino:
                    if not isinstance(cell, (list, tuple)) or len(cell) != 2:
                        return None
                    if not all(isinstance(c, int) for c in cell):
                        return None
            return solution
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        shaded = set(map(tuple, identity['shaded_cells']))
        rows = identity['rows']
        cols = identity['cols']
        all_cells = set()

        # 验证骨牌结构有效性
        for domino in solution:
            if len(domino) != 2:
                return False
            c1, c2 = tuple(domino[0]), tuple(domino[1])
            # 检查坐标范围
            if not (0 <= c1[0] < rows and 0 <= c1[1] < cols):
                return False
            if not (0 <= c2[0] < rows and 0 <= c2[1] < cols):
                return False
            # 检查相邻性
            if abs(c1[0]-c2[0]) + abs(c1[1]-c2[1]) != 1:
                return False
            # 检查重复
            if c1 in all_cells or c2 in all_cells:
                return False
            all_cells.update({c1, c2})

        # 检查全覆盖
        if len(all_cells) != rows * cols:
            return False

        # 检查着色配对
        for cell in shaded:
            cell = tuple(cell)
            found = False
            for domino in solution:
                if cell in domino:
                    other = domino[0] if domino[1] == cell else domino[1]
                    if other not in shaded:
                        return False
                    found = True
                    break
            if not found:
                return False

        # 检查非着色骨牌
        for domino in solution:
            shade_count = sum(1 for c in domino if tuple(c) in shaded)
            if shade_count not in (0, 2):
                return False

        return True
