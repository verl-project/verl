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

Dominosa is a logic puzzle where the goal is to partition a grid of numbers into non-overlapping dominoes (2x1 or 1x2 rectangles) following these rules:

1. **Grid Structure**: 
   - The grid has an even number of cells, arranged in rows and columns. The exact dimensions vary, but they must allow full coverage by dominoes (e.g., a 5x6 grid, 7x8 grid, etc.).

2. **Domino Formation**:
   - Each domino consists of two orthogonally adjacent cells (horizontal or vertical neighbors).
   - Every cell in the grid must belong to exactly one domino.

3. **Uniqueness Constraint**:
   - Each domino is defined by an unordered pair of numbers (e.g., a domino with numbers 3 and 5 is equivalent to one with 5 and 3).
   - Every domino in the solution must appear **exactly once**. If two dominoes share the same pair of numbers, the puzzle is invalid.

4. **Number Set**:
   - The numbers in the grid are derived from a contiguous set (e.g., 0 to N). The maximum number (N) determines the total number of possible unique domino pairs, which must match the grid size (e.g., for numbers 0-4, there are 15 unique domino pairs, requiring a 30-cell grid).

The challenge is to deduce the domino layout that satisfies all constraints without repetition. Logical elimination and pattern recognition are key to solving the puzzle.


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import itertools
import random
import re

class Dominosabootcamp(Basebootcamp):
    def __init__(self, n=4):
        """
        初始化Dominosa训练场环境，配置数字范围和网格参数。
        
        参数:
            n: 数字范围上限，生成数字0-n的连续集合，默认4对应5x6网格
        """
        self.n = n
        self.rows = n + 1
        self.cols = n + 2

    def case_generator(self):
        """
        生成合法Dominosa谜题实例，保证至少存在一个解。
        
        返回:
            dict: 包含网格布局和参数的字典，结构为{'grid': 二维列表, 'n': 数字范围}
        """
        # 生成所有可能的无序数字对并打乱顺序
        pairs = list(itertools.combinations_with_replacement(range(self.n + 1), 2))
        random.shuffle(pairs)
        # 创建网格并填充数字对
        grid = []
        pair_index = 0
        if self.cols % 2 == 0:
            for i in range(self.rows):
                row = []
                for j in range(0, self.cols, 2):
                    a, b = pairs[pair_index]
                    row.extend([a, b])  # 水平排列数字对
                    pair_index += 1
                grid.append(row)
        else:
            grid = [[] for _ in range(self.rows)]
            for i in range(0, self.cols):
                for j in range(0, self.rows, 2):
                    a, b = pairs[pair_index]
                    grid[j].append(a)
                    grid[j+1].append(b)
                    pair_index += 1

        return {
            'grid': grid,
            'n': self.n
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        """
        将数字网格转化为自然语言问题描述，包含格式说明。
        
        参数:
            question_case: case_generator生成的谜题实例
            
        返回:
            str: 包含网格布局和解答要求的提示文本
        """
        grid = question_case['grid']
        n = question_case['n']
        
        prompt = f"""你是Dominosa谜题专家，请将以下{len(grid)}x{len(grid[0])}网格划分为不重复的骨牌组合。每个骨牌必须覆盖两个相邻单元格（水平或垂直），且所有数字对唯一。

网格布局（行号从0开始）：
"""
        for i, row in enumerate(grid):
            prompt += f"行{i}:\t" + "\t".join(map(str, row)) + "\n"

        prompt += f"""
规则说明：
1. 数字范围：0-{n}，每个骨牌包含两个不同或相同的数字
2. 数对(a,b)与(b,a)视为相同，必须唯一
3. 必须完全覆盖所有单元格

答案格式要求：
将每个骨牌表示为两个坐标对，每行一个骨牌，如：
[answer]
(行号,列号),(行号,列号)
...[/answer]

请确保：
- 使用英文括号和逗号
- 按最后出现的答案块评分
- 坐标按行号、列号顺序"""
        return prompt

    @staticmethod
    def extract_output(output):
        """
        从模型输出中提取最后一个答案块并解析坐标。
        
        参数:
            output: 模型完整输出文本
            
        返回:
            list: 提取的骨牌坐标列表，格式[(坐标1, 坐标2), ...]
        """
        # 匹配最后一个答案块
        answer_blocks = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        
        dominoes = []
        last_block = answer_blocks[-1].strip()
        
        # 解析坐标对
        pattern = r'\((\d+)\s*,\s*(\d+)\)\s*,\s*\((\d+)\s*,\s*(\d+)\)'
        matches = re.findall(pattern, last_block)
        for m in matches:
            try:
                coord1 = (int(m[0]), int(m[1]))
                coord2 = (int(m[2]), int(m[3]))
                dominoes.append((coord1, coord2))
            except:
                continue
        
        return dominoes if dominoes else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        """
        验证答案的完整性和正确性。
        
        参数:
            solution: 提取的骨牌坐标列表
            identity: case_generator生成的谜题实例
            
        返回:
            bool: 是否满足所有谜题约束
        """
        if not solution:
            return False

        grid = identity['grid']
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        total_cells = rows * cols
        
        # 验证覆盖完整性
        covered = set()
        pairs = []
        
        for domino in solution:
            # 校验坐标数量
            if len(domino) != 2:
                return False
            (r1, c1), (r2, c2) = domino
            
            # 校验坐标有效性
            if not (0 <= r1 < rows and 0 <= c1 < cols):
                return False
            if not (0 <= r2 < rows and 0 <= c2 < cols):
                return False
            
            # 校验相邻性
            if not ((r1 == r2 and abs(c1 - c2) == 1) or 
                    (c1 == c2 and abs(r1 - r2) == 1)):
                return False
            
            # 检查重复覆盖
            if (r1, c1) in covered or (r2, c2) in covered:
                return False
            
            covered.update([(r1, c1), (r2, c2)])
            
            # 记录数字对
            a, b = grid[r1][c1], grid[r2][c2]
            pairs.append(tuple(sorted((a, b))))
        
        # 检查覆盖率
        if len(covered) != total_cells:
            return False
        
        # 检查唯一性
        return len(pairs) == len(set(pairs))
