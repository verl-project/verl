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

**Heyawake Puzzle Rules Explained:**

1. **Grid Structure**: 
   - Played on a rectangular grid divided into polyomino regions (\"rooms\"). 
   - Some rooms contain a number; others are unnumbered.

2. **Black Cell Placement**:
   - **Isolation**: No two black cells may be adjacent horizontally or vertically (diagonal is allowed).
   - **Room Constraints**: 
     - A room with a number **N** must contain exactly **N** black cells.
     - Unnumbered rooms can have any number of black cells (including zero).

3. **White Cell Connectivity**:
   - All white (uncolored) cells must form **a single contiguous area** connected edge-to-edge (diagonals do not count).

4. **Stripe Prevention (Three Rooms Rule)**:
   - In any row or column, a straight line of white cells cannot span **three or more rooms** without interruption by at least one black cell. This prevents white \"stripes\" crossing multiple rooms.


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
from collections import deque

class Heyawakebootcamp(Basebootcamp):
    def __init__(self, rows=3, cols=3):
        self.rows = rows
        self.cols = cols
    
    def case_generator(self):
        # 生成一个简单的谜题实例，房间0的number为0，确保有解
        rows = self.rows
        cols = self.cols
        rooms = [
            {
                'cells': [(0, 0)],
                'number': 0
            },
            {
                'cells': [(i, j) for i in range(rows) for j in range(cols) if (i, j) != (0, 0)]
                # 无number
            }
        ]
        return {
            'rows': rows,
            'cols': cols,
            'rooms': rooms
        }
    
    @staticmethod
    def prompt_func(question_case) -> str:
        rows = question_case['rows']
        cols = question_case['cols']
        rooms = question_case['rooms']
        problem_desc = "请解决以下Heyawake谜题：\n\n"
        problem_desc += "谜题规则：\n"
        problem_desc += "1. 黑格不能水平或垂直相邻。\n"
        problem_desc += "2. 每个带有数字的房间必须包含恰好该数字的黑格。未带数字的房间可以有任何数量的黑格。\n"
        problem_desc += "3. 所有白格必须形成一个连通的区域。\n"
        problem_desc += "4. 同一行或列中的连续白格不能跨越三个或更多不同的房间。\n\n"
        problem_desc += f"网格尺寸：{rows}行×{cols}列。\n"
        problem_desc += "房间划分及其数字说明：\n"
        for idx, room in enumerate(rooms):
            cells = room['cells']
            number = room.get('number', None)
            cells_str = ', '.join(f'({r},{c})' for r, c in cells)
            problem_desc += f"- 房间{idx+1}包含单元格 {cells_str}"
            if number is not None:
                problem_desc += f"，必须恰好有 {number} 个黑格"
            problem_desc += "。\n"
        problem_desc += "\n将答案以二维数组（0为白格，1为黑格）放在[answer]标签内，例如：\n[answer]\n[[0, 0, 0],\n[0, 1, 0],\n[0, 0, 0]]\n[/answer]"
        return problem_desc
    
    @staticmethod
    def extract_output(output):
        pattern = r'\[answer\](.*?)\[/answer\]'
        matches = re.findall(pattern, output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            solution = eval(last_match)
            if isinstance(solution, list) and all(isinstance(row, list) for row in solution):
                return solution
            return None
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        # 验证solution结构合法性
        rows = identity['rows']
        cols = identity['cols']
        rooms = identity['rooms']
        if (not isinstance(solution, list) or len(solution) != rows or
            any(not isinstance(row, list) or len(row) != cols for row in solution)):
            return False
        for row in solution:
            for cell in row:
                if cell not in (0, 1):
                    return False
        
        # 规则1：黑格不能相邻
        for i in range(rows):
            for j in range(cols):
                if solution[i][j] == 1:
                    if j+1 < cols and solution[i][j+1] == 1:
                        return False
                    if i+1 < rows and solution[i+1][j] == 1:
                        return False
        
        # 规则2：房间约束
        for room in rooms:
            cells = room['cells']
            if 'number' in room:
                required = room['number']
                actual = sum(solution[i][j] for (i, j) in cells)
                if actual != required:
                    return False
        
        # 规则3：白格连通性
        white = [(i, j) for i in range(rows) for j in range(cols) if solution[i][j] == 0]
        if not white:
            return False
        visited = set()
        queue = deque([white[0]])
        visited.add(white[0])
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        while queue:
            i, j = queue.popleft()
            for dx, dy in directions:
                ni, nj = i+dx, j+dy
                if 0 <= ni < rows and 0 <= nj < cols and solution[ni][nj] == 0:
                    if (ni, nj) not in visited:
                        visited.add((ni, nj))
                        queue.append((ni, nj))
        if len(visited) != len(white):
            return False
        
        # 规则4：条纹防止规则
        # 构建房间映射
        room_id = [[-1 for _ in range(cols)] for _ in range(rows)]
        for idx, room in enumerate(rooms):
            for i, j in room['cells']:
                room_id[i][j] = idx
        
        # 检查行
        for i in range(rows):
            current_rooms = []
            for j in range(cols):
                if solution[i][j] == 1:
                    if len(current_rooms) >= 3:
                        return False
                    current_rooms = []
                else:
                    rid = room_id[i][j]
                    if not current_rooms or rid != current_rooms[-1]:
                        current_rooms.append(rid)
                    if len(current_rooms) >= 3:
                        return False
            if len(current_rooms) >= 3:
                return False
        
        # 检查列
        for j in range(cols):
            current_rooms = []
            for i in range(rows):
                if solution[i][j] == 1:
                    if len(current_rooms) >= 3:
                        return False
                    current_rooms = []
                else:
                    rid = room_id[i][j]
                    if not current_rooms or rid != current_rooms[-1]:
                        current_rooms.append(rid)
                    if len(current_rooms) >= 3:
                        return False
            if len(current_rooms) >= 3:
                return False
        
        return True
