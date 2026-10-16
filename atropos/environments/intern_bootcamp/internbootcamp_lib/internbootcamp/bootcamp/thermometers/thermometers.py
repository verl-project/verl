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

**Thermometers Puzzle Rules (General Explanation):**

1. **Grid Structure**: The puzzle is played on a square grid (size can vary). Each cell may belong to one or more \"thermometers\" represented by connected paths starting at a bulb (●) and ending at a tip. Thermometers may overlap or intersect.

2. **Number Placement**: 
   - Fill the grid with numbers such that each row and column contains all integers from 1 to *N* (where *N* is the grid size), with no repetition (like a Latin square).
   - **Exception**: Some variants may omit row/column uniqueness, depending on the puzzle's design.

3. **Thermometer Constraint**:
   - For every thermometer, the numbers must **strictly increase** from the bulb to the tip. Each subsequent cell along the thermometer's path must be a larger number than the preceding cell.

4. **Clues**:
   - Pre-filled numbers (if provided) must remain unchanged and act as constraints for solving.

**Objective**: Fill the grid while satisfying all row/column uniqueness rules (if applicable) and ensuring all thermometer paths follow the strict increasing order.


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random

class Thermometersbootcamp(Basebootcamp):
    def __init__(self, size=5, num_thermometers=3, enforce_latin=True, max_retries=100):
        self.size = size
        self.num_thermometers = num_thermometers
        self.enforce_latin = enforce_latin
        self.max_retries = max_retries

    def case_generator(self):
        if self.enforce_latin:
            solution = self._generate_latin_square(self.size)
        else:
            raise NotImplementedError("Non-Latin square puzzles are not supported yet.")

        thermometers = []
        for _ in range(self.num_thermometers):
            thermometer = None
            for _ in range(self.max_retries):
                start_row = random.randint(0, self.size-1)
                start_col = random.randint(0, self.size-1)
                path = self._generate_thermometer_path(solution, start_row, start_col)
                if path:
                    thermometer = {'bulb': path[0], 'path': path}
                    break
            if not thermometer:
                raise ValueError(f"Failed to generate thermometer after {self.max_retries} attempts")
            thermometers.append(thermometer)
        
        return {
            "size": self.size,
            "thermometers": thermometers,
            "enforce_latin": self.enforce_latin
        }

    @staticmethod
    def _generate_latin_square(size):
        latin = []
        for i in range(size):
            row = [(i + j) % size + 1 for j in range(size)]
            latin.append(row)
        random.shuffle(latin)
        return latin

    @staticmethod
    def _generate_thermometer_path(solution, start_row, start_col):
        path = [(start_row, start_col)]
        current_value = solution[start_row][start_col]
        size = len(solution)
        
        while True:
            last_row, last_col = path[-1]
            neighbors = []
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = last_row + dr, last_col + dc
                if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in path:
                    next_value = solution[nr][nc]
                    if next_value > current_value:
                        neighbors.append((nr, nc, next_value))
            if not neighbors:
                break
            nr, nc, nv = random.choice(neighbors)
            path.append((nr, nc))
            current_value = nv
        
        return path if len(path) >= 2 else None

    @staticmethod
    def prompt_func(question_case) -> str:
        def format_coord(coord):
            r, c = coord
            return f"行{r+1}列{c+1}"
        
        size = question_case['size']
        thermometers = question_case['thermometers']
        enforce_latin = question_case['enforce_latin']
        
        rules = [
            f"1. 在{size}x{size}的网格中填入1到{size}的数字。",
            "2. 每个温度计的路径必须从灯泡(●)开始严格递增。",
            f"3. 每{'行和列必须包含不重复的1到{size}' if enforce_latin else '行和列允许重复但需满足温度计约束'}。"
        ]
        
        thermo_desc = []
        for i, thermo in enumerate(thermometers, 1):
            path = thermo['path']
            bulb = format_coord(thermo['bulb'])
            tip = format_coord(path[-1])
            path_str = " → ".join(format_coord(p) for p in path)
            thermo_desc.append(f"温度计{i}: 从{bulb}到{tip}, 路径: {path_str}")

        return (
            "解决以下温度计谜题：\n\n" +
            "\n".join(rules) + "\n\n" +
            "温度计列表：\n" + "\n".join(thermo_desc) + "\n\n" +
            "将答案按行排列，每行数字用空格分隔，置于[answer]和[/answer]之间。示例：\n" +
            "[answer]\n1 2 3\n2 3 1\n3 1 2\n[/answer]"
        )

    @staticmethod
    def extract_output(output):
        answer_blocks = re.findall(r'\[answer\](.*?)\[\/answer\]', output, re.DOTALL)
        if not answer_blocks:
            return None
        last_answer = answer_blocks[-1].strip()
        rows = [line.strip() for line in last_answer.split('\n') if line.strip()]
        try:
            grid = [list(map(int, row.split())) for row in rows]
            if all(len(row) == len(grid[0]) for row in grid) and len(grid) == len(grid[0]):
                return grid
        except ValueError:
            pass
        return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        size = identity['size']
        enforce_latin = identity.get('enforce_latin', True)
        thermometers = identity['thermometers']

        if len(solution) != size or any(len(row) != size for row in solution):
            return False
        if any(not (1 <= num <= size) for row in solution for num in row):
            return False

        if enforce_latin:
            expected = list(range(1, size+1))
            for row in solution:
                if sorted(row) != expected:
                    return False
            for col in range(size):
                if sorted(row[col] for row in solution) != expected:
                    return False

        for thermo in thermometers:
            path = thermo['path']
            values = [solution[r][c] for (r, c) in path]
            if any(values[i] >= values[i+1] for i in range(len(values)-1)):
                return False

        return True
