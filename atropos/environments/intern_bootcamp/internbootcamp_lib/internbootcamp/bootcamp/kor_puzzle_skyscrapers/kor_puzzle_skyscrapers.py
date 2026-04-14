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
1.The game is played on an n*n grid, with skyscrapers placed in all cells on the grid.
2.Skyscrapers have a height of 1 to the size of the grid, i.e. 1 to 4 for a 4x4 puzzle.
3.You cannot have two skyscrapers of the same height in the same row or column.
4.The numbers on the sides of the boxes indicate how many skyscrapers you would see if you looked in the direction of the arrows, since taller buildings will be blocked by shorter ones.
5.Fill in the numbers in each cell to indicate the height of the skyscrapers.
6.The topic consists of an n*n matrix filled with X, with the numerical constraints mentioned above attached to the top and bottom.Example questions are as follows:

<example 0>
Grid Layout:
	1	2	3	2	
1	X	X	X	X	4
2	X	X	X	X	1
2	X	X	X	X	3
2	X	X	X	X	2
	3	2	1	2
The answer should be given from left to right, top to bottom. Separate elements with a space and rows with a comma. Wrap the entire answer in double square brackets.
</example 0>

<example 1>
Grid Layout:
	2	1	2	3	
2	X	X	X	X	2
2	X	X	X	X	2
3	X	X	X	X	1
1	X	X	X	X	3
	1	3	2	2
The answer should be given from left to right, top to bottom. Separate elements with a space and rows with a comma. Wrap the entire answer in double square brackets.
</example 1>

<example 2>
Grid Layout:
	2	3	2	1	
3	X	X	X	X	1
1	X	X	X	X	3
2	X	X	X	X	2
2	X	X	X	X	2
	2	2	1	3
The answer should be given from left to right, top to bottom. Separate elements with a space and rows with a comma. Wrap the entire answer in double square brackets.
</example 2>

<example 3>
Grid Layout:
	2	4	2	1	
3	X	X	X	X	1
3	X	X	X	X	2
1	X	X	X	X	4
2	X	X	X	X	2
	2	2	1	3
The answer should be given from left to right, top to bottom. Separate elements with a space and rows with a comma. Wrap the entire answer in double square brackets.
</example 3>

<example 4>
Grid Layout:
	1	2	2	2	
1	X	X	X	X	3
2	X	X	X	X	2
3	X	X	X	X	1
2	X	X	X	X	2
	4	1	3	2
The answer should be given from left to right, top to bottom. Separate elements with a space and rows with a comma. Wrap the entire answer in double square brackets.
</example 4>

<example 5>
Grid Layout:
	2	1	2	3	
2	X	X	X	X	3
3	X	X	X	X	2
1	X	X	X	X	3
2	X	X	X	X	1
	2	3	2	1
The answer should be given from left to right, top to bottom. Separate elements with a space and rows with a comma. Wrap the entire answer in double square brackets.
</example 5>

<example 6>
Grid Layout:
	2	3	5	2	1	
3	X	X	X	X	X	1
1	X	X	X	X	X	4
2	X	X	X	X	X	2
4	X	X	X	X	X	2
2	X	X	X	X	X	2
	2	3	1	2	3
The answer should be given from left to right, top to bottom. Separate elements with a space and rows with a comma. Wrap the entire answer in double square brackets.
</example 6>

<example 7>
Grid Layout:
   2  4  3  1  2  
4  X  X  X  X  X  1
1  X  X  X  X  X  2
3  X  X  X  X  X  3
2  X  X  X  X  X  4
3  X  X  X  X  X  5
   3  2  2  2  1  
The answer should be given from left to right, top to bottom. Separate elements with a space and rows with a comma. Wrap the entire answer in double square brackets.
</example 7>

<example 8>
Grid Layout:
	3	4	2	4	1	
3	X	X	X	X	X	1
2	X	X	X	X	X	2
1	X	X	X	X	X	3
3	X	X	X	X	X	2
2	X	X	X	X	X	2
	2	1	3	2	2
The answer should be given from left to right, top to bottom. Separate elements with a space and rows with a comma. Wrap the entire answer in double square brackets.
</example 8>

<example 9>
Grid Layout:
	2	2	3	2	1	
2	X	X	X	X	X	1
4	X	X	X	X	X	2
2	X	X	X	X	X	2
1	X	X	X	X	X	5
2	X	X	X	X	X	3
	2	3	1	2	3
The answer should be given from left to right, top to bottom. Separate elements with a space and rows with a comma. Wrap the entire answer in double square brackets.
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
from bootcamp import Basebootcamp

class KorPuzzleSkyscrapersbootcamp(Basebootcamp):
    def __init__(self, **params):
        super().__init__(**params)
        self.n = params.get('n', 4)  # Default to 4x4 grid
    
    def case_generator(self):
        solution = self.generate_solution()
        left = [self.count_visible(row) for row in solution]
        right = [self.count_visible(reversed(row)) for row in solution]
        top = []
        bottom = []
        for col in range(self.n):
            column = [solution[row][col] for row in range(self.n)]
            top.append(self.count_visible(column))
            bottom.append(self.count_visible(reversed(column)))
        return {
            'n': self.n,
            'top': top,
            'bottom': bottom,
            'left': left,
            'right': right
        }
    
    def generate_solution(self):
        n = self.n
        # Generate base Latin square with shifted rows
        base = [[(i + j) % n + 1 for j in range(n)] for i in range(n)]
        random.shuffle(base)  # Shuffle rows
        
        # Shuffle columns
        cols = list(range(n))
        random.shuffle(cols)
        solution = []
        for row in base:
            new_row = [row[col] for col in cols]
            solution.append(new_row)
        
        # Additional row and column permutations for enhanced randomness
        for _ in range(n):
            i, j = random.sample(range(n), 2)
            solution[i], solution[j] = solution[j], solution[i]
        
        for _ in range(n):
            i, j = random.sample(range(n), 2)
            for row in solution:
                row[i], row[j] = row[j], row[i]
        
        return solution
    
    @staticmethod
    def count_visible(sequence):
        max_h, count = 0, 0
        for num in sequence:
            if num > max_h:
                count += 1
                max_h = num
        return count
    
    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case['n']
        grid_layout = "Grid Layout:\n"
        grid_layout += "\t" + "\t".join(map(str, question_case['top'])) + "\n"
        for i in range(n):
            left = question_case['left'][i]
            right = question_case['right'][i]
            x_part = "\t".join(['X'] * n)
            grid_layout += f"{left}\t{x_part}\t{right}\n"
        grid_layout += "\t" + "\t".join(map(str, question_case['bottom'])) + "\n"
        
        prompt = (
            "You are a city planner trying to arrange skyscrapers on an {n}x{n} grid. Each cell must contain a skyscraper with a height from 1 to {n}.\n"
            "The rules are:\n"
            "1. Each row and column must contain exactly one of each height (1-{n}).\n"
            "2. The numbers around the grid indicate how many skyscrapers are visible from that direction (taller buildings block shorter ones behind them).\n\n"
            "Given the following grid layout with visibility constraints:\n"
            "{grid_layout}\n"
            "Fill in the grid correctly. Format your answer as numbers arranged left to right, top to bottom, each row separated by a comma and space, enclosed in double square brackets.\n"
            "Example: [[1 2 3 4, 2 3 4 1, 3 4 1 2, 4 1 2 3]]\n"
        ).format(n=n, grid_layout=grid_layout)
        return prompt
    
    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[\[(.*?)\]\]', output, re.DOTALL)
        if not matches:
            return None
        last_match = matches[-1].strip()
        try:
            # Handle multi-line formatting
            cleaned = ' '.join(last_match.splitlines()).replace(' , ', ', ').replace(', ', ',')
            rows = [r.strip() for r in cleaned.split(',')]
            solution = []
            for row in rows:
                solution.append([int(num) for num in row.split()])
            return solution
        except Exception as e:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        n = identity['n']
        # Validate solution structure
        if not (isinstance(solution, list) and len(solution) == n and all(len(row) == n for row in solution)):
            return False
        
        # Verify Latin square properties
        expected = list(range(1, n+1))
        for row in solution:
            if sorted(row) != expected:
                return False
        for col in range(n):
            column = [solution[row][col] for row in range(n)]
            if sorted(column) != expected:
                return False
        
        # Verify visibility constraints
        for i in range(n):
            row = solution[i]
            if cls.count_visible(row) != identity['left'][i]:
                return False
            if cls.count_visible(reversed(row)) != identity['right'][i]:
                return False
        
        for j in range(n):
            column = [solution[i][j] for i in range(n)]
            if cls.count_visible(column) != identity['top'][j]:
                return False
            if cls.count_visible(reversed(column)) != identity['bottom'][j]:
                return False
        
        return True
