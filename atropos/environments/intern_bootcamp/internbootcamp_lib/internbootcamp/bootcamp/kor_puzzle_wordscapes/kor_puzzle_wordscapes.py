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
1.The puzzles are played on a grid and the questions are given in the form of a matrix, consisting of X and 0.
2.The player needs to replace X for filling in d letters and 0 for separating words that cannot be filled in with letters.
3.Two lists of words are given, across and down. Across means fill in the words from left to right, and down means fill in the words from top to bottom.
4.During the game, many words will cross each other and share some letters.The letters that cross must match.
5.The question consists of a list of words and a matrix, where X denotes a grid to be filled with letters and 0 denotes a grid that does not need to be filled with letters,e.g.
X X X
0 X 0
0 X 0Example questions are as follows:

<example 0>
across:ACT
down:CAT
X        X        X
0        X        0
0        X        0
The answer should be given from left to right, top to bottom. Separate elements with a space and rows with a comma. Wrap the entire answer in double square brackets.
</example 0>

<example 1>
across: SAD SAVE
down: ADS VASE SAVED
X	X	X	X	0
X	0	X	0	X
X	0	X	X	X
X	0	X	0	X
X	0	0	0	0
The answer should be given from left to right, top to bottom. Separate elements with a space and rows with a comma. Wrap the entire answer in double square brackets.
</example 1>

<example 2>
across:WON
down:NOW OWN
X	0	0
X	X	X
X	0	X
0	0	X
The answer should be given from left to right, top to bottom. Separate elements with a space and rows with a comma. Wrap the entire answer in double square brackets.
</example 2>

<example 3>
across:EAR
down:ARE ERA
X	0	0
X	0	X
X	X	X
0	0	X
The answer should be given from left to right, top to bottom. Separate elements with a space and rows with a comma. Wrap the entire answer in double square brackets.
</example 3>

<example 4>
across:PAT
down:APT TAP
X	0	X
X	0	X
X	X	X
The answer should be given from left to right, top to bottom. Separate elements with a space and rows with a comma. Wrap the entire answer in double square brackets.
</example 4>

<example 5>
across: RID RIP
down:DIP DRIP
0        0        0        0        X        0
X        X        X        0        X        0
0        0        X        X        X        0
0        0        X        0        0        0
0        0        X        0        0        0
The answer should be given from left to right, top to bottom. Separate elements with a space and rows with a comma. Wrap the entire answer in double square brackets.
</example 5>

<example 6>
across:FAR FAIR
down: AIR FIR
0        0        0        0        X        0
0        0        0        0        X        0
0        X        X        X        X        0
0        0        X        0        0        0
X        X        X        0        0        0
The answer should be given from left to right, top to bottom. Separate elements with a space and rows with a comma. Wrap the entire answer in double square brackets.
</example 6>

<example 7>
across:DEN TEN DENT
down: END NET TEND
0        0        X        0        0        X
X        X        X        0        0        X
X        0        X        X        X        X
X        0        0        0        0        0
X        X        X        0        0        0
The answer should be given from left to right, top to bottom. Separate elements with a space and rows with a comma. Wrap the entire answer in double square brackets.
</example 7>

<example 8>
across: ARK PAR
down: RAP PARK
X        0        X        0        0        0
X        0        X        X        X        0
X        X        X        0        0        0
0        0        X        0        0        0
The answer should be given from left to right, top to bottom. Separate elements with a space and rows with a comma. Wrap the entire answer in double square brackets.
</example 8>

<example 9>
aross: LAD LADY
down:DAY LAY
X	X	X	0	X	0
0	0	0	0	X	0
0	X	X	X	X	0
0	0	0	X	0	0
0	0	0	X	0	0
The answer should be given from left to right, top to bottom. Separate elements with a space and rows with a comma. Wrap the entire answer in double square brackets.
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import string
import re

class KorPuzzleWordscapesbootcamp(Basebootcamp):
    def __init__(self, min_size=3, max_size=6):
        self.min_size = min_size
        self.max_size = max_size
    
    def case_generator(self):
        # 生成保证合法尺寸的网格
        rows = random.randint(max(3, self.min_size), self.max_size)
        cols = random.randint(max(3, self.min_size), self.max_size)
        
        def generate_valid_puzzle():
            while True:
                # 生成主横向单词
                main_word = random.choice(['CAT', 'DOG', 'CAR', 'BAT', 'ANT', 'OWL', 'BEE'])
                max_start_col = cols - len(main_word)
                if max_start_col < 0: continue
                start_col = random.randint(0, max_start_col)
                start_row = random.randint(0, rows-1)
                
                # 创建初始网格
                grid = [['0' for _ in range(cols)] for _ in range(rows)]
                for i in range(len(main_word)):
                    grid[start_row][start_col+i] = 'X'
                
                # 计算纵向单词的可行长度
                cross_pos = random.randint(0, len(main_word)-1)
                max_down_length = min(
                    start_row + 1,  # 向上可扩展空间
                    rows - start_row  # 向下可扩展空间
                )
                if max_down_length < 2: continue
                
                # 生成纵向单词（确保包含交叉字母）
                vertical_word = main_word[cross_pos] + ''.join(
                    random.choice(string.ascii_uppercase) 
                    for _ in range(max_down_length-1)
                )
                
                # 检查纵向单词布局的合法性
                valid = True
                vertical_length = len(vertical_word)
                for i in range(vertical_length):
                    r = start_row - cross_pos + i
                    if r < 0 or r >= rows:
                        valid = False
                        break
                    if grid[r][start_col + cross_pos] == 'X' and i != cross_pos:
                        valid = False
                if not valid: continue
                
                # 更新网格布局
                for i in range(vertical_length):
                    r = start_row - cross_pos + i
                    grid[r][start_col + cross_pos] = 'X'
                
                return {
                    "grid": grid,
                    "across": [main_word],
                    "down": [vertical_word],
                    "__solution__": self._generate_solution(
                        grid, main_word, vertical_word, 
                        start_row, start_col, cross_pos
                    )
                }

        return generate_valid_puzzle()
    
    def _generate_solution(self, grid, across_word, down_word, start_row, start_col, cross_pos):
        solution = []
        for row in grid:
            solution.append(['0' if cell == '0' else '_' for cell in row])
        
        # 填充横向单词
        for i, c in enumerate(across_word):
            solution[start_row][start_col+i] = c
        
        # 填充纵向单词
        vertical_start_row = start_row - cross_pos
        for i, c in enumerate(down_word):
            r = vertical_start_row + i
            solution[r][start_col + cross_pos] = c
        
        return solution

    @staticmethod
    def prompt_func(question_case):
        grid = question_case["grid"]
        across = question_case["across"]
        down = question_case["down"]
        
        grid_str = '\n'.join(' '.join(row) for row in grid)
        return f"""Solve this crossword puzzle:
- Grid layout (X=fillable, 0=blocked):
{grid_str}

Word lists:
- Across: {', '.join(across)}
- Down: {', '.join(down)}

Format your answer as space-separated values per row, comma-separated rows enclosed in double square brackets.
Example: [[A B 0, 0 C D]]"""

    @staticmethod
    def extract_output(output):
        matches = re.findall(r'\[\[(.*?)\]\]', output, re.DOTALL)
        if not matches:
            return None
        try:
            last_match = matches[-1].strip()
            return [row.strip().split() for row in last_match.split(',')]
        except:
            return None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            expected = identity["__solution__"]
            return solution == expected
        except:
            return False
