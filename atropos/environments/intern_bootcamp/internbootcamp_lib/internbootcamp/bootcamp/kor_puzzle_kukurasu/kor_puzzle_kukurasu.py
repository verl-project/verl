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
1.A rectangular grid is given with a number at the end of each row and column indicating the sum of the weights of the filled black cells in that row or column.
2.In column i, the weight of the black grid is equal to its position in that column (i.e., 1 through n). Similarly, the weight of the black grid in row j is equal to its position in that row (i.e. 1 through n).
3.The player needs to blacken a number of cells in the grid such that the sum of the weights of the black cells in each row and column is equal to the given numbers.
4.The problem is given by a matrix in the form of a blank grid filled with X. Below and to the right of the matrix are attached the numerical constraints mentioned above. The player replaces the grid to be blacked out with 1.Example questions are as follows:

<example 0>
X  X  X  X  4
X  X  X  X  8
X  X  X  X  7
X  X  X  X  6
9  7  6  6  

The final answer should be given in order from left to right, top to bottom with each element separated by a space and different lines separated by \",\". Wrap your final answer in double square brackets, like this: [[your answer]].
</example 0>

<example 1>
X  X  X  X  5
X  X  X  X  7
X  X  X  X  6
X  X  X  X  5
3  9  4  6  

The final answer should be given in order from left to right, top to bottom with each element separated by a space and different lines separated by \",\". Wrap your final answer in double square brackets, like this: [[your answer]].
</example 1>

<example 2>
X  X  X  X  5
X  X  X  X  2
X  X  X  X  9
X  X  X  X  1
5  5  3  4      

The final answer should be given in order from left to right, top to bottom with each element separated by a space and different lines separated by \",\". Wrap your final answer in double square brackets, like this: [[your answer]].
</example 2>

<example 3>
X  X  X  X  2
X  X  X  X  9
X  X  X  X  6
X  X  X  X  6
4  10  6  5  

The final answer should be given in order from left to right, top to bottom with each element separated by a space and different lines separated by \",\". Wrap your final answer in double square brackets, like this: [[your answer]].
</example 3>

<example 4>
X  X  X  X  1
X  X  X  X  4
X  X  X  X  3
X  X  X  X  6
5  4  7  2

The final answer should be given in order from left to right, top to bottom with each element separated by a space and different lines separated by \",\". Wrap your final answer in double square brackets, like this: [[your answer]].
</example 4>

<example 5>
X  X  X  X  X  5
X  X  X  X  X  1
X  X  X  X  X  5
X  X  X  X  X  5
X  X  X  X  X  4
2  1  1  5  7  

The final answer should be given in order from left to right, top to bottom with each element separated by a space and different lines separated by \",\". Wrap your final answer in double square brackets, like this: [[your answer]].
</example 5>

<example 6>
X  X  X  X  X  13
X  X  X  X  X  3
X  X  X  X  X  14
X  X  X  X  X  13
X  X  X  X  X  12
5  3  15  13  13  

The final answer should be given in order from left to right, top to bottom with each element separated by a space and different lines separated by \",\". Wrap your final answer in double square brackets, like this: [[your answer]].
</example 6>

<example 7>
X  X  X  X  X  X  16
X  X  X  X  X  X  12
X  X  X  X  X  X  16
X  X  X  X  X  X  15
X  X  X  X  X  X  18
X  X  X  X  X  X  10
15  19  14  13  17  11

The final answer should be given in order from left to right, top to bottom with each element separated by a space and different lines separated by \",\". Wrap your final answer in double square brackets, like this: [[your answer]].
</example 7>

<example 8>
X  X  X  X  X  X  12
X  X  X  X  X  X  4
X  X  X  X  X  X  10
X  X  X  X  X  X  12
X  X  X  X  X  X  3
3  4  13  7  1  8

The final answer should be given in order from left to right, top to bottom with each element separated by a space and different lines separated by \",\". Wrap your final answer in double square brackets, like this: [[your answer]].
</example 8>

<example 9>
X  X  X  X  X  X  5
X  X  X  X  X  X  17
X  X  X  X  X  X  9
X  X  X  X  X  X  13
X  X  X  X  X  X  9
X  X  X  X  X  X  8
14  10  12  4  13  10

The final answer should be given in order from left to right, top to bottom with each element separated by a space and different lines separated by \",\". Wrap your final answer in double square brackets, like this: [[your answer]].
</example 9>


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
from bootcamp import Basebootcamp
import random
import re

class KorPuzzleKukurasubootcamp(Basebootcamp):
    def __init__(self, min_size=4, max_size=7):
        self.min_size = min_size
        self.max_size = max_size

    def case_generator(self):
        """生成保证有解且具备可玩性的谜题"""
        while True:
            n = random.randint(self.min_size, self.max_size)
            grid = self._generate_valid_grid(n)
            
            row_sums = [sum(j+1 for j in range(n) if grid[i][j]) for i in range(n)]
            col_sums = [sum(i+1 for i in range(n) if grid[i][j]) for j in range(n)]
            
            # 确保至少每行/列有1个填充
            if 0 not in row_sums and 0 not in col_sums:
                return {
                    "row_sums": row_sums,
                    "col_sums": col_sums,
                    "size": n,
                    "solution": grid
                }

    def _generate_valid_grid(self, n):
        """生成有效初始解"""
        grid = []
        for _ in range(n):
            # 动态调整填充概率
            base_prob = 0.3 + random.random()*0.4  # 30%-70%
            grid.append([
                1 if random.random() < base_prob else 0 
                for _ in range(n)
            ])
        return grid

    @staticmethod
    def prompt_func(question_case) -> str:
        n = question_case["size"]
        row_str = "\n".join(
            " ".join(["X"]*n) + f" {s}" 
            for s in question_case["row_sums"]
        )
        col_str = " ".join(map(str, question_case["col_sums"]))
        
        return f"""你是一个数学谜题专家，需要解决以下网格填充问题：

网格规格：{n}x{n}正方形网格
数值说明：
- 每行右侧数字表示该行黑格子的列坐标之和（列号从左到右为1~{n}）
- 底部数字序列表示各列黑格子的行坐标之和（行号从上到下为1~{n}）

当前谜题：
{row_str}
{col_str}

答案要求：
1. 用0表示白格，1表示黑格
2. 各行数字用空格连接，行间用英文逗号分隔
3. 将最终答案包含在双中括号内

示例（4x4）：
[[1 0 0 0, 0 1 1 1, 1 0 1 0, 0 1 0 1]]"""

    @staticmethod 
    def extract_output(output):
        # 兼容中文括号和超长上下文
        pattern = r'\[{2}[^\[\]]*\]{2}'
        matches = re.findall(pattern, output)
        if not matches:
            return None
            
        last_match = matches[-1].strip('[]')
        # 统一处理中文标点
        processed = re.sub(r'[，]', ',', last_match)
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        # 二次清洗
        rows = []
        for r in processed.split(','):
            clean_row = re.sub(r'[^\d\s]', '', r).strip()
            if clean_row:
                rows.append(clean_row)
                
        return ', '.join(rows) if rows else None

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            n = identity["size"]
            # 格式验证
            if not re.fullmatch(r'([01] )+[01](, ([01] )+[01])*', solution):
                return False
                
            grid = []
            for row in solution.split(', '):
                elements = list(map(int, row.split()))
                if len(elements) != n:
                    return False
                grid.append(elements)
            
            # 数学验证
            row_valid = all(
                sum(j+1 for j in range(n) if grid[i][j]) == identity["row_sums"][i]
                for i in range(n)
            )
            col_valid = all(
                sum(i+1 for i in range(n) if grid[i][j]) == identity["col_sums"][j]
                for j in range(n)
            )
            
            return row_valid and col_valid
        except:
            return False
