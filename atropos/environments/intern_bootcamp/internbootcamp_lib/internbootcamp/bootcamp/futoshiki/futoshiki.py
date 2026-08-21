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

Futoshiki is a logic puzzle played on an N×N grid where the objective is to fill each cell with a unique number from 1 to N, adhering to three core principles:

1. **Row and Column Uniqueness**:  
   Every row and column must contain all numbers from 1 to N exactly once, with no repetitions (similar to Sudoku).

2. **Inequality Constraints**:  
   Between certain pairs of adjacent cells (horizontally or vertically), there are inequality symbols (< or >). These symbols enforce relationships:  
   - **A > B** means the number in cell A must be greater than the number in cell B.  
   - **A < B** means the number in cell A must be less than the number in cell B.  
   The direction of the symbol determines the comparison (e.g., a \">\" between two horizontal cells means the left cell is greater; a \">\" between vertical cells means the upper cell is greater).

3. **Deductive Logic**:  
   The puzzle begins with some numbers pre-filled and inequalities provided. Solvers must deduce the remaining numbers by ensuring all uniqueness and inequality rules are satisfied simultaneously. Inequalities override potential number placements even if row/column uniqueness is temporarily met.

The challenge lies in balancing the constraints of uniqueness and inequalities to uniquely determine the solution.


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
import re
import ast
from typing import List, Dict, Any

class Futoshikibootcamp(Basebootcamp):
    def __init__(self, size=5, inequality_prob=0.3, retain_ratio=0.3):
        self.size = size
        self.inequality_prob = inequality_prob
        self.retain_ratio = retain_ratio

    def case_generator(self) -> dict:
        """生成带有唯一解的Futoshiki谜题实例"""
        solution = self._generate_latin_square()
        inequalities = self._generate_inequalities(solution)
        initial = self._generate_initial_grid(solution)
        return {
            'size': self.size,
            'initial': initial,
            'inequalities': inequalities
        }

    def _generate_latin_square(self) -> List[List[int]]:
        """生成随机拉丁方阵作为解"""
        n = self.size
        base_row = list(range(1, n+1))
        random.shuffle(base_row)
        rows = [base_row[i:] + base_row[:i] for i in range(n)]
        random.shuffle(rows)
        perm = list(range(1, n+1))
        random.shuffle(perm)
        return [[perm[x-1] for x in row] for row in rows]

    def _generate_inequalities(self, solution: List[List[int]]) -> List[Dict]:
        """根据解生成不等式约束"""
        inequalities = []
        for i in range(self.size):
            for j in range(self.size):
                if j+1 < self.size and random.random() < self.inequality_prob:
                    a, b = solution[i][j], solution[i][j+1]
                    inequalities.append({
                        'cell1': [i, j],
                        'cell2': [i, j+1],
                        'symbol': '>' if a > b else '<'
                    })
                if i+1 < self.size and random.random() < self.inequality_prob:
                    a, b = solution[i][j], solution[i+1][j]
                    inequalities.append({
                        'cell1': [i, j],
                        'cell2': [i+1, j],
                        'symbol': '>' if a > b else '<'
                    })
        return inequalities

    def _generate_initial_grid(self, solution: List[List[int]]) -> List[List[int]]:
        """生成初始谜题网格"""
        n = self.size
        indices = [(i, j) for i in range(n) for j in range(n)]
        retain_num = int(len(indices) * self.retain_ratio)
        selected = random.sample(indices, retain_num)
        grid = [[0]*n for _ in range(n)]
        for i, j in selected:
            grid[i][j] = solution[i][j]
        return grid

    @staticmethod
    def prompt_func(case) -> str:
        """生成面向用户的自然语言问题描述"""
        prompt = [
            "你是Futoshiki谜题专家，请根据以下条件解开谜题：",
            "\n规则说明：",
            "1. 填充1到N的整数（N为网格大小），满足：",
            "   - 每行和每列数字不重复",
            "   - 遵守所有不等式约束（>表示左边/上边数字更大）",
            f"\n初始网格（{case['size']}x{case['size']}，0表示空格）:"
        ]
        
        # 添加网格可视化
        for row in case['initial']:
            prompt.append("[" + " ".join(str(n) if n != 0 else "_" for n in row) + "]")
        
        # 添加不等式描述
        prompt.append("\n不等式约束：")
        for idx, ineq in enumerate(case['inequalities'], 1):
            c1, c2 = ineq['cell1'], ineq['cell2']
            direction = '右边' if c1[1]+1 == c2[1] else '下方'
            prompt.append(
                f"{idx}. 单元格({c1[0]}, {c1[1]}) {direction}的单元格应满足: "
                f"{ineq['symbol']}"
            )
        
        prompt.append(
            "\n将完整解答的二维数组放在[answer]标签内，例如：\n"
            "[answer]\n"
            "[[1,2,3],\n[2,3,1],\n[3,1,2]]\n"
            "[/answer]"
        )
        return "\n".join(prompt)

    @staticmethod
    def extract_output(output: str) -> List[List[int]]:
        """从模型输出中提取最后一个答案块"""
        answer_blocks = re.findall(
            r'\[answer\](.*?)\[/answer\]', 
            output, 
            re.DOTALL
        )
        if not answer_blocks:
            return None

        try:
            # 尝试解析最后一个答案块
            raw_answer = answer_blocks[-1].strip()
            return ast.literal_eval(raw_answer)
        except (SyntaxError, ValueError):
            return None

    @classmethod
    def _verify_correction(cls, solution: List[List[int]], case: dict) -> bool:
        """完整验证解的三个核心条件"""
        n = case['size']
        initial = case['initial']
        inequalities = case['inequalities']

        # 基础结构验证
        if len(solution) != n or any(len(row)!=n for row in solution):
            return False

        # 预填数字验证
        for i in range(n):
            for j in range(n):
                if initial[i][j] != 0 and solution[i][j] != initial[i][j]:
                    return False

        # 行列唯一性验证
        valid_numbers = set(range(1, n+1))
        for row in solution:
            if set(row) != valid_numbers:
                return False
        for col in zip(*solution):
            if set(col) != valid_numbers:
                return False

        # 不等式验证
        for ineq in inequalities:
            i1, j1 = ineq['cell1']
            i2, j2 = ineq['cell2']
            a, b = solution[i1][j1], solution[i2][j2]
            if not (a > b if ineq['symbol'] == '>' else a < b):
                return False

        return True
