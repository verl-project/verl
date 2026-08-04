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

**Stitches Puzzle Rules**

1. **Objective**: Create a single continuous, non-intersecting loop by drawing horizontal/vertical \"stitches\" (line segments) between adjacent dots on a grid.

2. **Grid Structure**: 
   - Dots are arranged in a grid (e.g., square lattice).
   - Some dots contain numbers (0–3), indicating how many stitches must connect to them.

3. **Stitches**:
   - A stitch is a straight line between two orthogonally adjacent dots.
   - Stitches cannot cross, branch, or overlap.

4. **Key Rules**:
   - **Numbered Dots**: A dot with a number **N** must have exactly **N stitches** connected to it.
   - **Unnumbered Dots**: May have 0–2 stitches (default: 2 if part of the loop).
   - **Loop Requirement**: All stitches must form a single closed loop. Every dot in the loop must have exactly 2 stitches (entering/exiting), except numbered dots with values other than 2 (e.g., 0, 1, 3), which override this rule.

5. **Special Cases**:
   - **0**: The dot cannot be part of the loop (no stitches).
   - **1**: The dot is an endpoint (dead end), but this contradicts loop closure. Thus, **1s are typically invalid in classic loop rules** and may indicate edge-case mechanics (e.g., partial lines).
   - **3**: The dot acts as a \"branch,\" but this is prohibited in standard loop puzzles. Such clues may signal unique logic (e.g., overlapping regions or special constraints).

6. **Logic**:
   - Use numbers to deduce which dots must connect to others while ensuring the loop remains continuous and uncrossed.
   - Dots with **2** must lie on the loop; dots with **0** must be isolated.


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import random
from collections import defaultdict

class Stitchesbootcamp(Basebootcamp):
    def __init__(self, rows=5, cols=5):
        self.rows = rows
        self.cols = cols
    
    def case_generator(self):
        numbered_cells = []
        # Generate outer perimeter as solution
        perimeter_points = self._get_perimeter_points()
        # Add numbered cells: select some perimeter points as 2, some inner points as 0
        for x in range(self.rows):
            for y in range(self.cols):
                if (x, y) in perimeter_points:
                    if random.random() < 0.3:  # 30% chance to mark as 2
                        numbered_cells.append({'x': x, 'y': y, 'num': 2})
                else:
                    if random.random() < 0.1:  # 10% chance to mark inner as 0
                        numbered_cells.append({'x': x, 'y': y, 'num': 0})
        return {
            'rows': self.rows,
            'cols': self.cols,
            'numbered_cells': numbered_cells
        }
    
    def _get_perimeter_points(self):
        points = set()
        for x in [0, self.rows-1]:
            for y in range(self.cols):
                points.add((x, y))
        for y in [0, self.cols-1]:
            for x in range(1, self.rows-1):
                points.add((x, y))
        return points
    
    @staticmethod
    def prompt_func(question_case) -> str:
        rows = question_case['rows']
        cols = question_case['cols']
        cells = question_case['numbered_cells']
        cells_desc = '\n'.join([f"- 坐标 ({c['x']}, {c['y']}) 的数值为 {c['num']}" for c in cells])
        return f"""你是一个Stitches Puzzle解题专家，请根据以下规则解决谜题：

**规则说明**
1. 目标：在{rows}x{cols}的点阵中绘制水平/垂直缝线，形成**唯一闭合环**（无交叉、无分支）。
2. 数字规则：
   - 数字N表示该点必须连接N条缝线
   - 数值0必须无连接，数值2必须连接两条
3. 未标数字的点属于环时必须有2条缝线
4. 缝线必须形成连续闭合环，所有点至多属于一个环

**当前谜题**
数字点列表（坐标从0开始）：
{cells_desc if cells else "无数字点"}

**答案格式**
请将答案包含在[answer]和[/answer]之间，格式为：
[[(x1,y1),(x2,y2)], [(x3,y3),(x4,y4)], ...]
确保每个缝线为相邻点坐标，如示例所示。"""

    @staticmethod
    def extract_output(output):
        import re
        import ast
        # Find last answer block
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', output, re.DOTALL)
        if not matches:
            return None
        last_answer = matches[-1].strip()
        try:
            solution = ast.literal_eval(last_answer)
            if not isinstance(solution, list):
                return None
            for stitch in solution:
                if len(stitch) != 2 or not all(isinstance(p, tuple) and len(p)==2 for p in stitch):
                    return None
            return solution
        except:
            return None
    
    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            rows = identity['rows']
            cols = identity['cols']
            numbered_cells = {(c['x'], c['y']): c['num'] for c in identity['numbered_cells']}
            
            # 验证基本结构
            stitches = set()
            for stitch in solution:
                # 验证缝线格式
                if len(stitch) != 2:
                    return False
                p1, p2 = stitch
                if not (isinstance(p1, tuple) and isinstance(p2, tuple) and len(p1)==2 and len(p2)==2):
                    return False
                x1, y1 = p1
                x2, y2 = p2
                # 验证坐标有效性
                if not (0 <= x1 < rows and 0 <= y1 < cols and 0 <= x2 < rows and 0 <= y2 < cols):
                    return False
                # 验证相邻性
                dx = abs(x1 - x2)
                dy = abs(y1 - y2)
                if not ((dx == 1 and dy == 0) or (dy == 1 and dx == 0)):
                    return False
                # 标准化缝线存储
                stitches.add(frozenset({p1, p2}))
            
            # 构建邻接表
            graph = defaultdict(list)
            for s in stitches:
                p1, p2 = s
                graph[p1].append(p2)
                graph[p2].append(p1)
            
            # 检查数字点约束
            for (x, y), num in numbered_cells.items():
                actual = len(graph.get((x, y), []))
                if actual != num:
                    return False
            
            # 检查所有节点的度数
            visited = set()
            for node in graph:
                # 处理未访问节点
                if node in visited:
                    continue
                # 检查是否为合法环结构
                current = node
                prev = None
                path = []
                while True:
                    next_nodes = [n for n in graph[current] if n != prev]
                    if len(next_nodes) != 1:
                        break  # 分支或末端
                    prev, current = current, next_nodes[0]
                    if current == node:  # 闭环检查
                        break
                    path.append(current)
                    if current in visited:
                        return False
                    visited.add(current)
                # 验证是否形成闭环
                if current != node or len(graph[node]) != 2:
                    return False
            
            # 检查未编号点度数
            for node in graph:
                if node not in numbered_cells:
                    if len(graph[node]) not in (0, 2):
                        return False
            
            # 检查单一环
            # 确保所有连接点被访问且构成单个环
            component = []
            stack = []
            if graph:
                start = next(iter(graph))
                stack.append(start)
                visited_nodes = set()
                while stack:
                    node = stack.pop()
                    if node in visited_nodes:
                        continue
                    visited_nodes.add(node)
                    for neighbor in graph[node]:
                        if neighbor not in visited_nodes:
                            stack.append(neighbor)
                if len(visited_nodes) != len(graph):
                    return False
            return True
        except:
            return False
