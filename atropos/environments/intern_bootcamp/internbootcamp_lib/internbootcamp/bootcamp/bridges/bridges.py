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

The objective of the Bridges puzzle (Hashiwokakero) is to connect all numbered \"islands\" on a grid using horizontal/vertical bridges, adhering to these principles:

1. **Island Numbers**: Each island (node) displays a number (1-8) indicating how many bridges must connect to it.  
   - Example: A \"3\" island must have exactly 3 bridges linked to it.

2. **Bridge Placement**:  
   - Bridges connect **two adjacent islands** horizontally/vertically.  
   - Bridges cannot cross islands, other bridges, or \"turn\" mid-connection.  

3. **Bridge Limits**:  
   - A maximum of **2 bridges** can connect any pair of islands.  
   - Bridges may overlap in straight lines if they connect different island pairs (no crossing).  

4. **Connectivity**:  
   - All islands must be interconnected into a **single continuous network** via bridges.  

Key Constraints:  
- Bridges cannot be placed diagonally.  
- Islands cannot have fewer/more bridges than their number.  
- Overlapping bridges (parallel lines) must align in the same direction without intersecting.  


请完成上述谜题的训练场环境类实现，包括所有必要的方法。
"""

from bootcamp import Basebootcamp
import re
import random
from collections import defaultdict

class Bridgesbootcamp(Basebootcamp):
    def __init__(self, width=5, height=5):
        self.width = width
        self.height = height

    def case_generator(self):
        direction = random.choice(['horizontal', 'vertical'])
        islands = []
        bridges = []

        if direction == 'horizontal':
            x = random.randint(0, self.width-1)
            y1 = random.randint(0, self.height-3)
            y2 = y1 + 2
            islands = [
                {'x': x, 'y': y1, 'num': 2},
                {'x': x, 'y': y2, 'num': 2},
            ]
            bridges = [{'from': (x, y1), 'to': (x, y2), 'count': 2}]
        else:
            y = random.randint(0, self.height-1)
            x1 = random.randint(0, self.width-3)
            x2 = x1 + 2
            islands = [
                {'x': x1, 'y': y, 'num': 2},
                {'x': x2, 'y': y, 'num': 2},
            ]
            bridges = [{'from': (x1, y), 'to': (x2, y), 'count': 2}]

        return {
            'islands': islands,
            'bridges': bridges
        }

    @staticmethod
    def prompt_func(question_case) -> str:
        islands = question_case['islands']
        islands_desc = []
        for island in islands:
            islands_desc.append(f"坐标({island['x']}, {island['y']})的岛屿数字为{island['num']}。")
        islands_text = '\n'.join(islands_desc)
        prompt = f"""你是Hashiwokakero谜题的解题专家，请根据以下规则连接所有岛屿：

规则：
1. 每个岛屿上的数字表示必须连接的桥梁数目。
2. 桥梁必须水平或竖直连接相邻的岛屿，中间不能有其他岛屿或桥梁阻挡。
3. 每对岛屿之间最多可以建造两座桥梁。
4. 所有岛屿必须通过桥梁连通成一个单一网络。
5. 桥梁不能交叉或转弯。

当前的岛屿分布如下：
{islands_text}

请建造桥梁以满足所有条件，并将答案按照以下格式放置于[answer]和[/answer]之间。每个桥梁的格式为：(x1,y1)-(x2,y2):数量，多个桥梁用换行分隔。

示例：
[answer]
(0,0)-(0,2):2
(1,3)-(3,3):1
[/answer]"""
        return prompt

    @staticmethod
    def extract_output(output):
        pattern = re.compile(r'\[answer\](.*?)\[/answer\]', re.DOTALL)
        matches = pattern.findall(output)
        if not matches:
            return None
        return matches[-1].strip()

    @classmethod
    def _verify_correction(cls, solution, identity):
        try:
            bridges = []
            pattern = re.compile(r'\((\d+)\s*,\s*(\d+)\)\s*-\s*\((\d+)\s*,\s*(\d+)\)\s*:\s*(\d+)')
            matches = pattern.findall(solution)
            for m in matches:
                x1, y1, x2, y2, cnt = map(int, m)
                if cnt not in (1, 2):
                    return False
                if (x1, y1) > (x2, y2):
                    x1, x2, y1, y2 = x2, x1, y2, y1
                bridges.append({'from': (x1, y1), 'to': (x2, y2), 'count': cnt})

            island_coords = {(i['x'], i['y']) for i in identity['islands']}
            for bridge in bridges:
                if bridge['from'] not in island_coords or bridge['to'] not in island_coords:
                    return False
                x1, y1 = bridge['from']
                x2, y2 = bridge['to']
                if not (x1 == x2 or y1 == y2):
                    return False

                if x1 == x2:
                    y_min, y_max = sorted([y1, y2])
                    for y in range(y_min+1, y_max):
                        if (x1, y) in island_coords:
                            return False
                else:
                    x_min, x_max = sorted([x1, x2])
                    for x in range(x_min+1, x_max):
                        if (x, y1) in island_coords:
                            return False

            bridge_counts = defaultdict(int)
            for bridge in bridges:
                pair = (bridge['from'], bridge['to'])
                bridge_counts[pair] += bridge['count']
            if any(v > 2 for v in bridge_counts.values()):
                return False

            island_num = {(i['x'], i['y']): i['num'] for i in identity['islands']}
            usage = defaultdict(int)
            for bridge in bridges:
                usage[bridge['from']] += bridge['count']
                usage[bridge['to']] += bridge['count']
            for coord, num in island_num.items():
                if usage.get(coord, 0) != num:
                    return False

            bridges_path = []
            for bridge in bridges:
                x1, y1 = bridge['from']
                x2, y2 = bridge['to']
                if x1 == x2:
                    y_start, y_end = sorted([y1, y2])
                    bridges_path.append(('vertical', x1, y_start, y_end))
                else:
                    x_start, x_end = sorted([x1, x2])
                    bridges_path.append(('horizontal', y1, x_start, x_end))

            for i in range(len(bridges_path)):
                ti, ai, si, ei = bridges_path[i]
                for j in range(i+1, len(bridges_path)):
                    tj, aj, sj, ej = bridges_path[j]
                    if ti == tj:
                        continue
                    if ti == 'horizontal':
                        y_h = ai
                        xh_s, xh_e = si, ej
                        x_v = aj
                        yv_s, yv_e = sj, ej
                    else:
                        x_v = ai
                        yv_s, yv_e = si, ei
                        y_h = aj
                        xh_s, xh_e = sj, ej
                    if (xh_s <= x_v <= xh_e) and (yv_s <= y_h <= yv_e):
                        return False

            coord_to_id = {(i['x'], i['y']): idx for idx, i in enumerate(identity['islands'])}
            parent = list(range(len(coord_to_id)))

            def find(u):
                while parent[u] != u:
                    parent[u] = parent[parent[u]]
                    u = parent[u]
                return u

            def union(u, v):
                pu, pv = find(u), find(v)
                if pu != pv:
                    parent[pu] = pv

            for bridge in bridges:
                u = coord_to_id[bridge['from']]
                v = coord_to_id[bridge['to']]
                union(u, v)

            roots = {find(i) for i in range(len(parent))}
            return len(roots) == 1

        except Exception as e:
            return False
