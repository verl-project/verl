import re
import json
import random
from internbootcamp.bootcamp.base import Basebootcamp
from internbootcamp.libs.starbattle.get_grid import generate_star_battle_grid
from internbootcamp.libs.starbattle.dfs_solver_cn import print_grid_in_kor

class Starbattlebootcamp(Basebootcamp):

    def __init__(self, size=5):
        self.size = size

    def generator(self):
        self.grid, self.star_positions = generate_star_battle_grid(self.size)
        return self.grid

    def case_generator(self, size:tuple = (8,8), expect_camp_number:int = 8, random_rate:float = 0.1 , seed:int = None):
        grid = self.generator()

        return {
            "input_grid": grid
        }

    
    def prompt_func(self, identity) -> str:
        """
        Process the input_data and return the processed prompt.
        
        Args:
            question_ori: The question to be processed.
        
        Returns:
            str: The processed prompt.
        """

        statements = [f"""### 核心职责:

1. 规则分析:
   - 解码并理解所提供的谜题规则 
   - 创建一个系统化的规则实施方法
   - 识别潜在的规则相互作用或依赖关系

2. 解决方案开发:
   - 有条理地将规则应用于谜题场景
   - 开发分步解决策略
   - 在整个解题过程中保持准确性

3. 质量保证:
   - 反复检查解决方案的有效性
   - 确保遵守所有规定规则
   - 根据初始条件验证最终结果

### 谜题参数:
1. 网格结构: 游戏区域由不同的区域(分区)组成,每个区域包含多个方格。

2. 星星放置指南:
   - 星星必须均匀分布,每行每列1个星星
   - 每个分区需要恰好1个星星
   - 星星必须保持分离(不能相邻,包括对角线)

3. 输入格式:
   - 采用字母指定区域的矩阵表示
   - 每个独特字母代表一个不同的分区 
   - 示例: A区域包括所有标记为'A'的方格

4. 解决方案格式:
   - 基于坐标的报告系统
   - 格式: [区域字母]:(行坐标,列坐标)
   - 每个区域根据需要列出多个坐标
   
   ###Question

   初始网格为:\n {print_grid_in_kor(identity['input_grid'])}

   """,
   f"""star battle是一种逻辑解谜游戏，其规则简单，解题过程富有挑战性。
游戏规则 很简单。
按如下要求在格子上放置星星：
- 任意两颗星星不能在横向、纵向或对角上相邻。
- 每行、每列及每个区域上需放置1颗星星。
网格由矩阵表示，每个字母表示该位置所在区域的ID
请完成该star battle，初始网格为:\n {print_grid_in_kor(identity['input_grid'])}"""]
        
        instruction_following = """Please note that the coordinate system used in this task starts from (1, 1). This means:
1. The first row and first column are both numbered 1, not 0.
2. In any coordinate (x, y), x represents the row number, and y represents the column number, with both x and y having a minimum value of 1.
3. When processing coordinate data, ensure all calculations and logic are based on (1, 1) as the starting point to avoid errors caused by misunderstandings of the coordinate system.

Let's think step by step and output the final answer with an example json formatting for a 5x5 board: 
Final-answer: ```json
{'A': [(row_a, col_a)],'B': [(row_b, col_b)],'C': [(row_c, col_c)],'D': [(row_d, col_d)],'E': [(row_e, col_e)]}
```"""
        prompt = statements[random.randint(0,len(statements)-1)] + '\n' + instruction_following
        return prompt
    
    @staticmethod
    def extract_output(output):
        """
        Extract the output from the solution.
        
        Args:
            output: Model output to be processed.
        
        Returns:
            The processed output.
        """
        pattern = pattern = r'```json\s*(\{[\s\S]*?\})\s*```'
        matches = re.findall(pattern, output)
        # Final-answer:{'A': [(3, 1)],
        # 'B': [(1, 4)],
        # 'C': [(2, 3)],
        # 'D': [(5, 2)],
        # 'E': [(4, 5)]}
        if matches:
            # 获取 JSON 字符串
            json_str = matches[-1]
            # print('match?', json_str)
            # print('solution generated? first lines', output[:200])
            # print('solution generated? last lines', output[-200:])
            # 替换单引号为双引号，将元组表示改为列表表示
            json_str = json_str.replace("'", '"').replace("(", "[").replace(")", "]")
            try:
                # 解析 JSON 字符串为 Python 字典
                result_dict = json.loads(json_str)
                return result_dict
            except json.JSONDecodeError as e:
                return json_str
        else:
            return None
        
    @classmethod 
    def _verify_correction(cls, solution, identuty:dict)->bool:
        """
        Verify the correction of the solution.
        """ 
        input_grid = json.loads(identuty['input_grid']) if type(identuty['input_grid']) == str else identuty['input_grid']
        n = identuty.pop('n', 1)
        # Check each star is in the correct region
        for region, coords in solution.items():
            for (row, col) in coords:
                if row < 1 or row > len(input_grid) or col < 1 or col > len(input_grid[0]):
                    return False
                if input_grid[row-1][col-1] != region:
                    return False
        
        # Check each region has exactly n stars
        for region, coords in solution.items():
            if len(coords) != n:
                return False
        
        # Collect all stars and check row/column counts
        all_stars = [coord for coords in solution.values() for coord in coords]
        rows = {}
        cols = {}
        for (r, c) in all_stars:
            rows[r] = rows.get(r, 0) + 1
            cols[c] = cols.get(c, 0) + 1
        if any(v != n for v in rows.values()) or any(v != n for v in cols.values()):
            return False
        
        # Check adjacency
        for i in range(len(all_stars)):
            for j in range(i+1, len(all_stars)):
                r1, c1 = all_stars[i]
                r2, c2 = all_stars[j]
                if abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1:
                    return False
        
        return True
        
        

if __name__ == '__main__':
    model_output = """ at (5,7).\n\nThus, Row 4 has no possible squares for a star.\n\nThis suggests that it's impossible to place stars in this configuration without violating the rules.\n\n### Conclusion\n\nAfter several iterations and attempts to place stars in each zone while adhering to the rules, it appears that there's an inherent conflict that prevents a valid solution. Specifically, Row 4 consistently lacks a suitable square for a star without violating the adjacency or zone constraints.\n\nTherefore, the puzzle, as presented, does not have a valid solution under the given rules.\n\nHowever, if we assume that there might have been a misunderstanding or a typo in the grid layout or zone definitions, we could revisit the initial assumptions. But based on the provided information and thorough analysis, it's concluded that the puzzle is unsolvable.\n\n**Final Answer**:\n\n```json\n{}\n```"""
    identity = {"input_grid": [["E", "E", "F", "F", "F", "F", "F"], ["E", "E", "D", "D", "F", "F", "F"], ["E", "E", "A", "A", "F", "F", "F"], ["E", "A", "A", "A", "F", "F", "G"], ["E", "A", "A", "A", "G", "G", "G"], ["B", "B", "B", "A", "A", "A", "C"], ["B", "B", "B", "B", "A", "A", "C"]]}
    
    score = StarBattlebootcamp.verify_score(model_output=model_output, identity=identity)
    print(score)