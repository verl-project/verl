from bootcamp.base import Basebootcamp
import re
import random
from typing import List, Tuple, Dict, Set, Optional

class InternGObootcamp(Basebootcamp):
    def __init__(self, min_length=8, max_length=20, error_prob=0.8):
        """
        初始化Dyck语言训练场
        
        参数:
            min_length: 生成序列的最小长度
            max_length: 生成序列的最大长度
            error_prob: 注入错误的概率
        """
        self.min_length = min_length
        self.max_length = max_length
        self.error_prob = error_prob
        self.bracket_pairs = {'(': ')', '[': ']', '{': '}', '<': '>'}
        self.open_brackets = set(self.bracket_pairs.keys())
        self.close_brackets = set(self.bracket_pairs.values())
        
    def case_generator(self) -> Dict:
        """
        GO bootcamp暂时没有case generator。
        """

        pass 
    
    def prompt_func(self, identity: Dict) -> str:
        """
        生成提示
        """
        
        pass 
    
    @staticmethod
    def extract_output(output: str) -> Optional[str]:
        
        """
        从模型输出中提取坐标和颜色
        处理步骤：
        1. 提取assistant回复部分
        2. 检查是否存在<think>标签
        3. 从<answer>标签或原始格式中提取坐标和颜色
        
        期望输出形式 : 
        <reasoning>
        先分析当前局面：黑棋刚刚下了Q3，这是一个试探性的手段，... ...
        </reasoning>
        
        <answer>
        \\boxed{下一步颜色:白}
        \\boxed{下一步位置:Q4}
        </answer>
        """
        
        ASSISTANT_PATTERN = re.compile(r'<\|im_start\|>assistant\n(.*)', re.DOTALL) # 跳过system_prompt
        REASONING_PATTERN = re.compile(r'<reasoning>(.*?)</reasoning>', re.DOTALL)
        try:
            # 提取assistant回复部分
            assistant_match = ASSISTANT_PATTERN.search(output)
            # print(f"assistant_match: {assistant_match}")
            if not assistant_match:
                content = output
            else :
                content = assistant_match.group(1)
            
            # # 检查是否存在<think>标签
            # has_think = bool(re.search(r'<think>.*?</think>', content, re.DOTALL))
            
            # 检查是否存在<reasoning>标签
            has_think = bool(REASONING_PATTERN.search(content))
            
            # 提取<reasoning>中包含的内容
            reasoning_match = REASONING_PATTERN.search(content)
            if reasoning_match:
                reasoning_content = reasoning_match.group(1)
            else :
                reasoning_content = ""
            
            # 尝试从<answer>标签中提取
            answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            if answer_match:
                content = answer_match.group(1)
            else :
                return None
            
            # 提取颜色和坐标
            color_match = re.search(r'\\boxed\{下一步颜色:(黑|白)\}', content)
            if not color_match:
                return None
            color = color_match.group(1)
            
            coord_match = re.search(r'\\boxed\{下一步位置:([A-HJ-T]\d+)\}', content)
            if not coord_match:
                # 尝试旧格式匹配
                coord_match = re.search(r'\\boxed\{([A-HJ-T]\d+)\}', content)
                if not coord_match:
                    return None
                
            coordinate = coord_match.group(1)
            # 验证数字部分是否在1-19范围内
            letter, number = coordinate[0], int(coordinate[1:])
            if 1 <= number <= 19:
                # 返回坐标、颜色和是否包含think标签
                return coordinate, color, has_think, reasoning_content
            return None
            
        except Exception as e:
            # print(f"Error extracting coordinate and color: {str(e)}")
            return None

    @classmethod
    def verify_score(cls, model_output, identity:dict, format_score=0.2) -> float:
        """
        验证LLM的输出是否符合要求
        
        参数:
            model_output: LLM的输出
            identity: KataGO标注信息
            format_score: 格式得分
        """
            
        score = 0.0
        try:
            # 提取坐标
            result = cls.extract_output(model_output)
            if not result:
                # print("提取信息失败")
                return score * (1/(1+format_score))
            
            move, color, has_think, think_content = result
            
            # if not has_think or think_content == "":
            #     return score * (1/(1+format_score))
            
            score += format_score
            
            # 判断黑白是否正确
            original_move_number = len(identity['former_moves']) + 1
            if original_move_number % 2 == 1:
                gt_color = '黑' # 最后一步是白棋，因此当前是黑棋
            else:
                gt_color = '白'
            
            if gt_color != color:
                # print(f"黑白颜色错误: {gt_color} != {color}")
                score -= format_score
                return score * (1/(1+format_score))
            
            # 获取当前这一步的最优落子
            candidates = {str(move_info['move']): move_info for move_info in identity['candidate_moves']}
            gt_best_move, gt_best_win_rate = None, 0
            for move_info in identity['candidate_moves']:
                win_rate_value = float(move_info['win_rate'])  # 确保转换为Python float
                if gt_best_win_rate < win_rate_value:
                    gt_best_move = str(move_info['move'])
                    gt_best_win_rate = win_rate_value
            
            if str(move) in candidates: # 如果在候选落子中，则不需要再调用模型
                in_state = 0
                if str(move) == gt_best_move: # 如果落子是top 1落子，则奖励
                    # print(f"{move}是top 1落子。")
                    in_state = 0
                    score += 0.5
                    
                # 奖励和top 1差距在0.9以内
                elif float(candidates[str(move)]['win_rate']) > gt_best_win_rate*0.9 :
                    # print(f"{move}和top 1落子差距在0.9以内。")
                    in_state = 1
                    score += 0.3 
                else:
                    # print(f"{move}和top 1落子差距在0.9以外。")
                    in_state = 2
                    score += 0.1
                win_rate = candidates[str(move)]['win_rate']
                
                # 使用sigmoid类函数来计算分数
                diff = win_rate - gt_best_win_rate
                score += 0.5 * (1 / (1 + 10 * abs(diff)))  # 系数可以调整惩罚的严格程度
                if in_state == 0:
                    move_str = f"当前落子是top 1落子。"
                elif in_state == 1:
                    move_str = f"当前落子胜率在top 1落子的0.9以内。"
                elif in_state == 2:
                    move_str = f"当前落子胜率在top 1落子的0.9以外。"
                # print(f"{move_str} 当前这一步的胜率: {win_rate}, 最优胜率: {gt_best_win_rate}, 得分: {score}")
            else : 
                # print(f"当前落子不在top 10落子中!")
                return score * (1/(1+format_score))
                
            return score * (1/(1+format_score))
                
        except Exception as e:
            # print(f"Error in compute_score: {type(e).__name__}: {str(e)}")
            return score * (1/(1+format_score))
            
        return score * (1/(1+format_score))
    
if __name__ == "__main__":
    
    # 测试
    import argparse
    import json
    import os
    
    bootcamp = InternGObootcamp()
    parser = argparse.ArgumentParser(description='Intern GO bootcamp DEV')
    parser.add_argument('--num_cases', type=int, default=100, help='Number of cases to generate')
    parser.add_argument('--output_dir', type=str, default='./data', help='Output directory')
    args = parser.parse_args()
    
    # 生成数据
    # case = bootcamp.case_generator()
    # print(case)
    # prompt = bootcamp.prompt_func(case)
    # print(prompt)
    
    # 测试是否能正确评分
    import json
    test_solution_1 = """
<reasoning>\n先分析当前局面：白棋刚刚下了Q3，这是一个试探性的手段，看黑棋如何应对。我需要评估几种可能的黑棋应手。\n\n首先考虑R3这个选择：\n如果黑棋下R3，从局部看似乎是对白Q3的直接回应，看起来很自然。但白棋后续可能会有N3的手段，引发一系列变化：黑M4，白P3，黑Q4，白O4，黑O5，白N4，黑N5，白M2，黑L2，白O2。这条变化线对黑棋并不有利，白棋能够获得良好的形状和厚势。我估计这样下的胜率大约47.9%。\n\n再看P3选项：\nP3是一个积极的选择，试图影响白棋Q3的发展。后续可能的变化是：白Q9，黑R9，白P10，黑P9，白Q8，黑R10，白O10。这条变化线会在右侧中央区域展开战斗，但黑棋形势略显被动，整体实地估计不如白棋。胜率大约47.7%，稍低于R3的选择。\n\n最后分析Q4这个选择：\nQ4是对白Q3的有力应对，既防止白棋向右下角发展，又为自己在右边建立基础。如果白棋以N3应对，黑棋可以N4反击，保持右侧局势的平衡。后续可能的变化是：白N3，黑N4，白P10，黑Q9，白M11，黑C3，白E2。这条变化线黑棋既照顾了右下角的利益，又能在左下角获得一定的主动权。整体来看，这是最平衡的选择，胜率大约49.6%。\n\n此外，我还需要考虑全局平衡。当前局面黑棋在上方已有一定布局，Q4这步既能与右上配合，又能制约白棋在右下角的发展，符合围棋\"取势\"的原则。\n\n综合各方面因素，Q4是当前局面最佳的选择，既有局部的实际效果，又兼顾了全局的战略平衡。\n</reasoning>\n\n<answer>\n\\boxed{下一步颜色:白}\n\\boxed{下一步位置:Q4}\n</answer><|im_end|>"""
    # test_solution_2 = """<Step1>\n目前进行了8步，下一步是第9步，轮到黑棋落子，标记为X。\n预测的下一步是：\n9.X-D3\n<Step2>\n后续的走法为：\n9.X-D3\n10.O-C7\n11.X-B2\n12.O-B4\n13.X-F5\n14.O-J4\n15.X-C9\n16.O-D8\n\n<Step3>\n对以上的变化进行数值分析，黑棋的胜率是49.06%。"""
        

    
    # 保存数据
    # with open(os.path.join(args.output_dir, 'cases.json'), 'w') as f:
    #     json.dump(cases, f)