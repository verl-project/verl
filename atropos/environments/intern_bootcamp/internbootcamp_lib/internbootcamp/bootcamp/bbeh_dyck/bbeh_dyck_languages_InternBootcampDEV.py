from bootcamp.base import Basebootcamp
import re
import random
from typing import List, Tuple, Dict, Set, Optional

class BbehDyckLanguagesbootcamp(Basebootcamp):
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
        生成包含错误步骤的Dyck语言案例
        
        返回:
            包含输入序列、正确步骤、修改后步骤和错误步骤的字典
        """
        # 生成合法Dyck序列及其正确步骤
        input_seq, correct_steps = self._generate_valid_sequence()
        modified_steps = correct_steps.copy()
        error_step, error_type = None, None
        
        # 注入错误
        if random.random() < self.error_prob:
            error_step, error_type = self._inject_error(modified_steps, input_seq)
            
            # 根据错误类型传播错误影响
            if error_type == 'corrupt_stack_state': # 堆栈状态描述错误
                self._propagate_error_effects(modified_steps, error_step)
            elif error_type == 'swap_bracket_type': # 括号类型错误
                self._propagate_error_effects(modified_steps, error_step)
            elif error_type == 'skip_closing': # 跳过闭合步骤
                self._propagate_error_effects(modified_steps, error_step)
            elif error_type == 'incorrect_stack_update': # 堆栈更新错误
                self._propagate_error_effects(modified_steps, error_step)
            elif error_type == 'misinterpret_input': # 输入解释错误
                self._propagate_error_effects(modified_steps, error_step)
            
                self._propagate_error_effects(modified_steps, error_step)
        
        # 添加结论步骤
        self._add_conclusion_step(modified_steps)
        
        return {
            'input_sequence': input_seq,
            'correct_steps': correct_steps,
            'modified_steps': modified_steps,
            'error_step': error_step,
            'error_type': error_type
        }
    
    def _generate_valid_sequence(self) -> Tuple[str, List[str]]:
        """
        生成合法Dyck序列及处理步骤
        
        返回:
            元组(序列字符串, 处理步骤列表)
        """
        stack = []
        sequence = []
        steps = [
            "Thought 1: We should process each input one by one and keep track of the stack configuration.",
            "Thought 2: stack: empty"
        ]
        current_stack = []
        
        length_target = random.randint(self.min_length, self.max_length)
        
        while len(sequence) < length_target:
            # 如果栈为空或者随机决定添加开括号（保持序列合法）
            if not stack or (random.random() < 0.6 and len(sequence) < length_target - len(stack)):
                br = random.choice(list(self.bracket_pairs.keys()))
                sequence.append(br)
                stack.append(br)  # 存储开括号
                current_stack.append(br)
            else:
                # 添加闭括号
                open_br = stack.pop()
                close_br = self.bracket_pairs[open_br]
                sequence.append(close_br)
                current_stack.pop()
            
            # 记录步骤
            step_num = len(steps) + 1
            stack_str = ' '.join(current_stack) if current_stack else 'empty'
            step = f"Thought {step_num}: {sequence[-1]} ; stack: {stack_str}"
            steps.append(step)
            
            # 如果达到最小长度且栈为空，可以提前结束
            if len(sequence) >= self.min_length and not stack:
                if random.random() < 0.3:  # 30%概率提前结束
                    break
        
        return ''.join(sequence), steps
    
    def _inject_error(self, steps: List[str], input_seq: str) -> Optional[int]:
        """
        注入错误，返回错误步骤号
        
        参数:
            steps: 步骤列表
            input_seq: 输入序列
            
        返回:
            错误步骤号(1-based)或None
        """
        error_types = [
            self._corrupt_stack_state,      # 堆栈状态描述错误
            self._swap_bracket_type,        # 括号类型错误
            self._skip_closing,             # 跳过闭合步骤
            self._incorrect_stack_update,   # 堆栈更新错误
            self._misinterpret_input        # 输入解释错误
        ]
        
        # 随机选择错误类型
        error_func = random.choice(error_types)
        error_type = error_func.__name__
        return error_func(steps, input_seq), error_type
    
    def _corrupt_stack_state(self, steps: List[str], _) -> Optional[int]:
        """
        错误类型1：堆栈状态描述错误, 堆栈状态描述错误是指步骤中的堆栈状态描述与实际不符
        
        参数:
            steps: 步骤列表
            _: 未使用的参数
            
        返回:
            错误步骤号(1-based)或None
        """
        # 跳过前几个步骤以增加难度
        valid_steps = [i for i in range(5, len(steps)) if '; stack:' in steps[i]]
        if not valid_steps:
            return None
            
        step_idx = random.choice(valid_steps)
        step = steps[step_idx]
        
        parts = step.split(';')
        bracket = parts[0].split()[-1]
        stack_desc = parts[1].split(': ')[-1]
        
        # 修改堆栈状态
        new_stack = self._modify_stack_description(stack_desc)
        steps[step_idx] = f"{parts[0]}; stack: {new_stack}"
        return step_idx + 1  # 步骤从1开始计数
    
    def _modify_stack_description(self, stack_desc: str) -> str:
        """
        修改堆栈描述
        
        参数:
            stack_desc: 原堆栈描述
            
        返回:
            修改后的堆栈描述
        """
        if stack_desc == 'empty':
            return random.choice(['[', '(', '{', '<'])
            
        elements = stack_desc.split()
        
        # 多种错误类型
        error_type = random.randint(1, 4)
        
        if error_type == 1 and elements:  # 删除元素
            remove_idx = random.randint(0, len(elements)-1)
            return ' '.join(elements[:remove_idx] + elements[remove_idx+1:]) or 'empty'
            
        elif error_type == 2:  # 添加元素
            new_element = random.choice(list(self.open_brackets))
            insert_idx = random.randint(0, len(elements))
            new_elements = elements[:insert_idx] + [new_element] + elements[insert_idx:]
            return ' '.join(new_elements)
            
        elif error_type == 3 and len(elements) >= 2:  # 交换元素
            idx1, idx2 = random.sample(range(len(elements)), 2)
            elements[idx1], elements[idx2] = elements[idx2], elements[idx1]
            return ' '.join(elements)
            
        else:  # 替换元素
            if not elements:
                return random.choice(list(self.open_brackets))
            replace_idx = random.randint(0, len(elements)-1)
            old_element = elements[replace_idx]
            new_element = random.choice([b for b in self.open_brackets if b != old_element])
            elements[replace_idx] = new_element
            return ' '.join(elements)
    
    def _swap_bracket_type(self, steps: List[str], input_seq: str) -> Optional[int]:
        """
        错误类型2：括号类型错误, 括号类型错误是指步骤中的括号类型与输入序列不符
        
        参数:
            steps: 步骤列表
            input_seq: 输入序列
            
        返回:
            错误步骤号(1-based)或None
        """
        # 找到所有闭括号位置
        close_positions = [(i, c) for i, c in enumerate(input_seq) if c in self.close_brackets]
        if not close_positions:
            return None
            
        pos, original = random.choice(close_positions)
        
        # 找到对应的步骤
        step_idx = pos + 2  # 步骤偏移
        if step_idx >= len(steps):
            return None
            
        # 找到一个不同的闭括号
        new_char = random.choice([c for c in self.close_brackets if c != original])
        
        # 修改步骤中的括号类型
        step = steps[step_idx]
        parts = step.split(';')
        new_step = f"Thought {step_idx+1}: {new_char} ;{parts[1]}"
        steps[step_idx] = new_step
        
        return step_idx + 1  # 返回1-based索引
    
    def _skip_closing(self, steps: List[str], _) -> Optional[int]:
        """
        错误类型3：跳过闭合步骤, 跳过闭合步骤是指步骤中没有正确闭合应该闭合的括号
        
        参数:
            steps: 步骤列表
            _: 未使用的参数
            
        返回:
            错误步骤号(1-based)或None
        """
        # 找到包含闭括号的步骤
        close_steps = [(i, s) for i, s in enumerate(steps) if i > 2 and any(c in s.split(';')[0] for c in self.close_brackets)]
        if not close_steps:
            return None
            
        step_idx, step = random.choice(close_steps)
        parts = step.split(';')
        
        # 获取当前括号和堆栈
        current_bracket = parts[0].split()[-1]
        stack_desc = parts[1].split(': ')[-1]
        
        # 错误处理：不弹出应该弹出的括号
        if stack_desc != 'empty':
            stack_elements = stack_desc.split()
            # 保持堆栈不变，但应该减少
            steps[step_idx] = f"{parts[0]};{parts[1]}"
            return step_idx + 1
        
        return None
    
    def _incorrect_stack_update(self, steps: List[str], _) -> Optional[int]:
        """
        错误类型4：堆栈更新错误, 堆栈更新错误是指步骤中的堆栈更新与实际不符
        
        参数:
            steps: 步骤列表
            _: 未使用的参数
            
        返回:
            错误步骤号(1-based)或None
        """
        # 跳过前几个步骤
        valid_steps = [i for i in range(5, len(steps)-1) if '; stack:' in steps[i]]
        if not valid_steps:
            return None
            
        step_idx = random.choice(valid_steps)
        current_step = steps[step_idx]
        next_step = steps[step_idx + 1]
        
        # 解析当前步骤
        current_parts = current_step.split(';')
        current_bracket = current_parts[0].split()[-1]
        current_stack = current_parts[1].split(': ')[-1]
        
        # 解析下一步骤
        next_parts = next_step.split(';')
        next_bracket = next_parts[0].split()[-1]
        next_stack = next_parts[1].split(': ')[-1]
        
        # 根据括号类型确定堆栈应该如何变化
        if next_bracket in self.open_brackets:
            # 开括号应该入栈
            if current_stack == 'empty':
                expected_next_stack = next_bracket
            else:
                expected_next_stack = f"{current_stack} {next_bracket}"
                
            # 错误：没有正确入栈
            if random.random() < 0.5:
                # 不添加新括号
                steps[step_idx + 1] = f"{next_parts[0]}; stack: {current_stack}"
            else:
                # 添加错误的括号
                wrong_bracket = random.choice([b for b in self.open_brackets if b != next_bracket])
                if current_stack == 'empty':
                    steps[step_idx + 1] = f"{next_parts[0]}; stack: {wrong_bracket}"
                else:
                    steps[step_idx + 1] = f"{next_parts[0]}; stack: {current_stack} {wrong_bracket}"
                    
        elif next_bracket in self.close_brackets:
            # 闭括号应该出栈
            if current_stack != 'empty':
                current_stack_list = current_stack.split()
                if current_stack_list:
                    # 错误：没有正确出栈或出栈错误的括号
                    if random.random() < 0.5 and len(current_stack_list) > 1:
                        # 出栈错误的括号
                        wrong_idx = random.randint(0, len(current_stack_list)-2)
                        new_stack = ' '.join(current_stack_list[:wrong_idx] + current_stack_list[wrong_idx+1:])
                        steps[step_idx + 1] = f"{next_parts[0]}; stack: {new_stack or 'empty'}"
                    else:
                        # 不出栈
                        steps[step_idx + 1] = f"{next_parts[0]}; stack: {current_stack}"
        
        return step_idx + 2  # 返回1-based索引
    
    def _misinterpret_input(self, steps: List[str], input_seq: str) -> Optional[int]:
        """
        错误类型5：输入解释错误, 输入解释错误是指步骤中的括号与输入序列不符
        
        参数:
            steps: 步骤列表
            input_seq: 输入序列
            
        返回:
            错误步骤号(1-based)或None
        """
        # 跳过前几个步骤
        valid_steps = [i for i in range(3, min(8, len(steps))) if '; stack:' in steps[i]]
        if not valid_steps:
            return None
            
        step_idx = random.choice(valid_steps)
        step = steps[step_idx]
        
        parts = step.split(';')
        current_bracket = parts[0].split()[-1]
        
        # 找到一个不同的括号
        all_brackets = list(self.open_brackets) + list(self.close_brackets)
        new_bracket = random.choice([b for b in all_brackets if b != current_bracket])
        
        # 修改步骤中的括号
        steps[step_idx] = f"Thought {step_idx+1}: {new_bracket} ;{parts[1]}"
        
        return step_idx + 1  # 返回1-based索引
    
    def _propagate_error_effects(self, steps: List[str], error_step: int) -> None:
        """
        传播错误影响到后续步骤
        
        参数:
            steps: 步骤列表
            error_step: 错误步骤号(1-based)
        """
        # 错误步骤的索引
        error_idx = error_step - 1
        
        # 如果错误步骤不在范围内或没有堆栈信息，则不处理
        if error_idx < 0 or error_idx >= len(steps) or '; stack:' not in steps[error_idx]:
            return
            
        # 解析错误步骤的堆栈状态
        error_parts = steps[error_idx].split(';')
        error_stack = error_parts[1].split(': ')[-1]
        
        # 从错误步骤开始，更新后续步骤的堆栈状态
        for i in range(error_idx + 1, len(steps)):
            if '; stack:' not in steps[i]:
                continue
                
            parts = steps[i].split(';')
            bracket = parts[0].split()[-1]
            
            # 根据括号类型更新堆栈
            if bracket in self.open_brackets:
                # 开括号入栈
                if error_stack == 'empty':
                    error_stack = bracket
                else:
                    error_stack = f"{error_stack} {bracket}"
            elif bracket in self.close_brackets:
                # 闭括号出栈
                if error_stack != 'empty':
                    stack_elements = error_stack.split()
                    if stack_elements:
                        stack_elements.pop()
                        error_stack = ' '.join(stack_elements) or 'empty'
            
            # 更新步骤
            steps[i] = f"{parts[0]}; stack: {error_stack}"
    
    def _add_conclusion_step(self, steps: List[str]) -> None:
        """
        添加结论步骤
        
        参数:
            steps: 步骤列表
        """
        # 检查是否已有结论步骤
        for step in steps:
            if "So the answer is" in step:
                return
                
        # 获取最后一个堆栈状态
        last_stack = "empty"
        for step in reversed(steps):
            if '; stack:' in step:
                last_stack = step.split('; stack: ')[-1]
                break
        
        # 生成结论
        step_num = len(steps) + 1
        
        if last_stack == "empty":
            # 如果堆栈为空，添加明确的结论
            steps.append(f"Thought {step_num}: Now, we have reached the end. The final stack is empty.")
            steps.append(f"Thought {step_num+1}: So the answer is empty")
        else:
            # 如果堆栈不为空，生成一个基于堆栈的详细结论
            steps.append(f"Thought {step_num}: Now, we have reached the end. The final stack is \"{last_stack}\".")
            
            # 添加解释步骤
            stack_elements = last_stack.split()
            pop_description = ", ".join([f"\"{elem}\"" for elem in reversed(stack_elements)])
            steps.append(f"Thought {step_num+1}: We will need to pop out {pop_description} one by one in that order.")
            
            # 生成所需的闭合括号序列
            closing_brackets = [self.bracket_pairs.get(elem, ')') for elem in reversed(stack_elements)]
            closing_sequence = ' '.join(closing_brackets)
            needed_brackets = ", ".join([f"\"{b}\"" for b in closing_brackets])
            
            steps.append(f"Thought {step_num+2}: So, we need {needed_brackets}. So the answer is {closing_sequence}")
    
    @staticmethod
    def prompt_func(case: Dict) -> str:
        """
        生成问题描述
        
        参数:
            case: 包含问题信息的字典
            
        返回:
            格式化的问题字符串
        """
        
        input_sequence = case['input_sequence']
        steps = '\n'.join(case['modified_steps'])
        
        return_prompts = [
            f"""Dyck语言是计算机科学中的一种形式语言，由匹配的括号对组成。你是一个Dyck语言的专家，擅长分析括号序列中可能出现的错误。在这个任务中，你需要分析括号序列的处理过程，并找出其中的第一个错误步骤。

## 输入序列
{input_sequence}

## 分析步骤
{steps}

## 你的任务
请仔细检查上述步骤，找出第一个出现错误的步骤编号。如果所有步骤都正确，请回答"No"。

请逐步推理解答此问题，并且将最终答案放入[answer] and [/answer]中。例如：
[answer]
步骤编号或"No"
[/answer]
""", 

            f"""你是一个Dyck语言的专家。Dyck语言是计算机科学中的一种形式语言，由匹配的括号对组成。给定一个括号序列和对应于处理该序列的思考步骤，请你找出其中的第一个错误步骤。

## 括号序列
{input_sequence}

## 处理过程
{steps}

## 分析要求
1. 每个开括号必须与对应类型的闭括号匹配
2. 括号必须按"后开先闭"原则匹配
3. 堆栈状态必须准确反映当前未匹配的开括号

## 请回答
如果发现错误，请指出第一个错误步骤的编号；如果全部正确，请回答"No"。

请逐步推理解答此问题，并且将最终答案放入[answer] and [/answer]中。例如：
[answer]
你的答案
[/answer]
""", 

        f"""You are given a bracket sequence and the reasoning steps for processing it, find the first erroneous step, if any.

## Sequence
{input_sequence}

## Processing
{steps}

## Analysis Requirements
1. Each opening bracket must match with its corresponding closing bracket
2. Brackets must follow the "last-opened-first-closed" principle
3. Stack state must accurately reflect currently unmatched opening brackets

## Response Format
Identify the number of the first incorrect step, or answer "No" if all steps are correct.
Please follow the format below:
[answer]
Your answer here
[/answer]
Please think step by step.
""", 

        f"""You are debugging an algorithm that processes bracket sequences according to Dyck language rules. The algorithm tracks a stack of open brackets and pops them when matching closing brackets are encountered. Your task is to identify the first step where the algorithm makes a mistake in processing the sequence. 

## Given Sequence
{input_sequence}

## Algorithm Trace
{steps}

Please think step by step and put your final answer within [answer] and [/answer] tags.
For example:
[answer]
First error step number or "No" if error-free
[/answer]
Now generate your solution.
""", 

        f"""You are an expert in a language called dyck where you must complete the language sequence of unclosed brackets of all types (e.g., [], {{}}, <>). You are given an initial initial dyck language sequence and the steps, provided as\nthoughts, that were used to arrive at the closing bracket sequence in the dyck language. Your job is to identify the first step that was a mistake in reasoning about the closing bracket sequence in dyck language, if any.\nThis can be forgetting to close a bracket or closing with the wrong bracket or incorrectly copying the prior subsequence of closing brackets to the next step.\nTask: Complete the rest of the sequence, making sure that the parentheses are closed properly. Input: {input_sequence}
        {steps}
        Q: Is there a mistake in this sequence? Write \"No\" if there are no mistakes, or the number N if there is a mistake in Thought N.
        Please solve this task step by step and put your final answer within [answer] and [/answer] tags.
        For example:
        [answer]
        Your answer here
        [/answer]
        Now generate your solution:
        """
        ]   
        
        return random.choice(return_prompts)
    
    @staticmethod
    def extract_output(output: str) -> Optional[str]:
        """
        从LLM的回复中提取答案
        
        参数:
            output: LLM的完整输出
            
        返回:
            提取的答案或None
        """
        matches = re.findall(r'\[answer\]\s*(\d+|No)\s*\[/answer\]', output, re.IGNORECASE)
        return matches[-1].strip().capitalize() if matches else None
    
    @classmethod
    def _verify_correction(cls, solution: str, case: Dict) -> bool:
        """
        验证提取的答案是否正确
        
        参数:
            solution: extract_output提取的答案
            case: 包含问题信息的字典
            
        返回:
            答案是否正确
        """
        # 如果没有错误步骤
        if case['error_step'] is None:
            return solution.lower() == "no"
            
        # 如果有错误步骤
        return solution == str(case['error_step'])

if __name__ == "__main__":
    
    # 测试
    import argparse
    import json
    import os
    
    bootcamp = BbehDyckLanguagesbootcamp()
    parser = argparse.ArgumentParser(description='BBEH Dyck Languages Intern bootcamp DEV')
    parser.add_argument('--num_cases', type=int, default=100, help='Number of cases to generate')
    parser.add_argument('--output_dir', type=str, default='./data', help='Output directory')
    args = parser.parse_args()
    
    # 生成数据
    case = bootcamp.case_generator()
    print(case)
    prompt = bootcamp.prompt_func(case)
    print(prompt)
    
    # 保存数据
    # with open(os.path.join(args.output_dir, 'cases.json'), 'w') as f:
    #     json.dump(cases, f)
