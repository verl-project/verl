import math
import random
import fractions
import json
import tqdm
import os
import numpy as np
from multiprocessing import Pool, Queue, Process

class Game24Plus:
    """General version of game24 (Krypto), given N numbers less than X, find a way to get number M using 4 basic operations."""
    def __init__(self, num_numbers, range_max, target_max, seed=None):
        self.rng = np.random.default_rng(seed)
        self.num_numbers = num_numbers
        self.target_max = target_max
        self.range_max = range_max
        self.operations = ['+', '-', '*', '/']
        
    def sample_one_number(self, num_min, num_max):
        return self.rng.integers(num_min, num_max, size=1)[0]
        
    def get_numbers(self):
        numbers = self.rng.integers(0, self.range_max, size=self.num_numbers).tolist()
        numbers.sort()
        return numbers
    
    def enumerate_all_numbers(self, num_numbers):
        if num_numbers == 1:
            for i in range(1, self.range_max):
                yield [i]
            return
        for i in range(1, self.range_max):
            for j in self.enumerate_all_numbers(num_numbers - 1):
                yield [i] + j            

    def sample_operation(self, numbers: list):
        n_pos = self.rng.choice(len(numbers), size=2, replace=False)
        n1 = numbers[n_pos[0]]
        n2 = numbers[n_pos[1]]
        op = self.rng.choice(self.operations).item()
        if op == '/':
        #     if n2 == 0 or n1 % n2 != 0:
            if n2 == 0:
                return None
        return n1, op, n2, n_pos[0], n_pos[1]
    
    def calculate(self, n1, op, n2):
        if op == '+':
            return n1 + n2
        elif op == '-':
            return n1 - n2
        elif op == '*':
            return n1 * n2
        elif op == '/':
            if n1 % n2 != 0:
                return fractions.Fraction(n1, n2)
            return n1 // n2
    
    def get_target(self, numbers: list):
        current_numbers = numbers.copy()
        current_operations = []
        while len(current_numbers) > 1:
            r = self.sample_operation(current_numbers)
            if r is None:
                continue
            n1, op, n2, pos1, pos2 = r
            current_numbers.pop(max(pos1, pos2))
            current_numbers.pop(min(pos1, pos2))
            res = self.calculate(n1, op, n2)
            if isinstance(res, fractions.Fraction):
                a, b = res.as_integer_ratio()
                if a % b == 0:
                    res = int(a/b)
            current_operations.append((str(n1), op, str(n2), str(res)))
            current_numbers.append(res)
        return current_numbers[0], current_operations
    
    def get_target_limit_range(self, numbers):
        target, operations = self.get_target(numbers)
        num_try = 0
        while not isinstance(target, int) or target < 0 or target > self.target_max:
            target, operations = self.get_target(numbers)
            num_try += 1
            if num_try > 1000:
                return None, None
        return target, operations
    
    def solve(self, numbers, target):
        def precise_calculate(n1, op, n2):
            if op == '+':
                return n1 + n2
            elif op == '-':
                return n1 - n2
            elif op == '*':
                return n1 * n2
            elif op == '/':
                if n1 % n2 != 0:
                    return fractions.Fraction(n1, n2)
                return n1 // n2
            
        if len(numbers) == 1:
            if isinstance(numbers[0], int) and numbers[0] == target:
                return []
            else:
                a, b = numbers[0].as_integer_ratio()
                if a % b == 0 and int(a/b) == target:
                    return []
                return None
        for i, n1 in enumerate(numbers):
            for j, n2 in enumerate(numbers):
                if i == j:
                    continue
                for op in self.operations:
                    if op == '/':
                        # if n2 == 0 or n1 % n2 != 0:
                        if n2 == 0:
                            continue
                    res = precise_calculate(n1, op, n2)
                    new_num = [n for k, n in enumerate(numbers) if k != i and k != j] + [res]
                    new_how = [(str(n1), op, str(n2), str(res))]
                    part_solution = self.solve(new_num, target)
                    if part_solution is not None:
                        return new_how + part_solution
    def generate_ground_truth(self, numbers, target):
        calc_steps = self.solve(numbers, target)
        
        result_map = {}

        for step in calc_steps:
            operand1, operator, operand2, result = step
            
            # 如果操作数是之前的结果，则替换为对应的表达式
            if operand1 in result_map:
                operand1 = f"({result_map[operand1]})"
            else:
                operand1 = operand1
            
            if operand2 in result_map:
                operand2 = f"({result_map[operand2]})"
            else:
                operand2 = operand2
            
            # 构建当前步骤的表达式
            if operator == "":
                expression = operand1  # 如果没有操作符，直接取操作数
            else:
                expression = f"{operand1} {operator} {operand2}"
            
            # 将当前结果存入映射表
            result_map[result] = expression

        # 最终结果是最后一个元组的结果
        final_result = calc_steps[-1][-1]
        final_expression = result_map[final_result]
        
        return final_expression
            
                    
def construct_game24_v1(num_numbers=3, range_max=101, num_samples=10000, target_max=1000, seed=1234, output_dir=None):
    game = Game24Plus(num_numbers, range_max, target_max=target_max, seed=seed)
    with open(os.path.join(output_dir, f'train_m={num_numbers}.jsonl'), 'w') as fp:
        puzzle_dict = {}
        for idx in tqdm.trange(num_samples):
            num_try = 0
            while True:
                numbers = game.get_numbers()
                puzzle = ' '.join(str(n) for n in numbers)
                if puzzle in puzzle_dict:
                    num_try += 1
                    if num_try > 1000:
                        puzzle = None
                        break
                    continue
                else:
                    break
            if puzzle is None:
                print('Failed to generate unique puzzle')
                break
            puzzle_dict[puzzle] = True
            num_try = 0
            data = {}
            while True:
                target, operations = game.get_target_limit_range(numbers)
                target_str = str(target)
                key = puzzle + ' ' + target_str
                if key in data:
                    num_try += 1
                    if num_try > 10:
                        break
                else:
                    data[key] = {
                        'puzzle': puzzle,
                        'target': target_str,
                        'operations': operations
                    }
                    num_try = 0
            for key, ex in data.items():
                print(json.dumps(ex), file=fp)


def construct_helper(args):
    num_numbers, range_max, num_samples_per_target, target_min, target_max, seed = args
    game = Game24Plus(num_numbers, range_max, target_max=target_max, seed=seed)
    data = []
    for target in range(target_min, target_max):
        num_try = 0
        target_str = str(target)
        seen_data = {}
        while True:
            numbers = game.get_numbers()
            puzzle = ' '.join(str(n) for n in numbers)
            if puzzle in seen_data:
                continue
            operations = game.solve(numbers, target)
            if operations is None:
                num_try += 1
                if num_try > 10000:
                    break
                continue
            ex = {
                'puzzle': puzzle,
                'target': target_str,
                'operations': operations
            }
            data.append(ex)
            num_try = 0
            seen_data[puzzle] = 1
            if len(seen_data) > num_samples_per_target:
                break
            
    return data
    

def construct_game24_v3(num_numbers=3, range_max=101, num_samples=50000, target_max=1000, seed=1234, num_workers=64, output_dir=None):
    print(f'Args', num_numbers, range_max, num_samples, target_max, seed, num_workers)
    arg_list = []
    num_samples_per_target = (num_samples // target_max) + 1
    chunk_size = 1
    start_target = 0
    while start_target < target_max:
        _target_min = start_target
        _target_max = min(target_max, start_target + chunk_size)
        arg_list.append((
            num_numbers, range_max, num_samples_per_target, _target_min, _target_max, seed
        ))
        start_target = _target_max
    # print(arg_list)
    with open(os.path.join(output_dir, f'train_m={num_numbers}.jsonl'), 'w') as fp:
        with Pool(num_workers) as p:
            for data in tqdm.tqdm(p.imap_unordered(construct_helper, arg_list), total=len(arg_list)):
                for ex in data:
                    print(json.dumps(ex), file=fp)
                    
                    
def construct_helper_for_v4(args_queue, out_queue):
    while True:
        args = args_queue.get()
        if args is None:
            break
        numbers = args[0]
        game = args[1]
        puzzle = ' '.join(str(n) for n in numbers)
        num_try = 0
        data = {}
        while True:
            target, operations = game.get_target_limit_range(numbers)
            if target in data:
                num_try += 1
                if num_try > 10:
                    break
            else:
                data[target] = {
                    'puzzle': puzzle,
                    'target': str(target),
                    'operations': operations
                }
        out_queue.put(data)
    

def construct_game24_v4(num_numbers=3, range_max=101, num_samples=200000, target_max=1000, seed=1234, num_workers=64, output_dir=None):
    print(f'Args', num_numbers, range_max, num_samples, target_max, seed, num_workers)
    seen_numbers = set()
    procs = []
    args_queue = Queue()
    out_queue = Queue()
    for i in range(num_workers):
        p = Process(target=construct_helper_for_v4, args=(args_queue, out_queue))
        p.start()
        procs.append(p)
        
    game = Game24Plus(num_numbers, range_max, target_max=target_max, seed=seed)
    with open(os.path.join(output_dir, f'train_m={num_numbers}.jsonl'), 'w') as fp:
        for i in range(num_samples):
            numbers = game.get_numbers()
            puzzle = ' '.join(str(n) for n in numbers)
            if puzzle in seen_numbers:
                continue
            seen_numbers.add(puzzle)
            args_queue.put((numbers, game))
        for i in range(num_workers):
            args_queue.put(None)
            
        count = 0
        tqdm_bar = tqdm.tqdm(total=num_samples)
        while count < num_samples:
            data = out_queue.get()
            for target, ex in data.items():
                print(json.dumps(ex), file=fp)
                count += 1
                tqdm_bar.update(1)
                
        for p in procs:
            p.kill()
            

if __name__ == "__main__":
    # 初始化 Game24Plus 实例
    game = Game24Plus(num_numbers=6, range_max=10, target_max=24, seed=46000767)
    
    # 获取一组随机数字
    numbers = game.get_numbers()
    print("Numbers:", numbers)
    
    # 设置目标值
    target = 24
    print("Target:", target)
    
    print("Trace:", game.solve(numbers, target))
    
    # 尝试生成一步答案算式
    expression = game.generate_ground_truth(numbers, target)
    if expression:
        print("expression:", expression)
    else:
        print("No expression found.")
    