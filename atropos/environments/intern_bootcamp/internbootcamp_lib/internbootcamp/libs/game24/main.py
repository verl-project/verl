import random
import numpy as np
import fractions
from game24 import Game24Plus


# 实例化 Game24Plus
game = Game24Plus(num_numbers=4, range_max=10, target_max=24, seed=random.randint(0, 1000))

# 生成一组数字
numbers = game.get_numbers()
print("生成的数字:", numbers)

# 计算目标值及其运算过程
target, operations = game.get_target_limit_range(numbers)
if target is not None:
    print("计算得到的目标值:", target)
    print("运算过程:")
    for step in operations:
        print(f"{step[0]} {step[1]} {step[2]} = {step[3]}")
else:
    print("未能生成符合条件的目标值。")

# 求解问题：找到一种运算方式得到目标值
if target is not None:
    solution = game.solve(numbers, target)
    if solution is not None:
        print("求解得到的运算步骤:")
        for step in solution:
            print(f"{step[0]} {step[1]} {step[2]} = {step[3]}")
    else:
        print("未能找到符合条件的运算步骤。")