# Game24Plus - 数学谜题生成与解决工具

`Game24Plus` 是一个用于生成和解决类似于“24点游戏”的数学谜题的 Python 工具。它通过加、减、乘、除四种基本运算，将给定的数字组合成目标数字。该工具支持生成不同数量和范围的数字组合，并提供多种生成方式，包括单进程和多进程并行处理。

---

## 功能特性

- **生成数学谜题**：生成指定数量和范围的数字组合，并通过随机运算生成目标数字。
- **解决谜题**：递归地尝试所有可能的运算组合，找到达到目标数字的解决方案。
- **多进程并行处理**：支持多进程并行生成谜题，提高生成效率。
- **数据保存**：生成的谜题和解决方案以 JSONL 格式保存到文件中。


## 代码结构

### 主要类与方法

#### `Game24Plus` 类

- **`__init__(self, num_numbers, range_max, target_max, seed=None)`**  
  初始化随机数生成器、数字数量、数字范围、目标数字范围等。

- **`sample_one_number(self, num_min, num_max)`**  
  生成一个指定范围内的随机整数。

- **`get_numbers(self)`**  
  生成一组随机整数并排序。

- **`enumerate_all_numbers(self, num_numbers)`**  
  枚举所有可能的数字组合（递归实现）。

- **`sample_operation(self, numbers)`**  
  从数字列表中随机选择两个数字和一个运算符，返回运算表达式。

- **`calculate(self, n1, op, n2)`**  
  根据运算符计算两个数字的结果，处理除法时使用分数。

- **`get_target(self, numbers)`**  
  通过随机运算将数字列表逐步减少，直到得到一个目标数字，并记录运算过程。

- **`get_target_limit_range(self, numbers)`**  
  在指定范围内生成目标数字，避免无效结果。

- **`solve(self, numbers, target)`**  
  递归地尝试所有可能的运算组合，找到达到目标数字的解决方案。

### 主要函数

- **`construct_game24_v1`**  
  生成指定数量的谜题，并将结果保存为 JSONL 文件。

- **`construct_helper`**  
  辅助函数，用于生成指定目标范围内的谜题。

- **`construct_game24_v3`**  
  使用多进程并行生成谜题，提高生成效率。

- **`construct_helper_for_v4`**  
  辅助函数，用于多进程版本的谜题生成。

- **`construct_game24_v4`**  
  使用多进程和多队列实现高效的谜题生成。

---

## 使用示例

### 生成谜题

```python
from game24plus import Game24Plus

# 初始化
game = Game24Plus(num_numbers=4, range_max=100, target_max=1000)

# 生成一组数字
numbers = game.get_numbers()
print("生成的数字：", numbers)

# 生成目标数字和运算过程
target, operations = game.get_target_limit_range(numbers)
print("目标数字：", target)
print("运算过程：", operations)
```

### 解决谜题

```python
# 解决谜题
solution = game.solve(numbers, target=24)
if solution:
    print("解决方案：", solution)
else:
    print("无解")
```

### 批量生成谜题

```bash
python game24plus.py --num_numbers 4 --output_dir data/gameX/raw_1104
```

---

## 参数说明

- **`num_numbers`**：生成谜题的数字数量（默认为 4）。
- **`range_max`**：生成数字的最大值（默认为 101）。
- **`target_max`**：目标数字的最大值（默认为 1000）。
- **`num_samples`**：生成的谜题数量（默认为 200000）。
- **`seed`**：随机数种子（默认为 1234）。
- **`num_workers`**：多进程并行处理的工作进程数量（默认为 64）。
- **`output_dir`**：输出文件的保存目录（默认为当前目录）。

---

## 输出文件格式

生成的谜题和解决方案以 JSONL 格式保存，每行包含一个 JSON 对象，例如：

```json
{
  "puzzle": "3 4 6 8",
  "target": "24",
  "operations": [
    ["3", "*", "8", "24"],
    ["4", "+", "6", "10"],
    ["10", "-", "3", "7"]
  ]
}
```

---

## 性能优化

- **多进程并行处理**：通过 `multiprocessing` 模块实现高效的谜题生成。
- **递归算法优化**：减少重复计算，提高解决谜题的效率。

---