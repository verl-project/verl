# 任务训练场构建说明

## 数据生成

对于使用已有的训练场生成任务数据，我们在pipelines文件夹内提供了根据config file生成bootcamp data的脚本。 
**具体用法参考 [pipelines](pipelines/README.md)。**

下面介绍新增训练场并生成新增训练场任务数据的流程。

### 1. 新增训练场

bootcamp的命名支持驼峰命名和下划线命名。

- 下划线命名

在 `internbootcamp/bootcamp` 下新增一个bootcamp类 `Bbehshuffobjectbootcamp`（只有首字母大写，然后加上bootcamp）。在该bootcamp类中实现case_generator,prompt_func,verify_score三个主要接口。

在 `examples/pipelines/puzzle_configs` 下新增一个bootcamp配置，例如 `bbeh_shuff_object_train.json` 和 `bbeh_shuff_object_test.json`，在两个json文件中完成参数配置。

注意，用户需要根据bootcamp的实际任务设置合理参数防止生成无法完成的问题，同时设置多样化的参数实现控制生成问题的难度分布。


- 驼峰命名

在 `internbootcamp/bootcamp` 下新增一个bootcamp类 `BbehGeometricShapesbootcamp`（驼峰命名，然后加上bootcamp）。在该bootcamp类中实现`__init__`以及·case_generator`,`prompt_func`,`verify_score`三个主要接口。

在 `examples/pipelines/puzzle_configs` 下新增一个bootcamp配置，例如 `BbehGeometricShapes_train.json` 和 `BbehGeometricShapes_test.json`，在两个json文件中完成参数配置，在数据生成过程中，配置中的参数将传入训练场`__init__`函数以构建训练场实例。

注意，用户需要根据bootcamp的实际任务设置合理参数防止生成无法完成的问题，同时设置多样化的参数实现控制生成问题的难度分布。


### 2. Data Config

根据puzzle_configs中的配置信息，通过执行

`python examples/pipelines/quickgen_data_configs.py`

生成train和test的jsonl文件，每个jsonl包含了所有的bootcamp meta信息。


### 3. 生成bootcamp数据

完成上述步骤后，执行

`bash examples/pipelines/run_pipeline.sh` 

可自动生成全量的bootcamp data，包括**"prompt"，"ground_truth"和"data_source"**字段。
生成的路径在 `examples/bootcamp_generator_outputs` 下。

数据示例：
```json
{{"data_source": "Aquarium", "prompt": "You are to solve an Aquarium puzzle. The puzzle is played on a grid divided into aquarium regions. Each aquarium must be filled up to a horizontal level such that all its columns are filled to the same level. Here are the details:\n\n- The grid has 5 rows and 3 columns.\n\n- Aquarium regions are as follows (each number represents the aquarium ID for that cell):\nRow 0: 0 1 2\nRow 1: 0 1 2\nRow 2: 0 1 2\nRow 3: 0 1 2\nRow 4: 0 1 2\n\n- Each row has a clue on the right indicating the total filled cells in that row. The row clues are: [3, 2, 2, 1, 1].\n\n- Each column has a clue at the bottom indicating the total filled cells in that column. The column clues are: [5, 1, 3].\n\nYour task is to determine the water level for each aquarium. The water level is the highest row number filled (0-based from the bottom). Each aquarium's water level must be such that all its columns are filled up to this level.\n\nProvide your answer as a list of integers in column order (from left to right), where each integer is the water level for the corresponding column's aquarium. Enclose your answer within [answer] and [/answer]. For example, if the solution is levels 2, 1, 0 for columns 0, 1, 2, write:\n[answer]2 1 0[/answer]", "ground_truth": {"regions": [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]], "row_clues": [3, 2, 2, 1, 1], "col_clues": [5, 1, 3]}}}
```

## Unittest Checking

为了检测已有bootcamp和新增sadnbox是否存在问题，可以使用unittests核查每个bootcamp的正确性。

unittest具体的用法参考[unittests](unittests/README_zh.md)。

## RL training

我们基于Xtuner提供了进行bootcamp-RL的训练框架, 具体的用法参考[xpuyu_usage](xpuyu_usage/README.md)。
