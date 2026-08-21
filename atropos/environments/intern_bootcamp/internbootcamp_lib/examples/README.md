# Examples

## bootcamp Data Generation

For generating data with existing bootcamp, we provide script to generate bootcamp data with the configs of the bootcamps under [examples/pipelines](examples/pipelines).
**Please refer to [pipelines](pipelines/README.md) for specific usages.**

In the following, we introduce how to add a new bootcamp and generate data samples with the bootcamps.

### 1. Constructing A New bootcamp

The naming of bootcamps allows either `underlined_naming` or `CamelCase`.

- underlined_naming

First, implement a class under `internbootcamp/bootcamp`, with only first letter capitalized and `bootcamp` appended, for instance, `Bbehshuffobjectbootcamp`. Then, implement the `__init__` method and three interfaces: `case_generator`,`prompt_func`, and `verify_score`. Then, import this class in `internbootcamp/bootcamp/__init__.py`.

For configuration, you can add a configuration file under `examples/pipelines/puzzle_configs` with underlined naming, such as `bbeh_shuff_object_train.json` and `bbeh_shuff_object_test.json`, then finish the configuration in the json files. During the data generation process, parameters in the configuration files will be feed into the `__init__` method of the bootcamp to create bootcamp instances.

Notice, unreasonable configurations may lead to failure or execution errors when using the bootcamp.

- CamelCase

First, implement a class under `internbootcamp/bootcamp`, with `CamelCase` naming ends with bootcamp, for instance, `BbehGeometricShapesbootcamp`. Then, implement the `__init__` method and three interfaces: `case_generator`,`prompt_func`, and `verify_score`. Then, import this class in `internbootcamp/bootcamp/__init__.py`.

For configuration, you can add a configuration file under `examples/pipelines/puzzle_configs`, such as `BbehGeometricShapes_train.json` and `BbehGeometricShapes_test.json`, then finish the configuration in the json files. During the data generation process, parameters in the configuration files will be feed into the `__init__` method of the bootcamp to create bootcamp instances.

Notice, unreasonable configurations may lead to failure or execution errors when using the bootcamp.


### 2. Data Config

To generate data from all bootcamps together, you can run

`python examples/pipelines/quickgen_data_configs.py`

to merge the bootcamp configs.


### 3. Generating Data

Run

`bash examples/pipelines/run_pipeline.sh` 

to generate all bootcamp data automatically. Each sample contains **"prompt", "ground_truth" and "data_source"**. The results will be under `examples/bootcamp_generator_outputs`.

**Example data**:
```json
{{"data_source": "Aquarium", "prompt": "You are to solve an Aquarium puzzle. The puzzle is played on a grid divided into aquarium regions. Each aquarium must be filled up to a horizontal level such that all its columns are filled to the same level. Here are the details:\n\n- The grid has 5 rows and 3 columns.\n\n- Aquarium regions are as follows (each number represents the aquarium ID for that cell):\nRow 0: 0 1 2\nRow 1: 0 1 2\nRow 2: 0 1 2\nRow 3: 0 1 2\nRow 4: 0 1 2\n\n- Each row has a clue on the right indicating the total filled cells in that row. The row clues are: [3, 2, 2, 1, 1].\n\n- Each column has a clue at the bottom indicating the total filled cells in that column. The column clues are: [5, 1, 3].\n\nYour task is to determine the water level for each aquarium. The water level is the highest row number filled (0-based from the bottom). Each aquarium's water level must be such that all its columns are filled up to this level.\n\nProvide your answer as a list of integers in column order (from left to right), where each integer is the water level for the corresponding column's aquarium. Enclose your answer within [answer] and [/answer]. For example, if the solution is levels 2, 1, 0 for columns 0, 1, 2, write:\n[answer]2 1 0[/answer]", "ground_truth": {"regions": [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]], "row_clues": [3, 2, 2, 1, 1], "col_clues": [5, 1, 3]}}}
```

## Unittest Checking

To help debugs potential errors in the bootcamps, we can use `unittest`. The specific usage can be found in [unittests](unittests/README.md).

## RL training

We implement RL training with XTuner/XPuyu, see [xpuyu_usage](xpuyu_usage/README.md).
