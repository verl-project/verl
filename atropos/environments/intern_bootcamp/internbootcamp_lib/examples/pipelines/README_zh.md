# Pipeline Usage

## 配置文件

**puzzle_configs**: 用户在该目录下配置puzzle任务的传入参数。具体来说，通过为每一个puzzle配置一个json文件来控制合成数据的分布，并且可以同时配置train和test（也可以将train和test的配置设置成一样，等同于两者服从同一个分布）。

**data_configs**: 用户运行run pipeline生成数据时所需的配置文件
- 可以手动新增设置 cipher 和各种puzzle任务
- 可以基于 `examples/pipelines/puzzle_configs/`来运行 `examples/pipelines/data_config_gen.py` 脚本，在data_configs路径下自动生成 **data_config_train.jsonl** 和 **data_config_test.jsonl**

futoshiki任务的配置示例：
```json
{"bootcamp_name": "futoshiki", "sample_number": 100, "config_file": "futoshiki", "bootcamp_cls_name": "Futoshikibootcamp"}
```
其中sample_number表示生成的数量，config_file表示配置文件名，bootcamp_cls_name表示bootcamp的类名。 

## 运行脚本

**run_pipeline.sh**: 包含了 cipher 和各种puzzle任务的统一generate pipeline，通过执行该脚本获取 generator puzzle data

cipher和其他puzzle的执行逻辑略有区别，因此会从两个不同的generator file（`data_generator.py`和 `cipher_data_generator.py`）中启动

## 快速启动
1. 配置puzzle_configs下的bootcamp参数，执行下面命令生成数据合成时的配置文件：
```bash
python examples/pipelines/quickgen_data_configs.py
```
其中，可以调整脚本中的train_sample_number和test_sample_number参数来控制训练集和测试集的样本数量。

2. 执行 `bash examples/pipelines/run_pipline.sh`，generator出来的数据保存在 **examples/bootcamp_generator_outputs**路径下。